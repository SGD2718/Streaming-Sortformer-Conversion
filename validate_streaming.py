import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import math
import librosa

# Import NeMo
from nemo.collections.asr.models import SortformerEncLabelModel


def streaming_feat_loader(modules, feat_seq, feat_seq_length, feat_seq_offset):
    """
    Load a chunk of feature sequence for streaming inference.
    Adapted from NeMo's SortformerModules.streaming_feat_loader
    """
    feat_len = feat_seq.shape[2]
    chunk_len = modules.chunk_len
    subsampling_factor = modules.subsampling_factor
    chunk_left_context = getattr(modules, 'chunk_left_context', 0)
    chunk_right_context = getattr(modules, 'chunk_right_context', 0)

    stt_feat, end_feat, chunk_idx = 0, 0, 0
    while end_feat < feat_len:
        left_offset = min(chunk_left_context * subsampling_factor, stt_feat)
        end_feat = min(stt_feat + chunk_len * subsampling_factor, feat_len)
        right_offset = min(chunk_right_context * subsampling_factor, feat_len - end_feat)
        
        chunk_feat_seq = feat_seq[:, :, stt_feat - left_offset : end_feat + right_offset]
        feat_lengths = (feat_seq_length + feat_seq_offset - stt_feat + left_offset).clamp(
            0, chunk_feat_seq.shape[2]
        )
        feat_lengths = feat_lengths * (feat_seq_offset < end_feat)
        stt_feat = end_feat
        
        chunk_feat_seq_t = torch.transpose(chunk_feat_seq, 1, 2)
        
        yield chunk_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset
        chunk_idx += 1


def run_custom_streaming_inference_forward_streaming_step(nemo_model, audio_path):
    """
    Custom streaming inference using forward_streaming_step directly.
    This should match NeMo's forward_streaming EXACTLY.
    """
    modules = nemo_model.sortformer_modules
    sample_rate = 16000
    
    # Load Audio
    full_audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    total_samples = len(full_audio)
    
    audio_tensor = torch.from_numpy(full_audio).unsqueeze(0).float()
    audio_length = torch.tensor([total_samples], dtype=torch.long)
    
    # === Use process_signal exactly as NeMo does ===
    with torch.no_grad():
        processed_signal, processed_signal_length = nemo_model.process_signal(
            audio_signal=audio_tensor, audio_signal_length=audio_length
        )
    
    # Trim to actual length (same as forward())
    processed_signal = processed_signal[:, :, :processed_signal_length.max()]
    
    # === Initialize streaming state ===
    state = modules.init_streaming_state(batch_size=1, device='cpu')
    
    # === Use streaming_feat_loader to chunk features ===
    batch_size = processed_signal.shape[0]
    processed_signal_offset = torch.zeros((batch_size,), dtype=torch.long)
    
    total_preds = torch.zeros((batch_size, 0, modules.n_spk), device='cpu')
    
    feat_loader = streaming_feat_loader(
        modules=modules,
        feat_seq=processed_signal,
        feat_seq_length=processed_signal_length,
        feat_seq_offset=processed_signal_offset,
    )
    
    for chunk_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in feat_loader:
        # Use forward_streaming_step directly - this is what NeMo's forward_streaming uses internally
        with torch.no_grad():
            state, total_preds = nemo_model.forward_streaming_step(
                processed_signal=chunk_feat_seq_t,
                processed_signal_length=feat_lengths,
                streaming_state=state,
                total_preds=total_preds,
                left_offset=left_offset,
                right_offset=right_offset,
            )
    
    return total_preds


def run_custom_streaming_inference_forward_for_export(nemo_model, audio_path):
    """
    Custom streaming inference using forward_for_export.
    This is for CoreML/ONNX export compatibility.
    """
    modules = nemo_model.sortformer_modules
    subsampling_factor = modules.subsampling_factor
    sample_rate = 16000
    
    # Load Audio
    full_audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    total_samples = len(full_audio)
    
    audio_tensor = torch.from_numpy(full_audio).unsqueeze(0).float()
    audio_length = torch.tensor([total_samples], dtype=torch.long)
    
    # === Use process_signal exactly as NeMo does ===
    with torch.no_grad():
        processed_signal, processed_signal_length = nemo_model.process_signal(
            audio_signal=audio_tensor, audio_signal_length=audio_length
        )
    
    # Trim to actual length (same as forward())
    processed_signal = processed_signal[:, :, :processed_signal_length.max()]
    
    # === Initialize streaming state ===
    state = modules.init_streaming_state(batch_size=1, device='cpu')
    
    # === Use streaming_feat_loader to chunk features ===
    batch_size = processed_signal.shape[0]
    processed_signal_offset = torch.zeros((batch_size,), dtype=torch.long)
    
    all_preds = []
    
    feat_loader = streaming_feat_loader(
        modules=modules,
        feat_seq=processed_signal,
        feat_seq_length=processed_signal_length,
        feat_seq_offset=processed_signal_offset,
    )
    
    for chunk_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in feat_loader:
        chunk_in = chunk_feat_seq_t
        chunk_len_in = feat_lengths.long()
        
        # Get actual lengths from state (for forward_for_export we need to pad but track real lengths)
        curr_spk_len = state.spkcache.shape[1]
        curr_fifo_len = state.fifo.shape[1]
        
        # Prepare SpkCache - Pad to fixed size for export model
        current_spkcache = state.spkcache
        req_spk_len = modules.spkcache_len
        
        if curr_spk_len < req_spk_len:
            pad_len = req_spk_len - curr_spk_len
            current_spkcache = torch.nn.functional.pad(current_spkcache, (0, 0, 0, pad_len))
        elif curr_spk_len > req_spk_len:
            current_spkcache = current_spkcache[:, :req_spk_len, :]

        spkcache_in = current_spkcache
        # Use actual length, not padded length
        spkcache_len_in = torch.tensor([curr_spk_len], dtype=torch.long)
        
        # Prepare FIFO - Pad to fixed size for export model
        current_fifo = state.fifo
        req_fifo_len = modules.fifo_len
        
        if curr_fifo_len < req_fifo_len:
            pad_len = req_fifo_len - curr_fifo_len
            current_fifo = torch.nn.functional.pad(current_fifo, (0, 0, 0, pad_len))
        elif curr_fifo_len > req_fifo_len:
            current_fifo = current_fifo[:, :req_fifo_len, :]
             
        fifo_in = current_fifo
        fifo_len_in = torch.tensor([curr_fifo_len], dtype=torch.long)
        
        with torch.no_grad():
            pred_logits, chunk_embs, emb_lens = nemo_model.forward_for_export(
                chunk=chunk_in,
                chunk_lengths=chunk_len_in,
                spkcache=spkcache_in,
                spkcache_lengths=spkcache_len_in,
                fifo=fifo_in,
                fifo_lengths=fifo_len_in
            )

        lc = round(left_offset / subsampling_factor)
        rc = math.ceil(right_offset / subsampling_factor)
        
        state, chunk_probs = modules.streaming_update(
            streaming_state=state,
            chunk=chunk_embs,
            preds=pred_logits,
            lc=lc,
            rc=rc
        )
        
        all_preds.append(chunk_probs)
        
    if len(all_preds) > 0:
        final_probs = torch.cat(all_preds, dim=1)
        return final_probs
    return None


def compare_results(ref_probs_np, cand_probs_np, ref_label, cand_label):
    """Compare two probability arrays and return error metrics."""
    min_len = min(ref_probs_np.shape[0], cand_probs_np.shape[0])
    ref_slice = ref_probs_np[:min_len, :]
    cand_slice = cand_probs_np[:min_len, :]
    
    diff = np.abs(ref_slice - cand_slice)
    mean_error = np.mean(diff)
    max_error = np.max(diff)
    
    print(f"\n{ref_label} vs {cand_label}:")
    print(f"  Length Match: ref={ref_probs_np.shape[0]}, cand={cand_probs_np.shape[0]}")
    print(f"  Mean Absolute Error: {mean_error:.10f}")
    print(f"  Max Absolute Error:  {max_error:.10f}")
    
    if mean_error < 0.001 and max_error < 0.001:
        print(f"  ✅ SUCCESS: Errors are within tolerance (< 0.001)")
    else:
        print(f"  ❌ FAIL: Errors exceed tolerance (>= 0.001)")
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"     Max error at frame {max_idx[0]}, speaker {max_idx[1]}")
        print(f"     Ref value: {ref_slice[max_idx]:.6f}")
        print(f"     Cand value: {cand_slice[max_idx]:.6f}")
    
    return mean_error, max_error, ref_slice, cand_slice, diff


def validate(model_name, audio_path):
    print("=" * 70)
    print("VALIDATION: Comparing NeMo forward_streaming vs Custom Implementations")
    print("=" * 70)
    
    # Load model once
    print(f"\nLoading NeMo Model: {model_name}")
    nemo_model = SortformerEncLabelModel.from_pretrained(model_name, map_location="cpu")
    nemo_model.eval()
    
    # Overrides for Low Latency (Match CoreML)
    print("Overriding Config to Low Latency (chunk_len=4)...")
    nemo_model.sortformer_modules.chunk_len = 4
    nemo_model.sortformer_modules.chunk_right_context = 1
    nemo_model.sortformer_modules.chunk_left_context = 0
    nemo_model.sortformer_modules.fifo_len = 125
    nemo_model.sortformer_modules.spkcache_len = 125
    nemo_model.sortformer_modules.spkcache_update_period = 63
    
    # === Match diarize() preprocessing ===
    # Disable dither and pad_to (as diarize does in _diarize_on_begin)
    if hasattr(nemo_model.preprocessor, 'featurizer'):
        if hasattr(nemo_model.preprocessor.featurizer, 'dither'):
            original_dither = nemo_model.preprocessor.featurizer.dither
            nemo_model.preprocessor.featurizer.dither = 0.0
            print(f"Disabled dither (was {original_dither})")
        if hasattr(nemo_model.preprocessor.featurizer, 'pad_to'):
            original_pad_to = nemo_model.preprocessor.featurizer.pad_to
            nemo_model.preprocessor.featurizer.pad_to = 0
            print(f"Disabled pad_to (was {original_pad_to})")
    
    # =========================================
    # 1. NeMo Reference using forward_streaming
    # =========================================
    print("\n" + "=" * 70)
    print("TEST 1: NeMo forward_streaming (Reference)")
    print("=" * 70)
    
    sample_rate = 16000
    full_audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio_tensor = torch.from_numpy(full_audio).unsqueeze(0).float()
    audio_length = torch.tensor([len(full_audio)], dtype=torch.long)
    
    with torch.no_grad():
        processed_signal, processed_signal_length = nemo_model.process_signal(
            audio_signal=audio_tensor, audio_signal_length=audio_length
        )
        processed_signal = processed_signal[:, :, :processed_signal_length.max()]
        ref_probs = nemo_model.forward_streaming(processed_signal, processed_signal_length)
    
    ref_probs_np = ref_probs.squeeze(0).detach().cpu().numpy()
    print(f"Reference Probs Shape: {ref_probs_np.shape}")
    
    # =========================================
    # 2. Custom using forward_streaming_step
    # =========================================
    print("\n" + "=" * 70)
    print("TEST 2: Custom Loop using forward_streaming_step")
    print("=" * 70)
    
    cand_step_tensor = run_custom_streaming_inference_forward_streaming_step(nemo_model, audio_path)
    cand_step_np = cand_step_tensor.squeeze(0).detach().cpu().numpy()
    print(f"forward_streaming_step Probs Shape: {cand_step_np.shape}")
    
    mean_err_step, max_err_step, ref_slice, cand_step_slice, diff_step = compare_results(
        ref_probs_np, cand_step_np, 
        "NeMo forward_streaming", "Custom forward_streaming_step"
    )
    
    # =========================================
    # 3. Custom using forward_for_export
    # =========================================
    print("\n" + "=" * 70)
    print("TEST 3: Custom Loop using forward_for_export")
    print("=" * 70)
    
    cand_export_tensor = run_custom_streaming_inference_forward_for_export(nemo_model, audio_path)
    cand_export_np = cand_export_tensor.squeeze(0).detach().cpu().numpy()
    print(f"forward_for_export Probs Shape: {cand_export_np.shape}")
    
    mean_err_export, max_err_export, _, cand_export_slice, diff_export = compare_results(
        ref_probs_np, cand_export_np,
        "NeMo forward_streaming", "Custom forward_for_export"
    )
    
    # =========================================
    # 4. Compare forward_streaming_step vs forward_for_export (to verify padding doesn't cause differences)
    # =========================================
    print("\n" + "=" * 70)
    print("TEST 4: Verify padding doesn't cause differences")
    print("=" * 70)
    
    mean_err_step_vs_export, max_err_step_vs_export, _, _, diff_step_export = compare_results(
        cand_step_np, cand_export_np,
        "forward_streaming_step", "forward_for_export"
    )
    
    # =========================================
    # 5. Summary
    # =========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"forward_streaming_step vs NeMo: Mean={mean_err_step:.10f}, Max={max_err_step:.10f}")
    print(f"forward_for_export vs NeMo:     Mean={mean_err_export:.10f}, Max={max_err_export:.10f}")
    print(f"step vs export (padding test):  Mean={mean_err_step_vs_export:.10f}, Max={max_err_step_vs_export:.10f}")
    
    all_pass = (mean_err_step < 0.001 and max_err_step < 0.001 and
                mean_err_export < 0.001 and max_err_export < 0.001)
    
    if all_pass:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️  SOME TESTS FAILED - see details above")
    
    # =========================================
    # 6. Plot
    # =========================================
    print("\nGenerating Comparison Plot...")
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    
    # Reference
    sns.heatmap(ref_slice.T, ax=axes[0], cmap="viridis", vmin=0, vmax=1, cbar=True)
    axes[0].set_title(f"NeMo forward_streaming (Reference). Shape: {ref_slice.shape}")
    axes[0].set_ylabel("Speaker")
    
    # forward_streaming_step
    sns.heatmap(cand_step_slice.T, ax=axes[1], cmap="viridis", vmin=0, vmax=1, cbar=True)
    axes[1].set_title(f"Custom forward_streaming_step. Mean Err={mean_err_step:.6f}")
    axes[1].set_ylabel("Speaker")
    
    # forward_for_export
    sns.heatmap(cand_export_slice.T, ax=axes[2], cmap="viridis", vmin=0, vmax=1, cbar=True)
    axes[2].set_title(f"Custom forward_for_export. Mean Err={mean_err_export:.6f}")
    axes[2].set_ylabel("Speaker")
    
    # Difference (export vs NeMo)
    sns.heatmap(diff_export.T, ax=axes[3], cmap="Reds", vmin=0, vmax=0.1, cbar=True)
    axes[3].set_title(f"Difference (forward_for_export vs NeMo). Max={max_err_export:.6f}")
    axes[3].set_ylabel("Speaker")
    axes[3].set_xlabel("Time Frames")
    
    plt.tight_layout()
    out_file = "validation_heatmap.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")
    
    return {
        'step_mean': mean_err_step, 'step_max': max_err_step,
        'export_mean': mean_err_export, 'export_max': max_err_export,
        'step_vs_export_mean': mean_err_step_vs_export, 'step_vs_export_max': max_err_step_vs_export,
    }


if __name__ == "__main__":
    model_name = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    audio_path = "audio.wav"
    validate(model_name, audio_path)
