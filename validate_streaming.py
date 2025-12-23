import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import librosa
import coremltools as ct
from itertools import permutations

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
    print(f"left context: {chunk_left_context}")
    print(f"right context: {chunk_right_context}")

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


def run_coreml_streaming_inference(nemo_model, pre_encode_model, head_model, audio_path, coreml_config):
    """
    Streaming inference using the CoreML model.
    """
    modules = nemo_model.sortformer_modules
    subsampling_factor = modules.subsampling_factor
    sample_rate = 16000
    
    COREML_CHUNK_FRAMES = coreml_config['chunk_frames']
    COREML_SPKCACHE_LEN = coreml_config['spkcache_len']
    COREML_FIFO_LEN = coreml_config['fifo_len']
    
    # Load Audio
    full_audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio_tensor = torch.from_numpy(full_audio).unsqueeze(0).float()
    audio_length = torch.tensor([len(full_audio)], dtype=torch.long)
    
    with torch.no_grad():
        processed_signal, processed_signal_length = nemo_model.process_signal(
            audio_signal=audio_tensor, audio_signal_length=audio_length
        )
    processed_signal = processed_signal[:, :, :processed_signal_length.max()]
    
    # Initialize streaming state
    state = modules.init_streaming_state(batch_size=1, device='cpu')
    
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
        # Pad chunk to fixed size
        chunk_actual_len = chunk_feat_seq_t.shape[1]
        if chunk_actual_len < COREML_CHUNK_FRAMES:
            pad_len = COREML_CHUNK_FRAMES - chunk_actual_len
            chunk_in = torch.nn.functional.pad(chunk_feat_seq_t, (0, 0, 0, pad_len))
        else:
            chunk_in = chunk_feat_seq_t[:, :COREML_CHUNK_FRAMES, :]
        chunk_len_in = feat_lengths.long()
        
        curr_spk_len = state.spkcache.shape[1]
        curr_fifo_len = state.fifo.shape[1]
        
        # Prepare SpkCache
        current_spkcache = state.spkcache
        if curr_spk_len < COREML_SPKCACHE_LEN:
            pad_len = COREML_SPKCACHE_LEN - curr_spk_len
            current_spkcache = torch.nn.functional.pad(current_spkcache, (0, 0, 0, pad_len))
        elif curr_spk_len > COREML_SPKCACHE_LEN:
            current_spkcache = current_spkcache[:, :COREML_SPKCACHE_LEN, :]
        spkcache_in = current_spkcache
        spkcache_len_in = torch.tensor([curr_spk_len], dtype=torch.long)
        
        # Prepare FIFO
        current_fifo = state.fifo
        if curr_fifo_len < COREML_FIFO_LEN:
            pad_len = COREML_FIFO_LEN - curr_fifo_len
            current_fifo = torch.nn.functional.pad(current_fifo, (0, 0, 0, pad_len))
        elif curr_fifo_len > COREML_FIFO_LEN:
            current_fifo = current_fifo[:, :COREML_FIFO_LEN, :]
        fifo_in = current_fifo
        fifo_len_in = torch.tensor([curr_fifo_len], dtype=torch.long)
        
        # Run CoreML Model
        coreml_inputs = {
            "chunk": chunk_in.numpy().astype(np.float32),
            "chunk_lengths": chunk_len_in.numpy().astype(np.int32),
            "spkcache": spkcache_in.numpy().astype(np.float32),
            "spkcache_lengths": spkcache_len_in.numpy().astype(np.int32),
            "fifo": fifo_in.numpy().astype(np.float32),
            "fifo_lengths": fifo_len_in.numpy().astype(np.int32)
        }
        
        pre_encode_out = pre_encode_model.predict(coreml_inputs)
        coreml_out = head_model.predict(pre_encode_out)

        pred_logits = torch.from_numpy(coreml_out["speaker_preds"])
        chunk_embs = torch.from_numpy(coreml_out["chunk_pre_encoder_embs"])
        chunk_emb_len = int(coreml_out["chunk_pre_encoder_lengths"][0])
        
        # Trim chunk_embs to actual length (drop padded frames)
        chunk_embs = chunk_embs[:, :chunk_emb_len, :]

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


def validate(model_name, coreml_dir, audio_path):
    print("=" * 70)
    print("VALIDATION: Comparing nemo_model.diarize vs CoreML Streaming")
    print("=" * 70)
    
    # Load NeMo model
    print(f"\nLoading NeMo Model: {model_name}")
    nemo_model = SortformerEncLabelModel.from_pretrained(model_name, map_location="cpu")
    nemo_model.eval()
    
    # CoreML export configuration
    COREML_CONFIG = {
        'chunk_len': 6,
        'chunk_right_context': 1,
        'chunk_left_context': 1,
        'fifo_len': 40,
        'spkcache_len': 120,
        'spkcache_update_period': 30,
        'chunk_frames': 64,  # chunk_len * subsampling_factor
    }
    
    # Apply config to modules
    modules = nemo_model.sortformer_modules
    modules.chunk_len = COREML_CONFIG['chunk_len']
    modules.chunk_right_context = COREML_CONFIG['chunk_right_context']
    modules.chunk_left_context = COREML_CONFIG['chunk_left_context']
    modules.fifo_len = COREML_CONFIG['fifo_len']
    modules.spkcache_len = COREML_CONFIG['spkcache_len']
    modules.spkcache_update_period = COREML_CONFIG['spkcache_update_period']
    
    print(f"Config: chunk_len={modules.chunk_len}, fifo={modules.fifo_len}, spkcache={modules.spkcache_len}")
    
    # Disable dither and pad_to (as diarize does)
    if hasattr(nemo_model.preprocessor, 'featurizer'):
        if hasattr(nemo_model.preprocessor.featurizer, 'dither'):
            nemo_model.preprocessor.featurizer.dither = 0.0
        if hasattr(nemo_model.preprocessor.featurizer, 'pad_to'):
            nemo_model.preprocessor.featurizer.pad_to = 0
    
    # Load CoreML model
    print(f"Loading CoreML Model from {coreml_dir}...")
    head_model = ct.models.MLModel(
        os.path.join(coreml_dir, "Pipeline_Head.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY
    )

    pre_encode_model = ct.models.MLModel(
        os.path.join(coreml_dir, "Pipeline_PreEncoder.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY
    )
    
    # =========================================
    # 1. NeMo Reference using forward_streaming
    # =========================================
    print("\n" + "=" * 70)
    print("TEST 1: NeMo forward_streaming (Reference)")
    print("=" * 70)
    
    try:
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
        print(f"Reference (forward_streaming) Probs Shape: {ref_probs_np.shape}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error running NeMo forward_streaming: {e}")
        return
    
    # =========================================
    # 2. CoreML Streaming Inference
    # =========================================
    print("\n" + "=" * 70)
    print("TEST 2: CoreML Streaming Inference")
    print("=" * 70)
    
    cand_probs_tensor = run_coreml_streaming_inference(
        nemo_model, pre_encode_model, head_model, audio_path, COREML_CONFIG
    )
    cand_probs_np = cand_probs_tensor.squeeze(0).detach().cpu().numpy()
    print(f"CoreML Streaming Probs Shape: {cand_probs_np.shape}")
    
    # =========================================
    # 3. Compare with Permutation Invariance
    # =========================================
    print("\n" + "=" * 70)
    print("COMPARISON (Permutation Invariant)")
    print("=" * 70)
    
    min_len = min(ref_probs_np.shape[0], cand_probs_np.shape[0])
    ref_slice = ref_probs_np[:min_len, :]
    cand_slice = cand_probs_np[:min_len, :]
    
    # Sortformer speakers may be in different order - find best permutation
    n_spk = 4
    best_mse = float('inf')
    best_perm = None
    best_cand_permuted = None
    
    for p in permutations(range(n_spk)):
        cand_permuted = cand_slice[:, p]
        mse = np.mean((ref_slice - cand_permuted)**2)
        if mse < best_mse:
            best_mse = mse
            best_perm = p
            best_cand_permuted = cand_permuted
    
    diff = np.abs(ref_slice - best_cand_permuted)
    mean_error = np.mean(diff)
    max_error = np.max(diff)
    
    print(f"Best Speaker Permutation: {best_perm}")
    print(f"Length Match: ref={ref_probs_np.shape[0]}, cand={cand_probs_np.shape[0]}")
    print(f"Mean Absolute Error: {mean_error:.8f}")
    print(f"Max Absolute Error:  {max_error:.8f}")
    
    if mean_error < 0.001 and max_error < 0.001:
        print("\n✅ SUCCESS: Errors are within tolerance (< 0.001)")
    else:
        print("\n⚠️  Errors exceed tolerance (>= 0.001)")
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"   Max error at frame {max_idx[0]}, speaker {max_idx[1]}")
        print(f"   Ref value: {ref_slice[max_idx]:.6f}")
        print(f"   Cand value: {best_cand_permuted[max_idx]:.6f}")
    
    # =========================================
    # 4. Plot: NeMo diarize, CoreML, and Error
    # =========================================
    print("\nGenerating Comparison Plot...")
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Reference (NeMo diarize)
    sns.heatmap(ref_slice.T, ax=axes[0], cmap="viridis", vmin=0, vmax=1, cbar=True)
    axes[0].set_title(f"NeMo diarize() (Reference). Shape: {ref_slice.shape}")
    axes[0].set_ylabel("Speaker")
    
    # CoreML (with best permutation)
    sns.heatmap(best_cand_permuted.T, ax=axes[1], cmap="viridis", vmin=0, vmax=1, cbar=True)
    axes[1].set_title(f"CoreML Streaming (Perm: {best_perm}). Shape: {best_cand_permuted.shape}")
    axes[1].set_ylabel("Speaker")
    
    # Absolute Difference
    sns.heatmap(diff.T, ax=axes[2], cmap="Reds", vmin=0, vmax=0.1, cbar=True)
    axes[2].set_title(f"Absolute Error. Mean={mean_error:.6f}, Max={max_error:.6f}")
    axes[2].set_ylabel("Speaker")
    axes[2].set_xlabel("Time Frames")
    
    plt.tight_layout()
    out_file = "validation_heatmap.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")
    
    return {
        'mean_error': mean_error,
        'max_error': max_error,
        'best_perm': best_perm,
    }


if __name__ == "__main__":
    model_name = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    coreml_dir = "coreml_models"
    audio_path = "audio.wav"
    validate(model_name, coreml_dir, audio_path)
