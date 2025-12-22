import torch
import numpy as np
import coremltools as ct
import librosa
import argparse
import os
import sys
import math

# Import NeMo components for State Logic
try:
    from nemo.collections.asr.models import SortformerEncLabelModel
    # Try importing SortformerModules directly for type hints if needed, but we can access via model instance
    from nemo.collections.asr.modules.sortformer_modules import SortformerModules
except ImportError as e:
    print(f"Error importing NeMo: {e}")
    sys.exit(1)


def streaming_feat_loader(modules, feat_seq, feat_seq_length, feat_seq_offset):
    """
    Load a chunk of feature sequence for streaming inference.
    Adapted from NeMo's SortformerModules.streaming_feat_loader
    
    Args:
        modules: SortformerModules instance with chunk_len, subsampling_factor, 
                 chunk_left_context, chunk_right_context
        feat_seq (torch.Tensor): Tensor containing feature sequence
            Shape: (batch_size, feat_dim, feat frame count)
        feat_seq_length (torch.Tensor): Tensor containing feature sequence lengths
            Shape: (batch_size,)
        feat_seq_offset (torch.Tensor): Tensor containing feature sequence offsets
            Shape: (batch_size,)

    Yields:
        chunk_idx (int): Index of the current chunk
        chunk_feat_seq (torch.Tensor): Tensor containing the chunk of feature sequence
            Shape: (batch_size, feat frame count, feat_dim)  # Transposed!
        feat_lengths (torch.Tensor): Tensor containing lengths of the chunk of feature sequence
            Shape: (batch_size,)
        left_offset (int): Left context offset in feature frames
        right_offset (int): Right context offset in feature frames
    """
    feat_len = feat_seq.shape[2]
    chunk_len = modules.chunk_len
    subsampling_factor = modules.subsampling_factor
    chunk_left_context = getattr(modules, 'chunk_left_context', 0)
    chunk_right_context = getattr(modules, 'chunk_right_context', 0)
    
    num_chunks = math.ceil(feat_len / (chunk_len * subsampling_factor))
    print(f"streaming_feat_loader: feat_len={feat_len}, num_chunks={num_chunks}, "
          f"chunk_len={chunk_len}, subsampling_factor={subsampling_factor}")

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
        
        # Transpose from (batch, feat_dim, frames) to (batch, frames, feat_dim)
        chunk_feat_seq_t = torch.transpose(chunk_feat_seq, 1, 2)
        
        print(f"  chunk_idx: {chunk_idx}, chunk_feat_seq_t shape: {chunk_feat_seq_t.shape}, "
              f"feat_lengths: {feat_lengths}, left_offset: {left_offset}, right_offset: {right_offset}")
        
        yield chunk_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset
        chunk_idx += 1


def run_streaming_inference(model_name, coreml_dir, audio_path):
    print(f"Loading NeMo Model (for Python Streaming Logic): {model_name}")
    if os.path.exists(model_name):
        nemo_model = SortformerEncLabelModel.restore_from(model_name, map_location="cpu")
    else:
        nemo_model = SortformerEncLabelModel.from_pretrained(model_name, map_location="cpu")
    nemo_model.eval()
    modules = nemo_model.sortformer_modules
    
    # --- Override Config to match CoreML Export (Low Latency) ---
    print("Overriding Config (Inference) to match CoreML...")
    modules.chunk_len = 4
    modules.chunk_right_context = 1
    modules.chunk_left_context = 0  # Typically 0 for low-latency streaming
    modules.fifo_len = 125
    modules.spkcache_len = 125
    modules.spkcache_update_period = 63
    
    # CoreML Models (loaded but using NeMo for now)
    print(f"Loading CoreML Models from {coreml_dir}...")
    preproc_model = ct.models.MLModel(os.path.join(coreml_dir, "SortformerPreprocessor.mlpackage"))
    main_model = ct.models.MLModel(os.path.join(coreml_dir, "Sortformer.mlpackage"))
    
    # Config
    chunk_len = modules.chunk_len  # Output frames (e.g., 4 for low latency)
    subsampling_factor = modules.subsampling_factor  # 8
    sample_rate = 16000
    
    print(f"Chunk Config: {chunk_len} output frames (diar), subsampling_factor={subsampling_factor}")

    # Load Audio
    print(f"Loading Audio: {audio_path}")
    full_audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    total_samples = len(full_audio)
    print(f"Total Samples: {total_samples} ({total_samples/sample_rate:.2f}s)")
    
    # === Step 1: Extract features for the ENTIRE audio using preprocessor ===
    # This matches NeMo's approach: process_signal -> forward_streaming
    print("Extracting features for entire audio...")
    audio_tensor = torch.from_numpy(full_audio).unsqueeze(0).float()  # [1, samples]
    audio_length = torch.tensor([total_samples], dtype=torch.long)
    
    with torch.no_grad():
        # processed_signal shape: [batch, feat_dim, feat_frames]
        processed_signal, processed_signal_length = nemo_model.preprocessor(
            input_signal=audio_tensor, length=audio_length
        )
    
    print(f"Processed signal shape: {processed_signal.shape}")  # [1, 128, T]
    print(f"Processed signal length: {processed_signal_length}")
    
    # Trim to actual length
    processed_signal = processed_signal[:, :, :processed_signal_length.max()]
    
    # === Step 2: Initialize streaming state ===
    print("Initializing Streaming State...")
    state = modules.init_streaming_state(batch_size=1, device='cpu')
    
    # === Step 3: Use streaming_feat_loader to chunk features (matches NeMo exactly) ===
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
        # chunk_feat_seq_t: [batch, feat_frames, feat_dim] e.g., [1, 32, 128] for chunk_len=4
        
        # Prepare inputs for forward_for_export
        chunk_in = chunk_feat_seq_t  # [1, T, 128]
        chunk_len_in = feat_lengths.long()  # actual length
        
        # Prepare SpkCache - Pad if needed
        current_spkcache = state.spkcache  # [1, L, 512]
        curr_spk_len = current_spkcache.shape[1]
        req_spk_len = modules.spkcache_len
        
        if curr_spk_len < req_spk_len:
            pad_len = req_spk_len - curr_spk_len
            current_spkcache = torch.nn.functional.pad(current_spkcache, (0, 0, 0, pad_len))
        elif curr_spk_len > req_spk_len:
            current_spkcache = current_spkcache[:, :req_spk_len, :]

        spkcache_in = current_spkcache
        spkcache_len_in = torch.tensor([max(1, curr_spk_len)], dtype=torch.long)
        
        # Prepare FIFO - Pad if needed
        current_fifo = state.fifo
        curr_fifo_len = current_fifo.shape[1]
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

        # Compute lc and rc for streaming_update (in embeddings/diar frames, not feature frames)
        # NeMo does: lc = round(left_offset / encoder.subsampling_factor)
        #            rc = math.ceil(right_offset / encoder.subsampling_factor)
        lc = round(left_offset / subsampling_factor)
        rc = math.ceil(right_offset / subsampling_factor)
        
        # Update state using streaming_update with proper lc/rc
        state, chunk_probs = modules.streaming_update(
            streaming_state=state,
            chunk=chunk_embs,
            preds=pred_logits,
            lc=lc,
            rc=rc
        )
        
        # chunk_probs is the prediction for the current chunk
        all_preds.append(chunk_probs)
        
        print(f"Processed chunk {chunk_idx + 1}, chunk_probs shape: {chunk_probs.shape}", end='\r')
        
    print(f"\nFinished. Total Chunks: {len(all_preds)}")
    if len(all_preds) > 0:
        final_probs = torch.cat(all_preds, dim=1)  # [1, TotalFrames, Spks]
        print(f"Final Predictions Shape: {final_probs.shape}")
        return final_probs
    return None
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    parser.add_argument("--coreml_dir", default="coreml_models")
    parser.add_argument("--audio_path", default="audio.wav")
    args = parser.parse_args()
    
    run_streaming_inference(args.model_name, args.coreml_dir, args.audio_path)
