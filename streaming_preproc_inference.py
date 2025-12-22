"""
True Streaming CoreML Diarization

This script implements true streaming inference:
    Audio chunks → CoreML Preprocessor → Feature Buffer → CoreML Main Model → Predictions

Audio is processed incrementally, features are accumulated with proper context handling.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import coremltools as ct
import librosa
import argparse
import math

# Import NeMo for state management (streaming_update) only
from nemo.collections.asr.models import SortformerEncLabelModel


# ============================================================
# Configuration for Sortformer16.mlpackage
# ============================================================
CONFIG = {
    'chunk_len': 4,                  # Diarization chunk length
    'chunk_right_context': 1,        # Right context chunks
    'chunk_left_context': 2,         # Left context chunks
    'fifo_len': 63,
    'spkcache_len': 63,
    'spkcache_update_period': 50,
    'subsampling_factor': 8,
    'sample_rate': 16000,
    
    # Derived values
    'chunk_frames': 56,              # (4+2+1)*8 = 56 feature frames for CoreML input
    'spkcache_input_len': 63,
    'fifo_input_len': 63,
    
    # Preprocessor settings
    'preproc_audio_samples': 9200,   # CoreML preprocessor fixed input size
    'mel_window': 400,               # 25ms @ 16kHz
    'mel_stride': 160,               # 10ms @ 16kHz
}


def run_true_streaming(nemo_model, preproc_model, main_model, audio_path, config):
    """
    True streaming inference: audio chunks → preproc → main model.
    
    Strategy:
    1. Process audio in chunks through CoreML preprocessor
    2. Accumulate features
    3. When enough features for a diarization chunk (with context), run main model
    """
    modules = nemo_model.sortformer_modules
    subsampling_factor = config['subsampling_factor']
    
    # Load full audio (simulating microphone input)
    full_audio, sr = librosa.load(audio_path, sr=config['sample_rate'], mono=True)
    total_samples = len(full_audio)
    
    print(f"Total audio samples: {total_samples}")
    
    # Preprocessing parameters
    mel_window = config['mel_window']
    mel_stride = config['mel_stride']
    preproc_len = config['preproc_audio_samples']
    
    # Audio hop for preprocessor (to avoid overlap in features)
    audio_hop = preproc_len - mel_window  # 8800 samples
    
    # Feature accumulator
    all_features = []
    audio_offset = 0
    preproc_chunk_idx = 0
    
    # Step 1: Process all audio through preprocessor to get features
    print("Step 1: Extracting features via CoreML preprocessor...")
    while audio_offset < total_samples:
        # Get audio chunk
        chunk_end = min(audio_offset + preproc_len, total_samples)
        audio_chunk = full_audio[audio_offset:chunk_end]
        actual_samples = len(audio_chunk)
        
        # Pad if needed
        if actual_samples < preproc_len:
            audio_chunk = np.pad(audio_chunk, (0, preproc_len - actual_samples))
        
        # Run preprocessor
        preproc_inputs = {
            "audio_signal": audio_chunk.reshape(1, -1).astype(np.float32),
            "length": np.array([actual_samples], dtype=np.int32)
        }
        
        preproc_out = preproc_model.predict(preproc_inputs)
        feat_chunk = np.array(preproc_out["features"])  # [1, 128, frames]
        feat_len = int(preproc_out["feature_lengths"][0])
        
        # Extract valid features and handle overlap
        if preproc_chunk_idx == 0:
            # First chunk: keep all
            valid_feats = feat_chunk[:, :, :feat_len]
        else:
            # Subsequent: skip overlap frames
            overlap_frames = (mel_window - mel_stride) // mel_stride + 1  # ~2-3 frames
            valid_feats = feat_chunk[:, :, overlap_frames:feat_len]
        
        all_features.append(valid_feats)
        
        audio_offset += audio_hop
        preproc_chunk_idx += 1
        
        print(f"\r  Processed audio chunk {preproc_chunk_idx}, features so far: {sum(f.shape[2] for f in all_features)}", end='')
    
    print()
    
    # Concatenate all features
    full_features = np.concatenate(all_features, axis=2)  # [1, 128, total_frames]
    processed_signal = torch.from_numpy(full_features).float()
    processed_signal_length = torch.tensor([full_features.shape[2]], dtype=torch.long)
    
    print(f"Total features extracted: {processed_signal.shape}")
    
    # Step 2: Run diarization streaming loop (same as NeMo reference)
    print("Step 2: Running diarization streaming...")
    
    state = modules.init_streaming_state(batch_size=1, device='cpu')
    all_preds = []
    
    feat_len = processed_signal.shape[2]
    chunk_len = modules.chunk_len
    left_ctx = modules.chunk_left_context
    right_ctx = modules.chunk_right_context
    
    stt_feat, end_feat, chunk_idx = 0, 0, 0
    
    while end_feat < feat_len:
        left_offset = min(left_ctx * subsampling_factor, stt_feat)
        end_feat = min(stt_feat + chunk_len * subsampling_factor, feat_len)
        right_offset = min(right_ctx * subsampling_factor, feat_len - end_feat)
        
        # Extract chunk with context
        chunk_feat = processed_signal[:, :, stt_feat - left_offset : end_feat + right_offset]
        actual_len = chunk_feat.shape[2]
        
        # Transpose to [B, T, D]
        chunk_t = chunk_feat.transpose(1, 2)
        
        # Pad to fixed size
        if actual_len < config['chunk_frames']:
            pad_len = config['chunk_frames'] - actual_len
            chunk_in = torch.nn.functional.pad(chunk_t, (0, 0, 0, pad_len))
        else:
            chunk_in = chunk_t[:, :config['chunk_frames'], :]
        
        # State preparation
        curr_spk_len = state.spkcache.shape[1]
        curr_fifo_len = state.fifo.shape[1]
        
        current_spkcache = state.spkcache
        if curr_spk_len < config['spkcache_input_len']:
            current_spkcache = torch.nn.functional.pad(
                current_spkcache, (0, 0, 0, config['spkcache_input_len'] - curr_spk_len)
            )
        elif curr_spk_len > config['spkcache_input_len']:
            current_spkcache = current_spkcache[:, :config['spkcache_input_len'], :]
        
        current_fifo = state.fifo
        if curr_fifo_len < config['fifo_input_len']:
            current_fifo = torch.nn.functional.pad(
                current_fifo, (0, 0, 0, config['fifo_input_len'] - curr_fifo_len)
            )
        elif curr_fifo_len > config['fifo_input_len']:
            current_fifo = current_fifo[:, :config['fifo_input_len'], :]
        
        # CoreML inference
        coreml_inputs = {
            "chunk": chunk_in.numpy().astype(np.float32),
            "chunk_lengths": np.array([actual_len], dtype=np.int32),
            "spkcache": current_spkcache.numpy().astype(np.float32),
            "spkcache_lengths": np.array([curr_spk_len], dtype=np.int32),
            "fifo": current_fifo.numpy().astype(np.float32),
            "fifo_lengths": np.array([curr_fifo_len], dtype=np.int32)
        }
        
        coreml_out = main_model.predict(coreml_inputs)
        
        pred_logits = torch.from_numpy(coreml_out["preds"])
        chunk_embs = torch.from_numpy(coreml_out["chunk_embs"])
        chunk_emb_len = int(coreml_out["chunk_emb_lengths"][0])
        
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
        stt_feat = end_feat
        chunk_idx += 1
        
        print(f"\r  Diarization chunk {chunk_idx}", end='')
    
    print()
    
    if len(all_preds) > 0:
        return torch.cat(all_preds, dim=1)
    return None


def run_reference(nemo_model, main_model, audio_path, config):
    """
    Reference implementation using NeMo preprocessing.
    """
    modules = nemo_model.sortformer_modules
    subsampling_factor = modules.subsampling_factor
    
    # Load full audio
    full_audio, _ = librosa.load(audio_path, sr=config['sample_rate'], mono=True)
    audio_tensor = torch.from_numpy(full_audio).unsqueeze(0).float()
    audio_length = torch.tensor([len(full_audio)], dtype=torch.long)
    
    # Extract features using NeMo preprocessor
    with torch.no_grad():
        processed_signal, processed_signal_length = nemo_model.process_signal(
            audio_signal=audio_tensor, audio_signal_length=audio_length
        )
    processed_signal = processed_signal[:, :, :processed_signal_length.max()]
    
    print(f"NeMo Preproc: features shape = {processed_signal.shape}")
    
    # Streaming loop
    state = modules.init_streaming_state(batch_size=1, device='cpu')
    all_preds = []
    
    feat_len = processed_signal.shape[2]
    chunk_len = modules.chunk_len
    left_ctx = modules.chunk_left_context
    right_ctx = modules.chunk_right_context
    
    stt_feat, end_feat, chunk_idx = 0, 0, 0
    
    while end_feat < feat_len:
        left_offset = min(left_ctx * subsampling_factor, stt_feat)
        end_feat = min(stt_feat + chunk_len * subsampling_factor, feat_len)
        right_offset = min(right_ctx * subsampling_factor, feat_len - end_feat)
        
        chunk_feat = processed_signal[:, :, stt_feat - left_offset : end_feat + right_offset]
        actual_len = chunk_feat.shape[2]
        
        chunk_t = chunk_feat.transpose(1, 2)
        
        if actual_len < config['chunk_frames']:
            pad_len = config['chunk_frames'] - actual_len
            chunk_in = torch.nn.functional.pad(chunk_t, (0, 0, 0, pad_len))
        else:
            chunk_in = chunk_t[:, :config['chunk_frames'], :]
        
        curr_spk_len = state.spkcache.shape[1]
        curr_fifo_len = state.fifo.shape[1]
        
        current_spkcache = state.spkcache
        if curr_spk_len < config['spkcache_input_len']:
            current_spkcache = torch.nn.functional.pad(
                current_spkcache, (0, 0, 0, config['spkcache_input_len'] - curr_spk_len)
            )
        elif curr_spk_len > config['spkcache_input_len']:
            current_spkcache = current_spkcache[:, :config['spkcache_input_len'], :]
        
        current_fifo = state.fifo
        if curr_fifo_len < config['fifo_input_len']:
            current_fifo = torch.nn.functional.pad(
                current_fifo, (0, 0, 0, config['fifo_input_len'] - curr_fifo_len)
            )
        elif curr_fifo_len > config['fifo_input_len']:
            current_fifo = current_fifo[:, :config['fifo_input_len'], :]
        
        coreml_inputs = {
            "chunk": chunk_in.numpy().astype(np.float32),
            "chunk_lengths": np.array([actual_len], dtype=np.int32),
            "spkcache": current_spkcache.numpy().astype(np.float32),
            "spkcache_lengths": np.array([curr_spk_len], dtype=np.int32),
            "fifo": current_fifo.numpy().astype(np.float32),
            "fifo_lengths": np.array([curr_fifo_len], dtype=np.int32)
        }
        
        coreml_out = main_model.predict(coreml_inputs)
        
        pred_logits = torch.from_numpy(coreml_out["preds"])
        chunk_embs = torch.from_numpy(coreml_out["chunk_embs"])
        chunk_emb_len = int(coreml_out["chunk_emb_lengths"][0])
        
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
        stt_feat = end_feat
        chunk_idx += 1
    
    if len(all_preds) > 0:
        return torch.cat(all_preds, dim=1)
    return None


def validate(model_name, coreml_dir, audio_path):
    """
    Validate true streaming against NeMo preprocessing.
    """
    print("=" * 70)
    print("VALIDATION: True Streaming vs NeMo Preprocessing")
    print("=" * 70)
    
    # Load NeMo model
    print(f"\nLoading NeMo Model: {model_name}")
    nemo_model = SortformerEncLabelModel.from_pretrained(model_name, map_location="cpu")
    nemo_model.eval()
    
    # Apply config
    modules = nemo_model.sortformer_modules
    modules.chunk_len = CONFIG['chunk_len']
    modules.chunk_right_context = CONFIG['chunk_right_context']
    modules.chunk_left_context = CONFIG['chunk_left_context']
    modules.fifo_len = CONFIG['fifo_len']
    modules.spkcache_len = CONFIG['spkcache_len']
    modules.spkcache_update_period = CONFIG['spkcache_update_period']
    
    # Disable dither and pad_to
    if hasattr(nemo_model.preprocessor, 'featurizer'):
        nemo_model.preprocessor.featurizer.dither = 0.0
        nemo_model.preprocessor.featurizer.pad_to = 0
    
    print(f"Config: chunk_len={modules.chunk_len}, left_ctx={modules.chunk_left_context}, "
          f"right_ctx={modules.chunk_right_context}")
    
    # Load CoreML models
    print(f"Loading CoreML Models from {coreml_dir}...")
    preproc_model = ct.models.MLModel(
        os.path.join(coreml_dir, "SortformerPreprocessor.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY
    )
    main_model = ct.models.MLModel(
        os.path.join(coreml_dir, "Sortformer16.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY
    )
    
    # Reference
    print("\n" + "=" * 70)
    print("TEST 1: NeMo Preprocessing + CoreML Inference (Reference)")
    print("=" * 70)
    
    ref_probs = run_reference(nemo_model, main_model, audio_path, CONFIG)
    if ref_probs is not None:
        ref_probs_np = ref_probs.squeeze(0).detach().cpu().numpy()
        print(f"Reference Probs Shape: {ref_probs_np.shape}")
    else:
        print("Reference inference failed!")
        return
    
    # True streaming
    print("\n" + "=" * 70)
    print("TEST 2: True Streaming (Audio → CoreML Preproc → CoreML Main)")
    print("=" * 70)
    
    streaming_probs = run_true_streaming(nemo_model, preproc_model, main_model, audio_path, CONFIG)
    
    if streaming_probs is not None:
        streaming_probs_np = streaming_probs.squeeze(0).detach().cpu().numpy()
        print(f"Streaming Probs Shape: {streaming_probs_np.shape}")
        
        # Compare
        min_len = min(ref_probs_np.shape[0], streaming_probs_np.shape[0])
        diff = np.abs(ref_probs_np[:min_len] - streaming_probs_np[:min_len])
        print(f"\nLength: ref={ref_probs_np.shape[0]}, streaming={streaming_probs_np.shape[0]}")
        print(f"Mean Absolute Error: {np.mean(diff):.8f}")
        print(f"Max Absolute Error: {np.max(diff):.8f}")
        
        if np.max(diff) < 0.01:
            print("\n✅ SUCCESS: True streaming matches reference!")
        else:
            print("\n⚠️  Errors exceed tolerance")
    else:
        print("True streaming inference produced no output!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    parser.add_argument("--coreml_dir", default="coreml_models")
    parser.add_argument("--audio_path", default="audio.wav")
    args = parser.parse_args()
    
    validate(args.model_name, args.coreml_dir, args.audio_path)
