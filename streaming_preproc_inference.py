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
from config import Config


# ============================================================
# Configuration for Sortformer16.mlpackage
# ============================================================
CONFIG = {
    'chunk_len': Config.chunk_len,                  # Diarization chunk length
    'chunk_right_context': Config.chunk_right_context,        # Right context chunks
    'chunk_left_context': Config.chunk_left_context,         # Left context chunks
    'fifo_len': Config.fifo_len,
    'spkcache_len': Config.spkcache_len,
    'spkcache_update_period': Config.spkcache_update_period,
    'subsampling_factor': Config.subsampling_factor,
    'sample_rate': Config.sample_rate,
    
    # Derived values
    'chunk_frames': Config.chunk_frames,              # (4+2+1)*8 = 56 feature frames for CoreML input
    'spkcache_input_len': Config.spkcache_len,
    'fifo_input_len': Config.fifo_len,
    
    # Preprocessor settings
    'preproc_audio_samples': Config.coreml_audio_samples,   # CoreML preprocessor fixed input size
    'mel_window': Config.mel_window,               # 25ms @ 16kHz
    'mel_stride': Config.mel_stride,               # 10ms @ 16kHz
}


def run_true_streaming(nemo_model, preproc_model, main_model, audio_path, config):
    """
    True streaming inference: audio chunks → preproc → main model.
    
    Strategy: Simulate microphone input by processing audio incrementally.
    - Audio arrives in small chunks (simulating mic capture)
    - Preprocessing runs when enough audio is buffered
    - Main model runs when enough features are available for a diarization chunk
    
    This interleaves preprocessing and diarization, matching real-time behavior.
    
    Key for accuracy: Features are extracted based on absolute audio position
    to match batch NeMo preprocessing within tolerance.
    """
    modules = nemo_model.sortformer_modules
    subsampling_factor = config['subsampling_factor']
    
    # Load full audio (will be fed incrementally to simulate microphone)
    full_audio, sr = librosa.load(audio_path, sr=config['sample_rate'], mono=True)
    total_samples = len(full_audio)
    
    print(f"Total audio samples: {total_samples} ({total_samples / config['sample_rate']:.2f}s)")
    
    # Preprocessing parameters
    mel_window = config['mel_window']  # 400 samples
    mel_stride = config['mel_stride']  # 160 samples
    preproc_len = config['preproc_audio_samples']  # CoreML preprocessor fixed input size (18160)
    
    # Audio hop for sliding window preprocessing
    # Process chunk_len * subsampling_factor = 48 new features per preproc call
    new_features_per_hop = Config.chunk_len * subsampling_factor  # 48 mel frames
    audio_hop = new_features_per_hop * mel_stride  # 7680 samples = 480ms
    
    # Overlap: how much audio is shared between consecutive preprocessing windows
    audio_overlap = preproc_len - audio_hop  # 10480 samples
    
    # The number of features in the overlap region
    # Features are computed starting at mel_stride intervals
    # For audio overlap of 10480 samples, features cover positions 0 to (10480-mel_window)/mel_stride
    overlap_features = (audio_overlap - mel_window) // mel_stride + 1  # 64 features
    
    # Diarization parameters  
    core_frames = Config.chunk_len * subsampling_factor  # 48 frames
    left_ctx = Config.chunk_left_context * subsampling_factor  # 8 frames
    right_ctx = Config.chunk_right_context * subsampling_factor  # 56 frames
    
    # Simulated microphone chunk size (~80ms = Config.frame_duration)
    mic_chunk_samples = int(config['sample_rate'] * Config.frame_duration)  # 1280 samples
    
    # Buffers
    audio_buffer = np.array([], dtype=np.float32)
    feature_buffer = None
    
    # State
    state = modules.init_streaming_state(batch_size=1, device='cpu')
    all_preds = []
    
    # Tracking
    audio_offset = 0  # How much audio has been "captured" from full_audio
    audio_processed = 0  # How much audio has had features extracted (global position)
    first_preproc_done = False
    diar_chunk_idx = 0
    preproc_count = 0
    
    print("\\nTrue Streaming: Audio → Preproc → Main Model (interleaved)")
    print("=" * 60)
    
    while audio_offset < total_samples or len(audio_buffer) >= preproc_len:
        # Step 1: Simulate microphone capture - add small audio chunk
        if audio_offset < total_samples:
            chunk_end = min(audio_offset + mic_chunk_samples, total_samples)
            new_audio = full_audio[audio_offset:chunk_end]
            audio_buffer = np.concatenate([audio_buffer, new_audio])
            audio_offset = chunk_end
        
        # Step 2: Run preprocessor when we have enough audio
        while len(audio_buffer) >= preproc_len:
            # Take preproc_len samples from the beginning
            audio_chunk = audio_buffer[:preproc_len].copy()
            
            # Run preprocessor
            preproc_inputs = {
                "audio_signal": audio_chunk.reshape(1, -1).astype(np.float32),
                "length": np.array([preproc_len], dtype=np.int32)
            }
            
            preproc_out = preproc_model.predict(preproc_inputs)
            feat_chunk = np.array(preproc_out["features"])  # [1, 128, frames]
            feat_len = int(preproc_out["feature_lengths"][0])
            
            if not first_preproc_done:
                # First run: take all features (this covers audio[0:preproc_len])
                valid_feats = feat_chunk[:, :, :feat_len]
                first_preproc_done = True
            else:
                # Subsequent runs: take only the NEW features from the end
                # Each hop adds audio_hop = 7680 samples of new audio
                # This produces audio_hop / mel_stride = 48 new mel frames
                # These are the LAST 48 frames of the output (they correspond to the newest audio)
                new_feat_count = audio_hop // mel_stride  # 48
                valid_feats = feat_chunk[:, :, feat_len - new_feat_count:feat_len]
            
            if feature_buffer is None:
                feature_buffer = valid_feats
            else:
                feature_buffer = np.concatenate([feature_buffer, valid_feats], axis=2)
            
            preproc_count += 1
            audio_processed += len(audio_buffer[:preproc_len]) if not first_preproc_done else audio_hop
            
            # Slide the window forward by audio_hop
            audio_buffer = audio_buffer[audio_hop:]
            
            # Step 3: Run diarization when we have enough features
            while feature_buffer is not None:
                total_features = feature_buffer.shape[2]
                
                # Calculate what we need for this chunk
                chunk_start = diar_chunk_idx * core_frames
                chunk_end = chunk_start + core_frames
                required_features = chunk_end + right_ctx  # Need full right context
                
                if required_features > total_features:
                    break  # Not enough features yet
                
                # Extract chunk with context
                left_offset = min(left_ctx, chunk_start)
                right_offset = right_ctx  # Always full right context since we checked
                
                feat_start = chunk_start - left_offset
                feat_end = chunk_end + right_offset
                
                chunk_feat = feature_buffer[:, :, feat_start:feat_end]
                chunk_feat_tensor = torch.from_numpy(chunk_feat).float()
                actual_len = chunk_feat.shape[2]
                
                # Transpose to [B, T, D]
                chunk_t = chunk_feat_tensor.transpose(1, 2)
                
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
                
                pred_logits = torch.from_numpy(coreml_out["speaker_preds"])
                chunk_embs = torch.from_numpy(coreml_out["chunk_pre_encoder_embs"])
                chunk_emb_len = int(coreml_out["chunk_pre_encoder_lengths"][0])
                
                chunk_embs = chunk_embs[:, :chunk_emb_len, :]
                
                lc = round(left_offset / subsampling_factor)
                rc = round(right_offset / subsampling_factor)
                
                state, chunk_probs = modules.streaming_update(
                    streaming_state=state,
                    chunk=chunk_embs,
                    preds=pred_logits,
                    lc=lc,
                    rc=rc
                )
                
                all_preds.append(chunk_probs)
                diar_chunk_idx += 1
                
                print(f"\\r  Preproc calls: {preproc_count}, Features: {total_features}, "
                      f"Diar chunks: {diar_chunk_idx}", end='')
    
    # Process any remaining audio that didn't fill a complete preprocessing chunk
    # This ensures we extract features for the final portion of audio
    if len(audio_buffer) > 0 and first_preproc_done:
        # The audio_buffer contains samples that overlap with the previous preprocessing window
        # plus new samples that haven't been processed yet.
        # 
        # After each preprocessing call, we keep (preproc_len - audio_hop) = 10480 samples as overlap
        # So if audio_buffer has N samples now:
        # - First 10480 samples overlap with the previous window
        # - Remaining (N - 10480) samples are new
        # 
        # But if N < 10480, all samples are overlap and we shouldn't add more features
        
        overlap_samples = preproc_len - audio_hop  # 10480
        new_samples = len(audio_buffer) - overlap_samples
        
        if new_samples > 0:
            # Pad the audio buffer to preproc_len
            actual_len = len(audio_buffer)
            audio_chunk = np.pad(audio_buffer, (0, preproc_len - actual_len))
            
            preproc_inputs = {
                "audio_signal": audio_chunk.reshape(1, -1).astype(np.float32),
                "length": np.array([actual_len], dtype=np.int32)
            }
            
            preproc_out = preproc_model.predict(preproc_inputs)
            feat_chunk = np.array(preproc_out["features"])
            feat_len = int(preproc_out["feature_lengths"][0])
            
            if feat_len > 0:
                # The first (overlap_samples // mel_stride) features correspond to already-processed audio
                # The remaining features are NEW
                skip_features = overlap_samples // mel_stride  # 65
                new_feat_count = feat_len - skip_features
                if new_feat_count > 0:
                    valid_feats = feat_chunk[:, :, skip_features:feat_len]
                    feature_buffer = np.concatenate([feature_buffer, valid_feats], axis=2)
                    preproc_count += 1
    
    # Final pass: process any remaining features with partial right context
    # (This handles the end of the audio where we can't get full right context)
    if feature_buffer is not None:
        total_features = feature_buffer.shape[2]
        
        while True:
            chunk_start = diar_chunk_idx * core_frames
            chunk_end = chunk_start + core_frames
            
            if chunk_start >= total_features:
                break  # No more core frames to process
            
            # At end of audio, use whatever right context is available
            chunk_end = min(chunk_end, total_features)
            left_offset = min(left_ctx, chunk_start)
            right_offset = min(right_ctx, total_features - chunk_end)
            
            feat_start = chunk_start - left_offset
            feat_end = chunk_end + right_offset
            
            chunk_feat = feature_buffer[:, :, feat_start:feat_end]
            chunk_feat_tensor = torch.from_numpy(chunk_feat).float()
            actual_len = chunk_feat.shape[2]
            
            chunk_t = chunk_feat_tensor.transpose(1, 2)
            
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
            
            pred_logits = torch.from_numpy(coreml_out["speaker_preds"])
            chunk_embs = torch.from_numpy(coreml_out["chunk_pre_encoder_embs"])
            chunk_emb_len = int(coreml_out["chunk_pre_encoder_lengths"][0])
            
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
            diar_chunk_idx += 1
    
    print(f"\\n  Total diarization chunks: {diar_chunk_idx}")
    
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
        
        pred_logits = torch.from_numpy(coreml_out["speaker_preds"])
        chunk_embs = torch.from_numpy(coreml_out["chunk_pre_encoder_embs"])
        chunk_emb_len = int(coreml_out["chunk_pre_encoder_lengths"][0])
        
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
        os.path.join(coreml_dir, "SortformerPipeline.mlpackage"),
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
        
        if np.max(diff) < 0.004:
            print("\\n✅ SUCCESS: True streaming matches reference (error < 0.004)!")
        elif np.max(diff) < 0.01:
            print("\\n⚠️  Minor errors, but usable (error < 0.01)")
        else:
            print("\\n⚠️  Errors exceed tolerance")
            # Show where the errors are
            frame_max_errors = np.max(diff, axis=1)
            worst_frames = np.argsort(frame_max_errors)[-5:][::-1]
            print("  Top 5 worst frames:")
            for idx in worst_frames:
                print(f"    Frame {idx}: max error = {frame_max_errors[idx]:.6f}")
    else:
        print("True streaming inference produced no output!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    parser.add_argument("--coreml_dir", default="coreml_models")
    parser.add_argument("--audio_path", default="audio.wav")
    args = parser.parse_args()
    
    validate(args.model_name, args.coreml_dir, args.audio_path)
