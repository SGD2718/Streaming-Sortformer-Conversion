"""
CoreML Streaming Diarization - Interactive Visualization

This script runs diarization using the exported CoreML model and displays
an interactive heatmap. Click on pink boxes to play the corresponding audio segment.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import seaborn as sns
import numpy as np
import coremltools as ct
import librosa
import threading
import math
import time
from pydub import AudioSegment
from pydub.playback import play as play_audio

# Use TkAgg for interactive plots on Mac
matplotlib.use("TkAgg")

# Import NeMo for state management (required for streaming_update)
from nemo.collections.asr.models import SortformerEncLabelModel


def streaming_feat_loader(modules, feat_seq, feat_seq_length, feat_seq_offset):
    """Load chunks of features for streaming inference."""
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


def run_coreml_streaming(nemo_model, pre_encoder_model, head_model, audio_path, config):
    """Run streaming diarization using CoreML model."""
    modules = nemo_model.sortformer_modules
    subsampling_factor = modules.subsampling_factor
    sample_rate = 16000
    
    # Load Audio
    full_audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio_tensor = torch.from_numpy(full_audio).unsqueeze(0).float()
    audio_length = torch.tensor([len(full_audio)], dtype=torch.long)
    
    # Extract features using NeMo preprocessor
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
    num_chunks = math.ceil(processed_signal.shape[2] / (modules.chunk_len * subsampling_factor))
    
    feat_loader = streaming_feat_loader(
        modules=modules,
        feat_seq=processed_signal,
        feat_seq_length=processed_signal_length,
        feat_seq_offset=processed_signal_offset,
    )
    
    for chunk_idx, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in feat_loader:
        # Pad chunk to fixed size
        chunk_actual_len = chunk_feat_seq_t.shape[1]
        if chunk_actual_len < config['chunk_frames']:
            pad_len = config['chunk_frames'] - chunk_actual_len
            chunk_in = torch.nn.functional.pad(chunk_feat_seq_t, (0, 0, 0, pad_len))
        else:
            chunk_in = chunk_feat_seq_t[:, :config['chunk_frames'], :]
        chunk_len_in = feat_lengths.long()
        
        # Get actual state lengths
        curr_spk_len = state.spkcache.shape[1]
        curr_fifo_len = state.fifo.shape[1]
        
        # Prepare SpkCache - Pad to CoreML fixed size
        current_spkcache = state.spkcache
        if curr_spk_len < config['spkcache_input_len']:
            current_spkcache = torch.nn.functional.pad(
                current_spkcache, (0, 0, 0, config['spkcache_input_len'] - curr_spk_len)
            )
        elif curr_spk_len > config['spkcache_input_len']:
            current_spkcache = current_spkcache[:, :config['spkcache_input_len'], :]
        
        # Prepare FIFO - Pad to CoreML fixed size
        current_fifo = state.fifo
        if curr_fifo_len < config['fifo_input_len']:
            current_fifo = torch.nn.functional.pad(
                current_fifo, (0, 0, 0, config['fifo_input_len'] - curr_fifo_len)
            )
        elif curr_fifo_len > config['fifo_input_len']:
            current_fifo = current_fifo[:, :config['fifo_input_len'], :]
        
        # Prepare CoreML inputs
        coreml_inputs = {
            "chunk": chunk_in.numpy().astype(np.float32),
            "chunk_lengths": chunk_len_in.numpy().astype(np.int32),
            "spkcache": current_spkcache.numpy().astype(np.float32),
            "spkcache_lengths": np.array([curr_spk_len], dtype=np.int32),
            "fifo": current_fifo.numpy().astype(np.float32),
            "fifo_lengths": np.array([curr_fifo_len], dtype=np.int32)
        }
        
        # Run CoreML model
        preenc_start = time.time()
        pre_encoder_out = pre_encoder_model.predict(coreml_inputs)
        preenc_end = time.time()
        coreml_out = head_model.predict(pre_encoder_out)
        out_end = time.time()
        print(f"Pre-Encoder time: {preenc_end - preenc_start}")
        print(f"Head time: {out_end - preenc_end}")

        pred_logits = torch.from_numpy(coreml_out["speaker_preds"])
        chunk_embs = torch.from_numpy(coreml_out["chunk_pre_encoder_embs"])
        chunk_emb_len = int(coreml_out["chunk_pre_encoder_lengths"][0])
        
        # Trim chunk_embs to actual length (drop padded frames)
        chunk_embs = chunk_embs[:, :chunk_emb_len, :]
        
        # Compute lc/rc for streaming update
        lc = round(left_offset / subsampling_factor)
        rc = math.ceil(right_offset / subsampling_factor)
        
        # Update state
        state, chunk_probs = modules.streaming_update(
            streaming_state=state,
            chunk=chunk_embs,
            preds=pred_logits,
            lc=lc,
            rc=rc
        )
        
        all_preds.append(chunk_probs)
        print(f"\rProcessing chunk {chunk_idx + 1}/{num_chunks}", end='')
    
    print()  # Newline after progress
    
    if len(all_preds) > 0:
        final_probs = torch.cat(all_preds, dim=1)
        return final_probs
    return None


def segments_from_probs(probs, threshold=0.5, frame_shift=0.08, min_duration=0.1):
    """
    Extract segments from probability matrix.
    
    Args:
        probs: numpy array [T, num_speakers]
        threshold: probability threshold for speaker activity
        frame_shift: time per frame in seconds
        min_duration: minimum segment duration to keep
    
    Returns:
        List of (start_time, end_time, speaker_id) tuples
    """
    segments = []
    num_frames, num_speakers = probs.shape
    
    for spk_id in range(num_speakers):
        in_segment = False
        segment_start = 0
        
        for frame_idx in range(num_frames):
            is_active = probs[frame_idx, spk_id] > threshold
            
            if is_active and not in_segment:
                # Start new segment
                in_segment = True
                segment_start = frame_idx
            elif not is_active and in_segment:
                # End segment
                in_segment = False
                start_time = segment_start * frame_shift
                end_time = frame_idx * frame_shift
                duration = end_time - start_time
                if duration >= min_duration:
                    segments.append((start_time, end_time, spk_id))
        
        # Handle segment that extends to the end
        if in_segment:
            start_time = segment_start * frame_shift
            end_time = num_frames * frame_shift
            duration = end_time - start_time
            if duration >= min_duration:
                segments.append((start_time, end_time, spk_id))
    
    return sorted(segments, key=lambda x: x[0])


def main():
    # --- Configuration ---
    audio_file = "audio.wav"
    coreml_dir = "coreml_models"
    model_name = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    
    # CoreML export configuration (must match export settings)
    CONFIG = {
        'chunk_len': 6,
        'chunk_right_context': 1,
        'chunk_left_context': 1,
        'fifo_len': 40,
        'spkcache_len': 120,
        'spkcache_update_period': 30,
        'chunk_frames': 64,  # (chunk_len + left_context + right_context) * subsampling_factor = (4+1+1)*8
        'fifo_input_len': 40,  # CoreML model input size
        'spkcache_input_len': 120,  # CoreML model input size
    }

    # Load audio for playback
    print("Loading audio file for playback...")
    full_audio = AudioSegment.from_wav(audio_file)
    
    # --- Load NeMo model (for state management) ---
    print(f"Loading NeMo Model: {model_name}")
    nemo_model = SortformerEncLabelModel.from_pretrained(model_name, map_location="cpu")
    nemo_model.eval()
    
    # Apply streaming config
    modules = nemo_model.sortformer_modules
    modules.chunk_len = CONFIG['chunk_len']
    modules.chunk_right_context = CONFIG['chunk_right_context']
    modules.chunk_left_context = CONFIG['chunk_left_context']
    modules.fifo_len = CONFIG['fifo_len']
    modules.spkcache_len = CONFIG['spkcache_len']
    modules.spkcache_update_period = CONFIG['spkcache_update_period']
    
    # Disable dither and pad_to
    if hasattr(nemo_model.preprocessor, 'featurizer'):
        if hasattr(nemo_model.preprocessor.featurizer, 'dither'):
            nemo_model.preprocessor.featurizer.dither = 0.0
        if hasattr(nemo_model.preprocessor.featurizer, 'pad_to'):
            nemo_model.preprocessor.featurizer.pad_to = 0
    
    # --- Load CoreML model ---
    print(f"Loading CoreML Model from {coreml_dir}...")
    pre_encoder_model = ct.models.MLModel(
        os.path.join(coreml_dir, "Pipeline_PreEncoder.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY  # For compatibility
    )
    head_model = ct.models.MLModel(
        os.path.join(coreml_dir, "Pipeline_Head.mlpackage"),
        compute_units=ct.ComputeUnit.ALL  # For compatibility
    )
    
    # --- Run Inference ---
    print("Running CoreML streaming inference...")
    st_time = time.time()
    probs_tensor = run_coreml_streaming(nemo_model, pre_encoder_model, head_model, audio_file, CONFIG)
    ed_time = time.time()
    print(f'duration: {ed_time - st_time}')
    
    if probs_tensor is None:
        print("Inference failed!")
        return
    
    probs = probs_tensor.squeeze(0).cpu().numpy()  # [T, 4]
    heatmap_data = probs.T  # [4, T] for heatmap
    
    print(f"Output shape: {probs.shape}")
    
    # --- Extract segments from probabilities ---
    segments = segments_from_probs(probs, threshold=0.5)
    
    # --- Interaction Logic ---
    def play_segment_thread(start_s, duration_s):
        """Plays audio in a separate thread"""
        start_ms = int(start_s * 1000)
        end_ms = int((start_s + duration_s) * 1000)
        print(f"▶️ Playing: {start_s:.2f}s -> {start_s + duration_s:.2f}s")
        segment = full_audio[start_ms:end_ms]
        play_audio(segment)
    
    def on_click(event):
        """Handle click events on the plot"""
        if event.inaxes is None:
            return
        
        for patch in ax.patches:
            contains, _ = patch.contains(event)
            if contains:
                meta = getattr(patch, 'audio_meta', None)
                if meta:
                    t = threading.Thread(target=play_segment_thread, args=(meta['start'], meta['dur']))
                    t.start()
                    return
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    sns.heatmap(
        heatmap_data,
        cmap="viridis",
        yticklabels=[f"Spk {i}" for i in range(heatmap_data.shape[0])],
        cbar_kws={'label': 'Activity Probability'},
        vmin=0, vmax=1,
        ax=ax
    )
    
    # --- Add segment rectangles ---
    frame_shift = 0.08  # 80ms per diarization frame
    
    print(f"Found {len(segments)} segments:")
    for start_time, end_time, spk_idx in segments:
        duration = end_time - start_time
        start_frame = start_time / frame_shift
        width_frame = duration / frame_shift
        
        print(f"  Speaker {spk_idx}: {start_time:.2f}s -> {end_time:.2f}s")
        
        rect = patches.Rectangle(
            (start_frame, spk_idx),
            width=width_frame,
            height=1,
            linewidth=2,
            edgecolor='#FF00FF',
            facecolor='none',
            picker=True
        )
        rect.audio_meta = {'start': start_time, 'dur': duration}
        ax.add_patch(rect)
    
    plt.title("CoreML Streaming Diarization (Click Pink Box to Play Audio)")
    plt.xlabel("Time Frames (80ms steps)")
    plt.ylabel("Speaker ID")
    plt.tight_layout()
    
    print("\nInference complete. Interactive window opening...")
    plt.show()


if __name__ == "__main__":
    main()
