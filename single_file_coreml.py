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
import argparse
import umap
from pydub import AudioSegment
from pydub.playback import play as play_audio
from config import Config

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


def run_coreml_streaming(nemo_model, main_model, audio_path, config):
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
    all_embeddings = []
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
        st_time = time.time()
        coreml_out = main_model.predict(coreml_inputs)
        ed_time = time.time()
        print(f"Inference time: {ed_time - st_time}")

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
        
        # Store the core embeddings (excluding left and right context)
        core_start = lc  # Skip left context
        core_end = chunk_emb_len - rc  # Exclude right context
        if core_end > core_start:
            core_embs = chunk_embs[:, core_start:core_end, :]
            all_embeddings.append(core_embs)
        
        print(f"\rProcessing chunk {chunk_idx + 1}/{num_chunks}", end='')
    
    print()  # Newline after progress
    
    if len(all_preds) > 0:
        final_probs = torch.cat(all_preds, dim=1)
        final_embeddings = torch.cat(all_embeddings, dim=1) if len(all_embeddings) > 0 else None
        return final_probs, final_embeddings, state  # Return final state with spkcache
    return None, None, None


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


def compute_speaker_centroids_and_distances(embeddings, probs, num_speakers=4):
    """
    Compute speaker centroids and per-frame cosine distances.
    
    Args:
        embeddings: numpy array [T, D] - frame embeddings
        probs: numpy array [T, num_speakers] - speaker probabilities per frame
        num_speakers: number of speakers
    
    Returns:
        centroids: [num_speakers, D] - centroid embedding for each speaker
        distances: [T, num_speakers] - cosine distance to each speaker's centroid per frame
    """
    T, D = embeddings.shape
    centroids = np.zeros((num_speakers, D))
    
    # Compute weighted centroid for each speaker
    for spk in range(num_speakers):
        weights = probs[:, spk]  # [T]
        weight_sum = weights.sum()
        if weight_sum > 0:
            # Weighted average of embeddings
            centroids[spk] = (embeddings * weights[:, np.newaxis]).sum(axis=0) / weight_sum
        else:
            # No activity for this speaker, use zero vector
            centroids[spk] = np.zeros(D)
    
    # Normalize centroids for cosine similarity
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroid_norms = np.where(centroid_norms > 0, centroid_norms, 1.0)
    normalized_centroids = centroids / centroid_norms
    
    # Normalize embeddings
    embedding_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embedding_norms = np.where(embedding_norms > 0, embedding_norms, 1.0)
    normalized_embeddings = embeddings / embedding_norms
    
    # Compute cosine similarity (dot product of normalized vectors)
    # [T, D] @ [D, num_speakers] -> [T, num_speakers]
    cosine_similarities = normalized_embeddings @ normalized_centroids.T
    
    # Cosine distance = 1 - cosine similarity
    cosine_distances = 1 - cosine_similarities
    
    return centroids, cosine_distances

def main(save_png=None):
    # --- Configuration ---
    audio_file = "multispeaker.wav"
    coreml_dir = "coreml_models"
    model_name = "nvidia/diar_streaming_sortformer_4spk-v2.1"
    
    # CoreML export configuration (must match export settings)
    CONFIG = {
        'chunk_len': Config.chunk_len,
        'chunk_right_context': Config.chunk_right_context,
        'chunk_left_context': Config.chunk_left_context,
        'fifo_len': Config.fifo_len,
        'spkcache_len': Config.spkcache_len,
        'spkcache_update_period': Config.spkcache_update_period,
        'chunk_frames': Config.chunk_frames,  # (chunk_len + left_context + right_context) * subsampling_factor = (4+1+1)*8
        'fifo_input_len': Config.fifo_len,  # CoreML model input size
        'spkcache_input_len': Config.spkcache_len,  # CoreML model input size
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
    main_model = ct.models.MLModel(
        os.path.join(coreml_dir, "SortformerPipeline.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY
    )

    # --- Run Inference ---
    print("Running CoreML streaming inference...")
    st_time = time.time()
    probs_tensor, embeddings_tensor, final_state = run_coreml_streaming(nemo_model, main_model, audio_file, CONFIG)
    ed_time = time.time()
    print(f'duration: {ed_time - st_time}')
    
    if probs_tensor is None:
        print("Inference failed!")
        return
    
    probs = probs_tensor.squeeze(0).cpu().numpy()  # [T, 4]
    embeddings = embeddings_tensor.squeeze(0).cpu().numpy() if embeddings_tensor is not None else None  # [T, D]
    heatmap_data = probs.T  # [4, T] for heatmap
    
    print(f"Output shape: {probs.shape}")
    if embeddings is not None:
        print(f"Embeddings shape: {embeddings.shape}")
    
    # --- Compute Speaker Centroids from Speaker Cache ---
    l2_distances = None
    cosine_distances = None
    num_speakers = probs.shape[1]
    
    # Use spkcache and spkcache_preds from final state to compute weighted centroids
    if final_state is not None and final_state.spkcache is not None and final_state.spkcache_preds is not None:
        spkcache = final_state.spkcache.squeeze(0).cpu().numpy()  # [cache_len, D]
        spkcache_preds = final_state.spkcache_preds.squeeze(0).cpu().numpy()  # [cache_len, num_speakers]
        
        print(f"Speaker cache shape: {spkcache.shape}")
        print(f"Speaker cache preds shape: {spkcache_preds.shape}")
        
        # Normalize spkcache embeddings for cosine centroid
        spkcache_norms = np.linalg.norm(spkcache, axis=1, keepdims=True)
        spkcache_norms = np.where(spkcache_norms > 0, spkcache_norms, 1.0)
        spkcache_normalized = spkcache / spkcache_norms
        
        # Compute weighted centroids for each speaker
        # L2 centroid: from raw embeddings
        # Cosine centroid: from normalized embeddings
        centroids_l2 = np.zeros((num_speakers, spkcache.shape[1]))
        centroids_cosine = np.zeros((num_speakers, spkcache.shape[1]))
        for spk in range(num_speakers):
            weights = spkcache_preds[:, spk]
            weight_sum = weights.sum()
            if weight_sum > 0:
                centroids_l2[spk] = (spkcache * weights[:, np.newaxis]).sum(axis=0) / weight_sum
                centroids_cosine[spk] = (spkcache_normalized * weights[:, np.newaxis]).sum(axis=0) / weight_sum
        
        # Normalize cosine centroids
        cosine_centroid_norms = np.linalg.norm(centroids_cosine, axis=1, keepdims=True)
        cosine_centroid_norms = np.where(cosine_centroid_norms > 0, cosine_centroid_norms, 1.0)
        centroids_cosine = centroids_cosine / cosine_centroid_norms
        
        print(f"L2 centroids computed (shape: {centroids_l2.shape})")
        print(f"Cosine centroids computed (shape: {centroids_cosine.shape})")
        
        # Compute distances
        if embeddings is not None:
            num_frames = min(len(probs), len(embeddings))
            
            # L2 distances from embeddings to L2 centroids
            l2_distances = np.zeros((num_frames, num_speakers))
            for s in range(num_speakers):
                diff = embeddings[:num_frames] - centroids_l2[s]
                l2_distances[:, s] = np.linalg.norm(diff, axis=1)
            
            # Normalize embeddings for cosine distance
            emb_norms = np.linalg.norm(embeddings[:num_frames], axis=1, keepdims=True)
            emb_norms = np.where(emb_norms > 0, emb_norms, 1.0)
            normalized_embs = embeddings[:num_frames] / emb_norms
            
            # Cosine distances from normalized embeddings to cosine centroids
            cos_sim = normalized_embs @ centroids_cosine.T
            cosine_distances = 1 - cos_sim
            
            print(f"L2 distances computed (shape: {l2_distances.shape})")
            print(f"Cosine distances computed (shape: {cosine_distances.shape})")
    else:
        print("WARNING: spkcache_preds is None - speaker cache not compressed yet. Using frame embeddings instead.")
        if embeddings is not None:
            num_frames = min(len(probs), len(embeddings))
            
            # Normalize embeddings
            emb_norms = np.linalg.norm(embeddings[:num_frames], axis=1, keepdims=True)
            emb_norms = np.where(emb_norms > 0, emb_norms, 1.0)
            normalized_embs = embeddings[:num_frames] / emb_norms
            
            # Compute weighted centroids
            centroids_l2 = np.zeros((num_speakers, embeddings.shape[1]))
            centroids_cosine = np.zeros((num_speakers, embeddings.shape[1]))
            for spk in range(num_speakers):
                weights = probs[:num_frames, spk]
                weight_sum = weights.sum()
                if weight_sum > 0:
                    centroids_l2[spk] = (embeddings[:num_frames] * weights[:, np.newaxis]).sum(axis=0) / weight_sum
                    centroids_cosine[spk] = (normalized_embs * weights[:, np.newaxis]).sum(axis=0) / weight_sum
            
            # Normalize cosine centroids
            cosine_centroid_norms = np.linalg.norm(centroids_cosine, axis=1, keepdims=True)
            cosine_centroid_norms = np.where(cosine_centroid_norms > 0, cosine_centroid_norms, 1.0)
            centroids_cosine = centroids_cosine / cosine_centroid_norms
            
            # L2 distances
            l2_distances = np.zeros((num_frames, num_speakers))
            for s in range(num_speakers):
                diff = embeddings[:num_frames] - centroids_l2[s]
                l2_distances[:, s] = np.linalg.norm(diff, axis=1)
            
            # Cosine distances
            cos_sim = normalized_embs @ centroids_cosine.T
            cosine_distances = 1 - cos_sim
            
            print(f"Speaker centroids computed (fallback)")
            print(f"L2 distances computed (shape: {l2_distances.shape})")
            print(f"Cosine distances computed (shape: {cosine_distances.shape})")
    
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
    # Create figure with subplots: 1 heatmap + L2 distance plot + UMAP scatter
    if l2_distances is not None and embeddings is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax = axes[0]
        l2_ax = axes[1]
        umap_ax = axes[2]
    elif l2_distances is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        l2_ax = axes[1]
        umap_ax = None
    else:
        fig, ax = plt.subplots(figsize=(15, 6))
        l2_ax = None
        umap_ax = None
    
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
    frame_shift = Config.frame_duration  # 80ms per diarization frame
    
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
    
    ax.set_title("CoreML Streaming Diarization (Click Pink Box to Play Audio)")
    ax.set_xlabel("Time Frames (80ms steps)")
    ax.set_ylabel("Speaker ID")
    
    # --- L2 Distance Plot (all speakers overlayed) ---
    if l2_ax is not None and l2_distances is not None:
        num_speakers = l2_distances.shape[1]
        num_frames = len(l2_distances)
        time_frames = np.arange(num_frames)
        time_seconds = time_frames * frame_shift
        colors = plt.cm.tab10(np.linspace(0, 1, num_speakers))
        
        for s in range(num_speakers):
            l2_ax.plot(time_seconds, l2_distances[:, s], color=colors[s], linewidth=1.5, label=f'Spk {s}')
        
        l2_ax.set_title("L2 Distance to Each Speaker Centroid")
        l2_ax.set_xlabel("Time (seconds)")
        l2_ax.set_ylabel("L2 Distance")
        l2_ax.set_xlim(0, time_seconds[-1] if len(time_seconds) > 0 else 1)
        l2_ax.set_ylim(0, np.nanmax(l2_distances) * 1.1)
        l2_ax.grid(True, alpha=0.3)
        l2_ax.legend(loc='upper right', fontsize='small', ncol=2)
    
    # --- UMAP Scatter Plot (colored by speaker powerset) ---
    if umap_ax is not None and embeddings is not None:
        num_frames = min(len(probs), len(embeddings))
        num_speakers = probs.shape[1]
        
        # Define distinct speaker colors using evenly spaced hues (same saturation/brightness)
        # Using HSV with S=0.8, V=0.9 for vibrant, distinguishable colors
        import colorsys
        speaker_colors = []
        for i in range(num_speakers):
            hue = i / num_speakers  # Evenly spaced hues: 0, 0.25, 0.5, 0.75
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            speaker_colors.append(np.array([r, g, b]))
        
        # Silence color (gray)
        silence_color = np.array([0.5, 0.5, 0.5])
        
        # Compute per-frame: active speakers, blended color, and opacity (max prob)
        frame_colors = []
        frame_alphas = []
        frame_labels = []
        
        for i in range(num_frames):
            active_speakers = tuple(s for s in range(num_speakers) if probs[i, s] > 0.5)
            frame_labels.append(active_speakers)
            
            # Opacity = max probability at this frame (clipped to [0.3, 1.0] for visibility)
            max_prob = probs[i, :].max()
            alpha = max(0.3, min(1.0, max_prob))
            frame_alphas.append(alpha)
            
            # Color = blend of active speaker colors, or gray for silence
            if len(active_speakers) == 0:
                color = silence_color
            else:
                color = np.mean([speaker_colors[s] for s in active_speakers], axis=0)
            frame_colors.append(color)
        
        frame_colors = np.array(frame_colors)
        frame_alphas = np.array(frame_alphas)
        
        # Perform UMAP
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.05, random_state=42)
        embs_2d = reducer.fit_transform(embeddings[:num_frames])
        
        # Get unique labels for legend
        unique_labels = sorted(set(frame_labels), key=lambda x: (len(x), x))
        
        # Compute legend colors (same blending logic)
        label_to_color = {}
        for label in unique_labels:
            if len(label) == 0:
                label_to_color[label] = silence_color
            else:
                label_to_color[label] = np.mean([speaker_colors[s] for s in label], axis=0)
        
        # Plot points individually to support per-point alpha
        for i in range(num_frames):
            umap_ax.scatter(embs_2d[i, 0], embs_2d[i, 1], 
                           c=[frame_colors[i]], s=20, alpha=frame_alphas[i])
        
        # Add legend manually with representative colors
        for label in unique_labels:
            label_str = '{' + ','.join(map(str, label)) + '}' if label else '∅'
            umap_ax.scatter([], [], c=[label_to_color[label]], s=40, label=label_str)
        
        umap_ax.set_title("UMAP of Frame Embeddings (colored by active speakers)")
        umap_ax.set_xlabel("UMAP 1")
        umap_ax.set_ylabel("UMAP 2")
        umap_ax.legend(loc='upper right', fontsize='small', ncol=2, title='Speakers')
        umap_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_png:
        plt.savefig(save_png, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {save_png}")
    else:
        print("\nInference complete. Interactive window opening...")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoreML Streaming Diarization Visualization")
    parser.add_argument("--save-png", type=str, default=None, 
                        help="Save plot to PNG file instead of showing interactively")
    args = parser.parse_args()
    main(save_png=args.save_png)
