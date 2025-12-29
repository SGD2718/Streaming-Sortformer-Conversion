"""
Real-Time Microphone Diarization with CoreML

This script captures audio from the microphone in real-time,
processes it through CoreML models, and displays a live updating
diarization heatmap.

Pipeline: Microphone → Audio Buffer → CoreML Preproc → CoreML Main → Live Plot

Requirements:
    pip install pyaudio matplotlib seaborn numpy coremltools

Usage:
    python mic_inference.py
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import coremltools as ct
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
import seaborn as sns
import queue
import time
import math
import argparse
from config import Config

# Import NeMo for state management
from nemo.collections.asr.models import SortformerEncLabelModel
import sounddevice as sd
SOUNDDEVICE_AVAILABLE = True

# ============================================================
# Configuration
# ============================================================


class MicrophoneStream:
    """Captures audio from microphone using sounddevice."""

    def __init__(self, sample_rate, chunk_size, audio_queue):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = audio_queue
        self.stream = None
        self.running = False

    def start(self):
        if not SOUNDDEVICE_AVAILABLE:
            print("sounddevice not available!")
            return False

        def callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            # indata is already float32 in range [-1, 1]
            audio = indata[:, 0].copy()  # Take first channel
            self.audio_queue.put(audio)

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=self.chunk_size,
            callback=callback
        )
        self.stream.start()
        self.running = True
        print("Microphone started...")
        return True

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("Microphone stopped.")


class StreamingPredictionBuffer:
    """
    Manages provisional predictions from right context that get replaced 
    with the actual core predictions on the next chunk update.
    
    Timeline example (chunk_len=6, right_context=7):
    - Chunk 0: core = frames 0-5, provisional = frames 6-12 (early guess)
    - Chunk 1: core = frames 6-11, provisional = frames 12-18
      → frames 6-11 were provisional, now replaced with actual core predictions
      → frames 12-18 are new provisional
    
    This gives lower perceived latency: provisional frames display immediately,
    then get replaced with proper predictions when processed as core.
    """
    
    def __init__(self, num_speakers=4):
        self.num_speakers = num_speakers
        self.display_probs = []  # All predictions to display (confirmed + provisional)
        self.confirmed_frame_count = 0  # Number of frames with confirmed (core) predictions
        self.provisional_frame_count = 0  # Current number of provisional frames displayed
        
    def update(self, core_probs: np.ndarray, provisional_probs: np.ndarray = None):
        """
        Update the buffer with new predictions.
        
        Args:
            core_probs: Actual predictions for the core chunk [T_core, num_speakers]
            provisional_probs: Early guess predictions for right context [T_rc, num_speakers]
        """
        if len(core_probs) == 0:
            return
        
        # Remove old provisional predictions (the last array if provisional_frame_count > 0)
        if self.provisional_frame_count > 0 and len(self.display_probs) > 0:
            self.display_probs.pop()  # Remove the provisional array
        
        # Add core predictions as confirmed
        self.display_probs.append(core_probs)
        self.confirmed_frame_count += len(core_probs)
        
        # Add provisional predictions for display (will be replaced next update)
        if provisional_probs is not None and len(provisional_probs) > 0:
            self.display_probs.append(provisional_probs)
            self.provisional_frame_count = len(provisional_probs)
        else:
            self.provisional_frame_count = 0
    
    def get_all_probs(self, include_provisional=True):
        """
        Get all accumulated probabilities.
        
        Args:
            include_provisional: If True, include provisional predictions
            
        Returns:
            numpy array of shape [total_frames, num_speakers]
        """
        if len(self.display_probs) == 0:
            return None
        
        all_probs = np.concatenate(self.display_probs, axis=0)
        
        if not include_provisional and self.provisional_frame_count > 0:
            return all_probs[:-self.provisional_frame_count]
        
        return all_probs
    
    def get_confirmed_count(self):
        """Return count of confirmed (core) frames."""
        return self.confirmed_frame_count
    
    @property
    def provisional_frames(self):
        """Number of provisional frames currently displayed."""
        return self.provisional_frame_count
    
    def get_total_count(self, include_provisional=True):
        """Return total frame count."""
        total = self.confirmed_frame_count
        if include_provisional:
            total += self.provisional_frame_count
        return total


class StreamingDiarizer:
    """Real-time streaming diarization using CoreML with provisional predictions."""

    def __init__(self, nemo_model, preproc_model, main_model):
        self.modules = nemo_model.sortformer_modules
        self.preproc_model = preproc_model
        self.main_model = main_model

        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)

        # Feature buffer
        self.feature_buffer = None
        self.features_processed = 0

        # Diarization state
        self.state = self.modules.init_streaming_state(batch_size=1, device='cpu')
        
        # NEW: Use prediction buffer with provisional support
        self.pred_buffer = StreamingPredictionBuffer(num_speakers=4)
        self.all_probs = []  # Keep for backward compatibility

        # Chunk tracking
        self.diar_chunk_idx = 0
        
        # Derived params
        self.subsampling = Config.subsampling_factor
        self.core_frames = Config.chunk_len * self.subsampling
        self.left_ctx = Config.chunk_left_context * self.subsampling
        self.right_ctx = Config.chunk_right_context * self.subsampling

        # Audio processing params
        self.coreml_audio_size = Config.coreml_audio_samples  # What CoreML expects (18160)
        self.audio_hop = Config.preproc_audio_hop  # New audio per update (~480ms = 7680)
        self.first_run_done = False  # Track first preprocessing run

    def add_audio(self, audio_chunk):
        """Add new audio samples."""
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

    def process(self):
        """
        Process available audio through preprocessor and diarizer.
        Returns new probability frames if available (including tentative).
        
        Strategy:
        - Wait for FULL right context before processing (for accuracy)
        - Output confirmed predictions for core frames
        - Output tentative predictions for right context frames (early guess for next chunk)
        
        Sliding window preprocessing for ~2Hz updates:
        - First run: Wait for coreml_audio_size samples (full window)
        - After: Run every audio_hop samples using sliding window
        """
        new_probs = None

        # Step 1: Run preprocessor with sliding window
        # Take audio from the BEGINNING of the buffer (not end) to avoid edge effects
        while len(self.audio_buffer) >= self.coreml_audio_size:
            # Take the first coreml_audio_size samples
            audio_chunk = self.audio_buffer[:self.coreml_audio_size].copy()
            
            preproc_inputs = {
                "audio_signal": audio_chunk.reshape(1, -1).astype(np.float32),
                "length": np.array([self.coreml_audio_size], dtype=np.int32)
            }

            preproc_out = self.preproc_model.predict(preproc_inputs)
            feat_chunk = np.array(preproc_out["features"])
            feat_len = int(preproc_out["feature_lengths"][0])

            if not self.first_run_done:
                # First run: take all features
                valid_feats = feat_chunk[:, :, :feat_len]
                self.first_run_done = True
            else:
                # Subsequent runs: take the LAST new_feat_count features
                # These correspond to the new audio at the end of the chunk
                new_feat_count = self.audio_hop // Config.mel_stride
                valid_feats = feat_chunk[:, :, feat_len - new_feat_count:feat_len]
            
            if self.feature_buffer is None:
                self.feature_buffer = valid_feats
            else:
                self.feature_buffer = np.concatenate([self.feature_buffer, valid_feats], axis=2)

            # Slide the window forward by removing audio_hop samples from the start
            self.audio_buffer = self.audio_buffer[self.audio_hop:]

        if self.feature_buffer is None:
            return None

        # Step 2: Run diarization on available features
        total_features = self.feature_buffer.shape[2]

        while True:
            # Calculate chunk boundaries
            chunk_start = self.diar_chunk_idx * self.core_frames
            chunk_end = chunk_start + self.core_frames

            # Wait for FULL right context before processing
            # This ensures accurate predictions (no partial context)
            required_features = chunk_end + self.right_ctx
            
            if required_features > total_features:
                break  # Not enough features yet - wait for more audio

            # Extract with full left and right context
            left_offset = min(self.left_ctx, chunk_start)
            right_offset = self.right_ctx  # Always full right context

            feat_start = chunk_start - left_offset
            feat_end = chunk_end + right_offset

            chunk_feat = self.feature_buffer[:, :, feat_start:feat_end]
            chunk_feat_tensor = torch.from_numpy(chunk_feat).float()
            actual_len = chunk_feat.shape[2]

            # Transpose to [B, T, D]
            chunk_t = chunk_feat_tensor.transpose(1, 2)

            # Pad to full chunk_frames
            if actual_len < Config.chunk_frames:
                pad_len = Config.chunk_frames - actual_len
                chunk_in = torch.nn.functional.pad(chunk_t, (0, 0, 0, pad_len))
            else:
                chunk_in = chunk_t[:, :Config.chunk_frames, :]

            # State preparation
            curr_spk_len = self.state.spkcache.shape[1]
            curr_fifo_len = self.state.fifo.shape[1]

            current_spkcache = self.state.spkcache
            if curr_spk_len < Config.spkcache_len:
                current_spkcache = torch.nn.functional.pad(
                    current_spkcache, (0, 0, 0, Config.spkcache_len - curr_spk_len)
                )
            elif curr_spk_len > Config.spkcache_len:
                current_spkcache = current_spkcache[:, :Config.spkcache_len, :]

            current_fifo = self.state.fifo
            if curr_fifo_len < Config.fifo_len:
                current_fifo = torch.nn.functional.pad(
                    current_fifo, (0, 0, 0, Config.fifo_len - curr_fifo_len)
                )
            elif curr_fifo_len > Config.fifo_len:
                current_fifo = current_fifo[:, :Config.fifo_len, :]

            # CoreML inference
            coreml_inputs = {
                "chunk": chunk_in.numpy().astype(np.float32),
                "chunk_lengths": np.array([actual_len], dtype=np.int32),
                "spkcache": current_spkcache.numpy().astype(np.float32),
                "spkcache_lengths": np.array([curr_spk_len], dtype=np.int32),
                "fifo": current_fifo.numpy().astype(np.float32),
                "fifo_lengths": np.array([curr_fifo_len], dtype=np.int32)
            }

            st_time = time.time_ns()
            coreml_out = self.main_model.predict(coreml_inputs)
            ed_time = time.time_ns()
            print(f"duration: {1e-6 * (ed_time - st_time):.2f}ms")

            pred_logits = torch.from_numpy(coreml_out["speaker_preds"])
            chunk_embs = torch.from_numpy(coreml_out["chunk_pre_encoder_embs"])
            chunk_emb_len = int(coreml_out["chunk_pre_encoder_lengths"][0])

            chunk_embs = chunk_embs[:, :chunk_emb_len, :]

            lc = round(left_offset / self.subsampling)
            rc = Config.chunk_right_context  # Full right context (7 diar frames)

            # Save state lengths BEFORE update for correct indexing
            pre_spkcache_len = curr_spk_len
            pre_fifo_len = curr_fifo_len
            core_len = Config.chunk_len  # 6 diar frames

            # Update streaming state - returns confirmed core predictions
            self.state, chunk_probs = self.modules.streaming_update(
                streaming_state=self.state,
                chunk=chunk_embs,
                preds=pred_logits,
                lc=lc,
                rc=rc
            )

            # Core predictions (confirmed) - these are the 6 core frames
            core_probs = chunk_probs.squeeze(0).detach().cpu().numpy()

            # Tentative predictions for the FULL right context (7 frames)
            # These are at: spkcache_len + fifo_len + lc + core_len ... + rc
            # (NOT at the end of the padded array!)
            tentative_start = pre_spkcache_len + pre_fifo_len + lc + core_len
            tentative_end = tentative_start + rc
            tentative_probs = pred_logits[0, tentative_start:tentative_end, :].detach().cpu().numpy()

            # Update the prediction buffer: confirmed core + tentative right context
            self.pred_buffer.update(core_probs, tentative_probs)
            
            # Also maintain backward compatibility with all_probs list
            self.all_probs.append(core_probs)

            new_probs = core_probs
            self.diar_chunk_idx += 1

        return new_probs

    def get_all_probs(self, include_provisional=True):
        """
        Get all accumulated probabilities.
        
        Args:
            include_provisional: If True, include provisional predictions from right context
                               These are shown with lower latency but may be refined later
        """
        return self.pred_buffer.get_all_probs(include_provisional=include_provisional)
    
    def process_remaining(self):
        """
        Process any remaining audio that didn't fill a complete preprocessing chunk.
        Call this at end-of-file to extract final features.
        """
        if len(self.audio_buffer) == 0 or not self.first_run_done:
            return
        
        overlap_samples = self.coreml_audio_size - self.audio_hop  # 10480
        new_samples = len(self.audio_buffer) - overlap_samples
        
        if new_samples > 0:
            # Pad the audio buffer to coreml_audio_size
            actual_len = len(self.audio_buffer)
            audio_chunk = np.pad(self.audio_buffer, (0, self.coreml_audio_size - actual_len))
            
            preproc_inputs = {
                "audio_signal": audio_chunk.reshape(1, -1).astype(np.float32),
                "length": np.array([actual_len], dtype=np.int32)
            }
            
            preproc_out = self.preproc_model.predict(preproc_inputs)
            feat_chunk = np.array(preproc_out["features"])
            feat_len = int(preproc_out["feature_lengths"][0])
            
            if feat_len > 0:
                # Skip the overlapping features, take only new ones
                skip_features = overlap_samples // Config.mel_stride  # 65
                new_feat_count = feat_len - skip_features
                if new_feat_count > 0:
                    valid_feats = feat_chunk[:, :, skip_features:feat_len]
                    if self.feature_buffer is None:
                        self.feature_buffer = valid_feats
                    else:
                        self.feature_buffer = np.concatenate([self.feature_buffer, valid_feats], axis=2)
    
    def process_final(self):
        """
        Final pass: process any remaining features with partial right context.
        Call this at end-of-file after process_remaining().
        """
        if self.feature_buffer is None:
            return
        
        total_features = self.feature_buffer.shape[2]
        
        while True:
            chunk_start = self.diar_chunk_idx * self.core_frames
            chunk_end = chunk_start + self.core_frames
            
            if chunk_start >= total_features:
                break  # No more core frames to process
            
            # At end of audio, use whatever right context is available
            chunk_end = min(chunk_end, total_features)
            left_offset = min(self.left_ctx, chunk_start)
            right_offset = min(self.right_ctx, total_features - chunk_end)
            
            feat_start = chunk_start - left_offset
            feat_end = chunk_end + right_offset
            
            chunk_feat = self.feature_buffer[:, :, feat_start:feat_end]
            chunk_feat_tensor = torch.from_numpy(chunk_feat).float()
            actual_len = chunk_feat.shape[2]
            
            chunk_t = chunk_feat_tensor.transpose(1, 2)
            
            if actual_len < Config.chunk_frames:
                pad_len = Config.chunk_frames - actual_len
                chunk_in = torch.nn.functional.pad(chunk_t, (0, 0, 0, pad_len))
            else:
                chunk_in = chunk_t[:, :Config.chunk_frames, :]
            
            curr_spk_len = self.state.spkcache.shape[1]
            curr_fifo_len = self.state.fifo.shape[1]
            
            current_spkcache = self.state.spkcache
            if curr_spk_len < Config.spkcache_len:
                current_spkcache = torch.nn.functional.pad(
                    current_spkcache, (0, 0, 0, Config.spkcache_len - curr_spk_len)
                )
            elif curr_spk_len > Config.spkcache_len:
                current_spkcache = current_spkcache[:, :Config.spkcache_len, :]
            
            current_fifo = self.state.fifo
            if curr_fifo_len < Config.fifo_len:
                current_fifo = torch.nn.functional.pad(
                    current_fifo, (0, 0, 0, Config.fifo_len - curr_fifo_len)
                )
            elif curr_fifo_len > Config.fifo_len:
                current_fifo = current_fifo[:, :Config.fifo_len, :]
            
            coreml_inputs = {
                "chunk": chunk_in.numpy().astype(np.float32),
                "chunk_lengths": np.array([actual_len], dtype=np.int32),
                "spkcache": current_spkcache.numpy().astype(np.float32),
                "spkcache_lengths": np.array([curr_spk_len], dtype=np.int32),
                "fifo": current_fifo.numpy().astype(np.float32),
                "fifo_lengths": np.array([curr_fifo_len], dtype=np.int32)
            }
            
            coreml_out = self.main_model.predict(coreml_inputs)
            
            pred_logits = torch.from_numpy(coreml_out["speaker_preds"])
            chunk_embs = torch.from_numpy(coreml_out["chunk_pre_encoder_embs"])
            chunk_emb_len = int(coreml_out["chunk_pre_encoder_lengths"][0])
            chunk_embs = chunk_embs[:, :chunk_emb_len, :]
            
            lc = round(left_offset / self.subsampling)
            rc = math.ceil(right_offset / self.subsampling)
            
            self.state, chunk_probs = self.modules.streaming_update(
                streaming_state=self.state,
                chunk=chunk_embs,
                preds=pred_logits,
                lc=lc,
                rc=rc
            )
            
            core_probs = chunk_probs.squeeze(0).detach().cpu().numpy()
            self.pred_buffer.update(core_probs, None)  # No provisional for final pass
            self.all_probs.append(core_probs)
            self.diar_chunk_idx += 1
    
    def get_all_probs_legacy(self):
        """Get all accumulated probabilities (legacy method without provisional)."""
        if len(self.all_probs) > 0:
            return np.concatenate(self.all_probs, axis=0)
        return None


def run_mic_inference(model_name, coreml_dir):
    """Run real-time microphone diarization."""

    if not SOUNDDEVICE_AVAILABLE:
        print("Cannot run mic inference without sounddevice!")
        return

    print("=" * 70)
    print("Real-Time Microphone Diarization")
    print("=" * 70)

    # Load NeMo model
    print(f"\nLoading NeMo Model: {model_name}")
    nemo_model = SortformerEncLabelModel.from_pretrained(model_name, map_location=torch.device("cpu"))
    nemo_model.eval()

    # Configure
    modules = nemo_model.sortformer_modules
    modules.chunk_len = Config.chunk_len
    modules.chunk_right_context = Config.chunk_right_context
    modules.chunk_left_context = Config.chunk_left_context
    modules.fifo_len = Config.fifo_len
    modules.spkcache_len = Config.spkcache_len
    modules.spkcache_update_period = Config.spkcache_update_period

    if hasattr(nemo_model.preprocessor, 'featurizer'):
        nemo_model.preprocessor.featurizer.dither = 0.0
        nemo_model.preprocessor.featurizer.pad_to = 0

    # Load CoreML models
    print(f"Loading CoreML Models from {coreml_dir}...")
    preproc_model = ct.models.MLModel(
        os.path.join(coreml_dir, "SortformerPreprocessor.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY
    )
    main_model = ct.models.MLModel(
        os.path.join(coreml_dir, "SortformerPipeline.mlpackage"),
        compute_units=ct.ComputeUnit.ALL
    )

    # Create diarizer
    diarizer = StreamingDiarizer(nemo_model, preproc_model, main_model)

    # Audio queue
    audio_queue = queue.Queue()

    # Start microphone
    mic = MicrophoneStream(
        sample_rate=Config.sample_rate,
        chunk_size=int(Config.sample_rate * Config.frame_duration),
        audio_queue=audio_queue
    )

    if not mic.start():
        return

    # Setup plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 4))
    im = None  # Will hold the imshow image object
    vline = None  # Will hold the provisional marker line

    print("\nListening... Press Ctrl+C to stop.\n")

    try:
        last_update = time.time()

        while True:
            # Get audio from queue
            while not audio_queue.empty():
                audio_chunk = audio_queue.get()
                diarizer.add_audio(audio_chunk)

            # Process
            new_probs = diarizer.process()

            # Update plot - use fast imshow instead of slow heatmap
            current_time = time.time()
            if current_time - last_update > 0.1:  # Update every 100ms (10Hz max)
                all_probs = diarizer.get_all_probs(include_provisional=True)
                confirmed_count = diarizer.pred_buffer.get_confirmed_count()
                provisional_count = diarizer.pred_buffer.provisional_frames

                if all_probs is not None and len(all_probs) > 0:
                    # Show last 200 frames (~16 seconds)
                    display_frames = min(200, len(all_probs))
                    display_probs = all_probs[-display_frames:]
                    
                    # Calculate where provisional predictions start
                    total_frames = len(all_probs)
                    display_start = total_frames - display_frames
                    provisional_start_in_display = max(0, confirmed_count - display_start)

                    # Initialize or update the image
                    if im is None:
                        ax.clear()
                        im = ax.imshow(display_probs.T, aspect='auto', cmap='viridis', 
                                       vmin=0, vmax=1, interpolation='nearest')
                        ax.set_yticks(range(4))
                        ax.set_yticklabels([f"Spk {i}" for i in range(4)])
                        ax.set_xlabel("Time (frames, 80ms each)")
                        ax.set_ylabel("Speaker")
                    else:
                        # Fast update - just change the data
                        im.set_data(display_probs.T)
                        im.set_extent([0, display_frames, 3.5, -0.5])
                    
                    # Update or create provisional marker line
                    if provisional_count > 0 and provisional_start_in_display < display_frames:
                        if vline is not None:
                            vline.set_xdata([provisional_start_in_display, provisional_start_in_display])
                        else:
                            vline = ax.axvline(x=provisional_start_in_display, color='red', 
                                               linestyle='--', linewidth=2, alpha=0.8)
                    elif vline is not None:
                        vline.set_xdata([-10, -10])  # Hide it off-screen
                    
                    # Update title
                    title = f"Live Diarization - Confirmed: {confirmed_count} | "
                    title += f"Provisional: {provisional_count} | "
                    title += f"Total: {len(all_probs)} frames ({len(all_probs) * 0.08:.1f}s)"
                    ax.set_title(title)

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                last_update = current_time

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop()
        plt.ioff()
        plt.close()

    # Final summary
    all_probs = diarizer.get_all_probs(include_provisional=False)  # Final = confirmed only
    provisional_count = diarizer.pred_buffer.provisional_frames
    if all_probs is not None:
        print(f"\nTotal confirmed: {len(all_probs)} frames ({len(all_probs) * 0.08:.1f} seconds)")
        if provisional_count > 0:
            print(f"(Plus {provisional_count} provisional frames that were not yet refined)")


def run_file_demo(model_name, coreml_dir, audio_path):
    """
    Run demo on audio file using NeMo preprocessing + CoreML inference.
    
    This uses NeMo's preprocessing and feature loader for proper feature alignment,
    ensuring the final predictions match what NeMo's forward_streaming produces.
    """

    print("=" * 70)
    print("File Demo with NeMo-Aligned CoreML Inference")
    print("=" * 70)

    # Load NeMo model
    print(f"\nLoading NeMo Model: {model_name}")
    nemo_model = SortformerEncLabelModel.from_pretrained(model_name, map_location=torch.device("cpu"))
    nemo_model.eval()

    # Configure streaming params
    modules = nemo_model.sortformer_modules
    modules.chunk_len = Config.chunk_len
    modules.chunk_right_context = Config.chunk_right_context
    modules.chunk_left_context = Config.chunk_left_context
    modules.fifo_len = Config.fifo_len
    modules.spkcache_len = Config.spkcache_len
    modules.spkcache_update_period = Config.spkcache_update_period

    if hasattr(nemo_model.preprocessor, 'featurizer'):
        nemo_model.preprocessor.featurizer.dither = 0.0
        nemo_model.preprocessor.featurizer.pad_to = 0

    # Load CoreML main model
    print(f"Loading CoreML Main Model from {coreml_dir}...")
    main_model = ct.models.MLModel(
        os.path.join(coreml_dir, "SortformerPipeline.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY
    )

    # Load audio file
    import librosa
    audio, _ = librosa.load(audio_path, sr=Config.sample_rate, mono=True)
    print(f"Loaded audio: {len(audio)} samples ({len(audio) / Config.sample_rate:.1f}s)")

    # Use NeMo preprocessing for proper feature alignment
    print("\nUsing NeMo preprocessing for feature alignment...")
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
    audio_length = torch.tensor([len(audio)], dtype=torch.long)
    
    with torch.no_grad():
        nemo_features, nemo_feat_len = nemo_model.process_signal(
            audio_signal=audio_tensor,
            audio_signal_length=audio_length
        )
    print(f"Features: {nemo_features.shape}, length: {nemo_feat_len.item()}")

    # Initialize state and prediction buffer
    state = modules.init_streaming_state(batch_size=1, device='cpu')
    pred_buffer = StreamingPredictionBuffer(num_speakers=4)

    # Setup plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 4))
    im = None
    vline = None

    print("\nRunning streaming inference...")

    # Use NeMo's streaming feature loader for proper chunk boundaries
    streaming_loader = modules.streaming_feat_loader(
        feat_seq=nemo_features,
        feat_seq_length=nemo_feat_len,
        feat_seq_offset=torch.zeros((1,), dtype=torch.long)
    )
    
    chunk_idx = 0
    try:
        for _, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in streaming_loader:
            lc = round(left_offset / modules.subsampling_factor)
            rc = round(right_offset / modules.subsampling_factor)
            
            # Prepare state for CoreML
            curr_spk_len = state.spkcache.shape[1]
            curr_fifo_len = state.fifo.shape[1]
            
            current_spkcache = state.spkcache
            if curr_spk_len < Config.spkcache_len:
                current_spkcache = torch.nn.functional.pad(
                    current_spkcache, (0, 0, 0, Config.spkcache_len - curr_spk_len)
                )
            elif curr_spk_len > Config.spkcache_len:
                current_spkcache = current_spkcache[:, :Config.spkcache_len, :]
                
            current_fifo = state.fifo
            if curr_fifo_len < Config.fifo_len:
                current_fifo = torch.nn.functional.pad(
                    current_fifo, (0, 0, 0, Config.fifo_len - curr_fifo_len)
                )
            elif curr_fifo_len > Config.fifo_len:
                current_fifo = current_fifo[:, :Config.fifo_len, :]
            
            # Pad chunk to fixed size for CoreML
            chunk_np = chunk_feat_seq_t.numpy().astype(np.float32)
            actual_len = chunk_np.shape[1]
            if actual_len < Config.chunk_frames:
                pad_len = Config.chunk_frames - actual_len
                chunk_np = np.pad(chunk_np, ((0,0), (0, pad_len), (0,0)), mode='constant')
            else:
                chunk_np = chunk_np[:, :Config.chunk_frames, :]
            
            # CoreML inference
            coreml_inputs = {
                "chunk": chunk_np,
                "chunk_lengths": np.array([actual_len], dtype=np.int32),
                "spkcache": current_spkcache.numpy().astype(np.float32),
                "spkcache_lengths": np.array([curr_spk_len], dtype=np.int32),
                "fifo": current_fifo.numpy().astype(np.float32),
                "fifo_lengths": np.array([curr_fifo_len], dtype=np.int32)
            }
            
            st_time = time.time_ns()
            coreml_out = main_model.predict(coreml_inputs)
            ed_time = time.time_ns()
            print(f"Chunk {chunk_idx}: {1e-6 * (ed_time - st_time):.2f}ms")
            
            coreml_preds = torch.from_numpy(coreml_out["speaker_preds"])
            chunk_embs = torch.from_numpy(coreml_out["chunk_pre_encoder_embs"])
            chunk_emb_len = int(coreml_out["chunk_pre_encoder_lengths"][0])
            chunk_embs = chunk_embs[:, :chunk_emb_len, :]
            
            # Update state using NeMo's streaming_update
            state, chunk_probs = modules.streaming_update(
                streaming_state=state,
                chunk=chunk_embs,
                preds=coreml_preds,
                lc=lc,
                rc=rc
            )
            
            # Core predictions (confirmed)
            core_probs = chunk_probs.squeeze(0).detach().cpu().numpy()
            
            # Provisional predictions from right context
            pred_len = coreml_preds.shape[1]
            rc_start = pred_len - rc if rc > 0 else pred_len
            provisional_probs = coreml_preds[0, rc_start:, :].detach().cpu().numpy() if rc > 0 else None
            
            # Update prediction buffer
            pred_buffer.update(core_probs, provisional_probs)
            
            # Update plot
            all_probs = pred_buffer.get_all_probs(include_provisional=True)
            confirmed_count = pred_buffer.get_confirmed_count()
            provisional_count = pred_buffer.provisional_frames

            if all_probs is not None and len(all_probs) > 0:
                if im is None:
                    ax.clear()
                    im = ax.imshow(all_probs.T, aspect='auto', cmap='viridis',
                                   vmin=0, vmax=1, interpolation='nearest')
                    ax.set_yticks(range(4))
                    ax.set_yticklabels([f"Spk {i}" for i in range(4)])
                    ax.set_xlabel("Time (frames, 80ms each)")
                    ax.set_ylabel("Speaker")
                else:
                    im.set_data(all_probs.T)
                    im.set_extent([0, len(all_probs), 3.5, -0.5])

                if provisional_count > 0:
                    if vline is not None:
                        vline.set_xdata([confirmed_count, confirmed_count])
                    else:
                        vline = ax.axvline(x=confirmed_count, color='red',
                                           linestyle='--', linewidth=2, alpha=0.8)
                elif vline is not None:
                    vline.set_xdata([-10, -10])

                title = f"Streaming Diarization - Confirmed: {confirmed_count} | "
                title += f"Provisional: {provisional_count} | "
                title += f"Total: {len(all_probs)} frames ({len(all_probs) * 0.08:.1f}s)"
                ax.set_title(title)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            chunk_idx += 1

    except KeyboardInterrupt:
        print("\nStopped.")

    plt.ioff()

    # Final results
    coreml_probs = pred_buffer.get_all_probs(include_provisional=False)
    provisional_count = pred_buffer.provisional_frames
    if coreml_probs is not None:
        print(f"\nTotal confirmed: {len(coreml_probs)} frames ({len(coreml_probs) * 0.08:.1f}s)")
        if provisional_count > 0:
            print(f"(Plus {provisional_count} provisional frames that were not yet refined)")
        plt.show()

    # ================================================================
    # VERIFICATION: Compare CoreML streaming output with NeMo streaming
    # ================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION: Comparing CoreML vs NeMo Streaming Inference")
    print("=" * 70)

    # Run NeMo streaming inference on the same audio
    nemo_model.streaming_mode = True

    with torch.no_grad():
        nemo_preds = nemo_model.forward_streaming(nemo_features, nemo_feat_len)

    nemo_probs = nemo_preds.squeeze(0).cpu().numpy()

    print(f"\nCoreML streaming frames: {len(coreml_probs)}")
    print(f"NeMo streaming frames:   {len(nemo_probs)}")

    # Align lengths for comparison
    min_len = min(len(coreml_probs), len(nemo_probs))
    coreml_aligned = coreml_probs[:min_len]
    nemo_aligned = nemo_probs[:min_len]

    # Compute differences
    abs_diff = np.abs(coreml_aligned - nemo_aligned)
    max_diff = abs_diff.max()
    mean_diff = abs_diff.mean()

    print(f"\nComparison (first {min_len} frames):")
    print(f"  Max absolute difference:  {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")

    for spk in range(coreml_aligned.shape[1]):
        corr = np.corrcoef(coreml_aligned[:, spk], nemo_aligned[:, spk])[0, 1]
        print(f"  Speaker {spk} correlation:  {corr:.6f}")

    if max_diff < 0.01:
        print("\n✓ PASS: CoreML streaming output matches NeMo streaming output!")
    elif max_diff < 0.05:
        print("\n⚠ WARNING: Minor differences detected (max diff < 0.05)")
        print("  This may be due to floating point precision differences.")
    else:
        print("\n✗ FAIL: Significant differences detected!")
        frame_max_diff = abs_diff.max(axis=1)
        worst_frames = np.argsort(frame_max_diff)[-5:][::-1]
        print(f"\n  Worst 5 frames:")
        for idx in worst_frames:
            print(f"    Frame {idx}: CoreML={coreml_aligned[idx]}, NeMo={nemo_aligned[idx]}")

    # Plot comparison if there are differences
    if max_diff >= 0.01:
        fig2, axes = plt.subplots(3, 1, figsize=(14, 10))

        im1 = axes[0].imshow(coreml_aligned.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title(f"CoreML Streaming ({len(coreml_aligned)} frames)")
        axes[0].set_ylabel("Speaker")
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(nemo_aligned.T, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title(f"NeMo Streaming ({len(nemo_aligned)} frames)")
        axes[1].set_ylabel("Speaker")
        plt.colorbar(im2, ax=axes[1])

        im3 = axes[2].imshow(abs_diff.T, aspect='auto', cmap='hot', vmin=0, vmax=max_diff)
        axes[2].set_title(f"Absolute Difference (max={max_diff:.4f}, mean={mean_diff:.4f})")
        axes[2].set_xlabel("Frame")
        axes[2].set_ylabel("Speaker")
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    parser.add_argument("--coreml_dir", default="coreml_models")
    parser.add_argument("--audio_path", default="audio.wav")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    args = parser.parse_args()

    if args.mic:
        run_mic_inference(args.model_name, args.coreml_dir)
    else:
        run_file_demo(args.model_name, args.coreml_dir, args.audio_path)
