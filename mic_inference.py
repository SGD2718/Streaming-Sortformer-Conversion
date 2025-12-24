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
        Returns new probability frames if available (including provisional).
        
        Sliding window preprocessing for ~2Hz updates:
        - First run: Wait for coreml_audio_size samples (full window)
        - After: Run every audio_hop samples using sliding window
        """
        new_probs = None

        # Step 1: Run preprocessor with sliding window
        total_audio = len(self.audio_buffer)
        
        # Always need coreml_audio_size samples to run
        # After first run, we keep (coreml_audio_size - audio_hop) samples,
        # so we need audio_hop new samples to reach coreml_audio_size again
        run_threshold = self.coreml_audio_size
        
        while total_audio >= run_threshold:
            # Use the most recent coreml_audio_size samples
            audio_chunk = self.audio_buffer[-self.coreml_audio_size:].copy()
            
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
                # Subsequent runs: only take the new features
                # audio_hop samples = audio_hop / mel_stride feature frames
                new_feat_count = self.audio_hop // Config.mel_stride
                valid_feats = feat_chunk[:, :, feat_len - new_feat_count:feat_len]
            
            if self.feature_buffer is None:
                self.feature_buffer = valid_feats
            else:
                self.feature_buffer = np.concatenate([self.feature_buffer, valid_feats], axis=2)

            # Remove processed audio (keep only what we need for next sliding window)
            keep_samples = self.coreml_audio_size - self.audio_hop
            if len(self.audio_buffer) > keep_samples:
                self.audio_buffer = self.audio_buffer[-keep_samples:]
            
            # Update for next iteration
            total_audio = len(self.audio_buffer)
            run_threshold = self.coreml_audio_size

        if self.feature_buffer is None:
            return None

        # Step 2: Run diarization on available features
        total_features = self.feature_buffer.shape[2]

        while True:
            # Calculate chunk boundaries
            chunk_start = self.diar_chunk_idx * self.core_frames
            chunk_end = chunk_start + self.core_frames

            # RIGHT CONTEXT CREATES LATENCY:
            # We need core + full right_ctx features before we can output CONFIRMED predictions
            # Confirmed predictions are always right_ctx frames BEHIND the latest audio
            required_features = chunk_end + self.right_ctx  # Need full right context
            
            if required_features > total_features:
                break  # Not enough features yet - wait for more audio

            # We have full context - extract with full left and right context
            left_offset = min(self.left_ctx, chunk_start)
            right_offset = self.right_ctx  # Always full right context now

            feat_start = chunk_start - left_offset
            feat_end = chunk_end + right_offset

            chunk_feat = self.feature_buffer[:, :, feat_start:feat_end]
            chunk_feat_tensor = torch.from_numpy(chunk_feat).float()
            actual_len = chunk_feat.shape[2]

            # Transpose to [B, T, D]
            chunk_t = chunk_feat_tensor.transpose(1, 2)

            # Pad to full chunk_frames (handles partial right context)
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
            rc = Config.chunk_right_context  # Always 7 now since we wait for full right context

            # Get state lengths for indexing into predictions
            spkcache_len = self.state.spkcache.shape[1]
            fifo_len = self.state.fifo.shape[1]
            core_len = Config.chunk_len  # 6
            
            # Extract core predictions (confirmed) and right context predictions (provisional)
            core_start = spkcache_len + fifo_len + lc
            core_end = core_start + core_len
            rc_end = core_end + rc  # Full 7 provisional frames
            
            # Core predictions - confirmed, always 6 frames behind right context
            core_probs = pred_logits[0, core_start:core_end, :].detach().cpu().numpy()
            
            # Provisional predictions - always 7 frames (full right context)
            provisional_probs = pred_logits[0, core_end:rc_end, :].detach().cpu().numpy()

            # Update streaming state
            self.state, chunk_probs = self.modules.streaming_update(
                streaming_state=self.state,
                chunk=chunk_embs,
                preds=pred_logits,
                lc=lc,
                rc=rc
            )

            # Update the prediction buffer with core and provisional predictions
            self.pred_buffer.update(core_probs, provisional_probs)
            
            # Also maintain backward compatibility with all_probs list
            probs_np = chunk_probs.squeeze(0).detach().cpu().numpy()
            self.all_probs.append(probs_np)

            new_probs = probs_np
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
    """Run demo on audio file with live updating plot."""

    print("=" * 70)
    print("File Demo with Live Updating Plot")
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

    # Load audio file
    import librosa
    audio, _ = librosa.load(audio_path, sr=Config.sample_rate, mono=True)
    print(f"Loaded audio: {len(audio)} samples ({len(audio) / Config.sample_rate:.1f}s)")

    # Create diarizer
    diarizer = StreamingDiarizer(nemo_model, preproc_model, main_model)

    # Setup plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 4))

    # Simulate streaming
    chunk_size = int(Config.sample_rate * Config.frame_duration)
    offset = 0

    print("\nStreaming audio with live plot...")

    try:
        while offset < len(audio):
            # Add audio chunk
            chunk_end = min(offset + chunk_size, len(audio))
            audio_chunk = audio[offset:chunk_end]
            diarizer.add_audio(audio_chunk)
            offset = chunk_end

            # Process
            diarizer.process()

            # Update plot
            all_probs = diarizer.get_all_probs()

            if all_probs is not None and len(all_probs) > 0:
                ax.clear()

                sns.heatmap(
                    all_probs.T,
                    ax=ax,
                    cmap="viridis",
                    vmin=0, vmax=1,
                    yticklabels=[f"Spk {i}" for i in range(4)],
                    cbar=False
                )

                ax.set_xlabel("Time (frames, 80ms each)")
                ax.set_ylabel("Speaker")
                ax.set_title(f"Streaming Diarization - {len(all_probs)} frames")

                plt.draw()
                plt.pause(0.05)

            # Simulate real-time (optional - comment out for fast mode)
            # time.sleep(chunk_size / CONFIG['sample_rate'])

    except KeyboardInterrupt:
        print("\nStopped.")

    plt.ioff()

    # Final plot
    all_probs = diarizer.get_all_probs()
    if all_probs is not None:
        print(f"\nTotal: {len(all_probs)} frames ({len(all_probs) * 0.08:.1f}s)")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="nvidia/diar_streaming_sortformer_4spk-v2.1")
    parser.add_argument("--coreml_dir", default="coreml_models")
    parser.add_argument("--audio_path", default="audio.wav")
    parser.add_argument("--mic", action="store_true", help="Use microphone input")
    args = parser.parse_args()

    run_mic_inference(args.model_name, args.coreml_dir)
    # if args.mic:
    # else:
    #     run_file_demo(args.model_name, args.coreml_dir, args.audio_path)
