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
import threading
import queue
import time
import math
import argparse

# Import NeMo for state management
from nemo.collections.asr.models import SortformerEncLabelModel

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not available. Install with: pip install sounddevice")


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'chunk_len': 6,
    'chunk_right_context': 1,
    'chunk_left_context': 1,
    'fifo_len': 40,
    'spkcache_len': 120,
    'spkcache_update_period': 32,
    'subsampling_factor': 8,
    'sample_rate': 16000,
    'mel_window': 400,
    'mel_stride': 160,
    
    # Audio settings
    'audio_chunk_samples': 1280,  # 80ms chunks from mic
    'channels': 1,
}

CONFIG['spkcache_input_len'] = CONFIG['spkcache_len']
CONFIG['fifo_input_len'] = CONFIG['fifo_len']
CONFIG['chunk_frames'] = (CONFIG['chunk_len'] + CONFIG['chunk_left_context'] + CONFIG['chunk_right_context']) * CONFIG['subsampling_factor']
CONFIG['preproc_audio_samples'] = (CONFIG['chunk_frames'] - 1) * CONFIG['mel_stride'] + CONFIG['mel_window']

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


class StreamingDiarizer:
    """Real-time streaming diarization using CoreML."""
    
    def __init__(self, nemo_model, preproc_model, main_model, config):
        self.modules = nemo_model.sortformer_modules
        self.preproc_model = preproc_model
        self.main_model = main_model
        self.config = config
        
        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # Feature buffer
        self.feature_buffer = None
        self.features_processed = 0
        
        # Diarization state
        self.state = self.modules.init_streaming_state(batch_size=1, device='cpu')
        self.all_probs = []  # List of [T, 4] arrays
        
        # Chunk tracking
        self.diar_chunk_idx = 0
        self.preproc_chunk_idx = 0
        
        # Derived params
        self.subsampling = config['subsampling_factor']
        self.core_frames = config['chunk_len'] * self.subsampling
        self.left_ctx = config['chunk_left_context'] * self.subsampling
        self.right_ctx = config['chunk_right_context'] * self.subsampling
        
        # Audio hop for preprocessor
        self.audio_hop = config['preproc_audio_samples'] - config['mel_window']
        self.overlap_frames = (config['mel_window'] - config['mel_stride']) // config['mel_stride'] + 1
        
    def add_audio(self, audio_chunk):
        """Add new audio samples."""
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        
    def process(self):
        """
        Process available audio through preprocessor and diarizer.
        Returns new probability frames if available.
        """
        new_probs = None
        
        # Step 1: Run preprocessor on available audio
        while len(self.audio_buffer) >= self.config['preproc_audio_samples']:
            audio_chunk = self.audio_buffer[:self.config['preproc_audio_samples']]
            
            preproc_inputs = {
                "audio_signal": audio_chunk.reshape(1, -1).astype(np.float32),
                "length": np.array([self.config['preproc_audio_samples']], dtype=np.int32)
            }
            
            preproc_out = self.preproc_model.predict(preproc_inputs)
            feat_chunk = np.array(preproc_out["features"])
            feat_len = int(preproc_out["feature_lengths"][0])
            
            if self.preproc_chunk_idx == 0:
                valid_feats = feat_chunk[:, :, :feat_len]
            else:
                valid_feats = feat_chunk[:, :, self.overlap_frames:feat_len]
            
            if self.feature_buffer is None:
                self.feature_buffer = valid_feats
            else:
                self.feature_buffer = np.concatenate([self.feature_buffer, valid_feats], axis=2)
            
            self.audio_buffer = self.audio_buffer[self.audio_hop:]
            self.preproc_chunk_idx += 1
        
        if self.feature_buffer is None:
            return None
            
        # Step 2: Run diarization on available features
        total_features = self.feature_buffer.shape[2]
        
        while True:
            # Calculate chunk boundaries
            chunk_start = self.diar_chunk_idx * self.core_frames
            chunk_end = chunk_start + self.core_frames
            
            # Need right context
            required_features = chunk_end + self.right_ctx
            
            if required_features > total_features:
                break  # Not enough features yet
            
            # Extract with context
            left_offset = min(self.left_ctx, chunk_start)
            right_offset = min(self.right_ctx, total_features - chunk_end)
            
            feat_start = chunk_start - left_offset
            feat_end = chunk_end + right_offset
            
            chunk_feat = self.feature_buffer[:, :, feat_start:feat_end]
            chunk_feat_tensor = torch.from_numpy(chunk_feat).float()
            actual_len = chunk_feat.shape[2]
            
            # Transpose to [B, T, D]
            chunk_t = chunk_feat_tensor.transpose(1, 2)
            
            # Pad if needed
            if actual_len < self.config['chunk_frames']:
                pad_len = self.config['chunk_frames'] - actual_len
                chunk_in = torch.nn.functional.pad(chunk_t, (0, 0, 0, pad_len))
            else:
                chunk_in = chunk_t[:, :self.config['chunk_frames'], :]
            
            # State preparation
            curr_spk_len = self.state.spkcache.shape[1]
            curr_fifo_len = self.state.fifo.shape[1]
            
            current_spkcache = self.state.spkcache
            if curr_spk_len < self.config['spkcache_input_len']:
                current_spkcache = torch.nn.functional.pad(
                    current_spkcache, (0, 0, 0, self.config['spkcache_input_len'] - curr_spk_len)
                )
            elif curr_spk_len > self.config['spkcache_input_len']:
                current_spkcache = current_spkcache[:, :self.config['spkcache_input_len'], :]
            
            current_fifo = self.state.fifo
            if curr_fifo_len < self.config['fifo_input_len']:
                current_fifo = torch.nn.functional.pad(
                    current_fifo, (0, 0, 0, self.config['fifo_input_len'] - curr_fifo_len)
                )
            elif curr_fifo_len > self.config['fifo_input_len']:
                current_fifo = current_fifo[:, :self.config['fifo_input_len'], :]
            
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
            print(f"duration: {1e-6 * (ed_time - st_time)}")

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
            
            # Store probabilities
            probs_np = chunk_probs.squeeze(0).detach().cpu().numpy()
            self.all_probs.append(probs_np)
            
            new_probs = probs_np
            self.diar_chunk_idx += 1
        
        return new_probs
    
    def get_all_probs(self):
        """Get all accumulated probabilities."""
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
    nemo_model = SortformerEncLabelModel.from_pretrained(model_name, map_location="cpu")
    nemo_model.eval()
    
    # Configure
    modules = nemo_model.sortformer_modules
    modules.chunk_len = CONFIG['chunk_len']
    modules.chunk_right_context = CONFIG['chunk_right_context']
    modules.chunk_left_context = CONFIG['chunk_left_context']
    modules.fifo_len = CONFIG['fifo_len']
    modules.spkcache_len = CONFIG['spkcache_len']
    modules.spkcache_update_period = CONFIG['spkcache_update_period']
    
    if hasattr(nemo_model.preprocessor, 'featurizer'):
        nemo_model.preprocessor.featurizer.dither = 0.0
        nemo_model.preprocessor.featurizer.pad_to = 0
    
    # Load CoreML models
    print(f"Loading CoreML Models from {coreml_dir}...")
    preproc_model = ct.models.MLModel(
        os.path.join(coreml_dir, "Pipeline_Preprocessor.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY
    )
    main_model = ct.models.MLModel(
        os.path.join(coreml_dir, "SortformerPipeline.mlpackage"),
        compute_units=ct.ComputeUnit.ALL
    )
    
    # Create diarizer
    diarizer = StreamingDiarizer(nemo_model, preproc_model, main_model, CONFIG)
    
    # Audio queue
    audio_queue = queue.Queue()
    
    # Start microphone
    mic = MicrophoneStream(
        sample_rate=CONFIG['sample_rate'],
        chunk_size=CONFIG['audio_chunk_samples'],
        audio_queue=audio_queue
    )
    
    if not mic.start():
        return
    
    # Setup plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 4))
    
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
            
            # Update plot periodically
            if time.time() - last_update > 0.16:  # Update every 160ms
                all_probs = diarizer.get_all_probs()
                
                if all_probs is not None and len(all_probs) > 0:
                    ax.clear()
                    
                    # Show last 200 frames (~16 seconds)
                    display_frames = min(200, len(all_probs))
                    display_probs = all_probs[-display_frames:]
                    
                    sns.heatmap(
                        display_probs.T,
                        ax=ax,
                        cmap="viridis",
                        vmin=0, vmax=1,
                        yticklabels=[f"Spk {i}" for i in range(4)],
                        cbar=False
                    )
                    
                    ax.set_xlabel("Time (frames, 80ms each)")
                    ax.set_ylabel("Speaker")
                    ax.set_title(f"Live Diarization - Total: {len(all_probs)} frames ({len(all_probs)*0.08:.1f}s)")
                    
                    plt.draw()
                    plt.pause(0.01)
                
                last_update = time.time()
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        mic.stop()
        plt.ioff()
        plt.close()
    
    # Final summary
    all_probs = diarizer.get_all_probs()
    if all_probs is not None:
        print(f"\nTotal processed: {len(all_probs)} frames ({len(all_probs)*0.08:.1f} seconds)")


def run_file_demo(model_name, coreml_dir, audio_path):
    """Run demo on audio file with live updating plot."""
    
    print("=" * 70)
    print("File Demo with Live Updating Plot")
    print("=" * 70)
    
    # Load NeMo model
    print(f"\nLoading NeMo Model: {model_name}")
    nemo_model = SortformerEncLabelModel.from_pretrained(model_name, map_location="cpu")
    nemo_model.eval()
    
    # Configure
    modules = nemo_model.sortformer_modules
    modules.chunk_len = CONFIG['chunk_len']
    modules.chunk_right_context = CONFIG['chunk_right_context']
    modules.chunk_left_context = CONFIG['chunk_left_context']
    modules.fifo_len = CONFIG['fifo_len']
    modules.spkcache_len = CONFIG['spkcache_len']
    modules.spkcache_update_period = CONFIG['spkcache_update_period']
    
    if hasattr(nemo_model.preprocessor, 'featurizer'):
        nemo_model.preprocessor.featurizer.dither = 0.0
        nemo_model.preprocessor.featurizer.pad_to = 0
    
    # Load CoreML models
    print(f"Loading CoreML Models from {coreml_dir}...")
    preproc_model = ct.models.MLModel(
        os.path.join(coreml_dir, "Pipeline_Preprocessor.mlpackage"),
        compute_units=ct.ComputeUnit.CPU_ONLY
    )
    main_model = ct.models.MLModel(
        os.path.join(coreml_dir, "SortformerPipeline.mlpackage"),
        compute_units=ct.ComputeUnit.ALL
    )
    
    # Load audio file
    import librosa
    audio, _ = librosa.load(audio_path, sr=CONFIG['sample_rate'], mono=True)
    print(f"Loaded audio: {len(audio)} samples ({len(audio)/CONFIG['sample_rate']:.1f}s)")
    
    # Create diarizer
    diarizer = StreamingDiarizer(nemo_model, preproc_model, main_model, CONFIG)
    
    # Setup plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Simulate streaming
    chunk_size = CONFIG['audio_chunk_samples']
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
        print(f"\nTotal: {len(all_probs)} frames ({len(all_probs)*0.08:.1f}s)")
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
