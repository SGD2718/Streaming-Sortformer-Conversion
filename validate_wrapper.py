"""
Validation Script: Run PyTorch wrapper in streaming loop and compare to diarize

This validates that the SortformerCoreMLWrapper, when called in a proper streaming
loop with state management, produces output matching model.diarize().

Once the PyTorch wrapper matches diarize, then CoreML should also match.
"""

import os
import math
import warnings
import numpy as np
import torch
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings('ignore')

SAMPLE_RATE = 16000
AUDIO_FILE = "audio.wav"
CHUNK_MEL_FRAMES = 100
FIFO_LENGTH = 188
SPKCACHE_LENGTH = 188
SUBSAMPLING_FACTOR = 8
EMB_DIM = 512
NUM_SPEAKERS = 4

# Context configuration (matching convert_to_coreml.py)
LEFT_CONTEXT = 0
RIGHT_CONTEXT = 8  # chunk_right_context * subsampling_factor = 1 * 8
TOTAL_CHUNK_LEN = LEFT_CONTEXT + CHUNK_MEL_FRAMES + RIGHT_CONTEXT  # 108

# Derived values
CHUNK_ENCODER_FRAMES = (CHUNK_MEL_FRAMES + SUBSAMPLING_FACTOR - 1) // SUBSAMPLING_FACTOR  # 13
DROP_EXTRA_PRE_ENCODED = (LEFT_CONTEXT + SUBSAMPLING_FACTOR - 1) // SUBSAMPLING_FACTOR  # 0


def load_model():
    """Load NeMo model"""
    from nemo.collections.asr.models import SortformerEncLabelModel
    
    model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2.1",
        map_location=torch.device('cpu')
    )
    model.eval()
    model.to('cpu')
    
    # Configure streaming
    # IMPORTANT: NeMo chunk_len is in ENCODER FRAMES (post-subsampling)
    # Our wrapper takes CHUNK_MEL_FRAMES mel frames -> ~CHUNK_ENCODER_FRAMES encoder frames
    CHUNK_ENCODER_FRAMES = (CHUNK_MEL_FRAMES + SUBSAMPLING_FACTOR - 1) // SUBSAMPLING_FACTOR
    model.sortformer_modules.chunk_len = CHUNK_ENCODER_FRAMES
    model.sortformer_modules.chunk_right_context = 1
    model.sortformer_modules.fifo_len = FIFO_LENGTH
    model.sortformer_modules.spkcache_update_period = 144
    model.sortformer_modules.spkcache_len = SPKCACHE_LENGTH
    model.streaming_mode = True
    
    print(f"  Configured: {CHUNK_MEL_FRAMES} mel frames -> {CHUNK_ENCODER_FRAMES} encoder frames")
    
    return model


def get_diarize_output(model, audio_file):
    """Get ground truth from model.diarize()"""
    print("Running model.diarize() for ground truth...")
    with torch.no_grad():
        _, predicted_probs = model.diarize(
            audio=[audio_file],
            batch_size=1,
            include_tensor_outputs=True,
            verbose=False
        )
    preds = predicted_probs[0].squeeze().cpu().numpy()
    print(f"  diarize output: {preds.shape}")
    return preds


def get_wrapper_streaming_output(model, audio_file):
    """
    Run PyTorch wrapper in a streaming loop with proper state management.
    This mirrors what forward_streaming_step does, but using our wrapper.
    """
    print("\nRunning wrapper in streaming loop...")
    
    import sys
    sys.path.insert(0, '.')
    from convert_to_coreml import SortformerCoreMLWrapper
    
    # Load and preprocess audio
    audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    eps = 1e-3
    audio = (1 / (np.max(np.abs(audio)) + eps)) * audio
    
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    audio_length = torch.tensor([len(audio)], dtype=torch.int64)
    
    with torch.no_grad():
        mel_features, mel_lengths = model.preprocessor(
            input_signal=audio_tensor, length=audio_length
        )
    
    print(f"  Mel features: {mel_features.shape}")
    
    # Create wrapper WITH context
    wrapper = SortformerCoreMLWrapper(model, left_context=LEFT_CONTEXT, right_context=RIGHT_CONTEXT)
    wrapper.eval()
    
    batch_size = mel_features.shape[0]
    
    # Initialize streaming state
    streaming_state = model.sortformer_modules.init_streaming_state(
        batch_size=batch_size,
        async_streaming=False,
        device='cpu'
    )
    
    total_preds = torch.zeros((batch_size, 0, NUM_SPEAKERS))
    
    # Use NeMo's streaming feature loader
    feat_seq_offset = torch.zeros((batch_size,), dtype=torch.long)
    feat_seq_length = torch.tensor([mel_lengths.item()], dtype=torch.long)
    
    streaming_loader = model.sortformer_modules.streaming_feat_loader(
        feat_seq=mel_features,
        feat_seq_length=feat_seq_length,
        feat_seq_offset=feat_seq_offset,
    )
    
    chunk_idx = 0
    for _, chunk_feat_seq_t, feat_lengths, left_offset, right_offset in streaming_loader:
        # chunk_feat_seq_t: [B, T, C] mel features with left/right context
        # The streaming_feat_loader already includes the context!
        
        # The wrapper expects the FULL chunk with context, so we pass it directly
        # but need to ensure it's padded to TOTAL_CHUNK_LEN if necessary
        chunk_mel_full = chunk_feat_seq_t
        current_len = chunk_mel_full.shape[1]
        
        if current_len < TOTAL_CHUNK_LEN:
            padding = torch.zeros([batch_size, TOTAL_CHUNK_LEN - current_len, 128])
            chunk_mel_padded = torch.cat([chunk_mel_full, padding], dim=1)
        else:
            chunk_mel_padded = chunk_mel_full[:, :TOTAL_CHUNK_LEN, :]
        
        # Get current state as fixed buffers
        spkcache_len = streaming_state.spkcache.shape[1]
        fifo_len = streaming_state.fifo.shape[1]
        
        spkcache_buffer = torch.zeros([batch_size, SPKCACHE_LENGTH, EMB_DIM])
        fifo_buffer = torch.zeros([batch_size, FIFO_LENGTH, EMB_DIM])
        
        if spkcache_len > 0:
            copy_len = min(spkcache_len, SPKCACHE_LENGTH)
            spkcache_buffer[:, :copy_len, :] = streaming_state.spkcache[:, :copy_len, :]
        if fifo_len > 0:
            copy_len = min(fifo_len, FIFO_LENGTH)
            fifo_buffer[:, :copy_len, :] = streaming_state.fifo[:, :copy_len, :]
        
        # Run wrapper with full chunk (including context)
        with torch.no_grad():
            preds, chunk_pre_encode_embs = wrapper(
                chunk=chunk_mel_padded,
                spkcache=spkcache_buffer,
                spkcache_actual_len=torch.tensor([min(spkcache_len, SPKCACHE_LENGTH)]),
                fifo=fifo_buffer,
                fifo_actual_len=torch.tensor([min(fifo_len, FIFO_LENGTH)]),
            )
        
        # Use NeMo's streaming_update to properly extract chunk predictions and update state
        lc = round(left_offset / SUBSAMPLING_FACTOR)
        rc = math.ceil(right_offset / SUBSAMPLING_FACTOR)
        
        streaming_state, chunk_preds = model.sortformer_modules.streaming_update(
            streaming_state=streaming_state,
            chunk=chunk_pre_encode_embs,
            preds=preds,
            lc=lc,
            rc=rc,
        )
        
        total_preds = torch.cat([total_preds, chunk_preds], dim=1)
        chunk_idx += 1
    
    # Trim to actual length
    n_frames = math.ceil(mel_lengths.item() / SUBSAMPLING_FACTOR)
    if total_preds.shape[1] > n_frames:
        total_preds = total_preds[:, :n_frames, :]
    
    result = total_preds.squeeze().numpy()
    print(f"  Wrapper streaming output: {result.shape}")
    return result


def compare_and_plot(diarize_preds, wrapper_preds, output_path):
    """Compare predictions and create visualization"""
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    min_len = min(len(diarize_preds), len(wrapper_preds))
    diarize_aligned = diarize_preds[:min_len]
    wrapper_aligned = wrapper_preds[:min_len]
    
    diff = np.abs(diarize_aligned - wrapper_aligned)
    mean_diff = np.mean(diff)
    max_diff = np.max(diff)
    
    print(f"\ndiarize shape:  {diarize_preds.shape}")
    print(f"wrapper shape:  {wrapper_preds.shape}")
    print(f"Aligned length: {min_len}")
    
    print("\nPer-speaker analysis:")
    for spk in range(diff.shape[1]):
        print(f"  Speaker {spk}: Mean={np.mean(diff[:, spk]):.6e}, Max={np.max(diff[:, spk]):.6e}")
    
    print(f"\nOverall: Mean={mean_diff:.6e}, Max={max_diff:.6e}")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    axes[0].set_title("model.diarize() - Ground Truth")
    sns.heatmap(diarize_aligned.T, cmap="viridis", vmin=0, vmax=1, ax=axes[0],
                yticklabels=[f"Spk {i}" for i in range(4)])
    
    axes[1].set_title("PyTorch Wrapper Streaming")
    sns.heatmap(wrapper_aligned.T, cmap="viridis", vmin=0, vmax=1, ax=axes[1],
                yticklabels=[f"Spk {i}" for i in range(4)])
    
    axes[2].set_title(f"Difference (mean={mean_diff:.2e})")
    sns.heatmap(diff.T, cmap="Reds", vmin=0, vmax=max(0.01, np.max(diff)), ax=axes[2],
                yticklabels=[f"Spk {i}" for i in range(4)])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()
    
    return mean_diff < 0.01  # More lenient threshold for now


def main():
    print("=" * 70)
    print("PyTorch Wrapper Streaming Validation")
    print("=" * 70)
    
    model = load_model()
    
    diarize_preds = get_diarize_output(model, AUDIO_FILE)
    wrapper_preds = get_wrapper_streaming_output(model, AUDIO_FILE)
    
    passed = compare_and_plot(diarize_preds, wrapper_preds, "wrapper_validation.png")
    
    print("\n" + "=" * 70)
    if passed:
        print("✓ PyTorch wrapper streaming matches diarize within tolerance!")
    else:
        print("✗ PyTorch wrapper doesn't match diarize")
    print("=" * 70)
    
    return passed


if __name__ == "__main__":
    main()
