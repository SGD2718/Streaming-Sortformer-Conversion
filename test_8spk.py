"""
Test the expanded 8-speaker model on audio.wav and save output as image.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import os
from nemo.collections.asr.models import SortformerEncLabelModel

# Use non-interactive backend
matplotlib.use("Agg")

# --- Config ---
device = torch.device("cpu")
script_dir = os.path.dirname(os.path.abspath(__file__))
audio_file = os.path.join(script_dir, "audio.wav")
model_path = os.path.join(script_dir, "streaming-sortformer-8spk-smart.nemo")
output_image = os.path.join(script_dir, "8spk_test_output.png")

print("=" * 60)
print("Testing 8-Speaker Model")
print("=" * 60)

# --- Load Model ---
print(f"\nLoading model from: {model_path}")
model = SortformerEncLabelModel.restore_from(model_path, map_location=device)
model.eval()
print(f"  n_spk: {model.sortformer_modules.n_spk}")

# --- Configure streaming params ---
model.sortformer_modules.chunk_len = 4
model.sortformer_modules.chunk_right_context = 1
model.sortformer_modules.chunk_left_context = 1
model.sortformer_modules.fifo_len = 188
model.sortformer_modules.spkcache_update_period = 144
model.sortformer_modules.spkcache_len = 188

# --- Run Inference ---
print(f"\nRunning inference on: {audio_file}")
predicted_segments, predicted_probs = model.diarize(
    audio=audio_file,
    batch_size=1,
    include_tensor_outputs=True
)

# --- Process Output ---
probs = predicted_probs[0].squeeze().cpu().numpy()
heatmap_data = probs.T

print(f"\n  Predictions shape: {probs.shape}")
print(f"  Heatmap shape: {heatmap_data.shape} (speakers x frames)")

# --- Plotting ---
fig, ax = plt.subplots(figsize=(15, 8))

sns.heatmap(
    heatmap_data,
    cmap="viridis",
    yticklabels=[f"Spk {i}" for i in range(heatmap_data.shape[0])],
    cbar_kws={'label': 'Activity Probability'},
    vmin=0, vmax=1,
    ax=ax
)

# Count active speakers per frame
active_per_frame = (probs > 0.5).sum(axis=1)
max_active = active_per_frame.max()
any_activity = (probs > 0.5).any(axis=0)
speakers_with_activity = any_activity.sum()

plt.title(f"8-Speaker Diarization Test\n(Max active at once: {max_active}, Speakers detected: {speakers_with_activity})")
plt.xlabel("Time Frames (80ms steps)")
plt.ylabel("Speaker ID")
plt.tight_layout()

# Save the figure
plt.savefig(output_image, dpi=150)
print(f"\n  Saved plot to: {output_image}")

# --- Print summary ---
print("\n" + "=" * 60)
print("Speaker Activity Summary:")
print("=" * 60)
for spk_idx in range(heatmap_data.shape[0]):
    activity = heatmap_data[spk_idx]
    active_frames = (activity > 0.5).sum()
    max_prob = activity.max()
    if active_frames > 0 or max_prob > 0.3:
        print(f"  Speaker {spk_idx}: {active_frames} frames active, max prob: {max_prob:.3f}")

print("\n✅ Test complete!")
