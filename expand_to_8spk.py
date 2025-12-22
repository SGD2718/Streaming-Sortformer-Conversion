"""
Expand 4-speaker Sortformer model to 8 speakers.

This script:
1. Loads the pretrained 4-speaker streaming Sortformer model
2. Replaces the output layers to support 8 speakers
3. Copies the pretrained weights for the first 4 speakers
4. Saves the expanded model ready for fine-tuning
"""

import torch
import torch.nn as nn
import os
from nemo.collections.asr.models import SortformerEncLabelModel
from omegaconf import OmegaConf

# Configuration
OLD_NUM_SPKS = 4
NEW_NUM_SPKS = 8
OUTPUT_PATH = "streaming-sortformer-8spk.nemo"

device = torch.device("cpu")  # Use CPU for model surgery

print("=" * 60)
print("Expanding Sortformer from 4 speakers to 8 speakers")
print("=" * 60)

# --- 1. Load the pretrained 4-speaker model ---
print("\n[1/5] Loading pretrained 4-speaker model...")
model = SortformerEncLabelModel.from_pretrained(
    "nvidia/diar_streaming_sortformer_4spk-v2.1",
    map_location=device
)
model.eval()

# Print original model info
print(f"  Original n_spk: {model.sortformer_modules.n_spk}")
print(f"  hidden_to_spks: {model.sortformer_modules.hidden_to_spks}")
print(f"  single_hidden_to_spks: {model.sortformer_modules.single_hidden_to_spks}")

# --- 2. Get the dimensions from existing layers ---
print("\n[2/5] Extracting layer dimensions...")
hidden_size = model.sortformer_modules.hidden_size
print(f"  hidden_size (tf_d_model): {hidden_size}")

# Get old layer weights
old_hidden_to_spks = model.sortformer_modules.hidden_to_spks
old_single_hidden_to_spks = model.sortformer_modules.single_hidden_to_spks

print(f"  old hidden_to_spks shape: weight={old_hidden_to_spks.weight.shape}, bias={old_hidden_to_spks.bias.shape}")
print(f"  old single_hidden_to_spks shape: weight={old_single_hidden_to_spks.weight.shape}, bias={old_single_hidden_to_spks.bias.shape}")

# --- 3. Create new layers with 8 speaker outputs ---
print("\n[3/5] Creating new 8-speaker output layers...")

# Create new hidden_to_spks: Linear(2 * hidden_size, 8)
new_hidden_to_spks = nn.Linear(2 * hidden_size, NEW_NUM_SPKS)
# Initialize with Xavier uniform (same as original)
torch.nn.init.xavier_uniform_(new_hidden_to_spks.weight)
new_hidden_to_spks.bias.data.fill_(0.01)

# Copy weights for first 4 speakers from pretrained model
with torch.no_grad():
    new_hidden_to_spks.weight[:OLD_NUM_SPKS, :] = old_hidden_to_spks.weight.clone()
    new_hidden_to_spks.bias[:OLD_NUM_SPKS] = old_hidden_to_spks.bias.clone()

print(f"  new hidden_to_spks shape: weight={new_hidden_to_spks.weight.shape}, bias={new_hidden_to_spks.bias.shape}")

# Create new single_hidden_to_spks: Linear(hidden_size, 8)
new_single_hidden_to_spks = nn.Linear(hidden_size, NEW_NUM_SPKS)
torch.nn.init.xavier_uniform_(new_single_hidden_to_spks.weight)
new_single_hidden_to_spks.bias.data.fill_(0.01)

# Copy weights for first 4 speakers from pretrained model
with torch.no_grad():
    new_single_hidden_to_spks.weight[:OLD_NUM_SPKS, :] = old_single_hidden_to_spks.weight.clone()
    new_single_hidden_to_spks.bias[:OLD_NUM_SPKS] = old_single_hidden_to_spks.bias.clone()

print(f"  new single_hidden_to_spks shape: weight={new_single_hidden_to_spks.weight.shape}, bias={new_single_hidden_to_spks.bias.shape}")

# --- 4. Replace the layers in the model ---
print("\n[4/5] Replacing layers in model...")

# Replace the linear layers
model.sortformer_modules.hidden_to_spks = new_hidden_to_spks
model.sortformer_modules.single_hidden_to_spks = new_single_hidden_to_spks

# Update the n_spk attribute
model.sortformer_modules.n_spk = NEW_NUM_SPKS

# Update the config
model._cfg.max_num_of_spks = NEW_NUM_SPKS
model._cfg.sortformer_modules.num_spks = NEW_NUM_SPKS

# Update speaker permutations (used for training with PIL loss)
import itertools
speaker_inds = list(range(NEW_NUM_SPKS))
model.speaker_permutations = torch.tensor(list(itertools.permutations(speaker_inds)))

print(f"  Updated n_spk: {model.sortformer_modules.n_spk}")
print(f"  Updated max_num_of_spks in config: {model._cfg.max_num_of_spks}")

# --- 5. Save the expanded model ---
print("\n[5/5] Saving expanded model...")

# Get script directory for output path
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, OUTPUT_PATH)

model.save_to(output_path)
print(f"  Saved to: {output_path}")

# --- Verification ---
print("\n" + "=" * 60)
print("Verification: Loading saved model...")
print("=" * 60)

# Reload and verify
loaded_model = SortformerEncLabelModel.restore_from(output_path, map_location=device)
print(f"  Loaded n_spk: {loaded_model.sortformer_modules.n_spk}")
print(f"  Loaded max_num_of_spks: {loaded_model._cfg.max_num_of_spks}")
print(f"  hidden_to_spks output dim: {loaded_model.sortformer_modules.hidden_to_spks.out_features}")
print(f"  single_hidden_to_spks output dim: {loaded_model.sortformer_modules.single_hidden_to_spks.out_features}")

# Quick inference test with dummy data
print("\n[Test] Running inference test...")
loaded_model.eval()
with torch.no_grad():
    # Create dummy audio (1 second at 16kHz)
    dummy_audio = torch.randn(1, 16000)
    dummy_length = torch.tensor([16000])
    
    try:
        preds = loaded_model.forward(dummy_audio, dummy_length)
        print(f"  ✓ Inference successful!")
        print(f"  Output shape: {preds.shape}")
        print(f"  Expected: (batch=1, frames, speakers={NEW_NUM_SPKS})")
        
        if preds.shape[-1] == NEW_NUM_SPKS:
            print(f"\n✅ SUCCESS: Model expanded to {NEW_NUM_SPKS} speakers!")
        else:
            print(f"\n❌ ERROR: Output has {preds.shape[-1]} speakers, expected {NEW_NUM_SPKS}")
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")

print("\n" + "=" * 60)
print("Done! The model is ready for fine-tuning on 8-speaker data.")
print("=" * 60)
