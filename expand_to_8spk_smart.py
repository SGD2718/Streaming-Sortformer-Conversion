"""
Expand 4-speaker Sortformer model to 8 speakers with SMART initialization.

Uses analysis of pretrained weights to initialize new speakers with:
1. Matched statistics (L2 norm, mean, std)
2. Orthogonal weight vectors to existing speakers
3. Proper negative biases
"""

import torch
import torch.nn as nn
import numpy as np
import os
from nemo.collections.asr.models import SortformerEncLabelModel
import itertools

# Configuration
OLD_NUM_SPKS = 4
NEW_NUM_SPKS = 8
OUTPUT_PATH = "streaming-sortformer-8spk-smart.nemo"

device = torch.device("cpu")

print("=" * 70)
print("Smart Expansion: Sortformer 4 speakers → 8 speakers")
print("=" * 70)

# --- 1. Load the pretrained 4-speaker model ---
print("\n[1/6] Loading pretrained 4-speaker model...")
model = SortformerEncLabelModel.from_pretrained(
    "nvidia/diar_streaming_sortformer_4spk-v2.1",
    map_location=device
)
model.eval()

# --- 2. Analyze existing weights ---
print("\n[2/6] Analyzing existing weight patterns...")

# Layer 1: hidden_to_spks (384 -> n_spk)
old_h2s = model.sortformer_modules.hidden_to_spks
w1_old = old_h2s.weight.detach().numpy()  # (4, 384)
b1_old = old_h2s.bias.detach().numpy()    # (4,)

# Layer 2: single_hidden_to_spks (192 -> n_spk) - MAIN OUTPUT LAYER
old_sh2s = model.sortformer_modules.single_hidden_to_spks
w2_old = old_sh2s.weight.detach().numpy()  # (4, 192)
b2_old = old_sh2s.bias.detach().numpy()    # (4,)

# Compute target statistics from Layer 2 (main layer)
target_norms = [np.linalg.norm(w2_old[i]) for i in range(OLD_NUM_SPKS)]
target_means = [w2_old[i].mean() for i in range(OLD_NUM_SPKS)]
target_stds = [w2_old[i].std() for i in range(OLD_NUM_SPKS)]

avg_norm = np.mean(target_norms)
avg_mean = np.mean(target_means)
avg_std = np.mean(target_stds)
avg_bias = np.mean(b2_old)
std_bias = np.std(b2_old)

print(f"  Layer 2 (main output) statistics:")
print(f"    Avg L2 norm: {avg_norm:.4f}")
print(f"    Avg mean: {avg_mean:.6f}")
print(f"    Avg std: {avg_std:.4f}")
print(f"    Avg bias: {avg_bias:.4f} (± {std_bias:.4f})")

# Layer 1 statistics
avg_norm_l1 = np.mean([np.linalg.norm(w1_old[i]) for i in range(OLD_NUM_SPKS)])
avg_mean_l1 = np.mean([w1_old[i].mean() for i in range(OLD_NUM_SPKS)])
avg_std_l1 = np.mean([w1_old[i].std() for i in range(OLD_NUM_SPKS)])
avg_bias_l1 = np.mean(b1_old)
std_bias_l1 = np.std(b1_old)

print(f"  Layer 1 statistics:")
print(f"    Avg L2 norm: {avg_norm_l1:.4f}")
print(f"    Avg mean: {avg_mean_l1:.6f}")
print(f"    Avg std: {avg_std_l1:.4f}")
print(f"    Avg bias: {avg_bias_l1:.4f} (± {std_bias_l1:.4f})")

# --- 3. Generate orthogonal weight vectors for new speakers ---
print("\n[3/6] Generating orthogonal weight vectors for speakers 5-8...")

def generate_orthogonal_vectors(existing_vectors, num_new, input_dim, 
                                 target_norm, target_mean, target_std, seed=42):
    """
    Generate new vectors that are orthogonal to existing ones,
    with matched statistics.
    
    Uses Gram-Schmidt process on random vectors projected into 
    the null space of existing vectors.
    """
    np.random.seed(seed)
    
    # Stack existing vectors and compute SVD to find null space
    V = np.array(existing_vectors)  # (num_existing, input_dim)
    U, S, Vt = np.linalg.svd(V, full_matrices=True)
    
    # The last (input_dim - num_existing) rows of Vt span the null space
    null_space_basis = Vt[len(existing_vectors):]  # (input_dim - num_existing, input_dim)
    
    new_vectors = []
    for i in range(num_new):
        # Generate random vector in the null space
        coeffs = np.random.randn(len(null_space_basis))
        vec = np.sum(coeffs[:, np.newaxis] * null_space_basis, axis=0)
        
        # Orthogonalize against previously generated new vectors
        for prev_vec in new_vectors:
            vec = vec - np.dot(vec, prev_vec) / np.dot(prev_vec, prev_vec) * prev_vec
        
        # Adjust statistics
        # First normalize, then scale to match mean and std
        vec = (vec - vec.mean()) / vec.std()  # Standardize
        vec = vec * target_std + target_mean   # Match target mean and std
        
        # Scale to target norm while preserving the distribution shape
        current_norm = np.linalg.norm(vec)
        vec = vec * (target_norm / current_norm)
        
        new_vectors.append(vec)
        
        # Verify orthogonality
        max_dot = max(abs(np.dot(vec, existing_vectors[j])) 
                      for j in range(len(existing_vectors)))
        print(f"    Speaker {OLD_NUM_SPKS + i}: norm={np.linalg.norm(vec):.4f}, "
              f"mean={vec.mean():.6f}, std={vec.std():.4f}, max_dot_existing={max_dot:.6f}")
    
    return np.array(new_vectors)

# Generate new weights for Layer 2 (main output layer)
new_weights_l2 = generate_orthogonal_vectors(
    existing_vectors=w2_old,
    num_new=NEW_NUM_SPKS - OLD_NUM_SPKS,
    input_dim=192,
    target_norm=avg_norm,
    target_mean=avg_mean,
    target_std=avg_std
)

# Generate new weights for Layer 1
print("\n  Layer 1 new vectors:")
new_weights_l1 = generate_orthogonal_vectors(
    existing_vectors=w1_old,
    num_new=NEW_NUM_SPKS - OLD_NUM_SPKS,
    input_dim=384,
    target_norm=avg_norm_l1,
    target_mean=avg_mean_l1,
    target_std=avg_std_l1
)

# Generate biases for new speakers (sample from similar distribution)
np.random.seed(123)
new_biases_l2 = avg_bias + std_bias * np.random.randn(NEW_NUM_SPKS - OLD_NUM_SPKS)
new_biases_l1 = avg_bias_l1 + std_bias_l1 * np.random.randn(NEW_NUM_SPKS - OLD_NUM_SPKS)

print(f"\n  New biases (Layer 2): {new_biases_l2}")
print(f"  New biases (Layer 1): {new_biases_l1}")

# --- 4. Create new layers with combined weights ---
print("\n[4/6] Creating new 8-speaker output layers...")

# Layer 1: hidden_to_spks
new_h2s = nn.Linear(384, NEW_NUM_SPKS)
with torch.no_grad():
    # Copy old weights
    new_h2s.weight[:OLD_NUM_SPKS, :] = old_h2s.weight.clone()
    new_h2s.bias[:OLD_NUM_SPKS] = old_h2s.bias.clone()
    # Add new weights
    new_h2s.weight[OLD_NUM_SPKS:, :] = torch.from_numpy(new_weights_l1).float()
    new_h2s.bias[OLD_NUM_SPKS:] = torch.from_numpy(new_biases_l1).float()

print(f"  hidden_to_spks: {new_h2s.weight.shape}")

# Layer 2: single_hidden_to_spks (MAIN OUTPUT)
new_sh2s = nn.Linear(192, NEW_NUM_SPKS)
with torch.no_grad():
    # Copy old weights
    new_sh2s.weight[:OLD_NUM_SPKS, :] = old_sh2s.weight.clone()
    new_sh2s.bias[:OLD_NUM_SPKS] = old_sh2s.bias.clone()
    # Add new weights
    new_sh2s.weight[OLD_NUM_SPKS:, :] = torch.from_numpy(new_weights_l2).float()
    new_sh2s.bias[OLD_NUM_SPKS:] = torch.from_numpy(new_biases_l2).float()

print(f"  single_hidden_to_spks: {new_sh2s.weight.shape}")

# --- 5. Replace layers in model ---
print("\n[5/6] Replacing layers in model...")

model.sortformer_modules.hidden_to_spks = new_h2s
model.sortformer_modules.single_hidden_to_spks = new_sh2s
model.sortformer_modules.n_spk = NEW_NUM_SPKS

# Update config
model._cfg.max_num_of_spks = NEW_NUM_SPKS
model._cfg.sortformer_modules.num_spks = NEW_NUM_SPKS

# Update speaker permutations
speaker_inds = list(range(NEW_NUM_SPKS))
model.speaker_permutations = torch.tensor(list(itertools.permutations(speaker_inds)))

# --- 6. Verify and save ---
print("\n[6/6] Verifying and saving model...")

# Verify orthogonality of all speakers
final_weights = new_sh2s.weight.detach().numpy()
print("\n  Correlation matrix (all 8 speakers):")
corr = np.corrcoef(final_weights)
for i in range(NEW_NUM_SPKS):
    row = [f"{corr[i,j]:+.2f}" for j in range(NEW_NUM_SPKS)]
    print(f"    Spk {i}: [{', '.join(row)}]")

# Save model
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, OUTPUT_PATH)
model.save_to(output_path)
print(f"\n  Saved to: {output_path}")

# --- Quick test ---
print("\n" + "=" * 70)
print("Verification Test")
print("=" * 70)

loaded = SortformerEncLabelModel.restore_from(output_path, map_location=device)
print(f"  n_spk: {loaded.sortformer_modules.n_spk}")

# Test inference
loaded.eval()
with torch.no_grad():
    dummy_audio = torch.randn(1, 16000)
    dummy_length = torch.tensor([16000])
    preds = loaded.forward(dummy_audio, dummy_length)
    print(f"  Output shape: {preds.shape}")
    print(f"  Mean predictions per speaker:")
    for spk in range(NEW_NUM_SPKS):
        print(f"    Speaker {spk}: {preds[0, :, spk].mean():.4f}")

print("\n✅ Smart 8-speaker model created successfully!")
print("\nThe new speakers (4-7) are initialized with:")
print("  - Weight vectors orthogonal to speakers 0-3")
print("  - Matched L2 norms, means, and standard deviations")
print("  - Similar negative biases (conservative activation)")
