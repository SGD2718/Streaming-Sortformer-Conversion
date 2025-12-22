"""
Analyze the weight patterns of the 4-speaker model's output layers
to see if we can infer optimal weights for speakers 5-8.
"""

import torch
import numpy as np
from nemo.collections.asr.models import SortformerEncLabelModel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

device = torch.device("cpu")

print("=" * 70)
print("Analyzing Sortformer Output Layer Weights")
print("=" * 70)

# Load the original 4-speaker model
model = SortformerEncLabelModel.from_pretrained(
    "nvidia/diar_streaming_sortformer_4spk-v2.1",
    map_location=device
)

# Extract the two output layers
hidden_to_spks = model.sortformer_modules.hidden_to_spks
single_hidden_to_spks = model.sortformer_modules.single_hidden_to_spks

print("\n" + "=" * 70)
print("Layer 1: hidden_to_spks (Linear: 384 -> 4)")
print("=" * 70)

w1 = hidden_to_spks.weight.detach().numpy()  # Shape: (4, 384)
b1 = hidden_to_spks.bias.detach().numpy()    # Shape: (4,)

print(f"\nWeight shape: {w1.shape}")
print(f"Bias shape: {b1.shape}")

# Statistics per speaker
print("\n--- Per-Speaker Statistics ---")
for spk in range(4):
    weights = w1[spk]
    print(f"  Speaker {spk}: mean={weights.mean():.6f}, std={weights.std():.6f}, "
          f"min={weights.min():.4f}, max={weights.max():.4f}, bias={b1[spk]:.4f}")

# Cross-speaker analysis
print("\n--- Cross-Speaker Correlations ---")
corr_matrix = np.corrcoef(w1)
print("  Correlation matrix:")
for i in range(4):
    print(f"    {[f'{c:.3f}' for c in corr_matrix[i]]}")

# L2 norms
print("\n--- Weight Vector Norms ---")
for spk in range(4):
    norm = np.linalg.norm(w1[spk])
    print(f"  Speaker {spk}: L2 norm = {norm:.4f}")

print("\n" + "=" * 70)
print("Layer 2: single_hidden_to_spks (Linear: 192 -> 4)")
print("=" * 70)

w2 = single_hidden_to_spks.weight.detach().numpy()  # Shape: (4, 192)
b2 = single_hidden_to_spks.bias.detach().numpy()    # Shape: (4,)

print(f"\nWeight shape: {w2.shape}")
print(f"Bias shape: {b2.shape}")

# Statistics per speaker
print("\n--- Per-Speaker Statistics ---")
for spk in range(4):
    weights = w2[spk]
    print(f"  Speaker {spk}: mean={weights.mean():.6f}, std={weights.std():.6f}, "
          f"min={weights.min():.4f}, max={weights.max():.4f}, bias={b2[spk]:.4f}")

# Cross-speaker analysis
print("\n--- Cross-Speaker Correlations ---")
corr_matrix_2 = np.corrcoef(w2)
print("  Correlation matrix:")
for i in range(4):
    print(f"    {[f'{c:.3f}' for c in corr_matrix_2[i]]}")

# L2 norms
print("\n--- Weight Vector Norms ---")
for spk in range(4):
    norm = np.linalg.norm(w2[spk])
    print(f"  Speaker {spk}: L2 norm = {norm:.4f}")

# Analyze weight distributions
print("\n" + "=" * 70)
print("Distribution Analysis")
print("=" * 70)

# Check if speakers have similar distributions (suggesting interchangeability)
print("\n--- Weight Distribution Similarity (KS test) ---")
from scipy import stats
for i in range(4):
    for j in range(i+1, 4):
        ks_stat, p_val = stats.ks_2samp(w2[i], w2[j])
        print(f"  Spk {i} vs Spk {j}: KS={ks_stat:.4f}, p={p_val:.4f}")

# Principal Component Analysis
print("\n--- PCA of Speaker Weight Vectors ---")
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
w2_pca = pca.fit_transform(w2)
print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"  Speaker positions in PC space:")
for spk in range(4):
    print(f"    Speaker {spk}: ({w2_pca[spk, 0]:.4f}, {w2_pca[spk, 1]:.4f})")

# Check for patterns suggesting optimal initialization
print("\n" + "=" * 70)
print("Insights for New Speakers (5-8)")
print("=" * 70)

# Mean weight pattern
mean_weights = w2.mean(axis=0)
std_weights = w2.std(axis=0)
mean_bias = b2.mean()
std_bias = b2.std()

print(f"\n--- Aggregate Statistics ---")
print(f"  Mean weight vector norm: {np.linalg.norm(mean_weights):.4f}")
print(f"  Std of weights across speakers: {std_weights.mean():.6f}")
print(f"  Mean bias: {mean_bias:.4f}")
print(f"  Std of biases: {std_bias:.4f}")

# Check orthogonality
print("\n--- Orthogonality Check ---")
for i in range(4):
    for j in range(i+1, 4):
        dot = np.dot(w2[i], w2[j])
        cos_sim = dot / (np.linalg.norm(w2[i]) * np.linalg.norm(w2[j]))
        print(f"  Spk {i} · Spk {j}: dot={dot:.4f}, cos_sim={cos_sim:.4f}")

# Suggestion based on analysis
print("\n" + "=" * 70)
print("RECOMMENDATIONS for Speakers 5-8")
print("=" * 70)

avg_norm = np.mean([np.linalg.norm(w2[i]) for i in range(4)])
avg_mean = np.mean([w2[i].mean() for i in range(4)])
avg_std = np.mean([w2[i].std() for i in range(4)])

print(f"""
Based on the weight analysis:

1. Weight Initialization:
   - Target L2 norm: ~{avg_norm:.4f}
   - Target mean: ~{avg_mean:.6f}
   - Target std: ~{avg_std:.4f}
   
2. Bias Initialization:
   - Mean bias: {mean_bias:.4f}
   - Std bias: {std_bias:.4f}

3. Orthogonality Strategy:
   - Existing speakers have moderate correlations ({corr_matrix_2.mean():.3f} avg)
   - New speakers should be initialized to be somewhat orthogonal
   - Consider Gram-Schmidt orthogonalization on the existing weight subspace
   
4. Practical Approaches:
   a) Random with matched statistics (current approach)
   b) Copy existing speakers with small perturbation
   c) Orthogonal projection from existing speaker space
   d) Use mean weight + random orthogonal component
""")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Weight heatmaps
ax = axes[0, 0]
im = ax.imshow(w2, aspect='auto', cmap='coolwarm')
ax.set_title('single_hidden_to_spks weights (192 dims)')
ax.set_ylabel('Speaker')
ax.set_xlabel('Hidden dim')
plt.colorbar(im, ax=ax)

# Weight distributions
ax = axes[0, 1]
for spk in range(4):
    ax.hist(w2[spk], bins=30, alpha=0.5, label=f'Spk {spk}')
ax.set_title('Weight Distributions')
ax.set_xlabel('Weight value')
ax.legend()

# Correlation heatmap
ax = axes[0, 2]
im = ax.imshow(corr_matrix_2, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_title('Speaker Correlation Matrix')
ax.set_xticks(range(4))
ax.set_yticks(range(4))
plt.colorbar(im, ax=ax)

# PCA plot
ax = axes[1, 0]
ax.scatter(w2_pca[:, 0], w2_pca[:, 1], s=200, c=['red', 'blue', 'green', 'orange'])
for i in range(4):
    ax.annotate(f'Spk {i}', (w2_pca[i, 0], w2_pca[i, 1]), fontsize=12)
ax.set_title('PCA of Speaker Weights')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

# Bias values
ax = axes[1, 1]
ax.bar(range(4), b2)
ax.set_title('Bias Values')
ax.set_xlabel('Speaker')
ax.set_ylabel('Bias')
ax.set_xticks(range(4))

# L2 norms
ax = axes[1, 2]
norms = [np.linalg.norm(w2[i]) for i in range(4)]
ax.bar(range(4), norms)
ax.set_title('Weight Vector L2 Norms')
ax.set_xlabel('Speaker')
ax.set_ylabel('L2 Norm')
ax.set_xticks(range(4))

plt.tight_layout()
plt.savefig('weight_analysis.png', dpi=150)
print("\nSaved visualization to: weight_analysis.png")
