#!/usr/bin/env python3
"""
Simple example: Compare topological metrics between DiRePyTorch and Kernel PCA embeddings.

Embeds a Swiss roll dataset (1500 points) using:
- DiRePyTorch (dimensionality reduction via DiRe)
- Kernel PCA from sklearn

Then computes:
- Betti curve-based topological metrics using the atlas approach with subsampling
- Procrustes analysis to compare geometric similarity of embeddings
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from scipy.spatial import procrustes

from dire_rapids import DiRePyTorch
from dire_rapids.metrics import compute_global_metrics

# Generate Swiss roll dataset
print("Generating Swiss roll dataset (1500 points)...")
data, color = make_swiss_roll(n_samples=1500, noise=0.1, random_state=42)
print(f"Data shape: {data.shape}")

# Embed with DiRePyTorch
print("\nEmbedding with DiRePyTorch...")
model_dire = DiRePyTorch(
    n_components=2,
    n_neighbors=15,
    max_iter_layout=100,
    random_state=42,
    verbose=True
)
embedding_dire = model_dire.fit_transform(data)
print(f"DiRePyTorch embedding shape: {embedding_dire.shape}")

# Embed with Kernel PCA
print("\nEmbedding with Kernel PCA...")
model_kpca = KernelPCA(
    n_components=2,
    kernel='rbf',
    gamma=0.1,
    random_state=42
)
embedding_kpca = model_kpca.fit_transform(data)
print(f"Kernel PCA embedding shape: {embedding_kpca.shape}")

# Procrustes analysis
print("\nComputing Procrustes analysis between embeddings...")
mtx1, mtx2, disparity = procrustes(embedding_dire, embedding_kpca)
print(f"Procrustes disparity: {disparity:.6f}")

# Compute topological metrics for DiRePyTorch embedding
print("\nComputing Betti curve metrics for DiRePyTorch embedding...")
print("(using atlas approach with subsample_threshold=0.1)")
metrics_dire = compute_global_metrics(
    data,
    embedding_dire,
    dimension=1,
    subsample_threshold=0.1,
    random_state=42,
    n_steps=100,
    k_neighbors=20,
    density_threshold=0.8,
    overlap_factor=1.5,
    use_gpu=False,
    metrics_only=False  # Get Betti curves for visualization
)

print("\nDiRePyTorch Betti curve distances:")
print(f"  β₀ (DTW): {metrics_dire['metrics']['dtw_beta0']:.6f}")
print(f"  β₁ (DTW): {metrics_dire['metrics']['dtw_beta1']:.6f}")

# Compute topological metrics for Kernel PCA embedding
print("\nComputing Betti curve metrics for Kernel PCA embedding...")
metrics_kpca = compute_global_metrics(
    data,
    embedding_kpca,
    dimension=1,
    subsample_threshold=0.1,
    random_state=42,
    n_steps=100,
    k_neighbors=20,
    density_threshold=0.8,
    overlap_factor=1.5,
    use_gpu=False,
    metrics_only=False  # Get Betti curves for visualization
)

print("\nKernel PCA Betti curve distances:")
print(f"  β₀ (DTW): {metrics_kpca['metrics']['dtw_beta0']:.6f}")
print(f"  β₁ (DTW): {metrics_kpca['metrics']['dtw_beta1']:.6f}")

# Compare
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print("\nGeometric similarity:")
print(f"  Procrustes disparity: {disparity:.6f}")

print("\nTopological preservation (DTW distances):")
print(f"  DiRePyTorch β₀: {metrics_dire['metrics']['dtw_beta0']:.6f}")
print(f"  Kernel PCA β₀:  {metrics_kpca['metrics']['dtw_beta0']:.6f}")
print(f"  DiRePyTorch β₁: {metrics_dire['metrics']['dtw_beta1']:.6f}")
print(f"  Kernel PCA β₁:  {metrics_kpca['metrics']['dtw_beta1']:.6f}")

# Visualize embeddings and Betti curves
print("\nCreating visualizations...")
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Row 1: Embeddings
ax_dire = fig.add_subplot(gs[0, 0])
ax_dire.scatter(embedding_dire[:, 0], embedding_dire[:, 1], c=color, cmap='viridis', s=10, alpha=0.7)
ax_dire.set_title('DiRePyTorch Embedding')
ax_dire.set_xlabel('Component 1')
ax_dire.set_ylabel('Component 2')
ax_dire.grid(True, alpha=0.3)

ax_kpca = fig.add_subplot(gs[0, 1])
ax_kpca.scatter(embedding_kpca[:, 0], embedding_kpca[:, 1], c=color, cmap='viridis', s=10, alpha=0.7)
ax_kpca.set_title('Kernel PCA Embedding')
ax_kpca.set_xlabel('Component 1')
ax_kpca.set_ylabel('Component 2')
ax_kpca.grid(True, alpha=0.3)

# Row 2: Betti curves
# Extract Betti curves from results
filtration_hd = metrics_dire['bettis']['data']['filtration']
beta0_hd = metrics_dire['bettis']['data']['beta_0']
beta1_hd = metrics_dire['bettis']['data']['beta_1']

filtration_dire = metrics_dire['bettis']['layout']['filtration']
beta0_dire = metrics_dire['bettis']['layout']['beta_0']
beta1_dire = metrics_dire['bettis']['layout']['beta_1']

filtration_kpca = metrics_kpca['bettis']['layout']['filtration']
beta0_kpca = metrics_kpca['bettis']['layout']['beta_0']
beta1_kpca = metrics_kpca['bettis']['layout']['beta_1']

# β₀ curves
ax_beta0 = fig.add_subplot(gs[1, 0])
ax_beta0.plot(filtration_hd, beta0_hd, 'k-', linewidth=2, label='High-dim data', alpha=0.8)
ax_beta0.plot(filtration_dire, beta0_dire, 'b-', linewidth=2, label='DiRePyTorch', alpha=0.8)
ax_beta0.plot(filtration_kpca, beta0_kpca, 'r--', linewidth=2, label='Kernel PCA', alpha=0.8)
ax_beta0.set_xlabel('Filtration value')
ax_beta0.set_ylabel('β₀ (connected components)')
ax_beta0.set_title('β₀ Betti Curves')
ax_beta0.legend()
ax_beta0.grid(True, alpha=0.3)

# β₁ curves
ax_beta1 = fig.add_subplot(gs[1, 1])
ax_beta1.plot(filtration_hd, beta1_hd, 'k-', linewidth=2, label='High-dim data', alpha=0.8)
ax_beta1.plot(filtration_dire, beta1_dire, 'b-', linewidth=2, label='DiRePyTorch', alpha=0.8)
ax_beta1.plot(filtration_kpca, beta1_kpca, 'r--', linewidth=2, label='Kernel PCA', alpha=0.8)
ax_beta1.set_xlabel('Filtration value')
ax_beta1.set_ylabel('β₁ (loops)')
ax_beta1.set_title('β₁ Betti Curves')
ax_beta1.legend()
ax_beta1.grid(True, alpha=0.3)

# Metrics summary
ax_summary = fig.add_subplot(gs[:, 2])
ax_summary.axis('off')
summary_text = f"""
COMPARISON SUMMARY

Geometric Similarity:
  Procrustes: {disparity:.6f}

Topological Preservation:

DTW for β₀ (connected components):
  DiRePyTorch: {metrics_dire['metrics']['dtw_beta0']:.6f}
  Kernel PCA:  {metrics_kpca['metrics']['dtw_beta0']:.6f}

DTW for β₁ (loops):
  DiRePyTorch: {metrics_dire['metrics']['dtw_beta1']:.6f}
  Kernel PCA:  {metrics_kpca['metrics']['dtw_beta1']:.6f}

"""
ax_summary.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig('swiss_roll_analysis.png', dpi=150, bbox_inches='tight')
print("Saved visualization to: swiss_roll_analysis.png")
