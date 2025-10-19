"""
Test Betti curve differences with Procrustes analysis for embeddings.
"""
import numpy as np
from sklearn.datasets import make_blobs
from scipy.spatial import procrustes
from dire_rapids import DiRePyTorch
from dire_rapids.metrics import compute_global_metrics

np.random.seed(42)

# Create 3 blobs with 150 points total in 10D (same as CI test)
data, labels = make_blobs(n_samples=750, n_features=10, centers=3, random_state=42)

print("="*70)
print("Test 1: 10D Blobs embedded TWICE with Euclidean metric")
print("="*70)
print(f"Data shape: {data.shape}")
print(f"Number of blobs: 3")

# Embed twice with euclidean metric (same seed) - use metric=None like CI test
print("\nEmbedding 1 (L2/euclidean, seed=42)...")
model1 = DiRePyTorch(n_components=2, metric=None, max_iter_layout=20, random_state=42, verbose=False)
embedding1 = model1.fit_transform(data)

print("Embedding 2 (L2/euclidean, seed=42)...")
model2 = DiRePyTorch(n_components=2, metric=None, max_iter_layout=20, random_state=42, verbose=False)
embedding2 = model2.fit_transform(data)

# Procrustes analysis
mtx1, mtx2, disparity = procrustes(embedding1, embedding2)
print(f"\nProcrustes disparity (euc vs euc): {disparity:.6f}")

# Compare Betti curves
print("Comparing Betti curves...")
result1 = compute_global_metrics(
    embedding1, embedding2,
    subsample_threshold=0.25,
    n_steps=25,
    k_neighbors=15,
    density_threshold=0.8,
    use_gpu=False,
    metrics_only=True
)

print(f"DTW β₀: {result1['metrics']['dtw_beta0']:.6f}")
print(f"DTW β₁: {result1['metrics']['dtw_beta1']:.6f}")

print("\n" + "="*70)
print("Test 2: 10D Blobs embedded with L2 vs Cosine metric")
print("="*70)

# Embed with L2 metric
print("\nEmbedding (L2, seed=42)...")
model_euc = DiRePyTorch(n_components=2, metric=None, max_iter_layout=20, random_state=42, verbose=False)
embedding_euc = model_euc.fit_transform(data)

# Embed with cosine metric - use same expression as CI test
print("Embedding (cosine, seed=42)...")
cosine_expr = "1 - (x * y).sum(-1) / (((x ** 2).sum(-1).sqrt() * (y ** 2).sum(-1).sqrt()) + 1e-8)"
model_cos = DiRePyTorch(n_components=2, metric=cosine_expr, max_iter_layout=20, random_state=42, verbose=False)
embedding_cos = model_cos.fit_transform(data)

# Procrustes analysis
mtx1, mtx2, disparity = procrustes(embedding_euc, embedding_cos)
print(f"\nProcrustes disparity (euc vs cosine): {disparity:.6f}")

# Compare Betti curves
print("Comparing Betti curves...")
result2 = compute_global_metrics(
    embedding_euc, embedding_cos,
    subsample_threshold=0.25,
    n_steps=25,
    k_neighbors=15,
    density_threshold=0.8,
    use_gpu=False,
    metrics_only=True
)

print(f"DTW β₀: {result2['metrics']['dtw_beta0']:.6f}")
print(f"DTW β₁: {result2['metrics']['dtw_beta1']:.6f}")

print("\n" + "="*70)
print("Summary")
print("="*70)
print("L2 vs L2 (same seed):")
print(f"  Procrustes disparity: {procrustes(embedding1, embedding2)[2]:.6f}")
print(f"  DTW β₀: {result1['metrics']['dtw_beta0']:.6f}")
print(f"  DTW β₁: {result1['metrics']['dtw_beta1']:.6f}")
print()
print("L2 vs Cosine:")
print(f"  Procrustes disparity: {procrustes(embedding_euc, embedding_cos)[2]:.6f}")
print(f"  DTW β₀: {result2['metrics']['dtw_beta0']:.6f}")
print(f"  DTW β₁: {result2['metrics']['dtw_beta1']:.6f}")
