"""
Test Betti curve differences for embeddings with different metrics.
"""
import numpy as np
from scipy.spatial import procrustes
from dire_rapids import DiRePyTorch
from dire_rapids.metrics import compute_global_metrics

np.random.seed(42)

# Create circle in x-y plane with noise in z-direction
n = 2000
theta = np.linspace(0, 2*np.pi, n, endpoint=False)
x = np.cos(theta)
y = np.sin(theta)
z = np.random.randn(n) * 0.1
data = np.column_stack([x, y, z]).astype(np.float32)

print("="*70)
print("Test 1: Circle embedded TWICE with Euclidean metric")
print("="*70)
print(f"Data shape: {data.shape}")

# Embed twice with euclidean metric (same seed)
print("\nEmbedding 1 (euclidean, seed=42)...")
model1 = DiRePyTorch(n_components=2, metric='euclidean', n_neighbors=15, random_state=42, verbose=False)
embedding1 = model1.fit_transform(data)

print("Embedding 2 (euclidean, seed=42)...")
model2 = DiRePyTorch(n_components=2, metric='euclidean', n_neighbors=15, random_state=42, verbose=False)
embedding2 = model2.fit_transform(data)

# Procrustes analysis
mtx1, mtx2, disparity = procrustes(embedding1, embedding2)
print(f"\nProcrustes disparity (euc vs euc): {disparity:.6f}")

# Compare Betti curves
print("Comparing Betti curves...")
result1 = compute_global_metrics(
    embedding1, embedding2,
    subsample_threshold=0.1,
    n_steps=25,
    k_neighbors=15,
    density_threshold=0.8,
    use_gpu=False,
    metrics_only=True
)

print(f"DTW β₀: {result1['metrics']['dtw_beta0']:.6f}")
print(f"DTW β₁: {result1['metrics']['dtw_beta1']:.6f}")

print("\n" + "="*70)
print("Test 2: Circle embedded with Euclidean vs Cosine metric")
print("="*70)

# Embed with euclidean metric
print("\nEmbedding (euclidean, seed=42)...")
model_euc = DiRePyTorch(n_components=2, metric='euclidean', n_neighbors=15, random_state=42, verbose=False)
embedding_euc = model_euc.fit_transform(data)

# Embed with cosine metric
print("Embedding (cosine, seed=42)...")
model_cos = DiRePyTorch(n_components=2, metric='cosine', n_neighbors=15, random_state=42, verbose=False)
embedding_cos = model_cos.fit_transform(data)

# Procrustes analysis
mtx1, mtx2, disparity = procrustes(embedding_euc, embedding_cos)
print(f"\nProcrustes disparity (euc vs cosine): {disparity:.6f}")

# Compare Betti curves
print("Comparing Betti curves...")
result2 = compute_global_metrics(
    embedding_euc, embedding_cos,
    subsample_threshold=0.1,
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
print("Euclidean vs Euclidean (same seed):")
print(f"  Procrustes disparity: {procrustes(embedding1, embedding2)[2]:.6f}")
print(f"  DTW β₀: {result1['metrics']['dtw_beta0']:.6f}")
print(f"  DTW β₁: {result1['metrics']['dtw_beta1']:.6f}")
print()
print("Euclidean vs Cosine:")
print(f"  Procrustes disparity: {procrustes(embedding_euc, embedding_cos)[2]:.6f}")
print(f"  DTW β₀: {result2['metrics']['dtw_beta0']:.6f}")
print(f"  DTW β₁: {result2['metrics']['dtw_beta1']:.6f}")
