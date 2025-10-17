"""
Analyze the sparsity structure of the atlas Laplacian matrices.
"""

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Generate circle
n = 100
theta = np.linspace(0, 2*np.pi, n, endpoint=False)
data = np.column_stack([np.cos(theta), np.sin(theta)]).astype(np.float32)

# Build atlas (same as in metrics.py)
k = 15
nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
nn.fit(data)
distances, indices = nn.kneighbors(data)

global_edges = set()
for i in range(n):
    local_neighborhood = indices[i, 1:]
    local_dists = distances[i, 1:]

    for j in local_neighborhood:
        global_edges.add(tuple(sorted([i, int(j)])))

    dist_threshold = np.percentile(local_dists, 80) * 1.5
    for idx_j in range(len(local_neighborhood)):
        for idx_k in range(idx_j + 1, len(local_neighborhood)):
            j = int(local_neighborhood[idx_j])
            k_pt = int(local_neighborhood[idx_k])
            dist_jk = np.linalg.norm(data[j] - data[k_pt])
            if dist_jk < dist_threshold:
                global_edges.add(tuple(sorted([j, k_pt])))

edges = sorted(list(global_edges))
n_edges = len(edges)
print(f"Circle: n={n}, k={k}, n_edges={n_edges}")

# Build B1
B1_rows, B1_cols, B1_data = [], [], []
for e_idx, (v0, v1) in enumerate(edges):
    B1_rows.extend([v0, v1])
    B1_cols.extend([e_idx, e_idx])
    B1_data.extend([-1, +1])

B1 = sparse.coo_matrix((B1_data, (B1_rows, B1_cols)), shape=(n, n_edges), dtype=np.float64)
B1 = B1.tocsr()

# Build L0 (vertex Laplacian)
adj_rows, adj_cols, adj_data = [], [], []
for v0, v1 in edges:
    adj_rows.extend([v0, v1])
    adj_cols.extend([v1, v0])
    adj_data.extend([1.0, 1.0])

A = sparse.coo_matrix((adj_data, (adj_rows, adj_cols)), shape=(n, n))
A = A.tocsr()
deg = np.array(A.sum(axis=1)).flatten()
D = sparse.diags(deg)
L0 = D - A

# Build L1 (edge Laplacian) - just B1^T @ B1 for simplicity
L1 = (B1.T @ B1).astype(np.float64)

# Analyze sparsity structure
print("\nL0 (vertex Laplacian):")
print(f"  Shape: {L0.shape}")
print(f"  Nonzeros: {L0.nnz}")
print(f"  Density: {L0.nnz / (n * n):.4f}")

print("\nL1 (edge Laplacian):")
print(f"  Shape: {L1.shape}")
print(f"  Nonzeros: {L1.nnz}")
print(f"  Density: {L1.nnz / (n_edges * n_edges):.4f}")

# Visualize sparsity pattern
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].spy(L0, markersize=1)
axes[0].set_title(f'L0 Sparsity Pattern (n={n})')
axes[0].set_xlabel('Column')
axes[0].set_ylabel('Row')

axes[1].spy(L1, markersize=0.5)
axes[1].set_title(f'L1 Sparsity Pattern (n_edges={n_edges})')
axes[1].set_xlabel('Column')
axes[1].set_ylabel('Row')

plt.tight_layout()
plt.savefig('atlas_sparsity_pattern.png', dpi=150)
print("\n✓ Saved sparsity pattern to atlas_sparsity_pattern.png")

# Analyze block structure
# Look at bandwidth (how far from diagonal are nonzeros?)
L0_coo = L0.tocoo()
L1_coo = L1.tocoo()

L0_bandwidth = np.max(np.abs(L0_coo.row - L0_coo.col))
L1_bandwidth = np.max(np.abs(L1_coo.row - L1_coo.col))

print("\nBandwidth analysis:")
print(f"  L0 bandwidth: {L0_bandwidth} (matrix size {n}x{n})")
print(f"  L1 bandwidth: {L1_bandwidth} (matrix size {n_edges}x{n_edges})")
print(f"  L0 bandwidth ratio: {L0_bandwidth/n:.3f}")
print(f"  L1 bandwidth ratio: {L1_bandwidth/n_edges:.3f}")

# Check if blocks are localized
print("\nBlock structure:")
for i in range(0, n, n//10):  # Sample every 10%
    row_nnz = L0[i].nonzero()[1]
    if len(row_nnz) > 0:
        span = np.max(row_nnz) - np.min(row_nnz)
        print(f"  Row {i}: {len(row_nnz)} nonzeros, span {span} (locality: {span/n:.3f})")
