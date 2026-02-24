# Fast Betti Curve Computation via Rank

## Overview

The `betti_curve.py` module computes filtered Betti curves (beta_0, beta_1) for
point cloud data using a kNN-based atlas complex. The key innovation is replacing
expensive eigenvalue computation (scipy `eigsh`) with exact algebraic topology
identities that reduce the problem to connected components and matrix rank.

## Mathematical Foundation

For a simplicial complex with V vertices, E edges, T triangles, and boundary
operators B1 (V x E) and B2 (E x T):

**beta_0** (connected components) is computed via union-find in O(E * alpha(E)),
essentially linear time.

**beta_1** (independent loops) uses the Hodge decomposition identity:

    beta_1 = E - rank(B1) - rank(B2)

where:
- **rank(B1) = V - beta_0** (standard graph theory: the rank of the incidence
  matrix of a graph equals the number of vertices minus the number of connected
  components)
- **rank(B2)** is computed via GPU-accelerated SVD (`torch.linalg.matrix_rank`)

This completely eliminates the need for eigenvalue computation of the Hodge
Laplacian L1 = B1^T B1 + B2 B2^T, which was the performance bottleneck.

## Implementation Details

### Rank of B2

B2 is the boundary operator from 2-simplices (triangles) to 1-simplices (edges),
an integer matrix with entries in {-1, 0, +1} and exactly 3 nonzero entries per
column.

**Precision requirement: float64.** In float32, genuine zero singular values of
B2 can be perturbed to ~1e-6, which may cross the data-dependent threshold used
by `matrix_rank`. In float64, the perturbation is ~1e-15, safely below any
reasonable threshold. This was discovered empirically: dense kNN complexes from
Gaussian point clouds produce B2 matrices with genuine nonzero singular values as
low as 0.24 (the initial assumption that all nonzero SVs >= 1 is false for
general simplicial complexes).

**Gram matrix optimization.** For large complexes (E * T > 5M elements), we
compute the Gram matrix G = B2 * B2^T (size E x E) instead of materializing the
full dense B2 (size E x T). The eigenvalues of G are the squared singular values
of B2, and in float64 the default threshold (~1e-10) is far below the smallest
genuine nonzero eigenvalue of G (~0.05), so this is safe.

### Backend Architecture

Three implementations share the same filtration infrastructure (atlas building,
edge filtering):

| Backend | kNN | Betti computation | Use case |
|---------|-----|-------------------|----------|
| `compute_betti_curve_fast` | sklearn (CPU) | union-find + GPU SVD rank | Default for CPU-only |
| `compute_betti_curve_gpu` | cuVS/cuML (GPU) | union-find + GPU SVD rank | Default when GPU available |
| `compute_betti_curve_cpu` | sklearn (CPU) | scipy eigsh (shift-invert) | Reference implementation |

The `compute_betti_curve` selector tries GPU first, then falls back to fast.

### Performance

On GH200 (480GB), with n=500 points, k=15, 30 filtration steps:

| Backend | Time | vs eigsh |
|---------|------|----------|
| fast (rank-based) | ~50-100s | 5-11x faster |
| cpu (eigsh reference) | ~500s | baseline |

The speedup grows with dataset size since eigsh scales poorly with matrix
dimension while SVD of the (smaller) Gram matrix scales as O(E^3).

### Correctness

The fast method is actually **more correct** than the eigsh reference at sparse
filtration steps:

- **beta_0**: Union-find counts all connected components exactly, including
  isolated vertices. The eigsh method is capped at k=50 eigenvalues and cannot
  detect more than 50 components.
- **beta_1**: The rank formula is exact (given sufficient numerical precision).
  The eigsh method is iterative and can miss near-zero eigenvalues in
  ill-conditioned Laplacians.

Small discrepancies (max 1-2) at transitional filtration steps are due to eigsh
approximation errors, not the rank method.
