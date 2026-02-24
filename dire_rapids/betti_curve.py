"""
Betti curve computation using filtered edge addition.

Computes Betti numbers (β₀, β₁) at multiple filtration thresholds by:
1. Building full atlas complex once with kNN graph
2. Recording all edge distances
3. Progressively adding edges by distance threshold
4. Recomputing Laplacians and Betti numbers at each step

Contains both CPU and GPU implementations with automatic backend selection.
"""

import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Shared helpers (used by both CPU and GPU paths)
# ---------------------------------------------------------------------------

def _build_atlas(data_np, distances_np, indices_np, k_neighbors,
                 density_threshold, overlap_factor):
    """
    Build the atlas complex from kNN graph data (CPU-only set operations).

    Returns
    -------
    global_edges : set of (int, int) tuples
    global_triangles : set of (int, int, int) tuples
    edge_distances : dict mapping edge -> float
    triangle_edges : list of (tri, e1, e2, e3) tuples
    """
    n = len(data_np)
    edge_distances = {}
    global_edges = set()
    global_triangles = set()

    for i in range(n):
        local_neighborhood = indices_np[i, 1:]
        local_dists = distances_np[i, 1:]

        # Edges from i to neighbors
        for j_idx, j in enumerate(local_neighborhood):
            edge = tuple(sorted([i, int(j)]))
            global_edges.add(edge)
            if edge not in edge_distances:
                edge_distances[edge] = local_dists[j_idx]

        # Dense local patch
        dist_threshold = (
            np.percentile(local_dists, density_threshold * 100) * overlap_factor
        )

        if len(local_neighborhood) > 1:
            neighborhood_coords = data_np[local_neighborhood]
            for idx_j, j_val in enumerate(local_neighborhood):
                j = int(j_val)
                dists_jk = np.linalg.norm(
                    neighborhood_coords[idx_j + 1:] - neighborhood_coords[idx_j],
                    axis=1,
                )
                close_enough = dists_jk < dist_threshold
                for offset, is_close in enumerate(close_enough):
                    if is_close:
                        k = int(local_neighborhood[idx_j + 1 + offset])
                        edge = tuple(sorted([j, k]))
                        global_edges.add(edge)
                        if edge not in edge_distances:
                            edge_distances[edge] = dists_jk[offset]

        # Build triangles
        for idx_j, j_val in enumerate(local_neighborhood):
            for idx_k in range(idx_j + 1, len(local_neighborhood)):
                j = int(j_val)
                k = int(local_neighborhood[idx_k])
                e1 = tuple(sorted([i, j]))
                e2 = tuple(sorted([i, k]))
                e3 = tuple(sorted([j, k]))
                if e1 in global_edges and e2 in global_edges and e3 in global_edges:
                    global_triangles.add(tuple(sorted([i, j, k])))

    # Pre-compute triangle edges for efficiency
    triangle_edges = []
    for tri in global_triangles:
        i, j, k = tri
        e1 = tuple(sorted([i, j]))
        e2 = tuple(sorted([i, k]))
        e3 = tuple(sorted([j, k]))
        triangle_edges.append((tri, e1, e2, e3))

    return global_edges, global_triangles, edge_distances, triangle_edges


def _build_boundary_data(active_edges, active_triangles, edge_to_idx):
    """
    Build the raw COO data for boundary operators B1, B2, and adjacency A.

    Returns
    -------
    b1 : (rows, cols, data) lists for B1
    b2 : (rows, cols, data) lists for B2  (empty lists if no triangles)
    adj : (rows, cols, data) lists for adjacency matrix
    """
    # B1 (edge boundary operator)
    B1_rows, B1_cols, B1_data = [], [], []
    for e_idx, (v0, v1) in enumerate(active_edges):
        B1_rows.extend([v0, v1])
        B1_cols.extend([e_idx, e_idx])
        B1_data.extend([-1.0, +1.0])

    # B2 (triangle boundary operator)
    B2_rows, B2_cols, B2_data = [], [], []
    for t_idx, (i, j, k) in enumerate(active_triangles):
        e1 = tuple(sorted([i, j]))
        e2 = tuple(sorted([i, k]))
        e3 = tuple(sorted([j, k]))

        if e1 in edge_to_idx:
            B2_rows.append(edge_to_idx[e1])
            B2_cols.append(t_idx)
            B2_data.append(+1.0 if i < j else -1.0)
        if e2 in edge_to_idx:
            B2_rows.append(edge_to_idx[e2])
            B2_cols.append(t_idx)
            B2_data.append(-1.0 if i < k else +1.0)
        if e3 in edge_to_idx:
            B2_rows.append(edge_to_idx[e3])
            B2_cols.append(t_idx)
            B2_data.append(+1.0 if j < k else -1.0)

    # Adjacency
    adj_rows, adj_cols, adj_data = [], [], []
    for v0, v1 in active_edges:
        adj_rows.extend([v0, v1])
        adj_cols.extend([v1, v0])
        adj_data.extend([1.0, 1.0])

    return (
        (B1_rows, B1_cols, B1_data),
        (B2_rows, B2_cols, B2_data),
        (adj_rows, adj_cols, adj_data),
    )


def _filter_at_threshold(filt_val, global_edges, edge_distances, triangle_edges):
    """Return active edges, triangles, and edge-to-index map at a filtration value."""
    active_edges = [e for e in global_edges if edge_distances[e] <= filt_val]
    active_edge_set = set(active_edges)

    active_triangles = [
        tri
        for tri, e1, e2, e3 in triangle_edges
        if e1 in active_edge_set and e2 in active_edge_set and e3 in active_edge_set
    ]

    edge_to_idx = {e: idx for idx, e in enumerate(active_edges)}
    return active_edges, active_triangles, edge_to_idx


# ---------------------------------------------------------------------------
# Fast Betti computation via union-find (β₀) and matrix rank (β₁)
# ---------------------------------------------------------------------------

class UnionFind:
    """Disjoint-set with path compression and union by rank."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.n_components = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_components -= 1


def _compute_rank_B2(active_triangles, edge_to_idx, n_edges):
    """
    Compute rank of boundary operator B2 via GPU-accelerated SVD.

    B2 is the (n_edges × n_triangles) boundary operator with entries in {-1, 0, +1}
    and exactly 3 nonzero entries per column. Uses the default data-dependent
    threshold from torch.linalg.matrix_rank (≈ max(E,T) * σ_max * eps), which
    correctly separates genuine nonzero singular values from numerical noise.

    Note: a fixed threshold like atol=0.5 is NOT safe here. Dense kNN complexes
    can produce B2 with genuine nonzero singular values well below 0.5.
    """
    n_triangles = len(active_triangles)
    if n_triangles == 0:
        return 0

    # Build B2 as scipy sparse COO.
    # Triangles are sorted tuples (i < j < k), so boundary signs are (+1, -1, +1).
    rows, cols, data = [], [], []
    for t_idx, (i, j, k) in enumerate(active_triangles):
        e_ij = (i, j)
        e_ik = (i, k)
        e_jk = (j, k)
        if e_ij in edge_to_idx:
            rows.append(edge_to_idx[e_ij])
            cols.append(t_idx)
            data.append(1.0)
        if e_ik in edge_to_idx:
            rows.append(edge_to_idx[e_ik])
            cols.append(t_idx)
            data.append(-1.0)
        if e_jk in edge_to_idx:
            rows.append(edge_to_idx[e_jk])
            cols.append(t_idx)
            data.append(1.0)

    B2_sparse = sparse.coo_matrix(
        (data, (rows, cols)), shape=(n_edges, n_triangles), dtype=np.float32
    ).tocsr()

    # Float64 is essential: float32 perturbs genuine zero singular values
    # above the data-dependent threshold, overcounting rank by O(1).
    #
    # For large matrices, use Gram matrix G = B2·B2ᵀ (E×E) instead of
    # materializing dense B2 (E×T). In float64, the default threshold
    # (≈ E * λ_max * eps ≈ 1e-10) is far below the smallest genuine
    # nonzero eigenvalue of G (which is σ_min²(B2) ≈ 0.05), so this is safe.
    use_gram = n_edges * n_triangles > 5_000_000  # ~40MB in float64

    try:
        import torch  # pylint: disable=import-outside-toplevel
        if torch.cuda.is_available():
            if use_gram:
                G = (B2_sparse @ B2_sparse.T).toarray().astype(np.float64)
                G_gpu = torch.from_numpy(G).cuda()
                return int(torch.linalg.matrix_rank(G_gpu).item())
            else:
                B2_dense = B2_sparse.toarray().astype(np.float64)
                B2_gpu = torch.from_numpy(B2_dense).cuda()
                return int(torch.linalg.matrix_rank(B2_gpu).item())
    except (ImportError, RuntimeError):
        pass

    # CPU fallback
    if use_gram:
        G = (B2_sparse @ B2_sparse.T).toarray().astype(np.float64)
        return int(np.linalg.matrix_rank(G))
    else:
        B2_dense = B2_sparse.toarray().astype(np.float64)
        return int(np.linalg.matrix_rank(B2_dense))


def _compute_betti_rank(n, active_edges, active_triangles, edge_to_idx):
    """
    Compute Betti numbers using union-find (β₀) and matrix rank (β₁).

    Mathematical foundation (Hodge decomposition + rank-nullity):
      β₀ = number of connected components (union-find, O(E·α(E)))
      β₁ = E - rank(B1) - rank(B2)
      rank(B1) = V - β₀  (standard graph theory: rank of incidence matrix)
      rank(B2) via GPU-accelerated SVD

    Parameters
    ----------
    n : int
        Total number of vertices (including isolated ones)
    active_edges : list of (int, int) tuples
    active_triangles : list of (int, int, int) tuples
    edge_to_idx : dict mapping edge tuple -> index

    Returns
    -------
    tuple : (beta_0, beta_1)
    """
    n_edges = len(active_edges)

    # β₀ via union-find
    active_vertices = set()
    for v0, v1 in active_edges:
        active_vertices.add(v0)
        active_vertices.add(v1)
    n_active = len(active_vertices)

    vertex_list = sorted(active_vertices)
    v_to_uf = {v: i for i, v in enumerate(vertex_list)}

    uf = UnionFind(n_active)
    for v0, v1 in active_edges:
        uf.union(v_to_uf[v0], v_to_uf[v1])

    # Total β₀ includes isolated vertices as singleton components
    beta_0 = (n - n_active) + uf.n_components

    # β₁ via Hodge decomposition: β₁ = E - rank(B1) - rank(B2)
    rank_B1 = n - beta_0  # equivalently: n_active - uf.n_components
    rank_B2 = _compute_rank_B2(active_triangles, edge_to_idx, n_edges)
    beta_1 = n_edges - rank_B1 - rank_B2

    return beta_0, max(0, beta_1)


# ---------------------------------------------------------------------------
# CPU implementation (reference, eigsh-based — kept for validation)
# ---------------------------------------------------------------------------

def compute_betti_curve_cpu(data, k_neighbors=20, density_threshold=0.8,
                            overlap_factor=1.5, n_steps=50):
    """
    CPU implementation of filtered Betti curve computation.

    Parameters
    ----------
    data : array-like
        Point cloud data (n_samples, n_features)
    k_neighbors : int
        Size of local neighborhood
    density_threshold : float
        Percentile threshold for edge inclusion (0-1)
    overlap_factor : float
        Factor for expanding local neighborhoods
    n_steps : int
        Number of filtration steps

    Returns
    -------
    dict : {
        'filtration_values': array of filtration thresholds,
        'beta_0': array of H0 Betti numbers,
        'beta_1': array of H1 Betti numbers,
        'n_edges_active': array of active edge counts,
        'n_triangles_active': array of active triangle counts
    }
    """
    data = np.asarray(data, dtype=np.float32)
    n = len(data)

    if n <= 2:
        return {
            'filtration_values': np.array([0.0]),
            'beta_0': np.array([1]),
            'beta_1': np.array([0]),
            'n_edges_active': np.array([0]),
            'n_triangles_active': np.array([0])
        }

    # Build kNN graph
    k_neighbors = min(k_neighbors, n - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
    nn.fit(data)
    distances, indices = nn.kneighbors(data)

    # Build atlas complex
    global_edges, _, edge_distances, triangle_edges = _build_atlas(
        data, distances, indices, k_neighbors, density_threshold, overlap_factor
    )

    # Create filtration values based on edge distances
    all_distances = np.array(list(edge_distances.values()))
    # Reverse order: from max (100) to min (0) so n_steps=1 gives full complex
    filtration_values = np.percentile(all_distances, np.linspace(100, 0, n_steps))

    # Arrays to store results
    beta_0_curve = []
    beta_1_curve = []
    n_edges_curve = []
    n_triangles_curve = []

    # Compute Betti numbers at each filtration value
    for filt_val in filtration_values:
        active_edges, active_triangles, edge_to_idx = _filter_at_threshold(
            filt_val, global_edges, edge_distances, triangle_edges
        )

        if len(active_edges) == 0:
            beta_0_curve.append(n)
            beta_1_curve.append(0)
            n_edges_curve.append(0)
            n_triangles_curve.append(0)
            continue

        n_edges_active = len(active_edges)
        n_triangles_active = len(active_triangles)

        # Build boundary operator data
        b1, b2, adj = _build_boundary_data(active_edges, active_triangles, edge_to_idx)

        # Construct sparse matrices (CPU / scipy)
        B1 = sparse.coo_matrix(
            (b1[2], (b1[0], b1[1])),
            shape=(n, n_edges_active), dtype=np.float64,
        ).tocsr()

        if n_triangles_active > 0:
            B2 = sparse.coo_matrix(
                (b2[2], (b2[0], b2[1])),
                shape=(n_edges_active, n_triangles_active), dtype=np.float64,
            ).tocsr()
            L1 = (B1.T @ B1 + B2 @ B2.T).astype(np.float64)
        else:
            L1 = (B1.T @ B1).astype(np.float64)

        A = sparse.coo_matrix(
            (adj[2], (adj[0], adj[1])), shape=(n, n), dtype=np.float64,
        ).tocsr()
        deg = np.array(A.sum(axis=1)).flatten()
        D = sparse.diags(deg)
        L0 = D - A

        # Compute eigenvalues
        n_eigs_h0 = min(50, n - 2)
        n_eigs_h1 = min(50, n_edges_active - 2) if n_edges_active > 2 else 0

        try:
            eigs_h0, _ = eigsh(L0, k=n_eigs_h0, which='SM', tol=1e-4)
            eigs_h0 = np.abs(eigs_h0)
        except Exception:  # pylint: disable=broad-exception-caught
            eigs_h0 = np.array([])

        if n_eigs_h1 > 0:
            try:
                eigs_h1, _ = eigsh(L1, k=n_eigs_h1, which='SM', tol=1e-4)
                eigs_h1 = np.abs(eigs_h1)
            except Exception:  # pylint: disable=broad-exception-caught
                eigs_h1 = np.array([])
        else:
            eigs_h1 = np.array([])

        # Count zero eigenvalues
        beta_0 = np.sum(eigs_h0 < 1e-6) if len(eigs_h0) > 0 else 1
        beta_1 = np.sum(eigs_h1 < 1e-6) if len(eigs_h1) > 0 else 0

        beta_0_curve.append(beta_0)
        beta_1_curve.append(beta_1)
        n_edges_curve.append(n_edges_active)
        n_triangles_curve.append(n_triangles_active)

    return {
        'filtration_values': filtration_values,
        'beta_0': np.array(beta_0_curve),
        'beta_1': np.array(beta_1_curve),
        'n_edges_active': np.array(n_edges_curve),
        'n_triangles_active': np.array(n_triangles_curve)
    }


# ---------------------------------------------------------------------------
# Fast implementation (union-find + rank, replaces eigsh)
# ---------------------------------------------------------------------------

def compute_betti_curve_fast(data, k_neighbors=20, density_threshold=0.8,
                              overlap_factor=1.5, n_steps=50):
    """
    Fast Betti curve computation using union-find (β₀) and matrix rank (β₁).

    Instead of computing eigenvalues of Hodge Laplacians via eigsh (the bottleneck
    in compute_betti_curve_cpu), this uses algebraic topology identities:
      β₀ = connected components via union-find, O(E·α(E))
      β₁ = E - rank(B1) - rank(B2) via Hodge decomposition
      rank(B1) = V - β₀ (graph theory, free)
      rank(B2) via GPU-accelerated SVD (torch.linalg.matrix_rank)

    Parameters
    ----------
    data : array-like
        Point cloud data (n_samples, n_features)
    k_neighbors : int
        Size of local neighborhood
    density_threshold : float
        Percentile threshold for edge inclusion (0-1)
    overlap_factor : float
        Factor for expanding local neighborhoods
    n_steps : int
        Number of filtration steps

    Returns
    -------
    dict : Same structure as compute_betti_curve_cpu
    """
    data = np.asarray(data, dtype=np.float32)
    n = len(data)

    if n <= 2:
        return {
            'filtration_values': np.array([0.0]),
            'beta_0': np.array([1]),
            'beta_1': np.array([0]),
            'n_edges_active': np.array([0]),
            'n_triangles_active': np.array([0])
        }

    # Build kNN graph
    k_neighbors = min(k_neighbors, n - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
    nn.fit(data)
    distances, indices = nn.kneighbors(data)

    # Build atlas complex
    global_edges, _, edge_distances, triangle_edges = _build_atlas(
        data, distances, indices, k_neighbors, density_threshold, overlap_factor
    )

    # Create filtration values based on edge distances
    all_distances = np.array(list(edge_distances.values()))
    filtration_values = np.percentile(all_distances, np.linspace(100, 0, n_steps))

    beta_0_curve = []
    beta_1_curve = []
    n_edges_curve = []
    n_triangles_curve = []

    for filt_val in filtration_values:
        active_edges, active_triangles, edge_to_idx = _filter_at_threshold(
            filt_val, global_edges, edge_distances, triangle_edges
        )

        n_edges_active = len(active_edges)
        n_triangles_active = len(active_triangles)

        if n_edges_active == 0:
            beta_0, beta_1 = n, 0
        else:
            beta_0, beta_1 = _compute_betti_rank(
                n, active_edges, active_triangles, edge_to_idx
            )

        beta_0_curve.append(beta_0)
        beta_1_curve.append(beta_1)
        n_edges_curve.append(n_edges_active)
        n_triangles_curve.append(n_triangles_active)

    return {
        'filtration_values': filtration_values,
        'beta_0': np.array(beta_0_curve),
        'beta_1': np.array(beta_1_curve),
        'n_edges_active': np.array(n_edges_curve),
        'n_triangles_active': np.array(n_triangles_curve)
    }


# ---------------------------------------------------------------------------
# GPU implementation (GPU kNN + rank-based Betti)
# ---------------------------------------------------------------------------

def compute_betti_curve_gpu(data, k_neighbors=20, density_threshold=0.8,
                            overlap_factor=1.5, n_steps=50):
    """
    GPU implementation of filtered Betti curve computation.

    Uses GPU kNN (cuVS/cuML) for graph construction and rank-based Betti
    computation (union-find for β₀, GPU SVD for β₁). Atlas building runs
    on CPU (set operations are faster there).

    Parameters
    ----------
    data : array-like
        Point cloud data (n_samples, n_features)
    k_neighbors : int
        Size of local neighborhood
    density_threshold : float
        Percentile threshold for edge inclusion (0-1)
    overlap_factor : float
        Factor for expanding local neighborhoods
    n_steps : int
        Number of filtration steps

    Returns
    -------
    dict : Same structure as compute_betti_curve_cpu
    """
    import cupy as cp  # pylint: disable=import-outside-toplevel

    # kNN backend: prefer cuVS, fallback to cuML
    try:
        from cuvs.neighbors import brute_force  # pylint: disable=import-outside-toplevel
        USE_CUVS = True
    except ImportError:
        from cuml.neighbors import NearestNeighbors as cumlNearestNeighbors  # pylint: disable=import-outside-toplevel
        USE_CUVS = False

    # Convert to CuPy array
    data_gpu = cp.asarray(data, dtype=cp.float32)
    n = len(data_gpu)

    if n <= 2:
        return {
            'filtration_values': np.array([0.0]),
            'beta_0': np.array([1]),
            'beta_1': np.array([0]),
            'n_edges_active': np.array([0]),
            'n_triangles_active': np.array([0])
        }

    # Build kNN graph on GPU
    k_neighbors = min(k_neighbors, n - 1)

    if USE_CUVS:
        index = brute_force.build(data_gpu, metric="euclidean")
        distances_gpu, indices_gpu = brute_force.search(index, data_gpu, k_neighbors + 1)
    else:
        nn = cumlNearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')  # pylint: disable=used-before-assignment
        nn.fit(data_gpu)
        distances_gpu, indices_gpu = nn.kneighbors(data_gpu)
        distances_gpu = cp.asarray(distances_gpu)
        indices_gpu = cp.asarray(indices_gpu)

    # Move to CPU for set operations (faster on CPU)
    indices_cpu = cp.asnumpy(indices_gpu).astype(np.int32)
    distances_cpu = cp.asnumpy(distances_gpu)
    data_cpu = cp.asnumpy(data_gpu)

    # Build atlas complex (shared helper)
    global_edges, _, edge_distances, triangle_edges = _build_atlas(
        data_cpu, distances_cpu, indices_cpu, k_neighbors,
        density_threshold, overlap_factor
    )

    # Create filtration values
    all_distances = np.array(list(edge_distances.values()))
    filtration_values = np.percentile(all_distances, np.linspace(100, 0, n_steps))

    beta_0_curve = []
    beta_1_curve = []
    n_edges_curve = []
    n_triangles_curve = []

    for filt_val in filtration_values:
        active_edges, active_triangles, edge_to_idx = _filter_at_threshold(
            filt_val, global_edges, edge_distances, triangle_edges
        )

        n_edges_active = len(active_edges)
        n_triangles_active = len(active_triangles)

        if n_edges_active == 0:
            beta_0, beta_1 = n, 0
        else:
            beta_0, beta_1 = _compute_betti_rank(
                n, active_edges, active_triangles, edge_to_idx
            )

        beta_0_curve.append(beta_0)
        beta_1_curve.append(beta_1)
        n_edges_curve.append(n_edges_active)
        n_triangles_curve.append(n_triangles_active)

    return {
        'filtration_values': filtration_values,
        'beta_0': np.array(beta_0_curve),
        'beta_1': np.array(beta_1_curve),
        'n_edges_active': np.array(n_edges_curve),
        'n_triangles_active': np.array(n_triangles_curve)
    }


# ---------------------------------------------------------------------------
# Backend selector
# ---------------------------------------------------------------------------

def compute_betti_curve(data, k_neighbors=20, density_threshold=0.8, overlap_factor=1.5,
                        n_steps=50, use_gpu=True):
    """
    Compute filtered Betti curve (backend selector).

    Automatically selects GPU or CPU implementation based on availability
    and use_gpu parameter.

    Parameters
    ----------
    data : array-like
        Point cloud data (n_samples, n_features)
    k_neighbors : int
        Size of local neighborhood
    density_threshold : float
        Percentile threshold for edge inclusion (0-1)
    overlap_factor : float
        Factor for expanding local neighborhoods
    n_steps : int
        Number of filtration steps
    use_gpu : bool
        Whether to use GPU acceleration (if available)

    Returns
    -------
    dict : {
        'filtration_values': array of filtration thresholds,
        'beta_0': array of H0 Betti numbers,
        'beta_1': array of H1 Betti numbers,
        'n_edges_active': array of active edge counts,
        'n_triangles_active': array of active triangle counts
    }
    """
    if use_gpu:
        try:
            return compute_betti_curve_gpu(
                data, k_neighbors=k_neighbors,
                density_threshold=density_threshold,
                overlap_factor=overlap_factor,
                n_steps=n_steps
            )
        except ImportError as e:
            warnings.warn(f"GPU not available ({e}), falling back to fast CPU", UserWarning)

    return compute_betti_curve_fast(
        data, k_neighbors=k_neighbors,
        density_threshold=density_threshold,
        overlap_factor=overlap_factor,
        n_steps=n_steps
    )
