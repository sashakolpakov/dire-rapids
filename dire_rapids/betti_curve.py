"""
Betti curve computation using filtered edge addition.

Computes Betti numbers (β₀, β₁) at multiple filtration thresholds by:
1. Building full atlas complex once with kNN graph
2. Recording edge and triangle filtration values
3. Progressively adding edges and triangles by threshold
4. Updating β₀ with union-find and rank(B₂) over GF(2) incrementally

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


class IncrementalF2Rank:
    """Incremental sparse rank over GF(2), represented as Python integer bitsets."""

    def __init__(self):
        self.pivots = {}
        self.rank = 0

    def add(self, column):
        """Add one boundary column. Return True if it increases rank."""
        while column:
            pivot = column.bit_length() - 1
            existing = self.pivots.get(pivot)
            if existing is None:
                self.pivots[pivot] = column
                self.rank += 1
                return True
            column ^= existing
        return False


def _compute_betti_curve_incremental(n, global_edges, edge_distances, triangle_edges,
                                     filtration_values):
    """
    Compute atlas Betti curves in one monotone filtration pass.

    Each simplex is added once. The pass uses union-find for beta_0 and maintains
    rank(B2) over GF(2) with sparse bitsets.
    Persistent homology is normally computed over a field; GF(2) avoids orientation
    bookkeeping and is the standard choice for fast Vietoris-Rips computations.
    """
    edge_list = sorted(global_edges, key=lambda e: (edge_distances[e], e))
    edge_to_idx = {edge: idx for idx, edge in enumerate(edge_list)}

    triangle_events = []
    for tri, e1, e2, e3 in triangle_edges:
        if e1 in edge_to_idx and e2 in edge_to_idx and e3 in edge_to_idx:
            filt_val = max(edge_distances[e1], edge_distances[e2], edge_distances[e3])
            col = (
                (1 << edge_to_idx[e1]) |
                (1 << edge_to_idx[e2]) |
                (1 << edge_to_idx[e3])
            )
            triangle_events.append((filt_val, tri, col))
    triangle_events.sort(key=lambda item: (item[0], item[1]))

    order = np.argsort(filtration_values)
    beta_0_curve = np.empty(len(filtration_values), dtype=np.int64)
    beta_1_curve = np.empty(len(filtration_values), dtype=np.int64)
    n_edges_curve = np.empty(len(filtration_values), dtype=np.int64)
    n_triangles_curve = np.empty(len(filtration_values), dtype=np.int64)

    uf = UnionFind(n)
    rank_b2 = IncrementalF2Rank()
    edge_pos = 0
    triangle_pos = 0
    n_edges_active = 0
    n_triangles_active = 0

    for out_idx in order:
        filt_val = filtration_values[out_idx]

        while edge_pos < len(edge_list) and edge_distances[edge_list[edge_pos]] <= filt_val:
            v0, v1 = edge_list[edge_pos]
            uf.union(v0, v1)
            edge_pos += 1
            n_edges_active += 1

        while triangle_pos < len(triangle_events) and triangle_events[triangle_pos][0] <= filt_val:
            rank_b2.add(triangle_events[triangle_pos][2])
            triangle_pos += 1
            n_triangles_active += 1

        beta_0 = uf.n_components
        rank_b1 = n - beta_0
        beta_1 = n_edges_active - rank_b1 - rank_b2.rank

        beta_0_curve[out_idx] = beta_0
        beta_1_curve[out_idx] = max(0, beta_1)
        n_edges_curve[out_idx] = n_edges_active
        n_triangles_curve[out_idx] = n_triangles_active

    return {
        'filtration_values': filtration_values,
        'beta_0': beta_0_curve,
        'beta_1': beta_1_curve,
        'n_edges_active': n_edges_curve,
        'n_triangles_active': n_triangles_curve
    }


# ---------------------------------------------------------------------------
# Betti computation helpers
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
# Fast implementation (incremental atlas)
# ---------------------------------------------------------------------------

def compute_betti_curve_fast(data, k_neighbors=20, density_threshold=0.8,
                              overlap_factor=1.5, n_steps=50):
    """
    Fast Betti curve computation using an incremental atlas filtration.

    Adds each edge and triangle once in filtration order:
      β₀ = connected components via union-find, O(E·α(E))
      β₁ = E - rank(B1) - rank(B2)
      rank(B1) = V - β₀
      rank(B2) is maintained incrementally over GF(2) using sparse bitsets

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

    return _compute_betti_curve_incremental(
        n, global_edges, edge_distances, triangle_edges, filtration_values
    )


# ---------------------------------------------------------------------------
# GPU implementation (GPU kNN + incremental atlas Betti)
# ---------------------------------------------------------------------------

def compute_betti_curve_gpu(data, k_neighbors=20, density_threshold=0.8,
                            overlap_factor=1.5, n_steps=50):
    """
    GPU implementation of filtered Betti curve computation.

    Uses GPU kNN (cuVS/cuML) for graph construction, then the shared
    incremental atlas computation (union-find for β₀ and GF(2) bitset
    elimination for rank(B₂)). Atlas building runs on CPU because the set-heavy
    merge step is faster and simpler there.

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

    return _compute_betti_curve_incremental(
        n, global_edges, edge_distances, triangle_edges, filtration_values
    )


# ---------------------------------------------------------------------------
# Ripser-based persistent-homology implementation (explicit reference backend)
# ---------------------------------------------------------------------------

def compute_betti_curve_ripser(data, n_steps=50, maxdim=1, thresh=None):
    """Betti curves via ripser's persistent-homology reduction.

    Much faster than the eigsh and dense-SVD paths at every scale: empirically
    ~4000× on MNIST n=3000 (eigsh path hangs for hours; ripser: ~6 s). The
    cost grows roughly O(N²) for maxdim=1 and O(N³) for maxdim=2, but the
    constants are small thanks to cohomology + apparent-pairs sparsification.

    Persistent-homology view: ripser returns, for each dimension, a list of
    (birth, death) intervals. The Betti curve is β_d(t) = #{intervals with
    birth ≤ t < death} — "number of d-cycles alive at filtration t".

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Point cloud.
    n_steps : int, default=50
        Number of filtration samples at which to evaluate each Betti curve.
    maxdim : int, default=1
        Highest homology dimension to compute. maxdim=1 returns β_0 and β_1;
        maxdim=2 additionally computes β_2 at O(N³) cost.
    thresh : float or None, default=None
        Maximum filtration radius. None lets ripser run to its default
        (≈ the enclosing radius of the data). Cap this on large data if you
        only care about small-scale topology.

    Returns
    -------
    dict
        Same shape as the other ``compute_betti_curve_*`` backends. The
        ``filtration_values`` array contains ``n_steps`` evenly spaced values
        from zero to the longest finite death time plus a 5% margin.
        ``beta_0`` and ``beta_1`` are length-``n_steps`` arrays, and ``beta_2``
        is present only when ``maxdim >= 2``. The active-edge and
        active-triangle counts are filled with NaN because ripser does not
        expose them; callers depending on those counts should use
        ``compute_betti_curve_fast`` instead.

    Notes
    -----
    Requires ``ripser``: install via ``pip install dire-rapids[persistent]``.
    """
    try:
        from ripser import ripser  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            "compute_betti_curve_ripser requires ripser. "
            "Install via `pip install dire-rapids[persistent]` "
            "or `pip install ripser`."
        ) from e

    data = np.asarray(data, dtype=np.float32)
    n = len(data)
    if n <= 2:
        zeros = np.zeros(1)
        return {
            'filtration_values': np.array([0.0]),
            'beta_0': np.array([1]),
            'beta_1': zeros.astype(int),
            'n_edges_active': np.array([np.nan]),
            'n_triangles_active': np.array([np.nan]),
        }

    ripser_kwargs = {'maxdim': maxdim}
    if thresh is not None:
        ripser_kwargs['thresh'] = thresh
    out = ripser(data, **ripser_kwargs)
    dgms = out['dgms']  # list: [H0_dgm, H1_dgm, ..., H{maxdim}_dgm]

    # Pick n_steps filtration values across the range where anything interesting
    # happens. Use the longest finite death time (across all dims), plus margin.
    finite_deaths = []
    for dgm in dgms:
        if len(dgm) > 0:
            fd = dgm[np.isfinite(dgm[:, 1]), 1]
            if len(fd) > 0:
                finite_deaths.append(float(fd.max()))
    max_filt = max(finite_deaths) * 1.05 if finite_deaths else 1.0
    filtration_values = np.linspace(0.0, max_filt, n_steps)

    def betti_curve_from_dgm(dgm):
        if len(dgm) == 0:
            return np.zeros(n_steps, dtype=np.int64)
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        curve = np.empty(n_steps, dtype=np.int64)
        for i, t in enumerate(filtration_values):
            curve[i] = int(((births <= t) & (deaths > t)).sum())
        return curve

    result = {
        'filtration_values': filtration_values,
        'beta_0': betti_curve_from_dgm(dgms[0]),
        'beta_1': (betti_curve_from_dgm(dgms[1]) if len(dgms) > 1
                   else np.zeros(n_steps, dtype=np.int64)),
        'n_edges_active': np.full(n_steps, np.nan),
        'n_triangles_active': np.full(n_steps, np.nan),
    }
    if maxdim >= 2 and len(dgms) > 2:
        result['beta_2'] = betti_curve_from_dgm(dgms[2])
    return result


# ---------------------------------------------------------------------------
# Backend selector
# ---------------------------------------------------------------------------

def compute_betti_curve(data, k_neighbors=20, density_threshold=0.8, overlap_factor=1.5,
                        n_steps=50, use_gpu=True, prefer_ripser=False):
    """
    Compute filtered Betti curve (backend selector).

    By default, try the GPU kNN + incremental atlas path
    (``compute_betti_curve_gpu``) when ``use_gpu`` is true, then fall back to
    the fast CPU incremental atlas path (``compute_betti_curve_fast``).
    ``prefer_ripser=True`` explicitly selects ripser first when it is installed.

    Parameters
    ----------
    data : array-like
        Point cloud data (n_samples, n_features).
    k_neighbors : int
        Size of local neighborhood (ignored by the ripser path, which uses
        Vietoris-Rips filtration on the full distance matrix up to ``thresh``).
    density_threshold : float
        Percentile threshold for edge inclusion, 0-1 (ignored by ripser).
    overlap_factor : float
        Factor for expanding local neighborhoods (ignored by ripser).
    n_steps : int
        Number of filtration steps.
    use_gpu : bool
        Whether to try the GPU kNN Atlas path before the CPU Atlas path.
    prefer_ripser : bool
        Explicitly prefer ripser when available. The default is False, so the
        public selector uses the atlas backends unless requested otherwise.

    Returns
    -------
    dict : {
        'filtration_values': array of filtration thresholds,
        'beta_0': array of H0 Betti numbers,
        'beta_1': array of H1 Betti numbers,
        'n_edges_active': array of active edge counts (NaN under ripser),
        'n_triangles_active': array of active triangle counts (NaN under ripser)
    }
    """
    if prefer_ripser:
        try:
            return compute_betti_curve_ripser(data, n_steps=n_steps)
        except ImportError:
            pass   # ripser not installed — fall through to the atlas paths

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
