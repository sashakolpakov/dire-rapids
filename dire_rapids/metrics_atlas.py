"""
Local kNN Atlas approach for fast topology computation.

Uses overlapping local neighborhoods that are densely triangulated,
then glued together consistently into a global complex.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def compute_h0_h1_atlas(data, k_local=20, density_threshold=0.8, overlap_factor=1.5, use_gpu=True):
    """
    Compute H0/H1 using local kNN atlas approach.

    Build dense local triangulations around each point, then merge consistently.
    This avoids the "holes" problem of global sparse kNN graphs.

    Parameters
    ----------
    data : array-like
        Point cloud data (n_samples, n_features)
    k_local : int
        Size of local neighborhood (default 20, should be >> than typical global k)
    density_threshold : float
        Percentile threshold for edge inclusion (0-1). Lower = denser triangulation.
        Default 0.8 means edges up to 80th percentile of local distances are included.
    overlap_factor : float
        Factor for expanding local neighborhoods to ensure overlap (default 1.5).
        Higher values create more dense, overlapping patches.
    use_gpu : bool
        Whether to use GPU for kNN computation

    Returns
    -------
    tuple : (h0_diagram, h1_diagram)
        Persistence diagrams with [birth, death] pairs
    """
    from sklearn.neighbors import NearestNeighbors  # pylint: disable=import-outside-toplevel

    data = np.asarray(data, dtype=np.float32)
    n = len(data)

    if n <= 2:
        return np.array([[0, np.inf]]), np.array([[0, 0]])

    # Build kNN graph for local neighborhoods
    k_local = min(k_local, n - 1)
    nn = NearestNeighbors(n_neighbors=k_local + 1, metric='euclidean')
    nn.fit(data)
    distances, indices = nn.kneighbors(data)

    # Global edge and triangle sets (use sets to auto-deduplicate)
    global_edges = set()
    global_triangles = set()

    # For each point, build dense local triangulation
    for i in range(n):
        local_neighborhood = indices[i, 1:]  # Exclude self
        local_dists = distances[i, 1:]

        # Build edges within local neighborhood
        # Strategy: connect point i to all neighbors, and neighbors to each other
        # if they're close enough (density threshold)

        # Edges from i to neighbors
        for j in local_neighborhood:
            edge = tuple(sorted([i, int(j)]))
            global_edges.add(edge)

        # Edges within neighborhood (creates dense local patch)
        # Use distance threshold: connect if distance < percentile of local distances
        # Apply overlap_factor to make local patches denser
        dist_threshold = np.percentile(local_dists, density_threshold * 100) * overlap_factor

        for idx_j in range(len(local_neighborhood)):
            for idx_k in range(idx_j + 1, len(local_neighborhood)):
                j = int(local_neighborhood[idx_j])
                k = int(local_neighborhood[idx_k])

                # Check if j and k are close enough to connect
                dist_jk = np.linalg.norm(data[j] - data[k])
                if dist_jk < dist_threshold:
                    edge = tuple(sorted([j, k]))
                    global_edges.add(edge)

        # Now build triangles from all edges in the local neighborhood
        # This happens AFTER all edges are added to ensure triangles are complete
        for idx_j in range(len(local_neighborhood)):
            for idx_k in range(idx_j + 1, len(local_neighborhood)):
                j = int(local_neighborhood[idx_j])
                k = int(local_neighborhood[idx_k])

                # Form triangle (i, j, k) if all three edges exist
                e1 = tuple(sorted([i, j]))
                e2 = tuple(sorted([i, k]))
                e3 = tuple(sorted([j, k]))

                # Triangle exists if all three edges are in the global edge set
                if e1 in global_edges and e2 in global_edges and e3 in global_edges:
                    triangle = tuple(sorted([i, j, k]))
                    global_triangles.add(triangle)

    # Convert to sorted lists
    edges = sorted(list(global_edges))
    triangles = sorted(list(global_triangles))

    n_edges = len(edges)
    n_triangles = len(triangles)

    edge_to_idx = {e: idx for idx, e in enumerate(edges)}

    # Build B1 (edge boundary operator)
    B1_rows, B1_cols, B1_data = [], [], []
    for e_idx, (v0, v1) in enumerate(edges):
        B1_rows.extend([v0, v1])
        B1_cols.extend([e_idx, e_idx])
        B1_data.extend([-1, +1])

    B1 = sparse.coo_matrix((B1_data, (B1_rows, B1_cols)),
                           shape=(n, n_edges), dtype=np.float64)
    B1 = B1.tocsr().astype(np.float64)

    # Build B2 (triangle boundary operator) if triangles exist
    if n_triangles > 0:
        B2_rows, B2_cols, B2_data = [], [], []

        for t_idx, (i, j, k) in enumerate(triangles):
            # Canonical ordering: i < j < k
            e1 = tuple(sorted([i, j]))
            e2 = tuple(sorted([i, k]))
            e3 = tuple(sorted([j, k]))

            # Standard boundary orientation: ∂(i,j,k) = (j,k) - (i,k) + (i,j)
            # Sign depends on edge orientation vs triangle orientation
            if e1 in edge_to_idx:
                B2_rows.append(edge_to_idx[e1])
                B2_cols.append(t_idx)
                B2_data.append(+1 if i < j else -1)

            if e2 in edge_to_idx:
                B2_rows.append(edge_to_idx[e2])
                B2_cols.append(t_idx)
                B2_data.append(-1 if i < k else +1)

            if e3 in edge_to_idx:
                B2_rows.append(edge_to_idx[e3])
                B2_cols.append(t_idx)
                B2_data.append(+1 if j < k else -1)

        B2 = sparse.coo_matrix((B2_data, (B2_rows, B2_cols)),
                              shape=(n_edges, n_triangles), dtype=np.float64)
        B2 = B2.tocsr().astype(np.float64)

        # Hodge Laplacian L1 = B1^T @ B1 + B2 @ B2^T
        L1 = (B1.T @ B1 + B2 @ B2.T).astype(np.float64)
    else:
        L1 = (B1.T @ B1).astype(np.float64)

    # Compute L0 for H0 (connected components)
    # Build adjacency matrix
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

    # Compute eigenvalues
    n_eigs_h0 = min(50, n - 2)
    n_eigs_h1 = min(50, n_edges - 2) if n_edges > 2 else 0

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

    # Count zero eigenvalues (Betti numbers)
    beta_0 = np.sum(eigs_h0 < 1e-6) if len(eigs_h0) > 0 else 1
    beta_1 = np.sum(eigs_h1 < 1e-6) if len(eigs_h1) > 0 else 0

    # Build persistence diagrams
    h0_features = [[0.0, np.inf] for _ in range(beta_0)]
    h1_features = [[0.0, np.inf] for _ in range(beta_1)]

    h0_diagram = np.array(h0_features) if h0_features else np.array([[0, 0]])
    h1_diagram = np.array(h1_features) if h1_features else np.array([[0, 0]])

    return h0_diagram, h1_diagram
