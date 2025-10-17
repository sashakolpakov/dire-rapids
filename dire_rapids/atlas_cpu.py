"""
CPU implementation of atlas topology computation.

Uses sklearn for kNN, numpy for arrays, and scipy sparse matrices.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


def compute_h0_h1_atlas_cpu(data, k_neighbors=20, density_threshold=0.8, overlap_factor=1.5,
                            return_distances=False):
    """
    CPU implementation of atlas-based H0/H1 computation.

    Build dense local triangulations around each point, then merge consistently.

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
    return_distances : bool
        If True, also return edge-to-distance mapping

    Returns
    -------
    tuple : (h0_diagram, h1_diagram) or (h0_diagram, h1_diagram, edge_distances)
    """
    data = np.asarray(data, dtype=np.float32)
    n = len(data)

    if n <= 2:
        result = (np.array([[0, np.inf]]), np.array([[0, 0]]))
        if return_distances:
            return result + ({},)
        return result

    # Build kNN graph
    k_neighbors = min(k_neighbors, n - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
    nn.fit(data)
    distances, indices = nn.kneighbors(data)

    # Track edge distances for persistence
    edge_distances = {} if return_distances else None

    # Global edge and triangle sets
    global_edges = set()
    global_triangles = set()

    # Build atlas patches
    for i in range(n):
        local_neighborhood = indices[i, 1:]  # Exclude self
        local_dists = distances[i, 1:]

        # Edges from i to neighbors
        for j_idx, j in enumerate(local_neighborhood):
            edge = tuple(sorted([i, int(j)]))
            global_edges.add(edge)
            if return_distances and edge not in edge_distances:
                edge_distances[edge] = local_dists[j_idx]

        # Dense local patch: connect neighbors if close enough
        dist_threshold = np.percentile(local_dists, density_threshold * 100) * overlap_factor

        # Vectorized distance computation within neighborhood
        if len(local_neighborhood) > 1:
            neighborhood_coords = data[local_neighborhood]
            for idx_j in range(len(local_neighborhood)):
                j = int(local_neighborhood[idx_j])
                # Vectorized distance from j to all k > j
                dists_jk = np.linalg.norm(
                    neighborhood_coords[idx_j+1:] - neighborhood_coords[idx_j], axis=1
                )
                close_enough = dists_jk < dist_threshold
                for offset, is_close in enumerate(close_enough):
                    if is_close:
                        k = int(local_neighborhood[idx_j + 1 + offset])
                        edge = tuple(sorted([j, k]))
                        global_edges.add(edge)
                        if return_distances and edge not in edge_distances:
                            edge_distances[edge] = dists_jk[offset]

        # Build triangles
        for idx_j in range(len(local_neighborhood)):
            for idx_k in range(idx_j + 1, len(local_neighborhood)):
                j = int(local_neighborhood[idx_j])
                k = int(local_neighborhood[idx_k])
                e1 = tuple(sorted([i, j]))
                e2 = tuple(sorted([i, k]))
                e3 = tuple(sorted([j, k]))
                if e1 in global_edges and e2 in global_edges and e3 in global_edges:
                    global_triangles.add(tuple(sorted([i, j, k])))

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

    # Build B2 (triangle boundary operator)
    if n_triangles > 0:
        B2_rows, B2_cols, B2_data = [], [], []

        for t_idx, (i, j, k) in enumerate(triangles):
            e1 = tuple(sorted([i, j]))
            e2 = tuple(sorted([i, k]))
            e3 = tuple(sorted([j, k]))

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

    # Build L0 for H0 (connected components)
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
    except:
        eigs_h0 = np.array([])

    if n_eigs_h1 > 0:
        try:
            eigs_h1, _ = eigsh(L1, k=n_eigs_h1, which='SM', tol=1e-4)
            eigs_h1 = np.abs(eigs_h1)
        except:
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

    if return_distances:
        return h0_diagram, h1_diagram, edge_distances
    return h0_diagram, h1_diagram
