"""
GPU implementation of atlas topology computation.

Uses cuVS/cuML for kNN, CuPy for arrays, and CuPy sparse matrices.
Requires: cupy, cupyx, cuvs (or cuml as fallback)
"""

import numpy as np
import cupy as cp
from cupyx.scipy import sparse as cp_sparse
from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh

# kNN backend: prefer cuVS, fallback to cuML
try:
    from cuvs.neighbors import brute_force
    USE_CUVS = True
except ImportError:
    from cuml.neighbors import NearestNeighbors as cumlNearestNeighbors
    USE_CUVS = False


def compute_h0_h1_atlas_gpu(data, k_neighbors=20, density_threshold=0.8, overlap_factor=1.5,
                            return_distances=False):
    """
    GPU implementation of atlas-based H0/H1 computation.

    Uses CuPy for array operations, cuVS/cuML for kNN, and CuPy sparse matrices.

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
    # Convert to CuPy array
    data_gpu = cp.asarray(data, dtype=cp.float32)
    n = len(data_gpu)

    if n <= 2:
        result = (np.array([[0, np.inf]]), np.array([[0, 0]]))
        if return_distances:
            return result + ({},)
        return result

    # Build kNN graph using cuVS or cuML
    k_neighbors = min(k_neighbors, n - 1)

    if USE_CUVS:
        # Use cuVS brute force kNN
        index = brute_force.build(data_gpu, metric="euclidean")
        distances_gpu, indices_gpu = brute_force.search(index, data_gpu, k_neighbors + 1)
    else:
        # Use cuML
        nn = cumlNearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
        nn.fit(data_gpu)
        distances_gpu, indices_gpu = nn.kneighbors(data_gpu)
        distances_gpu = cp.asarray(distances_gpu)
        indices_gpu = cp.asarray(indices_gpu)

    # Move to CPU for set operations and complex indexing (faster on CPU)
    indices_cpu = cp.asnumpy(indices_gpu).astype(np.int32)
    distances_cpu = cp.asnumpy(distances_gpu)
    data_cpu = cp.asnumpy(data_gpu)

    # Track edge distances
    edge_distances = {} if return_distances else None

    # Global edge and triangle sets
    global_edges = set()
    global_triangles = set()

    # Build atlas patches (on CPU, as set operations are faster)
    for i in range(n):
        local_neighborhood = indices_cpu[i, 1:]
        local_dists = distances_cpu[i, 1:]

        # Edges from i to neighbors
        for j_idx, j in enumerate(local_neighborhood):
            edge = tuple(sorted([i, int(j)]))
            global_edges.add(edge)
            if return_distances and edge not in edge_distances:
                edge_distances[edge] = local_dists[j_idx]

        # Dense local patch
        dist_threshold = np.percentile(local_dists, density_threshold * 100) * overlap_factor

        if len(local_neighborhood) > 1:
            neighborhood_coords = data_cpu[local_neighborhood]
            for idx_j in range(len(local_neighborhood)):
                j = int(local_neighborhood[idx_j])
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

    # Build sparse boundary operators on GPU using CuPy sparse matrices
    # B1: edge -> vertex
    B1_rows, B1_cols, B1_data = [], [], []
    for e_idx, (v0, v1) in enumerate(edges):
        B1_rows.extend([v0, v1])
        B1_cols.extend([e_idx, e_idx])
        B1_data.extend([-1.0, +1.0])

    B1_rows_gpu = cp.array(B1_rows, dtype=cp.int32)
    B1_cols_gpu = cp.array(B1_cols, dtype=cp.int32)
    B1_data_gpu = cp.array(B1_data, dtype=cp.float64)

    B1 = cp_sparse.coo_matrix(
        (B1_data_gpu, (B1_rows_gpu, B1_cols_gpu)),
        shape=(n, n_edges), dtype=cp.float64
    ).tocsr()

    # B2: triangle -> edge
    if n_triangles > 0:
        B2_rows, B2_cols, B2_data = [], [], []
        for t_idx, (i, j, k) in enumerate(triangles):
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

        B2_rows_gpu = cp.array(B2_rows, dtype=cp.int32)
        B2_cols_gpu = cp.array(B2_cols, dtype=cp.int32)
        B2_data_gpu = cp.array(B2_data, dtype=cp.float64)

        B2 = cp_sparse.coo_matrix(
            (B2_data_gpu, (B2_rows_gpu, B2_cols_gpu)),
            shape=(n_edges, n_triangles), dtype=cp.float64
        ).tocsr()

        # Hodge Laplacian on GPU
        L1 = (B1.T @ B1 + B2 @ B2.T).astype(cp.float64)
    else:
        L1 = (B1.T @ B1).astype(cp.float64)

    # L0 for H0
    adj_rows, adj_cols, adj_data = [], [], []
    for v0, v1 in edges:
        adj_rows.extend([v0, v1])
        adj_cols.extend([v1, v0])
        adj_data.extend([1.0, 1.0])

    adj_rows_gpu = cp.array(adj_rows, dtype=cp.int32)
    adj_cols_gpu = cp.array(adj_cols, dtype=cp.int32)
    adj_data_gpu = cp.array(adj_data, dtype=cp.float64)

    A = cp_sparse.coo_matrix(
        (adj_data_gpu, (adj_rows_gpu, adj_cols_gpu)),
        shape=(n, n), dtype=cp.float64
    ).tocsr()

    deg = cp.array(A.sum(axis=1)).flatten()
    D = cp_sparse.diags(deg)
    L0 = D - A

    # Eigenvalue computation on GPU
    n_eigs_h0 = min(50, n - 2)
    n_eigs_h1 = min(50, n_edges - 2) if n_edges > 2 else 0

    try:
        eigs_h0_gpu, _ = cp_eigsh(L0, k=n_eigs_h0, which='SM', tol=1e-4)
        eigs_h0 = cp.asnumpy(cp.abs(eigs_h0_gpu))
    except:
        eigs_h0 = np.array([])

    if n_eigs_h1 > 0:
        try:
            eigs_h1_gpu, _ = cp_eigsh(L1, k=n_eigs_h1, which='SM', tol=1e-4)
            eigs_h1 = cp.asnumpy(cp.abs(eigs_h1_gpu))
        except:
            eigs_h1 = np.array([])
    else:
        eigs_h1 = np.array([])

    # Count zero eigenvalues
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
