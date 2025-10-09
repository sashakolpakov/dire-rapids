# metrics.py

"""
Performance metrics for dimensionality reduction evaluation.

This module provides GPU-accelerated metrics using RAPIDS cuML for evaluating
the quality of dimensionality reduction embeddings, including:

- Distortion metrics (stress)
- Context preservation metrics (SVM, kNN classification)
- Topological metrics (persistence homology, Betti curves)

The module supports multiple backends for persistence computation:
- giotto-ph (fastest CPU, multi-threaded)
- ripser++ (GPU-accelerated)
- ripser (CPU fallback)
"""

import gc
import warnings
import numpy as np

# CuPy for GPU arrays
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    warnings.warn("CuPy not available. GPU acceleration disabled.", UserWarning)

# RAPIDS cuML imports
try:
    import cuml
    from cuml.neighbors import NearestNeighbors as cumlNearestNeighbors
    from cuml.model_selection import train_test_split as cuml_train_test_split
    from cuml.preprocessing import StandardScaler as cumlStandardScaler
    from cuml.svm import SVC as cumlSVC
    from cuml.neighbors import KNeighborsClassifier as cumlKNeighborsClassifier
    HAS_CUML = True
except ImportError:
    HAS_CUML = False
    warnings.warn(
        "cuML not available. Falling back to CPU-based scikit-learn. "
        "Install RAPIDS for GPU acceleration.",
        UserWarning
    )
    from sklearn.neighbors import NearestNeighbors
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier

# Persistence backends
_PERSISTENCE_BACKEND = None

try:
    from gph import ripser_parallel
    HAS_GIOTTO_PH = True
except ImportError:
    HAS_GIOTTO_PH = False

try:
    import ripserplusplus as rpp
    HAS_RIPSER_PP = True
except ImportError:
    HAS_RIPSER_PP = False

try:
    from ripser import ripser
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False

# Additional dependencies for persistence
try:
    from persim import wasserstein, bottleneck
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False
    warnings.warn("persim not available. Wasserstein and bottleneck distances will not be available.", UserWarning)

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False
    warnings.warn("POT (Python Optimal Transport) not available. EMD distance will not be available.", UserWarning)

try:
    from fastdtw import fastdtw
    HAS_DTW = True
except ImportError:
    HAS_DTW = False
    warnings.warn("fastdtw not available. DTW distance will not be available.", UserWarning)

try:
    from twed import twed
    HAS_TWED = True
except ImportError:
    HAS_TWED = False
    warnings.warn("twed not available. TWED distance will not be available.", UserWarning)


#
# Persistence backend management
#


def get_available_persistence_backends():
    """
    Get list of available persistence computation backends.

    Returns
    -------
    dict : Dictionary mapping backend names to availability status
    """
    return {
        'giotto-ph': HAS_GIOTTO_PH,
        'ripser++': HAS_RIPSER_PP,
        'ripser': HAS_RIPSER
    }


def set_persistence_backend(backend):
    """
    Set the persistence computation backend.

    Parameters
    ----------
    backend : str or None
        Backend to use: 'giotto-ph', 'ripser++', 'ripser', or None for auto-selection

    Raises
    ------
    ValueError
        If specified backend is not available
    """
    global _PERSISTENCE_BACKEND

    if backend is None:
        _PERSISTENCE_BACKEND = None
        return

    available = get_available_persistence_backends()

    if backend not in available:
        raise ValueError(
            f"Unknown backend '{backend}'. Available: {list(available.keys())}"
        )

    if not available[backend]:
        raise ValueError(
            f"Backend '{backend}' is not available. Install the required package."
        )

    _PERSISTENCE_BACKEND = backend


def get_persistence_backend():
    """
    Get the current persistence backend (with auto-selection if None).

    Returns
    -------
    str : Name of the selected backend

    Raises
    ------
    RuntimeError
        If no persistence backend is available
    """
    global _PERSISTENCE_BACKEND

    if _PERSISTENCE_BACKEND is not None:
        return _PERSISTENCE_BACKEND

    # Auto-select: giotto-ph > ripser++ > ripser
    if HAS_GIOTTO_PH:
        return 'giotto-ph'
    elif HAS_RIPSER_PP:
        return 'ripser++'
    elif HAS_RIPSER:
        return 'ripser'
    else:
        raise RuntimeError(
            "No persistence backend available. "
            "Install giotto-ph (recommended), ripserplusplus, or ripser."
        )


#
# Auxiliary functions
#


def welford_update_gpu(count, mean, M2, new_value, finite_threshold=1e12):
    """
    GPU-accelerated Welford's algorithm update step.

    Parameters
    ----------
    count : cupy.ndarray
        Running count of valid values
    mean : cupy.ndarray
        Running mean
    M2 : cupy.ndarray
        Running sum of squared differences
    new_value : cupy.ndarray
        New values to incorporate
    finite_threshold : float
        Maximum magnitude for inclusion

    Returns
    -------
    tuple : Updated (count, mean, M2)
    """
    is_finite = cp.isfinite(new_value) & (cp.abs(new_value) < finite_threshold)
    count = count + is_finite
    delta = new_value - mean
    mean = mean + cp.where(is_finite, delta / cp.maximum(count, 1), 0)
    delta2 = new_value - mean
    M2 = M2 + cp.where(is_finite, delta * delta2, 0)
    return count, mean, M2


def welford_finalize_gpu(count, mean, M2):
    """
    Finalize Welford's algorithm to compute mean and std.

    Parameters
    ----------
    count : cupy.ndarray
        Total count of valid values
    mean : cupy.ndarray
        Computed mean
    M2 : cupy.ndarray
        Sum of squared differences

    Returns
    -------
    tuple : (mean, std)
    """
    variance = cp.where(count > 1, M2 / (count - 1), 0.0)
    return mean, cp.sqrt(variance)


def welford_gpu(data):
    """
    GPU-accelerated computation of mean and std using Welford's algorithm.

    Parameters
    ----------
    data : cupy.ndarray
        Input data

    Returns
    -------
    tuple : (mean, std)
    """
    if not HAS_CUPY:
        # Fallback to NumPy
        data_flat = np.asarray(data).ravel()
        mean = np.nanmean(data_flat)
        std = np.nanstd(data_flat)
        return float(mean), float(std)

    if isinstance(data, np.ndarray):
        data = cp.asarray(data)

    data_flat = data.ravel()
    count = cp.zeros(1, dtype=cp.int32)
    mean = cp.zeros(1, dtype=cp.float32)
    M2 = cp.zeros(1, dtype=cp.float32)

    # Process in chunks to avoid memory issues
    chunk_size = 1024 * 1024
    for i in range(0, len(data_flat), chunk_size):
        chunk = data_flat[i:i + chunk_size]
        for val in chunk:
            count, mean, M2 = welford_update_gpu(count, mean, M2, val)

    mean_final, std_final = welford_finalize_gpu(count, mean, M2)
    return float(mean_final), float(std_final)


def threshold_subsample_gpu(data, layout, labels=None, threshold=0.5, random_state=42):
    """
    GPU-accelerated Bernoulli subsampling of data.

    Parameters
    ----------
    data : array-like
        High-dimensional data
    layout : array-like
        Low-dimensional embedding
    labels : array-like, optional
        Data labels
    threshold : float
        Probability of keeping each sample
    random_state : int
        Random seed

    Returns
    -------
    tuple : Subsampled arrays
    """
    if HAS_CUPY:
        cp.random.seed(random_state)

        # Convert to CuPy if needed
        data_gpu = cp.asarray(data) if not isinstance(data, cp.ndarray) else data
        layout_gpu = cp.asarray(layout) if not isinstance(layout, cp.ndarray) else layout

        n_samples = data_gpu.shape[0]
        random_numbers = cp.random.uniform(0, 1, size=n_samples)
        selected_indices = random_numbers < threshold

        data_sub = data_gpu[selected_indices]
        layout_sub = layout_gpu[selected_indices]

        if labels is not None:
            labels_gpu = cp.asarray(labels) if not isinstance(labels, cp.ndarray) else labels
            labels_sub = labels_gpu[selected_indices]
            return data_sub, layout_sub, labels_sub

        return data_sub, layout_sub
    else:
        # CPU fallback
        np.random.seed(random_state)
        data_np = np.asarray(data)
        layout_np = np.asarray(layout)

        n_samples = data_np.shape[0]
        random_numbers = np.random.uniform(0, 1, size=n_samples)
        selected_indices = random_numbers < threshold

        data_sub = data_np[selected_indices]
        layout_sub = layout_np[selected_indices]

        if labels is not None:
            labels_np = np.asarray(labels)
            labels_sub = labels_np[selected_indices]
            return data_sub, layout_sub, labels_sub

        return data_sub, layout_sub


#
# kNN graph construction
#


def make_knn_graph_gpu(data, n_neighbors, batch_size=None):
    """
    GPU-accelerated kNN graph construction using cuML.

    Parameters
    ----------
    data : array-like
        Data points (n_samples, n_features)
    n_neighbors : int
        Number of nearest neighbors
    batch_size : int, optional
        Batch size for processing (currently unused with cuML)

    Returns
    -------
    tuple : (distances, indices) arrays of shape (n_samples, n_neighbors+1)
    """
    if not HAS_CUML:
        # Fallback to CPU
        return make_knn_graph_cpu(data, n_neighbors, batch_size)

    # Convert to CuPy if needed
    if HAS_CUPY:
        data_gpu = cp.asarray(data, dtype=cp.float32)
    else:
        data_gpu = np.asarray(data, dtype=np.float32)

    # Use cuML NearestNeighbors
    nn = cumlNearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
    nn.fit(data_gpu)

    # Query for neighbors (including self)
    distances, indices = nn.kneighbors(data_gpu)

    # Convert to CuPy arrays if not already
    if HAS_CUPY:
        distances = cp.asarray(distances)
        indices = cp.asarray(indices)

    return distances, indices


def make_knn_graph_cpu(data, n_neighbors, batch_size=None):
    """
    CPU fallback for kNN graph construction.

    Parameters
    ----------
    data : array-like
        Data points
    n_neighbors : int
        Number of nearest neighbors
    batch_size : int, optional
        Unused for CPU version

    Returns
    -------
    tuple : (distances, indices) arrays
    """
    data_np = np.asarray(data, dtype=np.float32)

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='euclidean')
    nn.fit(data_np)

    distances, indices = nn.kneighbors(data_np)

    return distances, indices


#
# Distortion metrics (stress)
#


def compute_stress(data, layout, n_neighbors, eps=1e-6, use_gpu=True):
    """
    Compute normalized stress (distortion) of an embedding.

    This metric measures how well distances are preserved between the
    high-dimensional data and low-dimensional layout.

    Parameters
    ----------
    data : array-like
        High-dimensional data (n_samples, n_features)
    layout : array-like
        Low-dimensional embedding (n_samples, n_components)
    n_neighbors : int
        Number of nearest neighbors to consider
    eps : float
        Small constant to prevent division by zero
    use_gpu : bool
        Whether to use GPU acceleration

    Returns
    -------
    float : Normalized stress value
    """
    if use_gpu and HAS_CUML and HAS_CUPY:
        # GPU version
        data_gpu = cp.asarray(data, dtype=cp.float32)
        layout_gpu = cp.asarray(layout, dtype=cp.float32)

        # Compute kNN graph in high-dimensional space
        distances, indices = make_knn_graph_gpu(data_gpu, n_neighbors)

        # Distances are already L2 (Euclidean) from cuML
        distances = distances[:, 1:]  # Remove self
        indices = indices[:, 1:]

        # Compute distances in low-dimensional space
        n_samples = layout_gpu.shape[0]
        distances_emb = cp.zeros_like(distances)

        for i in range(n_samples):
            neighbor_coords = layout_gpu[indices[i]]
            point_coords = layout_gpu[i:i+1]
            distances_emb[i] = cp.linalg.norm(neighbor_coords - point_coords, axis=1)

        # Compute normalized stress
        ratios = cp.abs(distances / cp.maximum(distances_emb, eps) - 1.0)
        stress_mean = float(cp.mean(ratios))
        stress_std = float(cp.std(ratios))

        stress_normalized = 0.0 if stress_mean < eps else stress_std / stress_mean

    else:
        # CPU version
        data_np = np.asarray(data, dtype=np.float32)
        layout_np = np.asarray(layout, dtype=np.float32)

        distances, indices = make_knn_graph_cpu(data_np, n_neighbors)

        distances = distances[:, 1:]
        indices = indices[:, 1:]

        # Compute distances in embedding
        n_samples = layout_np.shape[0]
        distances_emb = np.zeros_like(distances)

        for i in range(n_samples):
            neighbor_coords = layout_np[indices[i]]
            point_coords = layout_np[i:i+1]
            distances_emb[i] = np.linalg.norm(neighbor_coords - point_coords, axis=1)

        ratios = np.abs(distances / np.maximum(distances_emb, eps) - 1.0)
        stress_mean = float(np.mean(ratios))
        stress_std = float(np.std(ratios))

        stress_normalized = 0.0 if stress_mean < eps else stress_std / stress_mean

    return stress_normalized


def compute_neighbor_score(data, layout, n_neighbors, use_gpu=True):
    """
    Compute neighborhood preservation score.

    Measures how well k-nearest neighbor relationships are preserved
    from high-dimensional to low-dimensional space.

    Parameters
    ----------
    data : array-like
        High-dimensional data
    layout : array-like
        Low-dimensional embedding
    n_neighbors : int
        Number of neighbors to consider
    use_gpu : bool
        Whether to use GPU acceleration

    Returns
    -------
    list : [mean_score, std_score]
    """
    if use_gpu and HAS_CUML and HAS_CUPY:
        data_gpu = cp.asarray(data, dtype=cp.float32)
        layout_gpu = cp.asarray(layout, dtype=cp.float32)

        _, indices_data = make_knn_graph_gpu(data_gpu, n_neighbors)
        _, indices_layout = make_knn_graph_gpu(layout_gpu, n_neighbors)

        indices_data = indices_data[:, 1:]  # Remove self
        indices_layout = indices_layout[:, 1:]

        # Sort indices
        indices_data = cp.sort(indices_data, axis=1)
        indices_layout = cp.sort(indices_layout, axis=1)

        # Compute preservation scores
        preservation_scores = cp.mean(indices_data == indices_layout, axis=1)

        neighbor_mean = float(cp.mean(preservation_scores))
        neighbor_std = float(cp.std(preservation_scores))

    else:
        data_np = np.asarray(data, dtype=np.float32)
        layout_np = np.asarray(layout, dtype=np.float32)

        _, indices_data = make_knn_graph_cpu(data_np, n_neighbors)
        _, indices_layout = make_knn_graph_cpu(layout_np, n_neighbors)

        indices_data = indices_data[:, 1:]
        indices_layout = indices_layout[:, 1:]

        indices_data = np.sort(indices_data, axis=1)
        indices_layout = np.sort(indices_layout, axis=1)

        preservation_scores = np.mean(indices_data == indices_layout, axis=1)

        neighbor_mean = float(np.mean(preservation_scores))
        neighbor_std = float(np.std(preservation_scores))

    return [neighbor_mean, neighbor_std]


def compute_local_metrics(data, layout, n_neighbors, memory_efficient=None, use_gpu=True):
    """
    Compute local quality metrics (stress and neighborhood preservation).

    Parameters
    ----------
    data : array-like
        High-dimensional data
    layout : array-like
        Low-dimensional embedding
    n_neighbors : int
        Number of neighbors for kNN graph
    memory_efficient : bool, optional
        Whether to subsample for large datasets
    use_gpu : bool
        Whether to use GPU acceleration

    Returns
    -------
    dict : Dictionary containing 'stress' and 'neighbor' metrics
    """
    if memory_efficient is None:
        memory_efficient = len(data) > 32768

    if memory_efficient and len(data) > 131072:
        sample_size = 32768
        indices = np.random.choice(len(data), sample_size, replace=False)
        data_sample = data[indices]
        layout_sample = layout[indices]

        metrics = {
            'stress': compute_stress(data_sample, layout_sample, n_neighbors, use_gpu=use_gpu),
            'neighbor': compute_neighbor_score(data_sample, layout_sample, n_neighbors, use_gpu=use_gpu),
            'note': f"Metrics computed on {sample_size} randomly sampled points"
        }
    else:
        metrics = {
            'stress': compute_stress(data, layout, n_neighbors, use_gpu=use_gpu),
            'neighbor': compute_neighbor_score(data, layout, n_neighbors, use_gpu=use_gpu)
        }

    return metrics


#
# Context preservation metrics (SVM, kNN classification)
#


def compute_svm_accuracy(X, y, test_size=0.3, reg_param=1.0, max_iter=1000, random_state=42, use_gpu=True):
    """
    Compute SVM classification accuracy.

    Parameters
    ----------
    X : array-like
        Features
    y : array-like
        Labels
    test_size : float
        Test set proportion
    reg_param : float
        Regularization parameter
    max_iter : int
        Maximum iterations
    random_state : int
        Random seed
    use_gpu : bool
        Whether to use cuML GPU acceleration

    Returns
    -------
    float : Classification accuracy
    """
    if use_gpu and HAS_CUML:
        if HAS_CUPY:
            X_gpu = cp.asarray(X, dtype=cp.float32)
            y_gpu = cp.asarray(y)
        else:
            X_gpu = np.asarray(X, dtype=np.float32)
            y_gpu = np.asarray(y)

        # Train/test split
        X_train, X_test, y_train, y_test = cuml_train_test_split(
            X_gpu, y_gpu, test_size=test_size, random_state=random_state
        )

        # Standardize
        scaler = cumlStandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train SVM
        model = cumlSVC(C=reg_param, max_iter=max_iter, kernel='linear')
        model.fit(X_train, y_train)

        # Predict and compute accuracy
        y_pred = model.predict(X_test)
        if HAS_CUPY:
            accuracy = float(cp.mean(y_pred == y_test))
        else:
            accuracy = float(np.mean(y_pred == y_test))

    else:
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = LinearSVC(C=reg_param, max_iter=max_iter)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def compute_knn_accuracy(X, y, n_neighbors=16, test_size=0.3, random_state=42, use_gpu=True):
    """
    Compute kNN classification accuracy.

    Parameters
    ----------
    X : array-like
        Features
    y : array-like
        Labels
    n_neighbors : int
        Number of neighbors
    test_size : float
        Test set proportion
    random_state : int
        Random seed
    use_gpu : bool
        Whether to use cuML GPU acceleration

    Returns
    -------
    float : Classification accuracy
    """
    if use_gpu and HAS_CUML:
        if HAS_CUPY:
            X_gpu = cp.asarray(X, dtype=cp.float32)
            y_gpu = cp.asarray(y)
        else:
            X_gpu = np.asarray(X, dtype=np.float32)
            y_gpu = np.asarray(y)

        X_train, X_test, y_train, y_test = cuml_train_test_split(
            X_gpu, y_gpu, test_size=test_size, random_state=random_state
        )

        scaler = cumlStandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = cumlKNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        if HAS_CUPY:
            accuracy = float(cp.mean(y_pred == y_test))
        else:
            accuracy = float(np.mean(y_pred == y_test))

    else:
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def compute_svm_score(data, layout, labels, subsample_threshold=0.5, random_state=42, use_gpu=True, **kwargs):
    """
    Compute SVM context preservation score.

    Compares SVM classification accuracy on high-dimensional data
    vs low-dimensional embedding.

    Parameters
    ----------
    data : array-like
        High-dimensional data
    layout : array-like
        Low-dimensional embedding
    labels : array-like
        Class labels
    subsample_threshold : float
        Subsampling probability
    random_state : int
        Random seed
    use_gpu : bool
        Whether to use GPU acceleration
    **kwargs : dict
        Additional parameters for SVM

    Returns
    -------
    ndarray : [acc_hd, acc_ld, log_ratio]
    """
    # Subsample
    X_hd, X_ld, y = threshold_subsample_gpu(
        data, layout, labels, threshold=subsample_threshold, random_state=random_state
    )

    # Convert back to NumPy for sklearn/cuml
    if HAS_CUPY and isinstance(X_hd, cp.ndarray):
        X_hd = cp.asnumpy(X_hd)
        X_ld = cp.asnumpy(X_ld)
        y = cp.asnumpy(y)

    test_size = kwargs.pop('test_size', 0.3)
    reg_param = kwargs.pop('reg_param', 1.0)
    max_iter = kwargs.pop('max_iter', 1000)

    svm_acc_hd = compute_svm_accuracy(
        X_hd, y, test_size=test_size, reg_param=reg_param,
        max_iter=max_iter, random_state=random_state, use_gpu=use_gpu
    )

    svm_acc_ld = compute_svm_accuracy(
        X_ld, y, test_size=test_size, reg_param=reg_param,
        max_iter=max_iter, random_state=random_state, use_gpu=use_gpu
    )

    svm_score = np.log(np.minimum(svm_acc_hd / svm_acc_ld, svm_acc_ld / svm_acc_hd))

    return np.array([svm_acc_hd, svm_acc_ld, svm_score], dtype=np.float32)


def compute_knn_score(data, layout, labels, n_neighbors=16, subsample_threshold=0.5,
                      random_state=42, use_gpu=True, **kwargs):
    """
    Compute kNN context preservation score.

    Compares kNN classification accuracy on high-dimensional data
    vs low-dimensional embedding.

    Parameters
    ----------
    data : array-like
        High-dimensional data
    layout : array-like
        Low-dimensional embedding
    labels : array-like
        Class labels
    n_neighbors : int
        Number of neighbors for kNN
    subsample_threshold : float
        Subsampling probability
    random_state : int
        Random seed
    use_gpu : bool
        Whether to use GPU acceleration
    **kwargs : dict
        Additional parameters

    Returns
    -------
    ndarray : [acc_hd, acc_ld, log_ratio]
    """
    # Subsample
    X_hd, X_ld, y = threshold_subsample_gpu(
        data, layout, labels, threshold=subsample_threshold, random_state=random_state
    )

    # Convert back to NumPy for sklearn/cuml
    if HAS_CUPY and isinstance(X_hd, cp.ndarray):
        X_hd = cp.asnumpy(X_hd)
        X_ld = cp.asnumpy(X_ld)
        y = cp.asnumpy(y)

    test_size = kwargs.pop('test_size', 0.3)

    knn_acc_hd = compute_knn_accuracy(
        X_hd, y, n_neighbors=n_neighbors, test_size=test_size,
        random_state=random_state, use_gpu=use_gpu
    )

    knn_acc_ld = compute_knn_accuracy(
        X_ld, y, n_neighbors=n_neighbors, test_size=test_size,
        random_state=random_state, use_gpu=use_gpu
    )

    knn_score = np.log(knn_acc_ld / knn_acc_hd)

    return np.array([knn_acc_hd, knn_acc_ld, knn_score], dtype=np.float32)


def compute_context_measures(data, layout, labels, subsample_threshold=0.5, n_neighbors=16,
                             random_state=42, use_gpu=True, **kwargs):
    """
    Compute context preservation measures (SVM and kNN).

    Parameters
    ----------
    data : array-like
        High-dimensional data
    layout : array-like
        Low-dimensional embedding
    labels : array-like
        Class labels
    subsample_threshold : float
        Subsampling probability
    n_neighbors : int
        Number of neighbors for kNN
    random_state : int
        Random seed
    use_gpu : bool
        Whether to use GPU acceleration
    **kwargs : dict
        Additional parameters

    Returns
    -------
    dict : Dictionary with 'svm' and 'knn' scores
    """
    measures = {
        'svm': compute_svm_score(
            data, layout, labels, subsample_threshold, random_state, use_gpu, **kwargs
        ),
        'knn': compute_knn_score(
            data, layout, labels, n_neighbors, subsample_threshold, random_state, use_gpu, **kwargs
        )
    }

    return measures


#
# Persistence homology and Betti curves
#


def compute_persistence_diagrams(data, layout, max_dim=1, subsample_threshold=0.5,
                                random_state=42, backend=None, n_threads=-1, collapse_edges=True):
    """
    Compute persistence diagrams for data and layout.

    Parameters
    ----------
    data : array-like
        High-dimensional data
    layout : array-like
        Low-dimensional embedding
    max_dim : int
        Maximum homology dimension
    subsample_threshold : float
        Subsampling probability
    random_state : int
        Random seed
    backend : str, optional
        Persistence backend: 'giotto-ph', 'ripser++', 'ripser', or None for auto
    n_threads : int
        Number of threads for giotto-ph (-1 for all cores)
    collapse_edges : bool
        Whether to use edge collapse optimization (giotto-ph only)

    Returns
    -------
    dict : {'data': diagrams_hd, 'layout': diagrams_ld, 'backend': backend_used}
    """
    # Select backend
    if backend is None:
        backend = get_persistence_backend()
    else:
        # Validate backend
        available = get_available_persistence_backends()
        if backend not in available or not available[backend]:
            raise ValueError(f"Backend '{backend}' not available. Use one of {list(available.keys())}")

    # Subsample
    data_sub, layout_sub = threshold_subsample_gpu(
        data, layout, threshold=subsample_threshold, random_state=random_state
    )

    # Convert to NumPy for persistence computation
    if HAS_CUPY and isinstance(data_sub, cp.ndarray):
        data_np = cp.asnumpy(data_sub)
        layout_np = cp.asnumpy(layout_sub)
    else:
        data_np = np.asarray(data_sub, dtype=np.float32)
        layout_np = np.asarray(layout_sub, dtype=np.float32)

    # Compute persistence diagrams based on backend
    if backend == 'giotto-ph':
        # Use giotto-ph (fastest CPU option)
        result_hd = ripser_parallel(
            data_np,
            maxdim=max_dim,
            collapse_edges=collapse_edges,
            n_threads=n_threads,
            return_generators=False
        )
        result_ld = ripser_parallel(
            layout_np,
            maxdim=max_dim,
            collapse_edges=collapse_edges,
            n_threads=n_threads,
            return_generators=False
        )
        diags_hd = result_hd['dgms']
        diags_ld = result_ld['dgms']

    elif backend == 'ripser++':
        # Use GPU-accelerated ripser++
        diags_hd = rpp.run(data_np, maxdim=max_dim)['dgms']
        diags_ld = rpp.run(layout_np, maxdim=max_dim)['dgms']

    elif backend == 'ripser':
        # Use standard CPU ripser
        diags_hd = ripser(data_np, maxdim=max_dim)['dgms']
        diags_ld = ripser(layout_np, maxdim=max_dim)['dgms']

    else:
        raise ValueError(f"Unknown backend: {backend}")

    return {'data': diags_hd, 'layout': diags_ld, 'backend': backend}


def betti_curve(diagram, n_steps=100):
    """
    Compute Betti curve from a persistence diagram.

    A Betti curve shows the number of topological features that persist
    at different filtration values.

    Parameters
    ----------
    diagram : array-like
        Persistence diagram as list of (birth, death) tuples
    n_steps : int
        Number of points in the curve

    Returns
    -------
    tuple : (filtration_values, betti_numbers)
    """
    if len(diagram) == 0:
        return np.linspace(0, 1, n_steps), np.zeros(n_steps)

    # Filter out infinite death times
    finite_diagram = [x for x in diagram if x[1] != np.inf]

    if len(finite_diagram) == 0:
        return np.linspace(0, 1, n_steps), np.zeros(n_steps)

    max_dist = np.max([x[1] for x in finite_diagram])
    axis_x = np.linspace(0, max_dist, n_steps)
    axis_y = np.zeros(n_steps)

    for i, x in enumerate(axis_x):
        for b, d in diagram:
            if b < x < d:
                axis_y[i] += 1

    return axis_x, axis_y


def compute_dtw(axis_x_hd, axis_y_hd, axis_x_ld, axis_y_ld, norm_factor=1.0):
    """
    Compute Dynamic Time Warping distance between Betti curves.

    Parameters
    ----------
    axis_x_hd, axis_y_hd : array-like
        High-dimensional Betti curve
    axis_x_ld, axis_y_ld : array-like
        Low-dimensional Betti curve
    norm_factor : float
        Normalization factor

    Returns
    -------
    float : DTW distance
    """
    if not HAS_DTW:
        warnings.warn("fastdtw not available, returning NaN")
        return np.nan

    seq0 = np.array(list(zip(axis_x_hd, axis_y_hd)))
    seq1 = np.array(list(zip(axis_x_ld, axis_y_ld)))
    dist_dtw, _ = fastdtw(seq0, seq1, dist=2)

    return dist_dtw * norm_factor


def compute_twed(axis_x_hd, axis_y_hd, axis_x_ld, axis_y_ld, norm_factor=1.0):
    """
    Compute Time Warp Edit Distance between Betti curves.

    Parameters
    ----------
    axis_x_hd, axis_y_hd : array-like
        High-dimensional Betti curve
    axis_x_ld, axis_y_ld : array-like
        Low-dimensional Betti curve
    norm_factor : float
        Normalization factor

    Returns
    -------
    float : TWED distance
    """
    if not HAS_TWED:
        warnings.warn("twed not available, returning NaN")
        return np.nan

    dist_twed = twed(
        axis_y_hd.reshape(-1, 1),
        axis_y_ld.reshape(-1, 1),
        axis_x_hd,
        axis_x_ld,
        p=2
    )

    return dist_twed * norm_factor


def compute_emd(axis_x_hd, axis_y_hd, axis_x_ld, axis_y_ld, adjust_mass=False, norm_factor=1.0):
    """
    Compute Earth Mover's Distance between Betti curves.

    Parameters
    ----------
    axis_x_hd, axis_y_hd : array-like
        High-dimensional Betti curve
    axis_x_ld, axis_y_ld : array-like
        Low-dimensional Betti curve
    adjust_mass : bool
        Whether to adjust for different total masses
    norm_factor : float
        Normalization factor

    Returns
    -------
    float : EMD distance
    """
    if not HAS_OT:
        warnings.warn("POT not available, returning NaN")
        return np.nan

    sum_hd = np.sum(axis_y_hd)
    sum_ld = np.sum(axis_y_ld)

    if sum_hd == 0 or sum_ld == 0:
        return 0.0

    axis_y_hd_norm = axis_y_hd / sum_hd
    axis_y_ld_norm = axis_y_ld / sum_ld

    dist_emd = ot.emd2_1d(axis_x_hd, axis_x_ld, axis_y_hd_norm, axis_y_ld_norm, metric='euclidean')

    if adjust_mass:
        dist_emd *= np.max([sum_hd / sum_ld, sum_ld / sum_hd])

    return dist_emd * norm_factor


def compute_wasserstein(diag_hd, diag_ld, norm_factor=1.0):
    """
    Compute Wasserstein distance between persistence diagrams.

    Handles infinite death times by:
    1. Computing Wasserstein on finite features
    2. Adding birth time difference for infinite features

    Parameters
    ----------
    diag_hd, diag_ld : array-like
        Persistence diagrams (birth, death) pairs
    norm_factor : float
        Normalization factor

    Returns
    -------
    float : Wasserstein distance
    """
    if not HAS_PERSIM:
        warnings.warn("persim not available, returning NaN")
        return np.nan

    diag_hd = np.asarray(diag_hd)
    diag_ld = np.asarray(diag_ld)

    # Separate finite and infinite features
    finite_mask_hd = np.isfinite(diag_hd[:, 1]) if diag_hd.size > 0 else np.array([], dtype=bool)
    finite_mask_ld = np.isfinite(diag_ld[:, 1]) if diag_ld.size > 0 else np.array([], dtype=bool)

    diag_hd_finite = diag_hd[finite_mask_hd] if diag_hd.size > 0 else np.array([[0, 0]])
    diag_ld_finite = diag_ld[finite_mask_ld] if diag_ld.size > 0 else np.array([[0, 0]])

    # Compute Wasserstein on finite features
    dist_wass_finite = wasserstein(diag_hd_finite, diag_ld_finite)

    # Handle infinite features by comparing birth times
    diag_hd_inf = diag_hd[~finite_mask_hd] if diag_hd.size > 0 and np.any(~finite_mask_hd) else np.array([])
    diag_ld_inf = diag_ld[~finite_mask_ld] if diag_ld.size > 0 and np.any(~finite_mask_ld) else np.array([])

    dist_wass_inf = 0.0
    if diag_hd_inf.size > 0 or diag_ld_inf.size > 0:
        # Extract birth times of infinite features
        births_hd = diag_hd_inf[:, 0] if diag_hd_inf.size > 0 else np.array([])
        births_ld = diag_ld_inf[:, 0] if diag_ld_inf.size > 0 else np.array([])

        # Compute Wasserstein distance on birth times (treated as 1D distributions)
        # We create (birth, birth) pairs to use the same wasserstein function
        if births_hd.size > 0:
            births_hd_pairs = np.column_stack([births_hd, births_hd])
        else:
            births_hd_pairs = np.array([[0, 0]])

        if births_ld.size > 0:
            births_ld_pairs = np.column_stack([births_ld, births_ld])
        else:
            births_ld_pairs = np.array([[0, 0]])

        dist_wass_inf = wasserstein(births_hd_pairs, births_ld_pairs)

    dist_wass = dist_wass_finite + dist_wass_inf
    return dist_wass * norm_factor


def compute_bottleneck(diag_hd, diag_ld, norm_factor=1.0):
    """
    Compute bottleneck distance between persistence diagrams.

    Handles infinite death times by:
    1. Computing bottleneck on finite features
    2. Taking max with birth time difference for infinite features

    Parameters
    ----------
    diag_hd, diag_ld : array-like
        Persistence diagrams (birth, death) pairs
    norm_factor : float
        Normalization factor

    Returns
    -------
    float : Bottleneck distance
    """
    if not HAS_PERSIM:
        warnings.warn("persim not available, returning NaN")
        return np.nan

    diag_hd = np.asarray(diag_hd)
    diag_ld = np.asarray(diag_ld)

    # Separate finite and infinite features
    finite_mask_hd = np.isfinite(diag_hd[:, 1]) if diag_hd.size > 0 else np.array([], dtype=bool)
    finite_mask_ld = np.isfinite(diag_ld[:, 1]) if diag_ld.size > 0 else np.array([], dtype=bool)

    diag_hd_finite = diag_hd[finite_mask_hd] if diag_hd.size > 0 else np.array([[0, 0]])
    diag_ld_finite = diag_ld[finite_mask_ld] if diag_ld.size > 0 else np.array([[0, 0]])

    # Compute bottleneck on finite features
    dist_bott_finite = bottleneck(diag_hd_finite, diag_ld_finite)

    # Handle infinite features by comparing birth times
    diag_hd_inf = diag_hd[~finite_mask_hd] if diag_hd.size > 0 and np.any(~finite_mask_hd) else np.array([])
    diag_ld_inf = diag_ld[~finite_mask_ld] if diag_ld.size > 0 and np.any(~finite_mask_ld) else np.array([])

    dist_bott_inf = 0.0
    if diag_hd_inf.size > 0 or diag_ld_inf.size > 0:
        # Extract birth times of infinite features
        births_hd = diag_hd_inf[:, 0] if diag_hd_inf.size > 0 else np.array([])
        births_ld = diag_ld_inf[:, 0] if diag_ld_inf.size > 0 else np.array([])

        # Compute bottleneck distance on birth times
        if births_hd.size > 0:
            births_hd_pairs = np.column_stack([births_hd, births_hd])
        else:
            births_hd_pairs = np.array([[0, 0]])

        if births_ld.size > 0:
            births_ld_pairs = np.column_stack([births_ld, births_ld])
        else:
            births_ld_pairs = np.array([[0, 0]])

        dist_bott_inf = bottleneck(births_hd_pairs, births_ld_pairs)

    # Bottleneck distance is the maximum
    dist_bott = max(dist_bott_finite, dist_bott_inf)
    return dist_bott * norm_factor


def compute_global_metrics(data, layout, dimension=1, subsample_threshold=0.5, random_state=42,
                          n_steps=100, metrics_only=True, backend=None, n_threads=-1, collapse_edges=True):
    """
    Compute global topological metrics based on persistence homology.

    Computes distances between persistence diagrams and Betti curves:
    - DTW, TWED, EMD for Betti curves
    - Wasserstein, Bottleneck for persistence diagrams

    Parameters
    ----------
    data : array-like
        High-dimensional data
    layout : array-like
        Low-dimensional embedding
    dimension : int
        Maximum homology dimension
    subsample_threshold : float
        Subsampling probability
    random_state : int
        Random seed
    n_steps : int
        Number of points for Betti curves
    metrics_only : bool
        If True, return only metrics; otherwise include diagrams and curves
    backend : str, optional
        Persistence backend: 'giotto-ph', 'ripser++', 'ripser', or None for auto
    n_threads : int
        Number of threads for giotto-ph (-1 for all cores)
    collapse_edges : bool
        Whether to use edge collapse optimization (giotto-ph only)

    Returns
    -------
    dict : Dictionary containing metrics (and optionally diagrams and betti curves)
    """
    metrics = {
        'dtw': [],
        'twed': [],
        'emd': [],
        'wass': [],
        'bott': []
    }

    betti_curves = {
        'data': [],
        'layout': []
    }

    # Compute persistence diagrams
    result = compute_persistence_diagrams(
        data, layout, max_dim=dimension, subsample_threshold=subsample_threshold,
        random_state=random_state, backend=backend, n_threads=n_threads,
        collapse_edges=collapse_edges
    )

    diags = result
    backend_used = result.get('backend', 'unknown')

    # Subsample to get point count for normalization
    data_sub, layout_sub = threshold_subsample_gpu(
        data, layout, threshold=subsample_threshold, random_state=random_state
    )
    n_points = len(data_sub)

    # Compute metrics for each dimension
    for diag_hd, diag_ld in zip(diags['data'], diags['layout']):
        # Compute Betti curves
        axis_x_hd, axis_y_hd = betti_curve(diag_hd, n_steps=n_steps)
        axis_x_ld, axis_y_ld = betti_curve(diag_ld, n_steps=n_steps)

        betti_curves['data'].append((axis_x_hd, axis_y_hd))
        betti_curves['layout'].append((axis_x_ld, axis_y_ld))

        # Compute distances on Betti curves
        norm_factor = 1.0 / n_points
        dist_dtw = compute_dtw(axis_x_hd, axis_y_hd, axis_x_ld, axis_y_ld, norm_factor)
        dist_twed = compute_twed(axis_x_hd, axis_y_hd, axis_x_ld, axis_y_ld, norm_factor)
        dist_emd = compute_emd(axis_x_hd, axis_y_hd, axis_x_ld, axis_y_ld, adjust_mass=True, norm_factor=norm_factor)

        # Compute distances on persistence diagrams
        dist_wass = compute_wasserstein(diag_hd, diag_ld, norm_factor)
        dist_bott = compute_bottleneck(diag_hd, diag_ld, 1.0)  # No normalization for bottleneck

        metrics['dtw'].append(dist_dtw)
        metrics['twed'].append(dist_twed)
        metrics['emd'].append(dist_emd)
        metrics['wass'].append(dist_wass)
        metrics['bott'].append(dist_bott)

    if metrics_only:
        return {'metrics': metrics, 'backend': backend_used}

    return {'metrics': metrics, 'diags': diags, 'bettis': betti_curves, 'backend': backend_used}


#
# Convenience function for all metrics
#


def evaluate_embedding(data, layout, labels=None, n_neighbors=16, subsample_threshold=0.5,
                      max_homology_dim=1, random_state=42, use_gpu=True,
                      persistence_backend=None, n_threads=-1, **kwargs):
    """
    Comprehensive evaluation of a dimensionality reduction embedding.

    Computes distortion, context preservation, and topological metrics.

    Parameters
    ----------
    data : array-like
        High-dimensional data (n_samples, n_features)
    layout : array-like
        Low-dimensional embedding (n_samples, n_components)
    labels : array-like, optional
        Class labels for context metrics
    n_neighbors : int
        Number of neighbors for kNN metrics
    subsample_threshold : float
        Subsampling probability for expensive metrics
    max_homology_dim : int
        Maximum homology dimension for persistence
    random_state : int
        Random seed
    use_gpu : bool
        Whether to use GPU acceleration
    persistence_backend : str, optional
        Persistence backend: 'giotto-ph', 'ripser++', 'ripser', or None for auto
    n_threads : int
        Number of threads for giotto-ph (-1 for all cores)
    **kwargs : dict
        Additional parameters for specific metrics

    Returns
    -------
    dict : Dictionary with all computed metrics
    """
    results = {}

    # Local metrics (distortion)
    print("Computing local metrics (stress, neighbor preservation)...")
    results['local'] = compute_local_metrics(
        data, layout, n_neighbors, use_gpu=use_gpu
    )

    # Context metrics (if labels provided)
    if labels is not None:
        print("Computing context preservation metrics (SVM, kNN)...")
        results['context'] = compute_context_measures(
            data, layout, labels, subsample_threshold, n_neighbors,
            random_state, use_gpu, **kwargs
        )

    # Global topological metrics
    if HAS_GIOTTO_PH or HAS_RIPSER_PP or HAS_RIPSER:
        backend = persistence_backend or get_persistence_backend()
        print(f"Computing topological metrics using backend: {backend}...")
        results['topology'] = compute_global_metrics(
            data, layout, max_homology_dim, subsample_threshold,
            random_state, backend=backend, n_threads=n_threads
        )
    else:
        warnings.warn("Skipping topological metrics (no persistence backend available)")

    return results
