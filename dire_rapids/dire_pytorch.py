# dire_pytorch.py

"""
PyTorch/PyKeOps backend for DiRe dimensionality reduction.

This implementation features:
- Memory-efficient chunked k-NN computation for large datasets (>100K points)
- Attraction forces applied only between k-NN neighbors  
- Repulsion forces computed from random samples
- Automatic GPU memory management with adaptive chunk sizing
- Designed for high-performance processing on CUDA GPUs

Performance characteristics:
- Best for datasets >50K points on CUDA GPUs
- Memory-aware processing up to millions of points
- Chunked computation prevents GPU out-of-memory errors
"""

import gc
import importlib.util
from time import perf_counter

import numpy as np
import torch
from loguru import logger
from scipy.optimize import curve_fit
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA

from ._compat import import_pykeops_lazy_tensor, torch_tensor_to_numpy

# PyKeOps is imported lazily to avoid import-time CUDA probing warnings.
PYKEOPS_AVAILABLE = importlib.util.find_spec("pykeops") is not None
if not PYKEOPS_AVAILABLE:
    logger.trace("PyKeOps not available. Install with: pip install pykeops")

# cuVS for fast approximate k-NN at scale (optional RAPIDS dependency)
try:
    import cupy as cp
    from cuvs.neighbors import ivf_flat as cuvs_ivf_flat
    CUVS_AVAILABLE = True
except ImportError:
    CUVS_AVAILABLE = False


_KNN_BACKEND_ALIASES = {
    None: "auto",
    "auto": "auto",
    "torch": "pytorch",
    "pytorch": "pytorch",
    "keops": "pykeops",
    "pykeops": "pykeops",
    "rapids": "cuvs",
    "cuvs": "cuvs",
}

_MATRIX_MULTIPLY_METRICS = {"euclidean", "sqeuclidean", "inner_product", "cosine"}
_CUVS_NATIVE_METRICS = {"euclidean", "l2", "sqeuclidean", "inner_product", "cosine"}
_AUTO_MEMORY_EFFICIENT_MIN_SAMPLES = 20_000
_AUTO_MEMORY_EFFICIENT_MIN_DIMS = 2_048


def _normalize_knn_backend(knn_backend):
    """Normalize public k-NN backend names to internal identifiers."""
    if knn_backend is None:
        return "auto"
    if not isinstance(knn_backend, str):
        raise ValueError("knn_backend must be one of: 'auto', 'pytorch', 'pykeops', 'cuvs'")
    key = knn_backend.strip().lower()
    try:
        return _KNN_BACKEND_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown knn_backend: {knn_backend}. "
            "Choose from: 'auto', 'pytorch', 'pykeops', 'cuvs'"
        ) from exc


def _normalize_named_metric(metric_spec):
    """Return the normalized name for built-in metrics, or None for custom metrics."""
    if metric_spec is None:
        return "euclidean"
    if not isinstance(metric_spec, str):
        return None

    metric_name = metric_spec.strip().lower()
    if metric_name == "l2":
        return "euclidean"
    if metric_name in _MATRIX_MULTIPLY_METRICS:
        return metric_name
    return None


def _is_cuvs_native_metric(metric_spec):
    """Return True when a metric spec can be represented by cuVS directly."""
    if metric_spec is None:
        return True
    if not isinstance(metric_spec, str):
        return False
    return metric_spec.strip().lower() in _CUVS_NATIVE_METRICS


def _prefer_memory_efficient_for_shape(n_samples=None, n_dims=None, metric_spec=None):
    """Heuristic for auto fallback paths where standard PyTorch is too optimistic."""
    if n_samples is not None and n_samples >= _AUTO_MEMORY_EFFICIENT_MIN_SAMPLES:
        return True
    if n_dims is not None and n_dims > _AUTO_MEMORY_EFFICIENT_MIN_DIMS:
        return True
    return metric_spec is not None and not _is_cuvs_native_metric(metric_spec)


def _compute_forces_kernel(positions, knn_indices, neg_indices, a_val, b_val, cutoff):
    """Compute attraction + repulsion forces for all points in one shot.

    This is the hot loop kernel — torch.compile fuses it into a small number
    of CUDA kernels, eliminating the per-chunk launch overhead that dominates
    when the dataset fits comfortably in GPU memory (true for any modern GPU
    up to ~1M+ points).

    All arithmetic runs in bf16 for ~2x memory bandwidth savings on the
    gather/scatter-dominated workload.  The caller accumulates positions
    in fp32, so numerical drift stays bounded.

    Parameters
    ----------
    positions : (N, D)  — bf16 on CUDA, fp32 on CPU
    knn_indices : (N, k) long
    neg_indices : (N, n_neg) long
    a_val, b_val, cutoff : float

    Returns
    -------
    (N, D) forces in same dtype as positions
    """
    # Attraction: toward k-NN neighbors
    neighbor_pos = positions[knn_indices]                    # (N, k, D)
    diff = neighbor_pos - positions.unsqueeze(1)             # (N, k, D)
    dist_sq = (diff * diff).sum(dim=2, keepdim=True) + 1e-10  # (N, k, 1)
    inv_dist = torch.rsqrt(dist_sq)                          # 1/dist, no sqrt
    dist_sq_b = dist_sq ** b_val                              # dist^(2b)
    att_coeff = dist_sq_b / (dist_sq_b + a_val)
    forces = (att_coeff * diff * inv_dist).sum(dim=1)         # (N, D)

    # Repulsion: against random negative samples
    neg_pos = positions[neg_indices]                          # (N, n_neg, D)
    diff_n = neg_pos - positions.unsqueeze(1)                 # (N, n_neg, D)
    dist_sq_n = (diff_n * diff_n).sum(dim=2, keepdim=True) + 1e-10
    inv_dist_n = torch.rsqrt(dist_sq_n)
    dist_n = dist_sq_n * inv_dist_n                           # sqrt via sq * rsqrt
    dist_sq_b_n = dist_sq_n ** b_val
    rep_coeff = -1.0 / (1.0 + a_val * (dist_sq_b_n))
    rep_coeff = rep_coeff * torch.exp(-dist_n / cutoff)
    forces = forces + (rep_coeff * diff_n * inv_dist_n).sum(dim=1)

    return torch.clamp(forces, -cutoff, cutoff)


# torch.compile fuses the above into efficient CUDA kernels, eliminating
# intermediate tensor allocations.  We lazily compile on first CUDA call
# and fall back to eager mode if compilation fails.
_forces_compiled_cuda = None
_torch_compile_failed = False


def _compute_forces_compiled(positions, knn_indices, neg_indices, a_val, b_val, cutoff):
    """Dispatch to compiled or eager kernel."""
    global _forces_compiled_cuda, _torch_compile_failed
    if positions.is_cuda and not _torch_compile_failed:
        if _forces_compiled_cuda is None:
            try:
                _forces_compiled_cuda = torch.compile(
                    _compute_forces_kernel, mode="reduce-overhead"
                )
            except Exception:
                _torch_compile_failed = True
                return _compute_forces_kernel(positions, knn_indices, neg_indices, a_val, b_val, cutoff)
        try:
            return _forces_compiled_cuda(positions, knn_indices, neg_indices, a_val, b_val, cutoff)
        except Exception:
            _torch_compile_failed = True
    return _compute_forces_kernel(positions, knn_indices, neg_indices, a_val, b_val, cutoff)


def _attraction_forces_kernel(positions, knn_indices, a_val, b_val):
    """Compute attraction forces only (for memory-efficient backend).

    Parameters
    ----------
    positions : (N, D)
    knn_indices : (N, k) long
    a_val, b_val : float

    Returns
    -------
    (N, D) attraction forces
    """
    neighbor_pos = positions[knn_indices]                    # (N, k, D)
    diff = neighbor_pos - positions.unsqueeze(1)             # (N, k, D)
    dist_sq = (diff * diff).sum(dim=2, keepdim=True) + 1e-10  # (N, k, 1)
    inv_dist = torch.rsqrt(dist_sq)
    dist_sq_b = dist_sq ** b_val
    att_coeff = dist_sq_b / (dist_sq_b + a_val)
    return (att_coeff * diff * inv_dist).sum(dim=1)           # (N, D)


_attraction_compiled_cuda = None
_attraction_compile_failed = False


def _attraction_forces_compiled(positions, knn_indices, a_val, b_val):
    """Dispatch to compiled or eager attraction kernel."""
    global _attraction_compiled_cuda, _attraction_compile_failed
    if positions.is_cuda and not _attraction_compile_failed:
        if _attraction_compiled_cuda is None:
            try:
                _attraction_compiled_cuda = torch.compile(
                    _attraction_forces_kernel, mode="reduce-overhead"
                )
            except Exception:
                _attraction_compile_failed = True
                return _attraction_forces_kernel(positions, knn_indices, a_val, b_val)
        try:
            return _attraction_compiled_cuda(positions, knn_indices, a_val, b_val)
        except Exception:
            _attraction_compile_failed = True
    return _attraction_forces_kernel(positions, knn_indices, a_val, b_val)


def _compile_metric(spec):
    """
    Turn a metric spec into a callable metric(x, y) that returns a distance-like
    matrix with broadcasting:
      - Torch path:  x: (A, 1, D), y: (1, B, D)  -> (A, B) torch.Tensor
      - KeOps path:  x: LazyTensor(A,1,D), y: LazyTensor(1,B,D) -> LazyTensor(A,B)

    If spec is None or 'euclidean'/'l2', return None (fast-path Euclidean stays in backend).
    Other named metrics such as 'sqeuclidean', 'inner_product', and 'cosine'
    are compiled to explicit tensor callables for PyKeOps/custom-style paths;
    the standard PyTorch path handles them with matrix multiplication.
    If spec is str expression, it's eval'ed with {'x': x, 'y': y} and no builtins.
    If spec is callable, it's returned unchanged.

    Returns
    -------
    callable or None
        None for backend-native metrics, callable for custom metrics
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        expr = spec.strip().lower()
        # Euclidean/L2 can use the fastest backend-native path.
        metric_name = _normalize_named_metric(expr)
        if metric_name == "euclidean":
            return None
        if metric_name == "sqeuclidean":
            return lambda x, y: ((x - y) ** 2).sum(-1)
        if metric_name == "inner_product":
            return lambda x, y: -(x * y).sum(-1)
        if metric_name == "cosine":
            def _cosine_metric(x, y):
                numerator = (x * y).sum(-1)
                x_norm = (x ** 2).sum(-1).sqrt()
                y_norm = (y ** 2).sum(-1).sqrt()
                return 1.0 - numerator / (x_norm * y_norm + 1e-8)
            return _cosine_metric
        # Custom expression - compile to callable
        def _expr_metric(x, y, _expr=spec):
            # Use ONLY tensor methods like .sum(-1), .sqrt(), .abs(), etc.
            # Works for both torch.Tensor and KeOps LazyTensor.
            return eval(_expr, {"__builtins__": {}}, {"x": x, "y": y})  # pylint: disable=eval-used # Sandboxed eval for custom metrics
        return _expr_metric
    if callable(spec):
        return spec
    raise ValueError("metric must be None, a string expression, or a callable")


def _remove_self_from_knn(indices, distances, row_start, n_neighbors):
    """Remove each query row's own index from a top-k result, preserving ties."""
    out_indices = np.empty((indices.shape[0], n_neighbors), dtype=indices.dtype)
    out_distances = np.empty((distances.shape[0], n_neighbors), dtype=distances.dtype)

    for row in range(indices.shape[0]):
        self_idx = row_start + row
        keep = indices[row] != self_idx
        row_indices = indices[row][keep]
        row_distances = distances[row][keep]
        if row_indices.shape[0] < n_neighbors:
            raise RuntimeError(
                "k-NN result did not contain enough non-self neighbors; "
                "increase the internal candidate count."
            )
        out_indices[row] = row_indices[:n_neighbors]
        out_distances[row] = row_distances[:n_neighbors]

    return out_indices, out_distances


class _InstanceLogger:
    """Small adapter that makes per-instance verbosity deterministic."""

    def __init__(self, enabled, instance_id):
        self.enabled = enabled
        self._logger = logger.bind(dire_instance=instance_id)

    def debug(self, message, *args, **kwargs):
        if self.enabled:
            self._logger.debug(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        if self.enabled:
            self._logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        if self.enabled:
            self._logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        if self.enabled:
            self._logger.error(message, *args, **kwargs)


class DiRePyTorch(TransformerMixin):
    """
    Memory-efficient PyTorch/PyKeOps implementation of DiRe dimensionality reduction.
    
    This class provides a high-performance implementation of the DiRe algorithm using PyTorch
    as the computational backend. It features adaptive memory management for large datasets
    and automatic GPU optimization.
    
    Features
    --------
    - Chunked k-NN computation prevents GPU out-of-memory errors
    - Memory-aware force computation with automatic chunk sizing
    - Attraction forces between k-NN neighbors only
    - Repulsion forces from random sampling for efficiency
    - Automatic FP16 optimization for memory and speed
    - Optional PyKeOps integration for low-dimensional data
    
    Best suited for
    ---------------
    - Large datasets (>50K points) on CUDA GPUs
    - Production environments requiring reliable memory usage
    - High-performance dimensionality reduction workflows
    
    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in the target embedding space.
    n_neighbors : int, default=16
        Number of nearest neighbors to use for attraction forces.
    init : {'pca', 'spectral', 'random'}, default='pca'
        Method for initializing the embedding. 'pca' uses PCA initialization,
        'random' uses random projection.
    max_iter_layout : int, default=128
        Maximum number of optimization iterations.
    min_dist : float, default=1e-2
        Minimum distance between points in the embedding.
    spread : float, default=1.0
        Controls how tightly points are packed in the embedding.
    cutoff : float, default=42.0
        Distance cutoff for repulsion forces.
    n_sample_dirs : int, default=8
        Number of sampling directions (used by derived classes).
    sample_size : int, default=16
        Size of samples for force computation (used by derived classes).
    neg_ratio : int, default=8
        Ratio of negative samples to positive samples for repulsion.
    verbose : bool, default=True
        Whether to print progress information.
    random_state : int or None, default=None
        Random seed for reproducible results.
    use_exact_repulsion : bool, default=False
        If True, use exact all-pairs repulsion (memory intensive, for testing only).
    metric : str, callable, or None, default=None
        Custom distance metric for k-NN computation only (layout forces remain Euclidean):

        - None or 'euclidean'/'l2': Use fast built-in Euclidean distance
        - str: String expression evaluated with x and y tensors (e.g., '(x - y).abs().sum(-1)' for L1)
        - callable: Custom function taking (x, y) tensors and returning distance matrix

        Examples: '(x - y).abs().sum(-1)' (L1), '1 - (x*y).sum(-1)/(x.norm()*y.norm() + 1e-8)' (cosine).
    knn_backend : {'auto', 'pytorch', 'pykeops', 'cuvs'}, default='auto'
        k-NN engine selection independent of the DiRe implementation:

        - 'auto': Preserve built-in heuristics for cuVS/PyKeOps/PyTorch
        - 'pytorch': Force chunked PyTorch distance computation
        - 'pykeops': Force PyKeOps LazyTensor k-NN when PyKeOps is installed
        - 'cuvs': Force RAPIDS cuVS k-NN when RAPIDS and CUDA are available
        
    Attributes
    ----------
    device : torch.device
        The PyTorch device being used (CPU or CUDA).
    logger : loguru.Logger
        Instance-specific logger for this reducer.
        
    Examples
    --------
    Basic usage::
    
        from dire_rapids import DiRePyTorch
        import numpy as np
        
        # Create sample data
        X = np.random.randn(10000, 100)
        
        # Create and fit reducer
        reducer = DiRePyTorch(n_neighbors=32, verbose=True)
        embedding = reducer.fit_transform(X)
        
        # Visualize results
        fig = reducer.visualize()
        fig.show()
    
    With custom parameters::

        reducer = DiRePyTorch(
            n_components=3,
            n_neighbors=50,
            max_iter_layout=200,
            min_dist=0.1,
            random_state=42
        )
        embedding = reducer.fit_transform(X)

    With custom distance metric::

        # Using L1 (Manhattan) distance for k-NN
        reducer = DiRePyTorch(
            metric='(x - y).abs().sum(-1)',
            n_neighbors=32
        )
        embedding = reducer.fit_transform(X)

        # Using custom callable metric
        def cosine_distance(x, y):
            return 1 - (x * y).sum(-1) / (x.norm(dim=-1, keepdim=True) * y.norm(dim=-1, keepdim=True) + 1e-8)

        reducer = DiRePyTorch(metric=cosine_distance)
        embedding = reducer.fit_transform(X)
    """

    def __init__(
            self,
            n_components=2,
            n_neighbors=16,
            init="pca",
            max_iter_layout=128,
            min_dist=1e-2,
            spread=1.0,
            cutoff=42.0,
            n_sample_dirs=8,
            sample_size=16,
            neg_ratio=8,
            verbose=True,
            random_state=None,
            use_exact_repulsion=False,  # If True, use all-pairs repulsion (for testing)
            metric=None,
            knn_backend="auto",
            knn_chunk_size=None,
            knn_memory_fraction=0.20,
            knn_broadcast_memory_multiplier=3.0,
            normalize=True,
    ):
        """
        Initialize DiRePyTorch reducer with specified parameters.
        
        Parameters
        ----------
        n_components : int, default=2
            Number of dimensions in the target embedding space.
        n_neighbors : int, default=16
            Number of nearest neighbors to use for attraction forces.
        init : {'pca', 'spectral', 'random'}, default='pca'
            Method for initializing the embedding. 'spectral' uses the bottom
            eigenvectors of the normalized graph Laplacian built from the kNN
            graph (Laplacian Eigenmaps) — good at surfacing cluster structure
            on datasets whose PCA projection has overlapping classes. On CUDA
            this runs via ``cupyx.scipy.sparse.linalg.lobpcg`` (seconds at
            500K+ points); on CPU it falls back to ``scipy.sparse.linalg.eigsh``.
        max_iter_layout : int, default=128
            Maximum number of optimization iterations.
        min_dist : float, default=1e-2
            Minimum distance between points in the embedding.
        spread : float, default=1.0
            Controls how tightly points are packed in the embedding.
        cutoff : float, default=42.0
            Distance cutoff for repulsion forces.
        n_sample_dirs : int, default=8
            Number of sampling directions (reserved for future use).
        sample_size : int, default=16
            Size of samples for force computation (reserved for future use).
        neg_ratio : int, default=8
            Ratio of negative samples to positive samples for repulsion.
        verbose : bool, default=True
            Whether to print progress information.
        random_state : int or None, default=None
            Random seed for reproducible results.
        use_exact_repulsion : bool, default=False
            If True, use exact all-pairs repulsion (memory intensive, testing only).
        metric : str, callable, or None, default=None
            Custom distance metric for k-NN computation. See class docstring for details.
        knn_backend : {'auto', 'pytorch', 'pykeops', 'cuvs'}, default='auto'
            k-NN engine selection. ``'auto'`` keeps the built-in selector;
            explicit values bypass heuristic choices and either use the
            requested engine or raise if the engine cannot run.
        knn_chunk_size : int or None, default=None
            Chunk size for k-NN distance computation. If None, estimate a safe
            chunk from available memory and metric implementation.
        knn_memory_fraction : float, default=0.20
            Fraction of currently available device memory used by automatic
            k-NN chunk sizing.
        knn_broadcast_memory_multiplier : float, default=3.0
            Safety multiplier for arbitrary metric callables/string expressions
            that materialize broadcast tensors of shape ``(chunk, n_samples,
            n_features)``.
        normalize : bool, default=True
            If True, scale inputs to fit in [-1, 1] before kNN and PCA, and
            mean-center them for translation-invariant Euclidean metrics. Named
            cosine and inner-product metrics retain the original origin so their
            neighbor rankings are not changed. This keeps squared distances in a
            numerically safe range for fp16.
        """

        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.init = init
        self.max_iter_layout = max_iter_layout
        self.min_dist = min_dist
        self.spread = spread
        self.cutoff = cutoff
        self.n_sample_dirs = n_sample_dirs
        self.sample_size = sample_size
        self.neg_ratio = neg_ratio
        self.verbose = verbose
        self.random_state = random_state if random_state is not None else np.random.randint(0, 2 ** 32)
        self.use_exact_repulsion = use_exact_repulsion
        self.normalize = normalize
        self.knn_backend = _normalize_knn_backend(knn_backend)
        self.knn_chunk_size = knn_chunk_size
        self.knn_memory_fraction = knn_memory_fraction
        self.knn_broadcast_memory_multiplier = knn_broadcast_memory_multiplier

        # Store RNG state -- defer torch/cuda seeding to fit_transform
        # to avoid mutating global state from a library constructor.
        self._rng = np.random.default_rng(self.random_state)

        # Custom metric for k-NN only (layout forces remain Euclidean):
        self.metric_spec = metric
        self._metric_name = _normalize_named_metric(self.metric_spec)
        self._metric_fn = _compile_metric(self.metric_spec)

        # Do not add/remove Loguru handlers from a library class. The adapter
        # keeps verbose=False quiet without mutating global logger state.
        self.logger = _InstanceLogger(verbose, id(self))

        # Internal state
        self._data = None
        self._layout = None
        self._n_samples = None
        self._a = None
        self._b = None
        self._knn_indices = None
        self._knn_distances = None
        self._last_knn_backend = None
        self._last_knn_chunk_size = None
        self._last_knn_distance_strategy = None
        self._last_knn_reducer = type(self).__name__

        # Device management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            self.logger.warning("CUDA not available, using CPU")

    def _find_ab_params(self):
        """
        Find optimal a and b parameters for the distribution kernel.
        
        This private method fits a curve to determine the optimal parameters for the
        probability kernel used in force calculations. The parameters control the
        shape of the attraction/repulsion curve.
        
        Notes
        -----
        Private method, should not be called directly. Used by fit_transform().
        The kernel function is: 1 / (1 + a * x^(2*b))
        
        Side Effects
        ------------
        Sets self._a and self._b attributes with the fitted parameters.
        """

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))

        xv = np.linspace(0, 3 * self.spread, 300)
        yv = np.zeros(xv.shape)
        yv[xv < self.min_dist] = 1.0
        yv[xv >= self.min_dist] = np.exp(-(xv[xv >= self.min_dist] - self.min_dist) / self.spread)

        params, _ = curve_fit(curve, xv, yv)[:2]  # Extract only params and covariance
        self._a, self._b = params

        self.logger.info(f"Found kernel params: a={self._a:.4f}, b={self._b:.4f}")

    def _cuvs_knn_unavailable_reason(self, n_samples, n_dims):
        """Return a reason cuVS k-NN cannot run for this reducer, or None."""
        del n_samples  # Kept for symmetry with selectors in subclasses.
        if not CUVS_AVAILABLE:
            return "RAPIDS cuVS is not installed"
        if self.device.type != 'cuda':
            return "cuVS k-NN requires a CUDA device"
        if n_dims > 2048:
            return f"cuVS k-NN supports up to 2048 dimensions in this path (got {n_dims})"
        if self._metric_fn is not None:
            return "cuVS k-NN in DiRePyTorch only supports backend-native euclidean/l2 metrics"
        return None

    def _pykeops_knn_unavailable_reason(self, use_fp16):
        """Return a reason PyKeOps k-NN cannot run, or None."""
        if not PYKEOPS_AVAILABLE:
            return "PyKeOps is not installed"
        if use_fp16:
            return "PyKeOps k-NN does not run on the active FP16 path"
        return None

    def _select_knn_backend(self, X, use_fp16, allow_cuvs=True):
        """Resolve the k-NN backend from the user policy and auto heuristics."""
        n_samples, n_dims = X.shape

        if self.knn_backend == 'pytorch':
            return 'pytorch'

        if self.knn_backend == 'pykeops':
            reason = self._pykeops_knn_unavailable_reason(use_fp16)
            if reason is not None:
                raise RuntimeError(f"knn_backend='pykeops' requested but unavailable: {reason}")
            return 'pykeops'

        if self.knn_backend == 'cuvs':
            reason = self._cuvs_knn_unavailable_reason(n_samples, n_dims)
            if reason is not None:
                raise RuntimeError(f"knn_backend='cuvs' requested but unavailable: {reason}")
            return 'cuvs'

        # Auto policy: preserve the historical hard-coded choices unless the
        # caller explicitly requested an engine through knn_backend.
        if allow_cuvs:
            cuvs_threshold = 200000 if n_dims >= 200 else 100000
            if (
                n_samples >= cuvs_threshold
                and self._cuvs_knn_unavailable_reason(n_samples, n_dims) is None
            ):
                return 'cuvs'

        if (
            PYKEOPS_AVAILABLE
            and n_dims < 200
            and self.device.type == 'cuda'
            and not use_fp16
        ):
            return 'pykeops'

        return 'pytorch'

    def _get_available_knn_memory(self):
        """Return currently available memory for k-NN chunk sizing."""
        if self.device.type == 'cuda':
            return torch.cuda.mem_get_info()[0]

        try:
            import psutil  # pylint: disable=import-outside-toplevel
            return psutil.virtual_memory().available
        except ImportError:
            # Conservative fallback when psutil is not installed. This keeps
            # CPU custom-metric chunking from assuming it can materialize huge
            # broadcast tensors.
            return 1024 ** 3

    def _metric_requires_broadcast_tensors(self, use_pykeops=False):
        """Return True when the PyTorch path must materialize broadcast metric tensors."""
        if use_pykeops:
            return False
        return self._metric_fn is not None and self._metric_name not in _MATRIX_MULTIPLY_METRICS

    def _estimate_knn_memory_per_query(self, n_samples, n_dims, bytes_per_element, use_pykeops=False):
        """Estimate temporary k-NN memory per query row in bytes."""
        distance_matrix_bytes = n_samples * bytes_per_element
        if not self._metric_requires_broadcast_tensors(use_pykeops):
            return distance_matrix_bytes

        broadcast_bytes = n_samples * n_dims * bytes_per_element
        return distance_matrix_bytes + int(broadcast_bytes * self.knn_broadcast_memory_multiplier)

    def _compute_auto_knn_chunk_size(
        self,
        n_samples,
        n_dims,
        dtype,
        use_pykeops=False,
        available_memory=None,
    ):
        """Compute a bounded chunk size from available memory and metric strategy."""
        if available_memory is None:
            available_memory = self._get_available_knn_memory()

        bytes_per_element = 2 if dtype == torch.float16 else 4
        memory_per_query = self._estimate_knn_memory_per_query(
            n_samples,
            n_dims,
            bytes_per_element,
            use_pykeops=use_pykeops,
        )
        memory_budget = max(1, int(available_memory * self.knn_memory_fraction))
        chunk_size = max(1, int(memory_budget / max(1, memory_per_query)))
        return max(1, min(chunk_size, n_samples, 50000))

    def _compute_knn(self, X, chunk_size=None, use_fp16=None):  # pylint: disable=too-many-branches
        """
        Compute k-nearest neighbors with memory-efficient chunking.
        
        This private method computes the k-nearest neighbors graph using either PyTorch
        or PyKeOps backends. It intelligently selects the optimal backend based on
        data dimensionality and automatically manages memory through chunking.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).
        chunk_size : int, optional
            Size of chunks for processing. If None, automatically determined based
            on available memory.
        use_fp16 : bool, optional
            Use FP16 precision for computation. If None, automatically determined
            based on data size and GPU capabilities. FP16 provides 2x memory savings
            and 2-14x speedup on modern GPUs.
            
        Notes
        -----
        Private method, should not be called directly. Used by fit_transform().
        
        Backend Selection:
        - PyTorch: Used for high-dimensional data (>= 200D) or when PyKeOps unavailable
        - PyKeOps: Used for low-dimensional data (< 200D) on GPU for better performance
        
        Side Effects
        ------------
        Sets self._knn_indices and self._knn_distances with computed k-NN graph.
        """
        # A reducer can be fitted repeatedly. Any cached tensor belongs to the
        # previous graph and must not be reused merely because N and k match.
        self._knn_indices_torch = None

        n_samples = X.shape[0]
        n_dims = X.shape[1]
        self.logger.info(f"Computing {self.n_neighbors}-NN graph for {n_samples} points in {n_dims}D...")

        # Auto-detect FP16 usage based on data size and GPU
        if use_fp16 is None and self.device.type == 'cuda':
            # Use FP16 for high-dimensional data or large datasets
            use_fp16 = n_dims >= 500 or n_samples >= 100000
        elif self.device.type == 'cpu':
            use_fp16 = False  # CPU doesn't benefit from FP16

        if self.knn_backend == 'pykeops' and use_fp16:
            self.logger.warning(
                "Disabling FP16 k-NN because knn_backend='pykeops' was requested"
            )
            use_fp16 = False

        # Safety guard: fp16 has a finite range (|x|²-sums must stay below ~65504).
        # If the data scale would push squared distances near overflow, fall back
        # to fp32 rather than silently corrupting the neighbor graph.  The guard
        # triggers when the user passes un-normalized data directly to
        # _compute_knn (callers who do not go through fit_transform) or supplies
        # use_fp16=True explicitly on unsafe data.
        if use_fp16 and self.device.type == 'cuda':
            x_abs_max = float(np.abs(X).max())
            worst_sq_dist = 4.0 * n_dims * x_abs_max * x_abs_max  # |2 * max|^2 * D
            if worst_sq_dist > 30000.0:  # leave headroom below fp16 max (65504)
                self.logger.warning(
                    f"Disabling FP16 k-NN: data scale |x|_max={x_abs_max:.3g} at "
                    f"D={n_dims} would overflow fp16 squared distances "
                    f"(worst case {worst_sq_dist:.3g} > 30000). Falling back to FP32."
                )
                use_fp16 = False

        selected_knn_backend = self._select_knn_backend(X, use_fp16, allow_cuvs=True)

        if selected_knn_backend == 'cuvs':
            self._last_knn_backend = 'cuvs'
            try:
                return self._compute_knn_cuvs(X)
            except Exception as e:
                if self.knn_backend == 'cuvs':
                    raise RuntimeError("Forced cuVS k-NN failed") from e
                self.logger.warning(f"cuVS kNN failed ({e}), falling back to another k-NN backend")
                selected_knn_backend = self._select_knn_backend(X, use_fp16, allow_cuvs=False)

        # Choose precision
        if use_fp16 and self.device.type == 'cuda':
            dtype = torch.float16
            self.logger.info("Using FP16 for k-NN (2x memory, faster on H100/A100)")
        else:
            dtype = torch.float32
            self.logger.info("Using FP32 for k-NN")
        
        X_torch = torch.tensor(X, dtype=dtype, device=self.device)

        use_pykeops = selected_knn_backend == 'pykeops'
        self._last_knn_backend = selected_knn_backend
        self._last_knn_reducer = type(self).__name__

        if use_pykeops:
            if self.knn_backend == 'pykeops':
                self.logger.info("Using PyKeOps for k-NN (forced by knn_backend)")
            else:
                self.logger.info("Using PyKeOps for k-NN (low dimension, GPU available)")
        else:
            if self.knn_backend == 'pytorch':
                self.logger.info("Using PyTorch for k-NN (forced by knn_backend)")
            elif n_dims >= 200:
                self.logger.info(f"Using PyTorch for k-NN (high dimension: {n_dims}D)")
            else:
                self.logger.info("Using PyTorch for k-NN")

        # Resolve chunk size. None means memory-aware sizing; explicit
        # knn_chunk_size/argument values are respected.
        if chunk_size is None:
            chunk_size = self.knn_chunk_size
        if chunk_size is None:
            available_memory = self._get_available_knn_memory()
            chunk_size = self._compute_auto_knn_chunk_size(
                n_samples,
                n_dims,
                dtype,
                use_pykeops=use_pykeops,
                available_memory=available_memory,
            )
        elif chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self._last_knn_chunk_size = int(chunk_size)
        memory_label = "GPU" if self.device.type == 'cuda' else "system"
        available_memory = self._get_available_knn_memory()
        self.logger.info(
            f"Using chunk size: {chunk_size} "
            f"({memory_label} memory: {available_memory/1024**3:.1f}GB, dtype: {dtype})"
        )

        X_torch_T = X_torch.T if not use_pykeops else None
        X_norm_torch = None
        X_norm_torch_T = None
        X_sq_norms = None
        if not use_pykeops:
            if self._metric_name == "cosine":
                X_norm_torch = torch.nn.functional.normalize(X_torch, p=2, dim=1, eps=1e-8)
                X_norm_torch_T = X_norm_torch.T
            elif self._metric_name == "sqeuclidean":
                X_sq_norms = (X_torch * X_torch).sum(dim=1)
        
        # Initialize arrays for results
        all_knn_indices = []
        all_knn_distances = []
        
        # Process in chunks to avoid memory issues
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)

            if n_samples > 50000:  # Only log for large datasets
                self.logger.info(f"Processing chunk {start_idx//chunk_size + 1}/{(n_samples + chunk_size - 1)//chunk_size}")

            # Get chunk data
            X_chunk = X_torch[start_idx:end_idx]  # (chunk_size, D)
            distances = None

            if use_pykeops:
                # Use PyKeOps for LOW dimensional data
                LazyTensor = import_pykeops_lazy_tensor()

                # Ensure contiguity for PyKeOps
                X_i = LazyTensor(X_chunk[:, None, :].contiguous())  # (chunk_size, 1, D)
                X_j = LazyTensor(X_torch[None, :, :].contiguous())   # (1, N, D)

                # Compute distances using custom metric or default Euclidean
                if self._metric_fn is not None:
                    # Custom metric - works with LazyTensor
                    D_ij = self._metric_fn(X_i, X_j)  # (chunk_size, N) LazyTensor
                else:
                    # Fast built-in Euclidean (squared distances)
                    D_ij = ((X_i - X_j) ** 2).sum(-1)  # (chunk_size, N) LazyTensor

                # Find k+1 nearest neighbors (including self)
                knn_dists, knn_indices = D_ij.Kmin_argKmin(K=self.n_neighbors + 1, dim=1)
                self._last_knn_distance_strategy = (
                    f"pykeops_{self._metric_name}" if self._metric_fn is not None else "pykeops_euclidean"
                )

                # For custom metrics, distances are already in metric space
                # For Euclidean, convert from squared to actual distances
                if self._metric_fn is None:
                    knn_dists_np = torch_tensor_to_numpy(torch.sqrt(knn_dists))
                else:
                    knn_dists_np = torch_tensor_to_numpy(knn_dists)
                chunk_indices, chunk_distances = _remove_self_from_knn(
                    torch_tensor_to_numpy(knn_indices),
                    knn_dists_np,
                    start_idx,
                    self.n_neighbors,
                )
            else:
                # Use PyTorch for HIGH dimensional data (MUCH faster!)
                if self._metric_name == "cosine":
                    # Matmul cosine path avoids materializing (chunk, N, D).
                    distances = 1.0 - (X_norm_torch[start_idx:end_idx] @ X_norm_torch_T)
                    self._last_knn_distance_strategy = "matmul_cosine"
                elif self._metric_name == "inner_product":
                    distances = -(X_chunk @ X_torch_T)
                    self._last_knn_distance_strategy = "matmul_inner_product"
                elif self._metric_name == "sqeuclidean":
                    distances = (
                        X_sq_norms[start_idx:end_idx, None]
                        + X_sq_norms[None, :]
                        - 2.0 * (X_chunk @ X_torch_T)
                    )
                    distances = torch.clamp(distances, min=0.0)
                    self._last_knn_distance_strategy = "matmul_sqeuclidean"
                elif self._metric_fn is not None:
                    # Custom metric - compute pairwise distances manually.
                    # This can materialize broadcast tensors, so automatic
                    # chunk sizing accounts for the (chunk, N, D) footprint.
                    # Broadcast: X_chunk: (chunk, 1, D), X_torch: (1, N, D) -> (chunk, N)
                    X_i = X_chunk.unsqueeze(1)  # (chunk, 1, D)
                    X_j = X_torch.unsqueeze(0)  # (1, N, D)
                    distances = self._metric_fn(X_i, X_j)  # (chunk, N)
                    self._last_knn_distance_strategy = "broadcast_custom"
                else:
                    # Fast built-in Euclidean distance
                    distances = torch.cdist(X_chunk, X_torch, p=2)
                    self._last_knn_distance_strategy = "torch_cdist"

                local_rows = torch.arange(end_idx - start_idx, device=self.device)
                distances[local_rows, start_idx + local_rows] = -torch.inf

                knn_dists, knn_indices = torch.topk(distances, k=self.n_neighbors + 1,
                                                   dim=1, largest=False)

                chunk_indices, chunk_distances = _remove_self_from_knn(
                    torch_tensor_to_numpy(knn_indices),
                    torch_tensor_to_numpy(knn_dists),
                    start_idx,
                    self.n_neighbors,
                )

            all_knn_indices.append(chunk_indices)
            all_knn_distances.append(chunk_distances)

            # Clear GPU memory after each chunk to avoid fragmentation
            if self.device.type == 'cuda':
                del knn_dists, knn_indices
                if distances is not None:
                    del distances
                torch.cuda.empty_cache()
        
        # Concatenate results
        self._knn_indices = np.vstack(all_knn_indices)
        self._knn_distances = np.vstack(all_knn_distances)

        self.logger.info(f"k-NN graph computed: shape {self._knn_indices.shape}")

    def _compute_knn_cuvs(self, X):
        """Use cuVS IVF-Flat for fast GPU-accelerated kNN.

        IVF-Flat partitions the dataset into Voronoi cells (n_lists clusters)
        and searches only the closest nprobe cells.  With nprobe high enough
        relative to n_lists the results are effectively exact.

        Sets self._knn_indices, self._knn_distances (same contract as _compute_knn).
        """
        n_samples, n_dims = X.shape
        k = self.n_neighbors + 1  # +1 because cuVS returns self as first neighbor

        self.logger.info(f"Using cuVS IVF-Flat for {n_samples} points in {n_dims}D")

        X_gpu = cp.asarray(X, dtype=cp.float32, order='C')

        # Scale n_lists with dataset size; search enough cells for high recall
        n_lists = min(int(np.sqrt(n_samples)), 1024)
        nprobe = min(n_lists, 64)

        build_params = cuvs_ivf_flat.IndexParams(n_lists=n_lists, metric="sqeuclidean")
        index = cuvs_ivf_flat.build(build_params, X_gpu)

        search_params = cuvs_ivf_flat.SearchParams(n_probes=nprobe)
        distances, indices = cuvs_ivf_flat.search(search_params, index, X_gpu, k)

        indices_np = cp.asnumpy(cp.asarray(indices))
        distances_np = cp.asnumpy(cp.asarray(distances))

        # Remove self and convert squared distances to distances. Self is not
        # guaranteed to be column 0 when there are duplicate/tied points.
        self._knn_indices, squared_distances = _remove_self_from_knn(
            indices_np,
            distances_np,
            0,
            self.n_neighbors,
        )
        self._knn_distances = np.sqrt(np.maximum(squared_distances, 0.0))

        self.logger.info(f"k-NN graph computed via cuVS: shape {self._knn_indices.shape}")

        # Clean up
        del X_gpu, index
        cp.get_default_memory_pool().free_all_blocks()

    def _spectral_init(self):
        """Laplacian-Eigenmap initialization from the kNN graph.

        Builds the symmetrized unweighted kNN adjacency `A`, forms the
        normalized adjacency `A_norm = D^{-1/2} A D^{-1/2}`, and returns the
        eigenvectors of `A_norm` associated with its top `n_components + 1`
        eigenvalues (dropping the trivial constant eigenvector at λ = 1 on each
        connected component).  Because the normalized Laplacian is
        `L_sym = I − A_norm`, largest eigenvalues of `A_norm` correspond to
        smallest of `L_sym`; this avoids shift-invert factorization, which is
        intractable on a 500K-node graph.

        On CUDA, eigenvectors are computed with
        `cupyx.scipy.sparse.linalg.lobpcg` (seconds on a 580K-node graph).
        On CPU, falls back to `scipy.sparse.linalg.eigsh(which='LA')`.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_components)
            Initial embedding; the outer `_initialize_embedding` normalizes it
            to zero mean and unit std per component.
        """
        from scipy.sparse import csr_matrix, diags
        self.logger.info("Initializing with spectral embedding (Laplacian eigenmaps)")

        n = self._n_samples
        k = self._knn_indices.shape[1]
        n_eigs = self.n_components + 1  # +1 to drop the trivial constant

        # Build symmetric unweighted adjacency on CPU. OR with its transpose so
        # (i→j ∈ kNN) ∨ (j→i ∈ kNN) ⇒ edge. nnz ≤ 2*N*k.
        rows = np.repeat(np.arange(n, dtype=np.int32), k)
        cols = self._knn_indices.ravel().astype(np.int32)
        vals = np.ones(len(rows), dtype=np.float32)
        A = csr_matrix((vals, (rows, cols)), shape=(n, n))
        A = A.maximum(A.T).tocsr()
        A.setdiag(0)
        A.eliminate_zeros()

        deg = np.asarray(A.sum(axis=1)).ravel()
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)

        # Reproducible random start for the iterative eigensolver.
        rng = np.random.default_rng(self.random_state)
        X0 = rng.standard_normal((n, n_eigs)).astype(np.float32)

        use_gpu = self.device.type == 'cuda'
        if use_gpu:
            try:
                import cupy as cp
                import cupyx.scipy.sparse as cusp
                import cupyx.scipy.sparse.linalg as cuspl

                D_inv_sqrt_gpu = cusp.diags(cp.asarray(d_inv_sqrt),
                                            0, format='csr', dtype=cp.float32)
                A_gpu = cusp.csr_matrix(A.astype(np.float32))
                A_norm_gpu = (D_inv_sqrt_gpu @ A_gpu @ D_inv_sqrt_gpu).tocsr()
                X0_gpu = cp.asarray(X0)

                vals_gpu, vecs_gpu = cuspl.lobpcg(
                    A_norm_gpu, X0_gpu, largest=True, tol=1e-4, maxiter=200
                )
                vals_np = cp.asnumpy(vals_gpu)
                vecs_np = cp.asnumpy(vecs_gpu)
                del A_gpu, A_norm_gpu, D_inv_sqrt_gpu, X0_gpu, vals_gpu, vecs_gpu
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:  # pylint: disable=broad-exception-caught
                self.logger.warning(
                    f"GPU spectral solver failed ({e}); falling back to CPU eigsh"
                )
                use_gpu = False

        if not use_gpu:
            from scipy.sparse.linalg import eigsh
            D_inv_sqrt_cpu = diags(d_inv_sqrt, 0, format='csr', dtype=np.float32)
            A_norm_cpu = (D_inv_sqrt_cpu @ A.astype(np.float32) @ D_inv_sqrt_cpu).tocsr()
            vals_np, vecs_np = eigsh(
                A_norm_cpu, k=n_eigs, which='LA', tol=1e-4, maxiter=500,
                v0=X0[:, 0],
            )

        # Sort eigenvalues descending (largest on A_norm = smallest on L_sym)
        # and drop the single most-trivial eigenvector (the constant-per-component
        # mode at λ ≈ 1). On a disconnected graph, additional λ ≈ 1 eigenvectors
        # are per-component indicators — these are useful for separating clusters,
        # so keep them rather than discarding (matches UMAP's spectral init).
        order = np.argsort(-vals_np)
        vecs_np = vecs_np[:, order[1:1 + self.n_components]]
        return vecs_np.astype(np.float32)

    def _initialize_embedding(self, X):
        """
        Initialize the low-dimensional embedding.

        This private method creates the initial embedding using either PCA or random
        projection, then normalizes the result.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input high-dimensional data of shape (n_samples, n_features).
            
        Returns
        -------
        torch.Tensor
            Initial embedding of shape (n_samples, n_components) on the target device.
            
        Notes
        -----
        Private method, should not be called directly. Used by fit_transform().
        
        The embedding is normalized to have zero mean and unit standard deviation
        along each dimension.
        """

        if self.init == 'pca':
            self.logger.info("Initializing with PCA")
            if self.device.type == 'cuda' and X.shape[1] > 32:
                # GPU-accelerated PCA via truncated SVD
                X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
                X_t = X_t - X_t.mean(dim=0)
                # Use randomized SVD (pca_lowrank) — O(N*D*q) instead of full SVD,
                # and avoids cusolver limits on very wide matrices (D >> N).
                U, S, _ = torch.pca_lowrank(X_t, q=self.n_components)
                embedding = torch_tensor_to_numpy(U * S)
                del X_t, U, S
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                pca = PCA(n_components=self.n_components, random_state=self.random_state)
                embedding = pca.fit_transform(X)

        elif self.init == 'random':
            self.logger.info("Initializing with random projection")
            rng = np.random.default_rng(self.random_state)
            projection = rng.standard_normal((X.shape[1], self.n_components))
            projection /= np.linalg.norm(projection, axis=0)
            embedding = X @ projection

        elif self.init == 'spectral':
            embedding = self._spectral_init()

        else:
            raise ValueError(f"Unknown init method: {self.init}")

        # Normalize
        embedding -= embedding.mean(axis=0)
        embedding /= embedding.std(axis=0)

        return torch.tensor(embedding, dtype=torch.float32, device=self.device)

    def _compute_forces(self, positions, iteration, max_iterations):
        """
        Compute attraction and repulsion forces for layout optimization.

        Uses a single vectorized pass over all points (no chunking). On any
        GPU with >= 2 GB free VRAM this handles > 500K points comfortably.
        Falls back to a chunked path only on true out-of-memory.

        Parameters
        ----------
        positions : torch.Tensor
            Current positions, shape (n_samples, n_components).
        iteration : int
            Current iteration number (0-indexed).
        max_iterations : int
            Total number of iterations planned.

        Returns
        -------
        torch.Tensor
            Forces of shape (n_samples, n_components).
        """
        n_samples = positions.shape[0]
        a_val = float(self._a)
        b_val = float(self._b)
        n_neg = min(int(self.neg_ratio * self.n_neighbors), n_samples - 1)

        knn_indices_torch = getattr(self, '_knn_indices_torch', None)
        if knn_indices_torch is None:
            knn_indices_torch = torch.tensor(
                self._knn_indices, dtype=torch.long, device=self.device
            )

        # Generate negative samples (randint is cheap on GPU)
        neg_indices = torch.randint(
            0, n_samples, (n_samples, n_neg), device=self.device
        )

        # Run force kernel in bf16 for ~2x bandwidth savings on gather-heavy ops;
        # positions stay fp32 in the caller, so accumulated drift is bounded.
        use_bf16 = positions.is_cuda and torch.cuda.is_bf16_supported()
        if use_bf16:
            pos_lo = positions.bfloat16()
        else:
            pos_lo = positions

        try:
            forces = _compute_forces_compiled(
                pos_lo, knn_indices_torch, neg_indices,
                a_val, b_val, self.cutoff,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            self.force_chunked_fallback_calls_ = (
                getattr(self, "force_chunked_fallback_calls_", 0) + 1
            )
            self.force_chunked_fallback_used_ = True
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            self.logger.warning(
                "Vectorized force computation OOM — falling back to chunked path"
            )
            forces = self._compute_forces_chunked(
                pos_lo, knn_indices_torch, neg_indices,
                a_val, b_val, n_neg,
            )

        if use_bf16:
            forces = forces.float()

        return forces

    def _compute_forces_chunked(self, positions, knn_indices_torch, neg_indices,
                                a_val, b_val, n_neg):
        """Chunked fallback for GPUs that cannot fit the full computation."""
        n_samples = positions.shape[0]
        forces = torch.zeros_like(positions)
        chunk_size = max(100, n_samples // 10)

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            s = slice(start, end)
            chunk_pos = positions[s]

            # Attraction
            neighbor_pos = positions[knn_indices_torch[s]]
            diff = neighbor_pos - chunk_pos.unsqueeze(1)
            dist_sq = (diff * diff).sum(dim=2, keepdim=True) + 1e-10
            inv_dist = torch.rsqrt(dist_sq)
            dist_sq_b = dist_sq ** b_val
            att = dist_sq_b / (dist_sq_b + a_val)
            forces[s] += (att * diff * inv_dist).sum(dim=1)

            # Repulsion
            neg_pos = positions[neg_indices[s]]
            diff_n = neg_pos - chunk_pos.unsqueeze(1)
            dist_sq_n = (diff_n * diff_n).sum(dim=2, keepdim=True) + 1e-10
            inv_dist_n = torch.rsqrt(dist_sq_n)
            dist_n = dist_sq_n * inv_dist_n
            dist_sq_b_n = dist_sq_n ** b_val
            rep = -1.0 / (1.0 + a_val * dist_sq_b_n)
            rep = rep * torch.exp(-dist_n / self.cutoff)
            forces[s] += (rep * diff_n * inv_dist_n).sum(dim=1)

        return torch.clamp(forces, -self.cutoff, self.cutoff)

    def _synchronize_for_timing(self):
        """Synchronize pending accelerator work before reading a stage timer."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _time_fit_stage(self, name, operation, *args, **kwargs):
        """Run one fit stage and retain a wall-clock duration in seconds."""
        self._synchronize_for_timing()
        started = perf_counter()
        result = operation(*args, **kwargs)
        self._synchronize_for_timing()
        self.stage_timings_[name] = perf_counter() - started
        return result

    def get_diagnostics(self):
        """Return JSON-serializable fitted backend and pipeline diagnostics.

        Returns
        -------
        dict
            Requested/effective k-NN backend, per-stage wall-clock timings,
            and whether the vectorized force implementation fell back to its
            chunked implementation during the most recent fit.
        """
        return {
            "requested_knn_backend": self.knn_backend,
            "effective_knn_backend": getattr(
                self, "effective_knn_backend_", None
            ),
            "stage_timings_seconds": dict(
                getattr(self, "stage_timings_", {})
            ),
            "force_chunked_fallback_used": getattr(
                self, "force_chunked_fallback_used_", False
            ),
            "force_chunked_fallback_calls": getattr(
                self, "force_chunked_fallback_calls_", 0
            ),
        }

    def _optimize_layout(self, initial_positions):
        """
        Optimize the embedding layout using iterative force computation.

        This private method performs the main optimization loop, iteratively
        computing and applying forces to refine the embedding layout.

        Parameters
        ----------
        initial_positions : torch.Tensor
            Initial embedding positions of shape (n_samples, n_components).

        Returns
        -------
        torch.Tensor
            Optimized final positions of shape (n_samples, n_components),
            normalized to zero mean and unit standard deviation.

        Notes
        -----
        Private method, should not be called directly. Used by fit_transform().

        Forces are computed in bf16 for bandwidth efficiency, accumulated
        into fp32 positions via linear cooling: alpha = 1 - iter/max_iter.
        """
        positions = initial_positions.clone()

        self.logger.info(f"Optimizing layout for {self._n_samples} points...")

        # Pre-convert kNN indices once. GPU graph builders may already have
        # populated this cache through DLPack, avoiding a device→host→device
        # round trip for large all-neighbors graphs.
        knn_indices_torch = getattr(self, '_knn_indices_torch', None)
        if (
                knn_indices_torch is None or
                knn_indices_torch.device != positions.device or
                tuple(knn_indices_torch.shape) != tuple(self._knn_indices.shape)
        ):
            self._knn_indices_torch = torch.as_tensor(
                self._knn_indices, dtype=torch.long, device=positions.device
            )

        # Optimization loop with linear cooling
        for iteration in range(self.max_iter_layout):
            forces = self._compute_forces(positions, iteration, self.max_iter_layout)

            alpha = 1.0 - iteration / self.max_iter_layout
            positions.add_(forces, alpha=alpha)

            if iteration % 20 == 0:
                self.logger.info(f"Iteration {iteration}/{self.max_iter_layout}")

        # Final normalization
        positions -= positions.mean(dim=0)
        positions /= positions.std(dim=0)

        return positions

    def fit_transform(self, X, y=None):  # pylint: disable=unused-argument,arguments-differ
        """
        Fit the DiRe model and transform data to low-dimensional embedding.
        
        This method performs the complete dimensionality reduction pipeline:
        1. Computes k-nearest neighbors graph
        2. Fits kernel parameters
        3. Initializes embedding with PCA or random projection
        4. Optimizes layout using force-directed algorithm
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            High-dimensional input data to transform.
        y : array-like of shape (n_samples,), optional
            Ignored. Present for scikit-learn API compatibility.
            
        Returns
        -------
        numpy.ndarray of shape (n_samples, n_components)
            Low-dimensional embedding of the input data.
            
        Examples
        --------
        Transform high-dimensional data::
        
            import numpy as np
            from dire_rapids import DiRePyTorch
            
            X = np.random.randn(1000, 100)
            reducer = DiRePyTorch(n_neighbors=16)
            embedding = reducer.fit_transform(X)
            print(embedding.shape)  # (1000, 2)
        """
        fit_started = perf_counter()
        self.stage_timings_ = {}
        self.effective_knn_backend_ = None
        self.force_chunked_fallback_calls_ = 0
        self.force_chunked_fallback_used_ = False

        # Seed torch RNGs for reproducibility (deferred from __init__
        # to avoid mutating global state at construction time).
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        # Store data
        self._data = np.asarray(X, dtype=np.float32)
        self._n_samples = self._data.shape[0]

        if self.normalize:
            # Mean-centering preserves Euclidean rankings but changes cosine and
            # inner-product neighborhoods because those metrics depend on the
            # origin. A single positive global rescale preserves all three.
            origin_sensitive = (
                isinstance(self.metric_spec, str) and
                self.metric_spec.strip().lower() in ('cosine', 'inner_product')
            )
            if not origin_sensitive:
                self._data = self._data - self._data.mean(axis=0, keepdims=True)
            max_abs = float(np.abs(self._data).max())
            if max_abs > 0:
                self._data /= max_abs

        self.logger.info(f"Processing {self._n_samples} samples with {self._data.shape[1]} features")

        # Validate n_components
        if self.n_components <= 0:
            raise ValueError(f"n_components must be positive, got {self.n_components}")

        # Validate n_neighbors
        if self.n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive, got {self.n_neighbors}")

        # Validate and adjust n_neighbors if necessary
        if self.n_neighbors >= self._n_samples:
            old_n_neighbors = self.n_neighbors
            self.n_neighbors = self._n_samples - 1
            self.logger.warning(
                f"n_neighbors={old_n_neighbors} is >= n_samples={self._n_samples}. "
                f"Adjusting n_neighbors to {self.n_neighbors}"
            )

        # Find distribution kernel parameters
        self._find_ab_params()

        # Compute k-NN graph
        self._time_fit_stage(
            "graph_construction",
            self._compute_knn,
            self._data,
            chunk_size=self.knn_chunk_size,
        )
        self.effective_knn_backend_ = self._last_knn_backend

        # Initialize embedding
        initial_embedding = self._time_fit_stage(
            "initialization", self._initialize_embedding, self._data
        )

        # Optimize layout
        final_embedding = self._time_fit_stage(
            "layout", self._optimize_layout, initial_embedding
        )

        # Convert back to numpy and store
        self._layout = torch_tensor_to_numpy(final_embedding)

        # Clear GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        self._synchronize_for_timing()
        self.stage_timings_["total"] = perf_counter() - fit_started
        self.logger.info(
            "Fit diagnostics: "
            f"backend={self.effective_knn_backend_}, "
            f"graph={self.stage_timings_['graph_construction']:.3f}s, "
            f"initialization={self.stage_timings_['initialization']:.3f}s, "
            f"layout={self.stage_timings_['layout']:.3f}s, "
            f"chunked_force_fallbacks={self.force_chunked_fallback_calls_}"
        )

        return self._layout

    def fit(self, X: np.ndarray, y=None):  # pylint: disable=unused-argument,arguments-differ
        """
        Fit the DiRe model to data without transforming.
        
        This method fits the model by computing the k-NN graph, kernel parameters,
        and optimized embedding, but primarily serves for scikit-learn compatibility.
        For practical use, fit_transform() is recommended.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            High-dimensional data to fit the model to.
        y : array-like of shape (n_samples,), optional
            Ignored. Present for scikit-learn API compatibility.

        Returns
        -------
        self : DiRePyTorch
            The fitted DiRe instance.
            
        Notes
        -----
        This method calls fit_transform() internally. The embedding result
        is stored in self._layout and can be accessed after fitting.
        """
        self.fit_transform(X, y)
        return self

    def visualize(self, labels=None, point_size=2, title=None, max_points=10000,
                  mode="auto", density_threshold=50000, categorical_labels=None,
                  **kwargs):
        """
        Create an interactive visualization of the embedding.

        Uses WebGL scatter for moderate point counts and automatically subsamples
        to ``max_points``. For large 2D embeddings it switches to a binned density
        (2D histogram) so the figure payload stays bounded regardless of the
        number of points; see ``mode`` and ``density_threshold``.

        Parameters
        ----------
        labels : array-like of shape (n_samples,), optional
            Labels for coloring points in the visualization.
        point_size : int, default=2
            Size of points in the scatter plot.
        mode : {'auto', 'scatter', 'density'}, default='auto'
            Rendering mode. ``'auto'`` switches a 2D embedding to density once it
            exceeds ``density_threshold`` points; ``'density'`` forces density
            (2D only); ``'scatter'`` always draws markers.
        density_threshold : int, default=50000
            Point count above which ``'auto'`` mode uses density.
        categorical_labels : bool, optional
            Treat labels as discrete classes (per-category colors / density
            layers). If None, inferred from the label dtype (numeric is treated
            as a continuous scalar).
        title : str, optional
            Title for the plot. If None, a default title is generated.
        max_points : int, default=10000
            Maximum number of points to display. Subsamples if larger.
        **kwargs : dict
            Additional keyword arguments passed to plotly.express plotting functions.

        Returns
        -------
        plotly.graph_objects.Figure or None
            Interactive Plotly figure, or None if no embedding is available.

        Examples
        --------
        Basic visualization::

            fig = reducer.visualize()
            fig.show()

        With labels and custom styling::

            fig = reducer.visualize(
                labels=y,
                point_size=3,
                title="My Embedding",
                max_points=20000,
                width=800,
                height=600
            )
            fig.show()

        Notes
        -----
        Requires a fitted model with available embedding (self._layout).
        Only supports 2D and 3D visualizations.
        """
        if self._layout is None:
            self.logger.warning("No layout available for visualization")
            return None

        from .utils import build_embedding_figure, _infer_categorical  # pylint: disable=import-outside-toplevel

        if title is None:
            title = f"PyTorch {self.n_components}D Embedding"
        if categorical_labels is None:
            categorical_labels = _infer_categorical(labels)

        return build_embedding_figure(
            self._layout, labels, title=title, n_dims=self.n_components,
            categorical_labels=categorical_labels, mode=mode,
            density_threshold=density_threshold, max_points=max_points,
            point_size=point_size, width=kwargs.get('width'),
            height=kwargs.get('height'), logger=self.logger,
        )


def create_dire(backend='auto', memory_efficient=False, knn_backend='auto', **kwargs):
    """
    Create DiRe instance with automatic backend selection.

    This factory function automatically selects the optimal DiRe implementation
    based on available hardware and software, or allows manual backend selection.
    It provides a convenient interface for creating DiRe instances without
    importing specific backend classes.

    Parameters
    ----------
    backend : {'auto', 'cuvs', 'pytorch', 'pytorch_gpu', 'pytorch_cpu'}, default='auto'
        Backend selection strategy:

        - 'auto': Automatically select best available backend based on hardware
          and metric policy; uses memory-efficient PyTorch when cuVS cannot
          serve the configured metric or k-NN policy on CUDA
        - 'cuvs': Force RAPIDS cuVS backend (requires RAPIDS installation)
        - 'pytorch': Force PyTorch backend with automatic device selection
        - 'pytorch_gpu': Force PyTorch backend on GPU (requires CUDA)
        - 'pytorch_cpu': Force PyTorch backend on CPU only

    memory_efficient : bool, default=False
        If True, use memory-efficient PyTorch implementation which provides:

        - Reduced memory usage for large datasets
        - FP16 support for additional memory savings
        - Enhanced chunking strategies
        - More aggressive memory cleanup

    knn_backend : {'auto', 'pytorch', 'pykeops', 'cuvs'}, default='auto'
        k-NN engine selection, independent of the DiRe implementation selected
        by ``backend``. ``'auto'`` preserves the built-in selector; explicit
        values force the requested k-NN engine or raise if it cannot run.

    **kwargs : dict
        Additional keyword arguments passed to the DiRe constructor.
        See individual backend documentation for available parameters.
        Common parameters include: n_components, n_neighbors, metric,
        max_iter_layout, min_dist, spread, verbose, random_state,
        knn_chunk_size, knn_memory_fraction, and
        knn_broadcast_memory_multiplier.

    Returns
    -------
    DiRe instance
        An instance of the selected DiRe backend (DiRePyTorch, DiRePyTorchMemoryEfficient,
        or DiReCuVS) configured with the specified parameters.

    Raises
    ------
    RuntimeError
        If a specific backend is requested but requirements are not met
        (e.g., requesting cuVS without RAPIDS, or GPU without CUDA).
    ValueError
        If an unknown backend name is specified.

    Examples
    --------
    Auto-select optimal backend::\n
        from dire_rapids import create_dire

        # Will use cuVS if available, otherwise PyTorch with GPU if available
        reducer = create_dire(n_neighbors=32, verbose=True)
        embedding = reducer.fit_transform(X)

    Force memory-efficient mode for large datasets::\n
        reducer = create_dire(
            memory_efficient=True,
            n_neighbors=50,
            max_iter_layout=200
        )

    Force specific backend::\n
        # CPU-only processing
        reducer = create_dire(backend='pytorch_cpu')

        # CPU implementation with an explicit k-NN engine
        reducer = create_dire(backend='pytorch_cpu', knn_backend='pytorch')

        # GPU processing with cuVS k-NN acceleration
        reducer = create_dire(backend='cuvs', knn_backend='cuvs')

        # With custom distance metric
        reducer = create_dire(
            metric='(x - y).abs().sum(-1)',  # L1 distance
            n_neighbors=32,
            verbose=True
        )

    Notes
    -----
    Backend Selection Priority (when backend='auto'):
    1. RAPIDS cuVS (if available and CUDA GPU present)
    2. PyTorch Memory-Efficient with CUDA (if CUDA GPU available, cuVS not available, or memory_efficient=True)
    3. PyTorch with CUDA (if CUDA GPU available and memory_efficient=False)
    4. PyTorch with CPU (fallback)

    When cuVS is not available but GPU is present, the memory-efficient PyTorch backend
    is automatically selected for better GPU memory management and to handle larger datasets.

    The function automatically handles import errors and missing dependencies,
    falling back to available alternatives when possible.
    """
    knn_backend = _normalize_knn_backend(knn_backend)

    # Handle verbose parameter early to disable logging if needed
    verbose = kwargs.get('verbose', True)
    use_cuvs_override = kwargs.pop('use_cuvs', None)

    # Import here to avoid circular imports
    try:
        from .dire_cuvs import DiReCuVS, CUVS_AVAILABLE  # pylint: disable=import-outside-toplevel
    except ImportError:
        CUVS_AVAILABLE = False

    from .dire_pytorch_memory_efficient import DiRePyTorchMemoryEfficient  # pylint: disable=import-outside-toplevel

    if knn_backend == 'pykeops' and not PYKEOPS_AVAILABLE:
        raise RuntimeError("knn_backend='pykeops' requested but PyKeOps is not installed")

    if knn_backend == 'cuvs':
        if not CUVS_AVAILABLE:
            raise RuntimeError(
                "knn_backend='cuvs' requested but RAPIDS cuVS is not installed. "
                "Follow the installation instructions at https://docs.rapids.ai/install/"
            )
        if backend == 'pytorch_cpu' or not torch.cuda.is_available():
            raise RuntimeError("knn_backend='cuvs' requested but CUDA GPU is not available")
        if not _is_cuvs_native_metric(kwargs.get('metric')):
            raise RuntimeError(
                "knn_backend='cuvs' requested but the configured metric is not a cuVS-native "
                "named metric. Use metric=None/'euclidean'/'l2'/'sqeuclidean'/'inner_product'/'cosine', "
                "or choose knn_backend='auto'/'pytorch'."
            )

    if backend == 'auto':
        metric_spec = kwargs.get('metric')
        # Auto-select best backend based on availability
        cuvs_allowed_by_knn_policy = knn_backend in ('auto', 'cuvs')
        cuvs_usable_for_auto = (
            CUVS_AVAILABLE
            and torch.cuda.is_available()
            and use_cuvs_override is not False
            and cuvs_allowed_by_knn_policy
            and _is_cuvs_native_metric(metric_spec)
        )
        if cuvs_usable_for_auto:
            if verbose:
                logger.info("Auto-selected RAPIDS cuVS backend (GPU acceleration)")
            return DiReCuVS(
                use_cuvs=True if use_cuvs_override is None else use_cuvs_override,
                knn_backend=knn_backend,
                **kwargs,
            )

        if torch.cuda.is_available():
            # Prefer memory-efficient PyTorch whenever auto cannot use cuVS
            # cleanly. This covers disabled cuVS, custom metrics, and explicit
            # PyTorch/PyKeOps k-NN requests.
            if (
                memory_efficient
                or not CUVS_AVAILABLE
                or use_cuvs_override is False
                or not _is_cuvs_native_metric(metric_spec)
                or knn_backend in ('pytorch', 'pykeops')
            ):
                if verbose:
                    logger.info("Auto-selected memory-efficient PyTorch backend (GPU)")
                return DiRePyTorchMemoryEfficient(knn_backend=knn_backend, **kwargs)
            if verbose:
                logger.info("Auto-selected PyTorch backend (GPU)")
            return DiRePyTorch(knn_backend=knn_backend, **kwargs)

        # CPU fallback
        if verbose:
            logger.info("Auto-selected PyTorch backend (CPU)")
        if memory_efficient:
            if verbose:
                logger.warning("Memory-efficient mode has limited benefits on CPU")
            return DiRePyTorchMemoryEfficient(knn_backend=knn_backend, **kwargs)
        return DiRePyTorch(knn_backend=knn_backend, **kwargs)

    if backend == 'cuvs':
        if not CUVS_AVAILABLE:
            raise RuntimeError(
                "cuVS backend requested but RAPIDS not installed. "
                "Follow the installation instructions at https://docs.rapids.ai/install/"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("cuVS backend requires CUDA GPU")
        if verbose:
            logger.info("Using RAPIDS cuVS backend")
        return DiReCuVS(
            use_cuvs=True if use_cuvs_override is None else use_cuvs_override,
            knn_backend=knn_backend,
            **kwargs,
        )

    if backend == 'pytorch':
        # Use PyTorch with auto device selection
        if memory_efficient:
            if verbose:
                logger.info("Using memory-efficient PyTorch backend")
            return DiRePyTorchMemoryEfficient(knn_backend=knn_backend, **kwargs)
        if verbose:
            logger.info("Using PyTorch backend")
        return DiRePyTorch(knn_backend=knn_backend, **kwargs)

    if backend == 'pytorch_gpu':
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA not available")
        if memory_efficient:
            if verbose:
                logger.info("Using memory-efficient PyTorch backend (GPU)")
            return DiRePyTorchMemoryEfficient(knn_backend=knn_backend, **kwargs)
        if verbose:
            logger.info("Using PyTorch backend (GPU)")
        return DiRePyTorch(knn_backend=knn_backend, **kwargs)

    if backend == 'pytorch_cpu':
        # Force CPU even if GPU is available
        if verbose:
            logger.info("Using PyTorch backend (forced CPU)")
        if memory_efficient:
            if verbose:
                logger.warning("Memory-efficient mode has limited benefits on CPU")
            # Create instance and force CPU
            reducer = DiRePyTorchMemoryEfficient(knn_backend=knn_backend, **kwargs)
        else:
            reducer = DiRePyTorch(knn_backend=knn_backend, **kwargs)
        reducer.device = torch.device('cpu')
        return reducer

    raise ValueError(
        f"Unknown backend: {backend}. "
        f"Choose from: 'auto', 'cuvs', 'pytorch', 'pytorch_gpu', 'pytorch_cpu'"
    )
