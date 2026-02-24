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
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from loguru import logger
from scipy.optimize import curve_fit
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA

# PyKeOps for efficient force computations
try:
    from pykeops.torch import LazyTensor

    PYKEOPS_AVAILABLE = True
except ImportError:
    PYKEOPS_AVAILABLE = False
    logger.warning("PyKeOps not available. Install with: pip install pykeops")

# cuVS for fast approximate k-NN at scale (optional RAPIDS dependency)
try:
    import cupy as cp
    from cuvs.neighbors import ivf_flat as cuvs_ivf_flat
    CUVS_AVAILABLE = True
except ImportError:
    CUVS_AVAILABLE = False


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
    Named metrics like 'euclidean', 'l2', 'inner_product' return None (handled by backend).
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
        # These named metrics can be handled by backends (cuVS, PyTorch)
        if expr in ("euclidean", "l2", "sqeuclidean", "inner_product", "cosine"):
            return None  # use built-in backend metric
        # Custom expression - compile to callable
        def _expr_metric(x, y, _expr=spec):
            # Use ONLY tensor methods like .sum(-1), .sqrt(), .abs(), etc.
            # Works for both torch.Tensor and KeOps LazyTensor.
            return eval(_expr, {"__builtins__": {}}, {"x": x, "y": y})  # pylint: disable=eval-used # Sandboxed eval for custom metrics
        return _expr_metric
    if callable(spec):
        return spec
    raise ValueError("metric must be None, a string expression, or a callable")


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
    init : {'pca', 'random'}, default='pca'
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
    ):
        """
        Initialize DiRePyTorch reducer with specified parameters.
        
        Parameters
        ----------
        n_components : int, default=2
            Number of dimensions in the target embedding space.
        n_neighbors : int, default=16
            Number of nearest neighbors to use for attraction forces.
        init : {'pca', 'random'}, default='pca'
            Method for initializing the embedding.
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

        # Store RNG state -- defer torch/cuda seeding to fit_transform
        # to avoid mutating global state from a library constructor.
        self._rng = np.random.default_rng(self.random_state)

        # Custom metric for k-NN only (layout forces remain Euclidean):
        self.metric_spec = metric
        self._metric_fn = _compile_metric(self.metric_spec)

        # Setup instance-specific logger
        # Use logger.bind() for context but track handler IDs to avoid
        # corrupting the global loguru logger for other users.
        self.logger = logger.bind(dire_instance=id(self))
        self._logger_handler_ids = []

        if verbose:
            # Add handler that outputs to stderr with formatting
            handler_id = self.logger.add(
                sys.stderr,
                level="INFO",
                filter=lambda record: record["extra"].get("dire_instance") == id(self)
            )
            self._logger_handler_ids.append(handler_id)
        else:
            # Add null handler that discards all messages
            handler_id = self.logger.add(
                lambda msg: None,
                level="TRACE",
                filter=lambda record: record["extra"].get("dire_instance") == id(self)
            )
            self._logger_handler_ids.append(handler_id)

        # Internal state
        self._data = None
        self._layout = None
        self._n_samples = None
        self._a = None
        self._b = None
        self._knn_indices = None
        self._knn_distances = None

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
        n_samples = X.shape[0]
        n_dims = X.shape[1]
        self.logger.info(f"Computing {self.n_neighbors}-NN graph for {n_samples} points in {n_dims}D...")

        # ── cuVS fast path: use GPU-accelerated kNN for large datasets ──
        # cuVS IVF-Flat is much faster than brute-force when N is large
        # relative to D.  For high-D data (D>200) the index build cost is
        # higher, so we raise the threshold.  Empirically:
        #   D<200:  cuVS wins at N >= 100K  (covertype 54D: 7s→2.5s at 200K)
        #   D>=200: cuVS wins at N >= 200K  (mnist 784D: slower at 50K)
        cuvs_threshold = 200000 if n_dims >= 200 else 100000
        if (CUVS_AVAILABLE
                and self.device.type == 'cuda'
                and n_samples >= cuvs_threshold
                and n_dims <= 2048
                and self._metric_fn is None):  # cuVS only supports built-in metrics
            try:
                return self._compute_knn_cuvs(X)
            except Exception as e:
                self.logger.warning(f"cuVS kNN failed ({e}), falling back to PyTorch")

        # Auto-detect FP16 usage based on data size and GPU
        if use_fp16 is None and self.device.type == 'cuda':
            # Use FP16 for high-dimensional data or large datasets
            use_fp16 = n_dims >= 500 or n_samples >= 100000
        elif self.device.type == 'cpu':
            use_fp16 = False  # CPU doesn't benefit from FP16
        
        # Choose precision
        if use_fp16 and self.device.type == 'cuda':
            dtype = torch.float16
            self.logger.info("Using FP16 for k-NN (2x memory, faster on H100/A100)")
        else:
            dtype = torch.float32
            self.logger.info("Using FP32 for k-NN")
        
        X_torch = torch.tensor(X, dtype=dtype, device=self.device)

        # CRITICAL: PyKeOps is slower than PyTorch for high dimensions!
        # Use PyTorch for high-D, PyKeOps for low-D
        use_pykeops = PYKEOPS_AVAILABLE and n_dims < 200 and self.device.type == 'cuda' and not use_fp16

        if n_dims >= 200:
            self.logger.info(f"Using PyTorch for k-NN (high dimension: {n_dims}D)")
        elif use_pykeops:
            self.logger.info("Using PyKeOps for k-NN (low dimension, GPU available)")
        else:
            self.logger.info("Using PyTorch for k-NN")

        # Set default chunk size if not provided
        if chunk_size is None:
            chunk_size = 50000

        # Adaptive chunk sizing based on available GPU memory
        if self.device.type == 'cuda':
            # Check available memory AFTER allocating X_torch
            gpu_mem_free = torch.cuda.mem_get_info()[0]
            # Estimate memory for k-NN: chunk_size * n_samples * bytes_per_element
            bytes_per_element = 2 if use_fp16 else 4  # FP16 uses 2 bytes, FP32 uses 4
            memory_per_chunk = chunk_size * n_samples * bytes_per_element

            # Only auto-adjust if using default chunk size
            if chunk_size == 50000:
                # Use conservative memory fraction: 20-25% of available memory
                # This accounts for PyTorch overhead, temp buffers, and fragmentation
                memory_fraction = 0.25 if use_fp16 else 0.20
                max_memory = gpu_mem_free * memory_fraction
                if memory_per_chunk > max_memory:
                    chunk_size = int(max_memory / (n_samples * bytes_per_element))
                    chunk_size = max(1000, chunk_size)  # Minimum chunk size

            self.logger.info(f"Using chunk size: {chunk_size} (GPU memory: {gpu_mem_free/1024**3:.1f}GB, dtype: {dtype})")
        
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

                # Remove self
                chunk_indices = knn_indices[:, 1:].cpu().numpy()
                # For custom metrics, distances are already in metric space
                # For Euclidean, convert from squared to actual distances
                if self._metric_fn is None:
                    chunk_distances = torch.sqrt(knn_dists[:, 1:]).cpu().numpy()
                else:
                    chunk_distances = knn_dists[:, 1:].cpu().numpy()
            else:
                # Use PyTorch for HIGH dimensional data (MUCH faster!)
                if self._metric_fn is not None:
                    # Custom metric - compute pairwise distances manually
                    # Broadcast: X_chunk: (chunk, 1, D), X_torch: (1, N, D) -> (chunk, N)
                    X_i = X_chunk.unsqueeze(1)  # (chunk, 1, D)
                    X_j = X_torch.unsqueeze(0)  # (1, N, D)
                    distances = self._metric_fn(X_i, X_j)  # (chunk, N)
                else:
                    # Fast built-in Euclidean distance
                    distances = torch.cdist(X_chunk, X_torch, p=2)

                knn_dists, knn_indices = torch.topk(distances, k=self.n_neighbors + 1,
                                                   dim=1, largest=False)

                # Remove self
                chunk_indices = knn_indices[:, 1:].cpu().numpy()
                chunk_distances = knn_dists[:, 1:].cpu().numpy()

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

        # Remove self (first neighbor) and convert squared distances to distances
        self._knn_indices = indices_np[:, 1:]
        self._knn_distances = np.sqrt(np.maximum(distances_np[:, 1:], 0.0))

        self.logger.info(f"k-NN graph computed via cuVS: shape {self._knn_indices.shape}")

        # Clean up
        del X_gpu, index
        cp.get_default_memory_pool().free_all_blocks()

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
                embedding = (U * S).cpu().numpy()
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

        # Pre-convert kNN indices to GPU tensor once (avoid re-creating every iteration)
        self._knn_indices_torch = torch.tensor(
            self._knn_indices, dtype=torch.long, device=self.device
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
        # Seed torch RNGs for reproducibility (deferred from __init__
        # to avoid mutating global state at construction time).
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        # Store data
        self._data = np.asarray(X, dtype=np.float32)
        self._n_samples = self._data.shape[0]

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
        self._compute_knn(self._data)

        # Initialize embedding
        initial_embedding = self._initialize_embedding(self._data)

        # Optimize layout
        final_embedding = self._optimize_layout(initial_embedding)

        # Convert back to numpy and store
        self._layout = final_embedding.cpu().numpy()

        # Clear GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

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

    def visualize(self, labels=None, point_size=2, title=None, max_points=10000, **kwargs):
        """
        Create an interactive visualization of the embedding.

        Uses WebGL rendering (Scattergl) for performance and automatically
        subsamples to max_points if dataset is larger.

        Parameters
        ----------
        labels : array-like of shape (n_samples,), optional
            Labels for coloring points in the visualization.
        point_size : int, default=2
            Size of points in the scatter plot.
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

        if title is None:
            title = f"PyTorch {self.n_components}D Embedding"

        # Subsample if needed
        n_points = self._layout.shape[0]
        if n_points > max_points:
            rng = np.random.default_rng(42)
            subsample_idx = rng.choice(n_points, max_points, replace=False)
            layout_vis = self._layout[subsample_idx]
            labels_vis = labels[subsample_idx] if labels is not None else None
        else:
            layout_vis = self._layout
            labels_vis = labels

        # Create dataframe
        if self.n_components == 2:
            df = pd.DataFrame(layout_vis, columns=['x', 'y'])
        elif self.n_components == 3:
            df = pd.DataFrame(layout_vis, columns=['x', 'y', 'z'])
        else:
            self.logger.error(f"Cannot visualize {self.n_components}D embedding")
            return None

        # Add labels if provided
        if labels_vis is not None:
            df['label'] = labels_vis

        # Create plot
        vis_params = {
            'color': 'label' if labels_vis is not None else None,
            'title': title,
        }
        vis_params.update(kwargs)

        if self.n_components == 2:
            fig = px.scatter(df, x='x', y='y', **vis_params)
            # Convert to WebGL for performance
            for trace in fig.data:
                trace.type = 'scattergl'
        else:
            fig = px.scatter_3d(df, x='x', y='y', z='z', **vis_params)

        fig.update_traces(marker={'size': point_size, 'opacity': 0.7})

        return fig


def create_dire(backend='auto', memory_efficient=False, **kwargs):
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

    **kwargs : dict
        Additional keyword arguments passed to the DiRe constructor.
        See individual backend documentation for available parameters.
        Common parameters include: n_components, n_neighbors, metric,
        max_iter_layout, min_dist, spread, verbose, random_state.

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

        # GPU processing with cuVS acceleration
        reducer = create_dire(backend='cuvs', use_cuvs=True)

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
    # Handle verbose parameter early to disable logging if needed
    verbose = kwargs.get('verbose', True)

    # Import here to avoid circular imports
    try:
        from .dire_cuvs import DiReCuVS, CUVS_AVAILABLE  # pylint: disable=import-outside-toplevel
    except ImportError:
        CUVS_AVAILABLE = False

    from .dire_pytorch_memory_efficient import DiRePyTorchMemoryEfficient  # pylint: disable=import-outside-toplevel

    if backend == 'auto':
        # Auto-select best backend based on availability
        if CUVS_AVAILABLE and torch.cuda.is_available():
            if verbose:
                logger.info("Auto-selected RAPIDS cuVS backend (GPU acceleration)")
            return DiReCuVS(use_cuvs=True, **kwargs)

        if torch.cuda.is_available():
            # When cuVS is not available, prefer memory-efficient backend for better GPU memory management
            if memory_efficient or not CUVS_AVAILABLE:
                if verbose:
                    logger.info("Auto-selected memory-efficient PyTorch backend (GPU)")
                return DiRePyTorchMemoryEfficient(**kwargs)
            if verbose:
                logger.info("Auto-selected PyTorch backend (GPU)")
            return DiRePyTorch(**kwargs)

        # CPU fallback
        if verbose:
            logger.info("Auto-selected PyTorch backend (CPU)")
        if memory_efficient:
            if verbose:
                logger.warning("Memory-efficient mode has limited benefits on CPU")
            return DiRePyTorchMemoryEfficient(**kwargs)
        return DiRePyTorch(**kwargs)

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
        return DiReCuVS(use_cuvs=True, **kwargs)

    if backend == 'pytorch':
        # Use PyTorch with auto device selection
        if memory_efficient:
            if verbose:
                logger.info("Using memory-efficient PyTorch backend")
            return DiRePyTorchMemoryEfficient(**kwargs)
        if verbose:
            logger.info("Using PyTorch backend")
        return DiRePyTorch(**kwargs)

    if backend == 'pytorch_gpu':
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA not available")
        if memory_efficient:
            if verbose:
                logger.info("Using memory-efficient PyTorch backend (GPU)")
            return DiRePyTorchMemoryEfficient(**kwargs)
        if verbose:
            logger.info("Using PyTorch backend (GPU)")
        return DiRePyTorch(**kwargs)

    if backend == 'pytorch_cpu':
        # Force CPU even if GPU is available
        if verbose:
            logger.info("Using PyTorch backend (forced CPU)")
        if memory_efficient:
            if verbose:
                logger.warning("Memory-efficient mode has limited benefits on CPU")
            # Create instance and force CPU
            reducer = DiRePyTorchMemoryEfficient(**kwargs)
        else:
            reducer = DiRePyTorch(**kwargs)
        reducer.device = torch.device('cpu')
        return reducer

    raise ValueError(
        f"Unknown backend: {backend}. "
        f"Choose from: 'auto', 'cuvs', 'pytorch', 'pytorch_gpu', 'pytorch_cpu'"
    )

