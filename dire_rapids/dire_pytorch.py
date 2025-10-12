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


def _compile_metric(spec):
    """
    Turn a metric spec into a callable metric(x, y) that returns a distance-like
    matrix with broadcasting:
      - Torch path:  x: (A, 1, D), y: (1, B, D)  -> (A, B) torch.Tensor
      - KeOps path:  x: LazyTensor(A,1,D), y: LazyTensor(1,B,D) -> LazyTensor(A,B)

    If spec is None or 'euclidean'/'l2', return None (fast-path Euclidean stays in backend).
    If spec is str, it's eval'ed with {'x': x, 'y': y} and no builtins.
    If spec is callable, it's returned unchanged.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        expr = spec.strip().lower()
        if expr in ("euclidean", "l2"):
            return None  # use built-in fast Euclidean
        def _expr_metric(x, y, _expr=spec):
            # Use ONLY tensor methods like .sum(-1), .sqrt(), .abs(), etc.
            # Works for both torch.Tensor and KeOps LazyTensor.
            return eval(_expr, {"__builtins__": {}}, {"x": x, "y": y})
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

        # Custom metric for k-NN only (layout forces remain Euclidean):
        self.metric_spec = metric
        self._metric_fn = _compile_metric(self.metric_spec)

        # Setup instance-specific logger
        # Create a completely new logger instance for this DiRe object
        self.logger = logger.bind(dire_instance=id(self))
        self.logger.remove()  # Remove all existing handlers
        
        if verbose:
            # Add handler that outputs to stderr with formatting
            self.logger.add(
                sys.stderr,
                level="INFO"
            )
        else:
            # Add null handler that discards all messages
            self.logger.add(lambda msg: None, level="TRACE")

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
            gpu_mem_free = torch.cuda.mem_get_info()[0]
            # Estimate memory for k-NN: chunk_size * n_samples * bytes_per_element
            bytes_per_element = 2 if use_fp16 else 4  # FP16 uses 2 bytes, FP32 uses 4
            memory_per_chunk = chunk_size * n_samples * bytes_per_element
            
            # Only auto-adjust if using default chunk size
            if chunk_size == 50000:
                # Use 30% of available memory for k-NN computation (40% for FP16 since it's more efficient)
                memory_fraction = 0.4 if use_fp16 else 0.3
                max_memory = gpu_mem_free * memory_fraction
                if memory_per_chunk > max_memory:
                    chunk_size = int(max_memory / (n_samples * bytes_per_element))
                    chunk_size = max(1000, chunk_size)  # Minimum chunk size
                
                # With FP16, we can use larger chunks!
                if use_fp16:
                    chunk_size = min(chunk_size * 2, 100000)  # Double chunk size for FP16
            
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
            
            if use_pykeops:
                # Use PyKeOps for LOW dimensional data
                # Ensure contiguity for PyKeOps
                X_i = LazyTensor(X_chunk[:, None, :].contiguous())  # (chunk_size, 1, D)
                X_j = LazyTensor(X_torch[None, :, :].contiguous())   # (1, N, D)
                
                # Compute squared distances
                D_ij = ((X_i - X_j) ** 2).sum(-1)  # (chunk_size, N) LazyTensor
                
                # Find k+1 nearest neighbors (including self)
                knn_dists, knn_indices = D_ij.Kmin_argKmin(K=self.n_neighbors + 1, dim=1)
                
                # Remove self and convert to actual distances
                chunk_indices = knn_indices[:, 1:].cpu().numpy()
                chunk_distances = torch.sqrt(knn_dists[:, 1:]).cpu().numpy()
            else:
                # Use PyTorch for HIGH dimensional data (MUCH faster!)
                distances = torch.cdist(X_chunk, X_torch, p=2)
                knn_dists, knn_indices = torch.topk(distances, k=self.n_neighbors + 1, 
                                                   dim=1, largest=False)
                
                # Remove self
                chunk_indices = knn_indices[:, 1:].cpu().numpy()  
                chunk_distances = knn_dists[:, 1:].cpu().numpy()
            
            all_knn_indices.append(chunk_indices)
            all_knn_distances.append(chunk_distances)
            
            # Clear GPU memory periodically
            if self.device.type == 'cuda' and start_idx % (chunk_size * 10) == 0:
                torch.cuda.empty_cache()
        
        # Concatenate results
        self._knn_indices = np.vstack(all_knn_indices)
        self._knn_distances = np.vstack(all_knn_distances)

        self.logger.info(f"k-NN graph computed: shape {self._knn_indices.shape}")

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
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            embedding = pca.fit_transform(X)

        elif self.init == 'random':
            self.logger.info("Initializing with random projection")
            rng = np.random.RandomState(self.random_state)  # pylint: disable=no-member
            projection = rng.randn(X.shape[1], self.n_components)
            projection /= np.linalg.norm(projection, axis=0)
            embedding = X @ projection

        else:
            raise ValueError(f"Unknown init method: {self.init}")

        # Normalize
        embedding -= embedding.mean(axis=0)
        embedding /= embedding.std(axis=0)

        return torch.tensor(embedding, dtype=torch.float32, device=self.device)

    def _compute_forces(self, positions, iteration, max_iterations, chunk_size=5000):  # pylint: disable=too-many-branches,too-many-locals
        """
        Compute attraction and repulsion forces for layout optimization.
        
        This private method computes forces between points in the embedding space
        using a memory-efficient chunked approach. Attraction forces are computed
        only between k-nearest neighbors, while repulsion forces use random sampling.
        
        Parameters
        ----------
        positions : torch.Tensor
            Current positions of points in embedding space, shape (n_samples, n_components).
        iteration : int
            Current iteration number (0-indexed).
        max_iterations : int
            Total number of iterations planned.
        chunk_size : int, default=5000
            Maximum chunk size for memory-efficient processing.
            
        Returns
        -------
        torch.Tensor
            Computed forces of shape (n_samples, n_components).
            
        Notes
        -----
        Private method, should not be called directly. Used by _optimize_layout().
        
        Force Computation:
        - **Attraction forces**: Applied only between k-nearest neighbors using
          the kernel function 1 / (1 + a * (1/d)^(2*b))
        - **Repulsion forces**: Applied between randomly sampled pairs using
          the kernel function -1 / (1 + a * d^(2*b)) with exponential cutoff
        - **Cooling**: Forces are scaled by (1 - iteration/max_iterations)
        - **Clipping**: Forces are clipped to [-cutoff, cutoff] range
        """
        # PyKeOps is optional - we can use pure PyTorch
        # if not PYKEOPS_AVAILABLE:
        #     raise RuntimeError("PyKeOps required for efficient force computation")

        n_samples = positions.shape[0]
        forces = torch.zeros_like(positions)

        # Linear cooling schedule
        alpha = 1.0 - iteration / max_iterations

        # Parameters
        a_val = float(self._a)
        b_val = float(self._b)
        b_exp = float(2 * b_val)
        
        # Adjust chunk size based on available memory
        # Estimate memory usage: chunk_size * (k + n_neg) * D * 4 bytes
        n_neg_samples = min(int(self.neg_ratio * self.n_neighbors), n_samples - 1)
        n_dims = positions.shape[1]
        
        # Memory estimate per point (in bytes):
        # k*D*4 for attraction + n_neg*D*4 for repulsion
        bytes_per_float = 4
        memory_per_point = (self.n_neighbors + n_neg_samples) * n_dims * bytes_per_float * 2  # x2 for safety
        
        if self.device.type == 'cuda':
            # Get available GPU memory and use 20% for force computation (more conservative)
            gpu_mem_free = torch.cuda.mem_get_info()[0]
            max_chunk_size = int(gpu_mem_free * 0.2 / memory_per_point)
            chunk_size = min(chunk_size, max_chunk_size, n_samples)
            # For very large datasets, be extra conservative
            if n_samples > 500000:
                chunk_size = min(chunk_size, 2000)
            # Ensure reasonable chunk size
            chunk_size = max(100, min(chunk_size, 5000))  # Between 100 and 5000
            self.logger.debug(f"Using chunk size: {chunk_size} (available memory: {gpu_mem_free/1e9:.1f} GB)")
        else:
            chunk_size = min(1000, n_samples)  # Smaller chunks for CPU

        # Process in chunks to manage memory
        knn_indices_torch = torch.tensor(self._knn_indices, device=self.device)
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            chunk_indices = slice(start_idx, end_idx)
            chunk_size_actual = end_idx - start_idx
            
            # ============ ATTRACTION FORCES (k-NN only) ============
            # Get chunk data
            chunk_positions = positions[chunk_indices]  # (chunk, D)
            chunk_knn_indices = knn_indices_torch[chunk_indices]  # (chunk, k)
            
            # Check if we have enough memory for vectorized computation
            attraction_memory = chunk_size_actual * self.n_neighbors * n_dims * bytes_per_float
            
            try:
                if self.device.type == 'cuda':
                    gpu_mem_free = torch.cuda.mem_get_info()[0]
                    if attraction_memory > gpu_mem_free * 0.4:
                        raise RuntimeError("Not enough memory for vectorized attraction")
                
                # Try vectorized version (faster)
                neighbor_positions = positions[chunk_knn_indices]  # (chunk, k, D)
                current_positions = chunk_positions.unsqueeze(1)  # (chunk, 1, D)
                
                # Compute differences and distances
                diff = neighbor_positions - current_positions  # (chunk, k, D)
                dist = torch.norm(diff, dim=2, keepdim=True) + 1e-10  # (chunk, k, 1)
                
                # Attraction kernel
                att_coeff = 1.0 / (1.0 + a_val * (1.0 / dist) ** b_exp)  # (chunk, k, 1)
                
                # Compute attraction forces and sum over neighbors
                att_forces = (att_coeff * diff / dist).sum(dim=1)  # (chunk, D)
                forces[chunk_indices] += att_forces
                
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                # Fall back to point-by-point if memory is tight
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                self.logger.debug("Falling back to point-by-point attraction due to memory constraints")
                
                for i in range(start_idx, end_idx):
                    neighbor_ids = self._knn_indices[i]
                    pos_i = positions[i:i+1]
                    pos_neighbors = positions[neighbor_ids]
                    
                    diff = pos_neighbors - pos_i
                    dist = torch.norm(diff, dim=1, keepdim=True) + 1e-10
                    att_coeff = 1.0 / (1.0 + a_val * (1.0 / dist) ** b_exp)
                    forces[i] += (att_coeff * diff / dist).sum(0)

            # ============ REPULSION FORCES (Random Sampling) ============
            if n_neg_samples > 0:
                # Check memory for repulsion
                repulsion_memory = chunk_size_actual * n_neg_samples * n_dims * bytes_per_float
                
                try:
                    if self.device.type == 'cuda':
                        gpu_mem_free = torch.cuda.mem_get_info()[0]
                        if repulsion_memory > gpu_mem_free * 0.4:
                            raise RuntimeError("Not enough memory for vectorized repulsion")
                    
                    # Try vectorized version
                    # Generate random samples for this chunk
                    neg_indices = torch.randint(0, n_samples, (chunk_size_actual, n_neg_samples + 5), 
                                              device=self.device)
                    
                    # Create mask to exclude points from the current chunk
                    chunk_range = torch.arange(start_idx, end_idx, device=self.device)
                    self_mask = neg_indices == chunk_range.unsqueeze(1)
                    
                    # Replace self indices with valid random ones
                    replacement_indices = torch.randint(0, n_samples, (chunk_size_actual, n_neg_samples + 5), 
                                                      device=self.device)
                    neg_indices = torch.where(self_mask, replacement_indices, neg_indices)
                    
                    # Take first n_neg_samples
                    neg_indices = neg_indices[:, :n_neg_samples]
                    
                    # Get negative sample positions
                    neg_positions = positions[neg_indices]  # (chunk, n_neg, D)
                    current_positions = chunk_positions.unsqueeze(1)  # (chunk, 1, D)
                    
                    # Compute differences and distances
                    diff = neg_positions - current_positions  # (chunk, n_neg, D)
                    dist = torch.norm(diff, dim=2, keepdim=True) + 1e-10  # (chunk, n_neg, 1)
                    
                    # Repulsion kernel
                    rep_coeff = -1.0 / (1.0 + a_val * (dist ** b_exp))  # (chunk, n_neg, 1)
                    
                    # Apply distance cutoff
                    cutoff_scale = torch.exp(-dist / self.cutoff)
                    rep_coeff = rep_coeff * cutoff_scale
                    
                    # Compute repulsion forces and sum over negative samples
                    rep_forces = (rep_coeff * diff / dist).sum(dim=1)  # (chunk, D)
                    forces[chunk_indices] += rep_forces
                    
                except (RuntimeError, torch.cuda.OutOfMemoryError):
                    # Fall back to point-by-point
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    self.logger.debug("Falling back to point-by-point repulsion due to memory constraints")
                    
                    for i in range(start_idx, end_idx):
                        # Random sample for repulsion
                        neg_samples = np.random.choice(n_samples, min(n_neg_samples, n_samples-1), replace=False)
                        neg_samples = neg_samples[neg_samples != i][:n_neg_samples]  # Exclude self
                        
                        if len(neg_samples) > 0:
                            pos_i = positions[i:i+1]
                            pos_neg = positions[neg_samples]
                            
                            # Compute differences and distances
                            diff = pos_neg - pos_i
                            dist = torch.norm(diff, dim=1, keepdim=True) + 1e-10
                            
                            # Repulsion kernel
                            rep_coeff = -1.0 / (1.0 + a_val * (dist ** b_exp))
                            
                            # Apply distance cutoff
                            cutoff_scale = torch.exp(-dist / self.cutoff)
                            rep_coeff = rep_coeff * cutoff_scale
                            
                            # Apply force
                            forces[i] += (rep_coeff * diff / dist).sum(0)

        # Apply cooling and clipping
        forces = alpha * forces
        forces = torch.clamp(forces, -self.cutoff, self.cutoff)

        return forces

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
        
        The optimization uses a simple force-directed layout algorithm with
        linear cooling schedule.
        """
        positions = initial_positions.clone()

        self.logger.info(f"Optimizing layout for {self._n_samples} points...")

        # Optimization loop
        for iteration in range(self.max_iter_layout):
            # Compute forces correctly
            forces = self._compute_forces(positions, iteration, self.max_iter_layout)

            # Update positions
            positions += forces

            # Log progress every 20th iteration
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
        # Store data
        self._data = np.asarray(X, dtype=np.float32)
        self._n_samples = self._data.shape[0]

        self.logger.info(f"Processing {self._n_samples} samples with {self._data.shape[1]} features")

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
            rng = np.random.RandomState(42)
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
    2. PyTorch with CUDA (if CUDA GPU available)
    3. PyTorch with CPU (fallback)

    The function automatically handles import errors and missing dependencies,
    falling back to available alternatives when possible.
    """
    # Handle verbose parameter early to disable logging if needed
    verbose = kwargs.get('verbose', True)

    # Import here to avoid circular imports
    try:
        from .dire_cuvs import DiReCuVS
        CUVS_AVAILABLE = True
    except ImportError:
        CUVS_AVAILABLE = False

    from .dire_pytorch_memory_efficient import DiRePyTorchMemoryEfficient

    if backend == 'auto':
        # Auto-select best backend based on availability
        if CUVS_AVAILABLE and torch.cuda.is_available():
            if verbose:
                logger.info("Auto-selected RAPIDS cuVS backend (GPU acceleration)")
            return DiReCuVS(use_cuvs=True, **kwargs)

        if torch.cuda.is_available():
            if memory_efficient:
                if verbose:
                    logger.info("Auto-selected memory-efficient PyTorch backend (GPU)")
                return DiRePyTorchMemoryEfficient(**kwargs)
            else:
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

    elif backend == 'cuvs':
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

    elif backend == 'pytorch':
        # Use PyTorch with auto device selection
        if memory_efficient:
            if verbose:
                logger.info(f"Using memory-efficient PyTorch backend")
            return DiRePyTorchMemoryEfficient(**kwargs)
        else:
            if verbose:
                logger.info(f"Using PyTorch backend")
            return DiRePyTorch(**kwargs)

    elif backend == 'pytorch_gpu':
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA not available")
        if memory_efficient:
            if verbose:
                logger.info("Using memory-efficient PyTorch backend (GPU)")
            return DiRePyTorchMemoryEfficient(**kwargs)
        else:
            if verbose:
                logger.info("Using PyTorch backend (GPU)")
            return DiRePyTorch(**kwargs)

    elif backend == 'pytorch_cpu':
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

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Choose from: 'auto', 'cuvs', 'pytorch', 'pytorch_gpu', 'pytorch_cpu'"
        )


# ============================================================================
# ReducerRunner Framework - Benchmarking and Dataset Loading Utilities
# ============================================================================

import inspect
import os
import re
import time
import gzip
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

from IPython.display import display, HTML  # for Colab/Notebook visualize

from sklearn import datasets as skds

try:
    from scipy import sparse as sp
except Exception:
    sp = None  # sklearn normally pulls scipy in; keep soft guard

TransformFn = Callable[[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, Optional[np.ndarray]]]


def _identity_transform(X: np.ndarray, y: Optional[np.ndarray]):
    return X, y


# --------- minimal display helpers (so .visualize renders in Colab) ---------

def _safe_init_plotly_renderer():
    try:
        import plotly.io as pio
        if pio.renderers.default in (None, "auto"):
            try:
                import google.colab  # noqa: F401
                pio.renderers.default = "colab"
            except Exception:
                pio.renderers.default = "notebook_connected"
    except Exception:
        pass

def _display_obj(obj) -> bool:
    if obj is None:
        return False
    if isinstance(obj, (list, tuple)):
        shown = False
        for it in obj:
            shown = _display_obj(it) or shown
        return shown
    # Plotly
    try:
        import plotly.graph_objects as go
        if isinstance(obj, go.Figure):
            _safe_init_plotly_renderer()
            obj.show()
            return True
    except Exception:
        pass
    # Matplotlib
    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.axes import Axes
        if isinstance(obj, (Figure, Axes)):
            plt.show()
            return True
    except Exception:
        pass
    # HTML / str
    if isinstance(obj, (str, bytes)):
        s = obj.decode("utf-8", "ignore") if isinstance(obj, bytes) else obj
        if "<" in s and ">" in s:
            display(HTML(s))
        else:
            print(s)
        return True
    try:
        display(obj)
        return True
    except Exception:
        return False


# --------- sklearn resolution ---------

_SKLEARN_ALIASES = {
    # loaders
    "iris": "load_iris",
    "digits": "load_digits",
    "wine": "load_wine",
    "breast_cancer": "load_breast_cancer",
    "diabetes": "load_diabetes",
    "linnerud": "load_linnerud",
    # generators
    "blobs": "make_blobs",
    "classification": "make_classification",
    "multilabel_classification": "make_multilabel_classification",
    "moons": "make_moons",
    "circles": "make_circles",
    "s_curve": "make_s_curve",
    "swiss_roll": "make_swiss_roll",
    "gaussian_quantiles": "make_gaussian_quantiles",
    "low_rank_matrix": "make_low_rank_matrix",
    "spd_matrix": "make_spd_matrix",
    "sparse_spd_matrix": "make_sparse_spd_matrix",
}

def _normalize_key(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.strip().lower())

def _resolve_sklearn_function(name: str):
    n = _normalize_key(name)
    if n.startswith(("load_", "fetch_", "make_")):
        fn = getattr(skds, n, None)
        if callable(fn):
            return n, fn
    alias = _SKLEARN_ALIASES.get(n)
    if alias and callable(getattr(skds, alias, None)):
        return alias, getattr(skds, alias)
    for pref in ("load_", "fetch_", "make_"):
        cand = pref + n
        fn = getattr(skds, cand, None)
        if callable(fn):
            return cand, fn
    candidates = [
        (attr, getattr(skds, attr))
        for attr in dir(skds)
        if attr.lower().endswith(n) and callable(getattr(skds, attr))
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        names = ", ".join(a for a, _ in candidates[:6])
        raise ValueError(f"Ambiguous sklearn dataset '{name}'. Candidates: {names} ...")
    all_names = ", ".join(a for a in dir(skds) if a.startswith(("load_", "fetch_", "make_")))
    raise ValueError(f"Unknown sklearn dataset '{name}'. Available include: {all_names}")


def _to_Xy_from_obj(obj) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if isinstance(obj, (tuple, list)) and len(obj) >= 1:
        X = obj[0]
        y = obj[1] if len(obj) > 1 else None
        return _coerce_Xy(X, y)
    if hasattr(obj, "get"):
        data = obj.get("data", None)
        target = obj.get("target", None)
        images = obj.get("images", None)
        if data is None and images is not None:
            imgs = np.asarray(images)
            data = imgs.reshape(len(imgs), -1)
        return _coerce_Xy(data, target)
    if hasattr(obj, "shape"):
        return _coerce_Xy(obj, None)
    raise ValueError("Unsupported sklearn return type; cannot coerce to (X, y).")


def _coerce_Xy(X, y) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if isinstance(X, list) and X and isinstance(X[0], str):
        raise TypeError("Loaded dataset contains text data; vectorize first.")
    if sp is not None and sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if y is None:
        return X, None
    y = np.asarray(y)
    if y.dtype.kind in {"U", "S", "O"}:
        uniq = {v: i for i, v in enumerate(np.unique(y))}
        y = np.array([uniq[v] for v in y], dtype=np.int32)
    return X, y


def _load_sklearn_any(name: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    _, fn = _resolve_sklearn_function(name)
    try:
        sig = inspect.signature(fn)
        if "return_X_y" in sig.parameters:
            obj = fn(return_X_y=True, **kwargs)
            X, y = _to_Xy_from_obj(obj)
        else:
            obj = fn(**kwargs)
            X, y = _to_Xy_from_obj(obj)
    except TypeError:
        obj = fn()
        X, y = _to_Xy_from_obj(obj)
    return X, y


# --------- file loader ---------

def _load_file(path: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    path = str(path)
    ext = Path(path).suffix.lower()

    if ext == ".csv":
        df = pd.read_csv(path)
        label_col = kwargs.pop("label_column", None)
        if label_col and label_col in df.columns:
            y = df[label_col].to_numpy()
            X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
        else:
            y = None
            X = df.to_numpy(dtype=np.float32)
        return X, y

    if ext == ".parquet":
        df = pd.read_parquet(path)
        label_col = kwargs.pop("label_column", None)
        if label_col and label_col in df.columns:
            y = df[label_col].to_numpy()
            X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
        else:
            y = None
            X = df.to_numpy(dtype=np.float32)
        return X, y

    if ext == ".npy":
        X = np.load(path, mmap_mode="r")
        y = None
        labels_path = kwargs.pop("labels_path", None)
        if labels_path:
            y = np.load(labels_path, mmap_mode="r")
        return np.asarray(X, dtype=np.float32), y

    if ext == ".npz":
        f = np.load(path, mmap_mode="r")
        if "X" not in f:
            raise ValueError(".npz must contain key 'X' (and optionally 'y').")
        X = np.asarray(f["X"], dtype=np.float32)
        y = f["y"] if "y" in f else None
        return X, y

    raise ValueError(f"Unsupported file type '{ext}'. Use .csv, .npy, .npz, or .parquet.")


# --------- DiRe geometric datasets ---------

def rand_point_disk(n_features, n_samples=1):
    """Generate uniformly distributed points in n-dimensional unit disk."""
    prepts = np.random.randn(n_samples, n_features)
    prenorms = np.linalg.norm(prepts, axis=1).reshape(-1, 1)
    rads = np.sqrt(np.random.rand(n_samples)).reshape(-1, 1)
    pts = prepts * rads / prenorms
    return pts


def rand_point_sphere(n_features, n_samples=1):
    """Generate uniformly distributed points on n-dimensional unit sphere."""
    prepts = np.random.randn(n_samples, n_features)
    prenorms = np.linalg.norm(prepts, axis=1).reshape(-1, 1)
    pts = prepts / prenorms
    return pts


class elgen:
    """Ellipsoid generator - transforms sphere points to ellipsoid."""
    def __init__(self, a):
        a = np.array(a)
        themat = np.diag(1 / (a * a))
        L = np.linalg.inv(np.linalg.cholesky(themat).T)
        self.L = L

    def __call__(self, ar):
        return (self.L @ ar.T).T


def rand_point_ell(semi_axes, n_features, n_samples=1):
    """Generate uniformly distributed points on n-dimensional ellipsoid with semi-axes."""
    spts = rand_point_sphere(n_features, n_samples)
    eg = elgen(semi_axes)
    return eg(spts)


def _load_dire_dataset(name: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load DiRe geometric datasets.

    Supported:
    - 'disk_uniform': Uniform in n-dimensional unit disk
    - 'sphere_uniform': Uniform on n-dimensional unit sphere
    - 'ellipsoid_uniform': Uniform on n-dimensional ellipsoid

    Options:
    - n_samples (default 1000)
    - n_features (default 10)
    - semi_axes (for ellipsoid, default [1, 2, ..., n])
    - random_state
    """
    key = _normalize_key(name)

    n_samples = kwargs.pop('n_samples', 1000)
    n_features = kwargs.pop('n_features', 10)
    random_state = kwargs.pop('random_state', None)

    if random_state is not None:
        np.random.seed(random_state)

    if key == 'disk_uniform':
        X = rand_point_disk(n_features, n_samples)
    elif key == 'sphere_uniform':
        X = rand_point_sphere(n_features, n_samples)
    elif key == 'ellipsoid_uniform':
        semi_axes = kwargs.pop('semi_axes', None)
        if semi_axes is not None:
            n_features = len(semi_axes)  # Infer n_features from semi_axes
        else:
            semi_axes = list(range(1, n_features + 1))  # Default semi_axes
        X = rand_point_ell(semi_axes, n_features, n_samples)
    else:
        raise ValueError(
            f"Unknown DiRe dataset '{name}'. Options: 'disk_uniform', 'sphere_uniform', 'ellipsoid_uniform'"
        )

    return X.astype(np.float32), None


# --------- cytof scheme (Levine13/32) ---------

_DEF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "reducer_runner", "cytof")
os.makedirs(_DEF_CACHE, exist_ok=True)

def _download(url: str, dest: str, *, overwrite=False):
    if (not overwrite) and os.path.exists(dest):
        return dest
    tmp = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest

def _safe_gunzip(path: str) -> str:
    if path.endswith(".gz"):
        out = path[:-3]
        if not os.path.exists(out):
            with gzip.open(path, "rb") as f_in, open(out, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return out
    return path

_CYTOF_REGISTRY = {
    "levine13": {
        "urls": [
            "https://raw.githubusercontent.com/lmweber/benchmark-data-Levine-13-dim/master/data/Levine_13dim.fcs",
            "https://raw.githubusercontent.com/lmweber/benchmark-data-Levine-13-dim/master/data/Levine_13dim.txt",
        ],
        "label_column": "label",
        "drop_columns": ("label", "individual"),
    },
    "levine32": {
        "urls": [
            "https://raw.githubusercontent.com/lmweber/benchmark-data-Levine-32-dim/master/data/Levine_32dim.fcs",
        ],
        "label_column": "label",
        "drop_columns": ("label", "individual"),
    },
}

def _load_cytof(name: str, **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    CyTOF loader:
      - 'levine13'
      - 'levine32'
    via built-in URLs/caching
    Supports .txt/.tsv/.csv (pandas).
    Options:
      - url / file / cache_dir
      - label_column (for txt/csv/tsv)
      - drop_columns
      - arcsinh_cofactor (if raw)
    """

    key = _normalize_key(name)
    spec = _CYTOF_REGISTRY.get(key)
    if spec is None:
        raise ValueError(f"Unknown cytof dataset '{name}'. Options: {tuple(_CYTOF_REGISTRY.keys())}")

    cache_dir = kwargs.pop("cache_dir", _DEF_CACHE)
    url = kwargs.pop("url", None)
    label_col = kwargs.pop("label_column", spec.get("label_column", "label"))
    drop_cols = tuple(kwargs.pop("drop_columns", spec.get("drop_columns", (label_col,))))
    drop_unassigned = bool(kwargs.pop("drop_unassigned", False))
    arcsinh_cofactor = kwargs.pop("arcsinh_cofactor", None)
    local_path = kwargs.pop("file", None)

    # Resolve local or download
    if local_path is None:
        urls = [url] if url else spec.get("urls", [])
        if not urls:
            raise ValueError(f"cytof:{name} requires 'url' or local 'file' path.")
        last_err = None
        for u in urls:
            try:
                fname = os.path.join(cache_dir, os.path.basename(u.split("?")[0]))
                local_path = _download(u, fname)
                break
            except Exception as e:
                last_err = e
                local_path = None
        if local_path is None:
            raise RuntimeError(f"Failed to download cytof:{name}: {last_err}")

    path = _safe_gunzip(local_path)
    ext = Path(path).suffix.lower()

    # ---------- FCS via flowio ----------
    if ext == ".fcs":
        try:
            import flowio
        except ImportError:
            raise ImportError("flowio required for FCS files. Install with: pip install flowio")

        fcs = flowio.FlowData(path)
        data = fcs.as_array()  # Get 2D numpy array with preprocessing

        # Get channel names from pnn_labels (parameter names)
        channel_names = fcs.pnn_labels if fcs.pnn_labels else [f'Ch{i}' for i in range(fcs.channel_count)]

        # Create DataFrame from FCS data
        df = pd.DataFrame(data, columns=channel_names)

        # Drop rows with null labels if requested
        if drop_unassigned and label_col in df.columns:
            before = len(df)
            df = df[df[label_col].notna()].copy()
            after = len(df)
            print(f"[cytof] dropped {before - after} rows with null labels")

        y = df[label_col].to_numpy() if label_col in df.columns else None
        drop = [c for c in drop_cols if c in df.columns]
        Xdf = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
        X = Xdf.to_numpy(dtype=np.float32, copy=False)

        if (arcsinh_cofactor is not None) and arcsinh_cofactor > 0:
            X = np.arcsinh(X / float(arcsinh_cofactor)).astype(np.float32)

        # map string labels to ints
        if y is not None:
            y = np.asarray(y)
            if y.dtype.kind in {"U", "S", "O"}:
                uniq = {v: i for i, v in enumerate(np.unique(y))}
                y = np.array([uniq[v] for v in y], dtype=np.int32)
            elif y.dtype.kind == "f":  # floating point labels
                y = y.astype(np.int32)

        return X, y

    # ---------- TXT/TSV/CSV via pandas ----------
    if ext in (".txt", ".tsv", ".csv"):
        sep = "\t" if ext in (".txt", ".tsv") else ","
        df = pd.read_csv(path, sep=sep)

        # Drop rows with null labels if requested
        if drop_unassigned and label_col in df.columns:
            before = len(df)
            df = df[df[label_col].notna()].copy()
            after = len(df)
            print(f"[cytof] dropped {before - after} rows with null labels")

        y = df[label_col].to_numpy() if label_col in df.columns else None
        drop = [c for c in drop_cols if c in df.columns]
        Xdf = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
        X = Xdf.to_numpy(dtype=np.float32, copy=False)

        if (arcsinh_cofactor is not None) and arcsinh_cofactor > 0:
            X = np.arcsinh(X / float(arcsinh_cofactor)).astype(np.float32)

        # map string labels to ints
        if y is not None:
            y = np.asarray(y)
            if y.dtype.kind in {"U", "S", "O"}:
                uniq = {v: i for i, v in enumerate(np.unique(y))}
                y = np.array([uniq[v] for v in y], dtype=np.int32)
            elif y.dtype.kind == "f":  # floating point labels
                y = y.astype(np.int32)

        return X, y

    raise ValueError(f"Unsupported cytof file: {path} (use .fcs, .txt/.tsv, or .csv)")



# --------- ReducerConfig ---------

@dataclass
class ReducerConfig:
    """
    Configuration for a dimensionality reduction algorithm.

    All fields are mutable and can be changed after creation:
        config.visualize = True
        config.categorical_labels = False
        config.max_points = 20000
    """
    name: str
    reducer_class: type
    reducer_kwargs: Dict[str, Any]
    visualize: bool = False
    categorical_labels: bool = True  # False for regression-style labels (swiss_roll, etc.)
    max_points: int = 10000  # Max points for visualization (subsamples if larger)


# --------- selector parsing ---------

def _parse_selector(selector: str) -> Tuple[str, str]:
    s = selector.strip()
    p = Path(s)
    if p.exists() or re.search(r"\.(csv|np[yz]|parquet)$", s, re.I):
        return "file", s
    m = re.match(r"^(?P<scheme>[A-Za-z0-9_]+)[:\.](?P<name>.+)$", s)
    if m:
        return m.group("scheme").lower(), m.group("name").strip()
    return "sklearn", s


# --------- Runner ---------

@dataclass
class ReducerRunner:
    """
    General-purpose runner for dimensionality reduction algorithms.

    Supports:
    - DiRe (create_dire, DiRePyTorch, DiRePyTorchMemoryEfficient, DiReCuVS)
    - cuML (UMAP, TSNE)
    - scikit-learn (any TransformerMixin-compatible class)

    Parameters
    ----------
    config : ReducerConfig
        Configuration object containing reducer_class, reducer_kwargs, name, and visualize flag.
    default_transform : callable
        Default transform to apply to data before reduction
    """
    config: ReducerConfig
    default_transform: TransformFn = _identity_transform

    def __post_init__(self):
        """Validate that config is provided."""
        if self.config is None:
            raise ValueError("Must provide 'config' (ReducerConfig)")

    def _get_reducer_info(self) -> Tuple[str, type, Dict[str, Any], bool, bool, int]:
        """Extract reducer info from config."""
        return (
            self.config.name,
            self.config.reducer_class,
            self.config.reducer_kwargs,
            self.config.visualize,
            self.config.categorical_labels,
            self.config.max_points
        )

    def run(
        self,
        dataset: str,
        *,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
        transform: Optional[TransformFn] = None,
    ) -> Dict[str, Any]:
        """
        Run dimensionality reduction on specified dataset.

        Parameters
        ----------
        dataset : str
            Dataset selector (sklearn:name, openml:name, cytof:name, dire:name, file:path)
        dataset_kwargs : dict, optional
            Arguments for dataset loader
        transform : callable, optional
            Custom transform function (X, y) -> (X', y')

        Returns
        -------
        dict
            Results containing:
            - embedding: reduced data
            - labels: data labels
            - reducer: fitted reducer instance
            - fit_time_sec: time taken for fit_transform
            - dataset_info: dataset metadata
        """
        # Get reducer configuration
        reducer_name, reducer_class, reducer_kwargs, should_visualize, categorical_labels, max_points = self._get_reducer_info()

        scheme, name = _parse_selector(dataset)
        dataset_kwargs = dataset_kwargs or {}

        if scheme == "sklearn":
            X, y = _load_sklearn_any(name, **dataset_kwargs)
        elif scheme == "file":
            X, y = _load_file(name, **dataset_kwargs)
        elif scheme == "openml":
            from sklearn.datasets import fetch_openml
            try:
                data_id = int(str(name))
                ds = fetch_openml(data_id=data_id, return_X_y=True, **dataset_kwargs)
            except Exception:
                ds = fetch_openml(name=name, return_X_y=True, **dataset_kwargs)
            X, y = _coerce_Xy(ds[0], ds[1])
        elif scheme == "cytof":
            X, y = _load_cytof(name, **dataset_kwargs)
        elif scheme == "dire":
            X, y = _load_dire_dataset(name, **dataset_kwargs)
        else:
            raise ValueError(f"Unsupported scheme '{scheme}'. Use 'sklearn', 'openml', 'cytof', 'dire', 'file'.")

        T = transform or self.default_transform
        X, y = T(X, y)

        # Instantiate reducer (handles both classes and factory functions)
        if callable(reducer_class):
            reducer = reducer_class(**reducer_kwargs)
        else:
            raise TypeError(f"reducer_class must be callable, got {type(reducer_class)}")

        t0 = time.perf_counter()
        embedding = reducer.fit_transform(X)
        t1 = time.perf_counter()

        # Handle visualization
        if should_visualize:
            # Only use ReducerRunner's plotly visualization (not the reducer's built-in visualize)
            n_dims = embedding.shape[1] if len(embedding.shape) > 1 else 1
            if n_dims in (2, 3):
                try:
                    self._visualize_with_plotly(embedding, y, reducer_name, n_dims, categorical_labels, max_points)
                except Exception as e:
                    print(f"[WARNING] plotly visualization failed: {e}")

        return {
            "embedding": embedding,
            "labels": y,
            "reducer": reducer,
            "fit_time_sec": float(t1 - t0),
            "dataset_info": {
                "selector": dataset,
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
            },
        }

    def _visualize_with_plotly(
        self,
        embedding: np.ndarray,
        labels: Optional[np.ndarray],
        title: str,
        n_dims: int,
        categorical_labels: bool = True,
        max_points: int = 10000
    ):
        """
        Create and display plotly visualization for 2D or 3D embeddings.

        Uses WebGL rendering (Scattergl) for performance. Automatically subsamples
        to max_points if dataset is larger.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("[WARNING] plotly not installed. Install with: pip install plotly")
            return

        _safe_init_plotly_renderer()

        n_points = embedding.shape[0]

        # Subsample if needed
        if n_points > max_points:
            rng = np.random.RandomState(42)
            subsample_idx = rng.choice(n_points, max_points, replace=False)
            embedding_vis = embedding[subsample_idx]
            labels_vis = labels[subsample_idx] if labels is not None else None
        else:
            embedding_vis = embedding
            labels_vis = labels

        if n_dims == 2:
            # Use Scattergl for WebGL acceleration
            if labels_vis is not None:
                if not categorical_labels:
                    fig = go.Figure(data=go.Scattergl(
                        x=embedding_vis[:, 0],
                        y=embedding_vis[:, 1],
                        mode='markers',
                        marker=dict(
                            size=4,
                            color=labels_vis,
                            colorscale='Viridis',
                            colorbar=dict(title="Label Value"),
                            showscale=True,
                            opacity=0.8
                        )
                    ))
                else:
                    unique_labels = np.unique(labels_vis)

                    if len(unique_labels) > 20:
                        label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
                        colors = np.array([label_to_idx[lbl] for lbl in labels_vis])

                        fig = go.Figure(data=go.Scattergl(
                            x=embedding_vis[:, 0],
                            y=embedding_vis[:, 1],
                            mode='markers',
                            marker=dict(
                                size=4,
                                color=colors,
                                colorscale='Viridis',
                                showscale=True,
                                opacity=0.8
                            ),
                            text=[f"Label: {lbl}" for lbl in labels_vis],
                            hovertemplate='%{text}<extra></extra>'
                        ))
                    else:
                        fig = go.Figure()
                        for label in unique_labels:
                            mask = labels_vis == label
                            fig.add_trace(go.Scattergl(
                                x=embedding_vis[mask, 0],
                                y=embedding_vis[mask, 1],
                                mode='markers',
                                name=str(label),
                                marker=dict(size=4, opacity=0.8)
                            ))
            else:
                fig = go.Figure(data=go.Scattergl(
                    x=embedding_vis[:, 0],
                    y=embedding_vis[:, 1],
                    mode='markers',
                    marker=dict(size=4, opacity=0.7)
                ))

            fig.update_layout(
                title=f"{title} - 2D Embedding",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                width=800,
                height=600,
                hovermode='closest'
            )

        elif n_dims == 3:
            if labels_vis is not None:
                if not categorical_labels:
                    fig = go.Figure(data=go.Scatter3d(
                        x=embedding_vis[:, 0],
                        y=embedding_vis[:, 1],
                        z=embedding_vis[:, 2],
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=labels_vis,
                            colorscale='Viridis',
                            colorbar=dict(title="Label Value"),
                            showscale=True,
                            opacity=0.8
                        )
                    ))
                else:
                    unique_labels = np.unique(labels_vis)

                    if len(unique_labels) > 20:
                        label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
                        colors = np.array([label_to_idx[lbl] for lbl in labels_vis])

                        fig = go.Figure(data=go.Scatter3d(
                            x=embedding_vis[:, 0],
                            y=embedding_vis[:, 1],
                            z=embedding_vis[:, 2],
                            mode='markers',
                            marker=dict(
                                size=2,
                                color=colors,
                                colorscale='Viridis',
                                showscale=True,
                                opacity=0.8
                            ),
                            text=[f"Label: {lbl}" for lbl in labels_vis],
                            hovertemplate='%{text}<extra></extra>'
                        ))
                    else:
                        fig = go.Figure()
                        for label in unique_labels:
                            mask = labels_vis == label
                            fig.add_trace(go.Scatter3d(
                                x=embedding_vis[mask, 0],
                                y=embedding_vis[mask, 1],
                                z=embedding_vis[mask, 2],
                                mode='markers',
                                name=str(label),
                                marker=dict(size=2, opacity=0.8)
                            ))
            else:
                fig = go.Figure(data=go.Scatter3d(
                    x=embedding_vis[:, 0],
                    y=embedding_vis[:, 1],
                    z=embedding_vis[:, 2],
                    mode='markers',
                    marker=dict(size=2, opacity=0.7)
                ))

            fig.update_layout(
                title=f"{title} - 3D Embedding",
                scene=dict(
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2",
                    zaxis_title="Dimension 3"
                ),
                width=900,
                height=700
            )

        fig.show()

    @staticmethod
    def available_sklearn() -> Dict[str, Tuple[str, ...]]:
        loads = tuple(a for a in dir(skds) if a.startswith("load_") and callable(getattr(skds, a)))
        fetches = tuple(a for a in dir(skds) if a.startswith("fetch_") and callable(getattr(skds, a)))
        makes = tuple(a for a in dir(skds) if a.startswith("make_") and callable(getattr(skds, a)))
        return {"load": loads, "fetch": fetches, "make": makes}

    @staticmethod
    def available_cytof() -> Tuple[str, ...]:
        return tuple(_CYTOF_REGISTRY.keys())
