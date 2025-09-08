# dire_pytorch_memory_efficient.py

"""
Memory-efficient PyTorch/PyKeOps backend for DiRe.

This implementation inherits from DiRePyTorch and overrides specific methods for:
- FP16 support for memory-efficient k-NN computation
- Point-by-point attraction force computation to avoid large tensor materialization
- More aggressive memory management and cache clearing
- Optional PyKeOps LazyTensors for repulsion when available
"""

import gc

import numpy as np
import torch
from loguru import logger

# Import base class
from .dire_pytorch import DiRePyTorch

# PyKeOps for efficient force computations
try:
    from pykeops.torch import LazyTensor
    PYKEOPS_AVAILABLE = True
except ImportError:
    PYKEOPS_AVAILABLE = False
    logger.warning("PyKeOps not available. Install with: pip install pykeops")


class DiRePyTorchMemoryEfficient(DiRePyTorch):
    """
    Memory-optimized PyTorch implementation of DiRe for large-scale datasets.
    
    This class extends DiRePyTorch with enhanced memory management capabilities,
    making it suitable for processing very large datasets that would otherwise
    cause out-of-memory errors with the standard implementation.
    
    Key Improvements over DiRePyTorch
    ---------------------------------
    - **FP16 Support**: Uses half-precision by default for 2x memory reduction
    - **Dynamic Chunking**: Automatically adjusts chunk sizes based on available memory
    - **Aggressive Cleanup**: More frequent garbage collection and cache clearing
    - **PyKeOps Integration**: Optional LazyTensors for memory-efficient exact repulsion
    - **Memory Monitoring**: Real-time memory usage tracking and warnings
    - **Point-wise Processing**: Falls back to point-by-point computation when needed
    
    Best Use Cases
    --------------
    - Datasets with >100K points
    - High-dimensional data (>500 features) 
    - Memory-constrained environments
    - Production systems requiring reliable memory usage
    
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
        If True, use exact all-pairs repulsion (memory intensive).
    use_fp16 : bool, default=True
        Enable FP16 precision for memory efficiency (recommended).
    use_pykeops_repulsion : bool, default=True
        Use PyKeOps LazyTensors for repulsion when beneficial.
    pykeops_threshold : int, default=50000
        Maximum dataset size for PyKeOps all-pairs computation.
    memory_fraction : float, default=0.25
        Fraction of available memory to use for computations.
        
    Examples
    --------
    Memory-efficient processing of large dataset::
    
        from dire_rapids import DiRePyTorchMemoryEfficient
        import numpy as np
        
        # Large dataset
        X = np.random.randn(500000, 512)
        
        # Memory-efficient reducer
        reducer = DiRePyTorchMemoryEfficient(
            use_fp16=True,
            memory_fraction=0.3,
            verbose=True
        )
        
        embedding = reducer.fit_transform(X)
        
    Custom memory settings::
    
        reducer = DiRePyTorchMemoryEfficient(
            use_pykeops_repulsion=False,  # Disable PyKeOps
            memory_fraction=0.15,         # Use less memory
            pykeops_threshold=20000       # Lower PyKeOps threshold
        )
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
        use_exact_repulsion=False,
        use_fp16=True,  # Enable FP16 by default for memory efficiency
        use_pykeops_repulsion=True,  # Use PyKeOps for repulsion when possible
        pykeops_threshold=50000,     # Max points for PyKeOps all-pairs
        memory_fraction=0.25,
    ):
        """
        Initialize memory-efficient DiRe reducer.
        
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
            If True, use exact all-pairs repulsion (memory intensive).
        use_fp16 : bool, default=True
            Enable FP16 precision for memory efficiency. Provides 2x memory
            reduction and significant speed improvements on modern GPUs.
        use_pykeops_repulsion : bool, default=True
            Use PyKeOps LazyTensors for memory-efficient repulsion computation
            when dataset size is below pykeops_threshold.
        pykeops_threshold : int, default=50000
            Maximum dataset size for PyKeOps all-pairs computation.
            Above this threshold, random sampling is used instead.
        memory_fraction : float, default=0.25
            Fraction of available memory to use for computations.
            Lower values are more conservative but may be slower.
        """
        
        # Call parent constructor
        super().__init__(
            n_components=n_components,
            n_neighbors=n_neighbors,
            init=init,
            max_iter_layout=max_iter_layout,
            min_dist=min_dist,
            spread=spread,
            cutoff=cutoff,
            n_sample_dirs=n_sample_dirs,
            sample_size=sample_size,
            neg_ratio=neg_ratio,
            verbose=verbose,
            random_state=random_state,
            use_exact_repulsion=use_exact_repulsion,
        )
        
        # Additional memory-efficient parameters
        self.use_fp16 = use_fp16
        self.use_pykeops_repulsion = use_pykeops_repulsion
        self.pykeops_threshold = pykeops_threshold
        self.memory_fraction = memory_fraction
        
        # Log memory-efficient settings
        if self.verbose:
            self.logger.info("Memory-efficient mode enabled")
            if self.use_fp16 and self.device.type == 'cuda':
                self.logger.info("FP16 enabled for k-NN computation")
            if self.use_pykeops_repulsion and PYKEOPS_AVAILABLE:
                self.logger.info(f"PyKeOps repulsion enabled (threshold: {self.pykeops_threshold} points)")
    
    def _get_available_memory(self):
        """
        Get available system memory in bytes.
        
        Private method that queries the available memory on the current device
        (GPU or CPU) to inform memory-aware chunk sizing decisions.
        
        Returns
        -------
        int
            Available memory in bytes.
            
        Notes
        -----
        Private method, should not be called directly. Used by _compute_optimal_chunk_size().
        
        For CUDA devices, returns free GPU memory.
        For CPU, returns available system RAM.
        """
        if self.device.type == 'cuda':
            return torch.cuda.mem_get_info()[0]  # Free memory
        else:
            import psutil
            return psutil.virtual_memory().available
    
    def _compute_optimal_chunk_size(self, n_samples, n_features, operation_type="knn", dtype=torch.float32):
        """
        Compute optimal chunk size based on available memory and operation type.
        
        This private method dynamically calculates the optimal chunk size for different
        operations based on available system memory, data characteristics, and the
        configured memory fraction.
        
        Parameters
        ----------
        n_samples : int
            Total number of samples in the dataset.
        n_features : int
            Number of features per sample.
        operation_type : {'knn', 'repulsion', 'general'}, default='knn'
            Type of operation to optimize for:
            - 'knn': k-nearest neighbors computation
            - 'repulsion': Repulsion force computation  
            - 'general': General tensor operations
        dtype : torch.dtype, default=torch.float32
            Data type for memory calculations.
            
        Returns
        -------
        int
            Optimal chunk size for the specified operation.
            
        Notes
        -----
        Private method, should not be called directly. Used by _compute_knn() and _compute_forces().
        
        The chunk size is bounded between reasonable minimum and maximum values
        to ensure both memory safety and computational efficiency.
        """
        available_memory = self._get_available_memory()
        usable_memory = available_memory * self.memory_fraction
        
        bytes_per_element = 2 if dtype == torch.float16 else 4
        
        if operation_type == "knn":
            # For k-NN: chunk_size × n_samples × bytes_per_element (distance matrix chunk)
            max_chunk_size = int(usable_memory / (n_samples * bytes_per_element))
            
        elif operation_type == "repulsion":
            # For repulsion: chunk_size × n_neg × n_components × bytes_per_element
            n_neg = min(int(self.neg_ratio * self.n_neighbors), n_samples - 1)
            memory_per_sample = n_neg * self.n_components * bytes_per_element
            max_chunk_size = int(usable_memory / memory_per_sample)
            
        else:  # general
            # Conservative estimate: chunk_size × n_features × bytes_per_element
            memory_per_sample = n_features * bytes_per_element
            max_chunk_size = int(usable_memory / memory_per_sample)
        
        # Apply reasonable bounds
        min_chunk_size = 100
        max_reasonable_chunk_size = min(20000, n_samples)
        
        optimal_chunk_size = max(min_chunk_size, min(max_chunk_size, max_reasonable_chunk_size))
        
        if self.verbose:
            self.logger.debug(f"Memory-aware chunk sizing for {operation_type}: "
                            f"{optimal_chunk_size} (available: {available_memory/1e9:.1f}GB)")
        
        return optimal_chunk_size
    
    def _compute_knn(self, X, chunk_size=None, use_fp16=None):
        """
        Compute k-nearest neighbors with enhanced memory management.
        
        This method overrides the parent implementation with memory-aware chunk
        sizing, automatic FP16 selection, and aggressive memory cleanup.
        
        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).
        chunk_size : int, optional
            Size of chunks for processing. If None, automatically computed
            based on available memory.
        use_fp16 : bool, optional
            Use FP16 precision. If None, automatically determined based on
            data size and GPU capabilities.
            
        Notes
        -----
        Private method, should not be called directly. Used by fit_transform().
        
        Enhancements over parent method:
        - Automatic FP16 selection for large/high-dimensional datasets
        - Memory-aware chunk size computation
        - More aggressive memory cleanup after processing
        
        Side Effects
        ------------
        Sets self._knn_indices and self._knn_distances with computed k-NN graph.
        """
        n_samples = X.shape[0]
        n_dims = X.shape[1]
        
        # Use instance setting if not explicitly provided
        if use_fp16 is None:
            use_fp16 = self.use_fp16
        
        # Force FP16 for large/high-dimensional datasets on GPU
        if self.device.type == 'cuda' and (n_dims >= 100 or n_samples >= 50000):
            use_fp16 = True
            self.logger.info(f"Forcing FP16 for large dataset ({n_samples} samples, {n_dims}D)")
        
        # Compute optimal chunk size based on available memory
        if chunk_size is None:
            chunk_size = self._compute_optimal_chunk_size(
                n_samples,
                n_dims,
                operation_type="knn",
                dtype=torch.float16 if use_fp16 else torch.float32
            )
        
        self.logger.info(f"Memory-efficient k-NN: chunk_size={chunk_size}, FP16={use_fp16}")
        
        # Call parent method with our settings
        super()._compute_knn(X, chunk_size=chunk_size, use_fp16=use_fp16)
        
        # Aggressive memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    def _compute_forces(self, positions, iteration, max_iterations, chunk_size=None):
        """
        Compute forces with memory-efficient strategies and PyKeOps integration.
        
        This method overrides the parent force computation with enhanced memory
        management, point-by-point fallback capabilities, and optional PyKeOps
        LazyTensors for exact repulsion computation.
        
        Parameters
        ----------
        positions : torch.Tensor
            Current positions of points in embedding space, shape (n_samples, n_components).
        iteration : int
            Current iteration number (0-indexed).
        max_iterations : int
            Total number of iterations planned.
        chunk_size : int, optional
            Maximum chunk size for processing. If None, automatically computed
            based on available memory.
            
        Returns
        -------
        torch.Tensor
            Computed forces of shape (n_samples, n_components).
            
        Notes
        -----
        Private method, should not be called directly. Used by _optimize_layout().
        
        Force Computation Strategy:
        - **Attraction forces**: Vectorized computation with fallback to point-by-point
        - **Repulsion forces**: Chooses between PyKeOps LazyTensors, exact computation,
          or chunked random sampling based on dataset size and available memory
        - **Memory management**: Aggressive cleanup of intermediate tensors
        
        Backend Selection for Repulsion:
        - PyKeOps LazyTensors: For datasets < pykeops_threshold on GPU
        - Exact computation: When use_exact_repulsion=True (testing only)
        - Random sampling: For large datasets or memory-constrained environments
        """
        n_samples = positions.shape[0]
        forces = torch.zeros_like(positions)
        
        # Auto-adjust chunk size based on available memory
        if chunk_size is None:
            chunk_size = self._compute_optimal_chunk_size(
                n_samples, 
                self.n_components, 
                operation_type="repulsion",
                dtype=torch.float16 if self.use_fp16 else torch.float32
            )
        
        # Linear cooling
        alpha = 1.0 - iteration / max_iterations
        
        # Parameters
        a_val = float(self._a)
        b_val = float(self._b)
        b_exp = float(2 * b_val)
        
        # ============ ATTRACTION FORCES (vectorized for speed) ============
        # Vectorize k-NN attraction forces for efficiency
        # Get all neighbor positions at once
        neighbor_positions = positions[self._knn_indices]  # shape: (n_samples, k, n_components)
        center_positions = positions.unsqueeze(1)          # shape: (n_samples, 1, n_components)
        
        # Compute differences and distances
        diff = neighbor_positions - center_positions       # shape: (n_samples, k, n_components)
        dist = torch.norm(diff, dim=2, keepdim=True) + 1e-10  # shape: (n_samples, k, 1)
        
        # Attraction coefficients
        att_coeff = 1.0 / (1.0 + a_val * (1.0 / dist) ** b_exp)  # shape: (n_samples, k, 1)
        
        # Sum attraction forces for each point
        attraction_forces = (att_coeff * diff / dist).sum(dim=1)  # shape: (n_samples, n_components)
        forces += attraction_forces
        
        # ============ REPULSION FORCES ============
        # Decide whether to use PyKeOps based on dataset size and availability
        use_pykeops = (
            PYKEOPS_AVAILABLE and 
            self.use_pykeops_repulsion and 
            n_samples < self.pykeops_threshold and
            self.device.type == 'cuda' and
            not self.use_exact_repulsion  # Don't use if exact repulsion is requested
        )
        
        if use_pykeops:
            # Use PyKeOps LazyTensors for efficient all-pairs repulsion
            self.logger.debug("Using PyKeOps LazyTensors for repulsion")
            
            X_i = LazyTensor(positions[:, None, :])  # (N, 1, D)
            X_j = LazyTensor(positions[None, :, :])  # (1, N, D)
            
            # Compute differences and distances
            diff = X_j - X_i  # (N, N, D) lazy
            D_ij = ((diff ** 2).sum(-1)).sqrt() + 1e-10  # (N, N) lazy
            
            # Repulsion kernel
            rep_kernel = -1.0 / (1.0 + a_val * (D_ij ** b_exp))
            
            # Apply cutoff
            cutoff_scale = (-D_ij / self.cutoff).exp()
            rep_kernel = rep_kernel * cutoff_scale
            
            # Compute forces (reduction happens efficiently in PyKeOps)
            # For LazyTensors, division broadcasts automatically
            force_dir = diff / D_ij
            rep_forces = (rep_kernel * force_dir).sum(1)
            forces += rep_forces
            
        elif self.use_exact_repulsion:
            # Use exact all-pairs repulsion (memory intensive, for testing)
            self.logger.debug("Using exact all-pairs repulsion (memory intensive)")
            # Fall back to parent implementation
            return super()._compute_forces(positions, iteration, max_iterations, chunk_size)
        else:
            # Use chunked random sampling to avoid memory issues with large datasets
            self.logger.debug("Using chunked random sampling for repulsion")
            n_neg = min(int(self.neg_ratio * self.n_neighbors), n_samples - 1)
            
            # Process repulsion in chunks to avoid large tensor allocation
            repulsion_chunk_size = min(chunk_size, n_samples)
            
            for start_idx in range(0, n_samples, repulsion_chunk_size):
                end_idx = min(start_idx + repulsion_chunk_size, n_samples)
                chunk_size_actual = end_idx - start_idx
                
                # Generate negative samples for this chunk only
                neg_indices = torch.randint(0, n_samples, (chunk_size_actual, n_neg), device=self.device)
                
                # Avoid self-selection more memory efficiently
                # Create arange only for chunk size
                chunk_indices = torch.arange(start_idx, end_idx, device=self.device).unsqueeze(1)
                mask = neg_indices == chunk_indices
                
                # Replace self-indices with random alternatives (avoid large tensor creation)
                if mask.any():
                    # Generate replacement indices on-the-fly
                    replacement_base = torch.randint(0, n_samples, (chunk_size_actual, n_neg), device=self.device)
                    # Ensure replacements are different from self
                    replacement_mask = replacement_base == chunk_indices
                    while replacement_mask.any():
                        replacement_base[replacement_mask] = torch.randint(0, n_samples, 
                                                                         (replacement_mask.sum(),), 
                                                                         device=self.device)
                        replacement_mask = replacement_base == chunk_indices
                    
                    neg_indices = torch.where(mask, replacement_base, neg_indices)
                
                # Compute repulsion for this chunk
                chunk_positions = positions[start_idx:end_idx]
                neg_positions = positions[neg_indices]  # shape: (chunk_size, n_neg, n_components)
                center_positions = chunk_positions.unsqueeze(1)  # shape: (chunk_size, 1, n_components)
                
                # Compute differences and distances
                diff = neg_positions - center_positions
                dist = torch.norm(diff, dim=2, keepdim=True) + 1e-10
                
                # Repulsion coefficients
                rep_coeff = -1.0 / (1.0 + a_val * (dist ** b_exp))
                cutoff_scale = torch.exp(-dist / self.cutoff)
                rep_coeff = rep_coeff * cutoff_scale
                
                # Sum repulsion forces for this chunk
                chunk_repulsion_forces = (rep_coeff * diff / dist).sum(dim=1)
                forces[start_idx:end_idx] += chunk_repulsion_forces
                
                # Clear intermediate tensors to free memory
                del neg_indices, neg_positions, diff, dist, rep_coeff, cutoff_scale, chunk_repulsion_forces
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Apply cooling and clipping
        forces = alpha * forces
        forces = torch.clamp(forces, -self.cutoff, self.cutoff)
        
        return forces
    
    def _optimize_layout(self, initial_positions):
        """
        Optimize embedding layout with enhanced memory monitoring and management.
        
        This method overrides the parent optimization loop with real-time memory
        monitoring, more frequent cleanup, and detailed progress reporting.
        
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
        
        Enhancements over parent method:
        - Real-time GPU memory monitoring and warnings
        - More frequent cache clearing (every 10 iterations)
        - Detailed memory usage reporting
        - Low memory warnings when free GPU memory < 2GB
        """
        positions = initial_positions.clone()
        
        self.logger.info(f"Memory-efficient optimization for {self._n_samples} points...")
        
        # Log initial memory usage
        if self.device.type == 'cuda':
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"Initial GPU memory: {mem_reserved:.2f}GB used / {mem_total:.1f}GB total")
        
        for iteration in range(self.max_iter_layout):
            # Monitor memory before computation
            if self.device.type == 'cuda' and iteration % 20 == 0:
                mem_available = torch.cuda.mem_get_info()[0] / 1e9
                
                # Warn if memory usage is getting high
                if mem_available < 2.0:  # Less than 2GB free
                    self.logger.warning(f"Low GPU memory: {mem_available:.1f}GB free, {mem_before:.1f}GB used")
            
            # Compute forces with our memory-efficient method
            forces = self._compute_forces(positions, iteration, self.max_iter_layout)
            
            # Update positions
            positions += forces
            
            # More frequent logging and memory management
            if iteration % 10 == 0:
                if self.verbose and iteration % 20 == 0:
                    force_mag = torch.norm(forces, dim=1).mean().item()
                    if self.device.type == 'cuda':
                        mem_reserved = torch.cuda.memory_reserved() / 1e9
                        mem_available = torch.cuda.mem_get_info()[0] / 1e9
                        self.logger.info(f"Iteration {iteration}/{self.max_iter_layout}, avg force: {force_mag:.6f}, "
                                       f"GPU memory: {mem_reserved:.1f}GB used, {mem_available:.1f}GB free")
                    else:
                        self.logger.info(f"Iteration {iteration}/{self.max_iter_layout}, avg force: {force_mag:.6f}")
                
                # Aggressive memory cleanup
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Final normalization
        positions -= positions.mean(dim=0)
        positions /= positions.std(dim=0)
        
        return positions
    
    def fit_transform(self, X, y=None):
        """
        Fit the model and transform data with memory-efficient processing.
        
        This method extends the parent implementation with memory-optimized data
        handling, enhanced logging, and aggressive cleanup procedures.
        
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
            
        Notes
        -----
        Memory Optimizations:
        - Automatic FP16 conversion for large datasets on GPU
        - Strategic backend selection based on dataset characteristics
        - Aggressive memory cleanup after processing
        - Real-time memory monitoring and reporting
        
        Examples
        --------
        Process large dataset with memory monitoring::
        
            import numpy as np
            from dire_rapids import DiRePyTorchMemoryEfficient
            
            # Large high-dimensional dataset
            X = np.random.randn(200000, 1000)
            
            reducer = DiRePyTorchMemoryEfficient(
                use_fp16=True,
                memory_fraction=0.3,
                verbose=True
            )
            
            embedding = reducer.fit_transform(X)
        """
        # Store data with potential dtype conversion
        if self.use_fp16 and self.device.type == 'cuda':
            # Keep data in float32 for CPU operations, convert to fp16 on GPU when needed
            self._data = np.asarray(X, dtype=np.float32)
        else:
            self._data = np.asarray(X, dtype=np.float32)
        
        self._n_samples = self._data.shape[0]
        
        self.logger.info(f"Memory-efficient processing: {self._n_samples} samples, {self._data.shape[1]} features")
        
        # Provide strategy information
        if self._n_samples > self.pykeops_threshold:
            self.logger.info(f"Large dataset ({self._n_samples} > {self.pykeops_threshold}): using random sampling for repulsion")
        elif PYKEOPS_AVAILABLE and self.use_pykeops_repulsion and self.device.type == 'cuda':
            self.logger.info("Using PyKeOps LazyTensors for exact repulsion (memory efficient)")
        else:
            self.logger.info("Using random sampling for repulsion")
        
        # Call parent fit_transform
        result = super().fit_transform(X, y)
        
        # Extra aggressive cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
        return result