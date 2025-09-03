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
    Memory-efficient PyTorch/PyKeOps implementation that inherits from DiRePyTorch.
    
    Key improvements over base class:
    - FP16 support by default for k-NN computation
    - Point-by-point attraction force computation
    - More aggressive GPU memory management
    - Optional PyKeOps LazyTensors for exact repulsion
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
        memory_fraction=0.15,         # More conservative memory usage (15% vs 30%)
    ):
        """Initialize with memory-efficient defaults."""
        
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
    
    def _compute_knn(self, X, chunk_size=20000, use_fp16=None):
        """
        Override k-NN computation with more aggressive memory management and FP16 by default.
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
        
        # Use smaller chunks for memory efficiency
        if self.device.type == 'cuda':
            gpu_mem_free = torch.cuda.mem_get_info()[0]
            bytes_per_element = 2 if use_fp16 else 4
            memory_per_chunk = chunk_size * n_samples * bytes_per_element
            
            # Use more conservative memory fraction
            max_memory = gpu_mem_free * self.memory_fraction
            if memory_per_chunk > max_memory:
                chunk_size = int(max_memory / (n_samples * bytes_per_element))
                chunk_size = max(500, min(chunk_size, 20000))  # Clamp between 500 and 20000
            
            self.logger.info(f"Memory-efficient k-NN: chunk_size={chunk_size}, FP16={use_fp16}")
        
        # Call parent method with our settings
        super()._compute_knn(X, chunk_size=chunk_size, use_fp16=use_fp16)
        
        # Aggressive memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
    
    def _compute_forces(self, positions, iteration, max_iterations, chunk_size=2000):
        """
        Override force computation with point-by-point processing for attraction
        and optional PyKeOps for repulsion.
        """
        n_samples = positions.shape[0]
        forces = torch.zeros_like(positions)
        
        # Linear cooling
        alpha = 1.0 - iteration / max_iterations
        
        # Parameters
        a_val = float(self._a)
        b_val = float(self._b)
        b_exp = float(2 * b_val)
        
        # ============ ATTRACTION FORCES (point-by-point for memory efficiency) ============
        # Always use point-by-point for attraction to minimize memory usage
        for i in range(n_samples):
            neighbor_ids = self._knn_indices[i]
            pos_i = positions[i:i+1]
            pos_neighbors = positions[neighbor_ids]
            
            diff = pos_neighbors - pos_i
            dist = torch.norm(diff, dim=1, keepdim=True) + 1e-10
            
            att_coeff = 1.0 / (1.0 + a_val * (1.0 / dist) ** b_exp)
            forces[i] += (att_coeff * diff / dist).sum(0)
            
            # Aggressive cache clearing
            if self.device.type == 'cuda' and i % 500 == 0:
                torch.cuda.empty_cache()
        
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
            force_dir = diff / D_ij.view(n_samples, n_samples, 1)
            rep_forces = (rep_kernel.view(n_samples, n_samples, 1) * force_dir).sum(1)
            forces += rep_forces
            
        elif self.use_exact_repulsion:
            # Use exact all-pairs repulsion (memory intensive, for testing)
            self.logger.debug("Using exact all-pairs repulsion (memory intensive)")
            # Fall back to parent implementation
            return super()._compute_forces(positions, iteration, max_iterations, chunk_size)
        else:
            # Use random sampling for large datasets or when PyKeOps unavailable
            self.logger.debug("Using random sampling for repulsion")
            n_neg = min(int(self.neg_ratio * self.n_neighbors), n_samples - 1)
            
            for i in range(n_samples):
                # Sample negative points
                neg_idx = np.random.choice(n_samples, min(n_neg + 1, n_samples), replace=False)
                neg_idx = neg_idx[neg_idx != i][:n_neg]
                
                if len(neg_idx) > 0:
                    pos_i = positions[i:i+1]
                    pos_neg = positions[neg_idx]
                    
                    diff = pos_neg - pos_i
                    dist = torch.norm(diff, dim=1, keepdim=True) + 1e-10
                    
                    rep_coeff = -1.0 / (1.0 + a_val * (dist ** b_exp))
                    cutoff_scale = torch.exp(-dist / self.cutoff)
                    rep_coeff = rep_coeff * cutoff_scale
                    
                    forces[i] += (rep_coeff * diff / dist).sum(0)
                
                # Aggressive cache clearing
                if self.device.type == 'cuda' and i % 500 == 0:
                    torch.cuda.empty_cache()
        
        # Apply cooling and clipping
        forces = alpha * forces
        forces = torch.clamp(forces, -self.cutoff, self.cutoff)
        
        return forces
    
    def _optimize_layout(self, initial_positions):
        """
        Override optimization with more frequent memory management.
        """
        positions = initial_positions.clone()
        
        self.logger.info(f"Memory-efficient optimization for {self._n_samples} points...")
        
        # Log initial memory usage
        if self.device.type == 'cuda':
            mem_used = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"Initial GPU memory: {mem_used:.2f}/{mem_total:.1f} GB")
        
        for iteration in range(self.max_iter_layout):
            # Compute forces with our memory-efficient method
            forces = self._compute_forces(positions, iteration, self.max_iter_layout)
            
            # Update positions
            positions += forces
            
            # More frequent logging and memory management
            if iteration % 10 == 0:
                if self.verbose and iteration % 20 == 0:
                    force_mag = torch.norm(forces, dim=1).mean().item()
                    self.logger.info(f"Iteration {iteration}/{self.max_iter_layout}, avg force: {force_mag:.6f}")
                
                # Aggressive memory cleanup
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
                    # Log memory usage periodically
                    if iteration % 40 == 0:
                        mem_used = torch.cuda.memory_allocated() / 1e9
                        self.logger.debug(f"GPU memory: {mem_used:.2f} GB")
        
        # Final normalization
        positions -= positions.mean(dim=0)
        positions /= positions.std(dim=0)
        
        return positions
    
    def fit_transform(self, X, y=None):
        """
        Override to add memory-efficient logging and cleanup.
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