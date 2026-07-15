# dire_cuvs.py

"""
DIRE with cuVS backend for GPU-accelerated k-NN at scale.

This module provides optional cuVS integration for massive datasets.
Falls back to PyTorch if cuVS is not available.

Requirements:
    Follow the installation instructions at https://docs.rapids.ai/install/
"""

import numpy as np
import torch  # pylint: disable=unused-import # Used via parent class (self.device from DiRePyTorch)
from loguru import logger

# Import base DIRE PyTorch implementation
from .dire_pytorch import (  # pylint: disable=cyclic-import
    DiRePyTorch,
    _is_cuvs_native_metric,
    _prefer_memory_efficient_for_shape,
    _remove_self_from_knn,
)

# Try to import cuVS and CuPy
try:
    import cupy as cp
    from cuvs.neighbors import cagra, ivf_pq, ivf_flat
    CUVS_AVAILABLE = True
    logger.trace("cuVS available - GPU-accelerated k-NN enabled")
except ImportError:
    CUVS_AVAILABLE = False
    logger.trace("cuVS not available. Install RAPIDS for GPU-accelerated k-NN: "
                 "Follow the installation instructions at https://docs.rapids.ai/install/")

# cuVS 25.10+ exposes a graph-construction API tailored to the all-vs-all
# workload used by manifold-learning algorithms. Keep this feature check
# separate from the core cuVS imports so older supported installations can
# still use the legacy index-build + self-query path.
try:
    from cuvs.common import MultiGpuResources, Resources
    from cuvs.neighbors import all_neighbors, nn_descent
    ALL_NEIGHBORS_AVAILABLE = True
except ImportError:
    ALL_NEIGHBORS_AVAILABLE = False


def _merged_params(defaults, overrides):
    """Return constructor kwargs with user values overriding defaults."""
    params = dict(defaults)
    if overrides:
        params.update(overrides)
    return params


_REMOVE_SELF_KERNEL = None


def _remove_self_from_knn_cupy(indices, distances, n_neighbors):
    """Remove each row's own id on-device while preserving neighbor order.

    cuVS all-neighbors normally includes self, but self is not guaranteed to
    occupy the first column (and may be absent for metrics such as inner
    product). Requesting one extra candidate and compacting with per-row ranks
    handles both cases without an O(N) Python loop or a second full graph
    allocation.
    """
    global _REMOVE_SELF_KERNEL  # pylint: disable=global-statement

    indices = cp.asarray(indices)
    distances = cp.asarray(distances)
    if indices.ndim != 2 or distances.shape != indices.shape:
        raise ValueError("cuVS indices and distances must be equally shaped 2-D arrays")
    if indices.dtype != cp.int64 or distances.dtype != cp.float32:
        raise TypeError("cuVS all-neighbors requires int64 indices and float32 distances")
    if not indices.flags.c_contiguous or not distances.flags.c_contiguous:
        raise ValueError("cuVS all-neighbors outputs must be C-contiguous")
    if indices.shape[1] < n_neighbors:
        raise RuntimeError("cuVS result has fewer columns than requested neighbors")

    if _REMOVE_SELF_KERNEL is None:
        _REMOVE_SELF_KERNEL = cp.RawKernel(
            r'''
            extern "C" __global__
            void remove_self(
                long long* indices,
                float* distances,
                long long n_rows,
                long long n_candidates,
                long long n_neighbors,
                int* failed)
            {
                long long row = (long long)blockDim.x * blockIdx.x + threadIdx.x;
                if (row >= n_rows) return;

                long long base = row * n_candidates;
                long long write = 0;
                for (long long col = 0; col < n_candidates; ++col) {
                    long long neighbor = indices[base + col];
                    float distance = distances[base + col];
                    bool finite_distance =
                        (__float_as_uint(distance) & 0x7f800000U) != 0x7f800000U;
                    if (
                        neighbor != row &&
                        neighbor >= 0 &&
                        neighbor < n_rows &&
                        finite_distance &&
                        write < n_neighbors
                    ) {
                        indices[base + write] = neighbor;
                        distances[base + write] = distance;
                        ++write;
                    }
                }
                if (write < n_neighbors) atomicExch(failed, 1);
            }
            ''',
            'remove_self',
        )

    n_rows, n_candidates = indices.shape
    failed = cp.zeros(1, dtype=cp.int32)
    threads = 256
    blocks = (n_rows + threads - 1) // threads
    _REMOVE_SELF_KERNEL(
        (blocks,),
        (threads,),
        (
            indices,
            distances,
            np.int64(n_rows),
            np.int64(n_candidates),
            np.int64(n_neighbors),
            failed,
        ),
    )
    if bool(failed.item()):
        raise RuntimeError(
            "cuVS all-neighbors returned an invalid or underfilled row; "
            "increase cluster capacity/overlap or reduce n_neighbors."
        )
    return indices[:, :n_neighbors], distances[:, :n_neighbors]

# Try to import cuML for GPU-accelerated PCA
try:
    from cuml.decomposition import PCA as cuPCA
    from cuml.decomposition import TruncatedSVD as cuTruncatedSVD
    CUML_AVAILABLE = True
    logger.trace("cuML available - GPU-accelerated PCA enabled")
except ImportError:
    CUML_AVAILABLE = False
    if CUVS_AVAILABLE:
        logger.trace("cuML not available but cuVS is. PCA will run on CPU.")


class DiReCuVS(DiRePyTorch):
    """
    RAPIDS cuVS/cuML accelerated implementation of DiRe for massive datasets.
    
    This class extends DiRePyTorch with optional RAPIDS cuVS (CUDA Vector Search)
    integration for GPU-accelerated k-nearest neighbors computation and cuML
    integration for GPU-accelerated PCA initialization. It provides substantial
    performance improvements for large-scale datasets.
    
    Performance Advantages over PyTorch/PyKeOps
    -------------------------------------------
    - **10-100x faster k-NN**: For large datasets (>100K points)
    - **Massive scale support**: Handles 10M+ points efficiently
    - **Tunable accuracy**: Exact brute force or approximate graph builders
    - **Multi-GPU ready**: Supports extreme scale processing
    - **GPU-accelerated PCA**: cuML PCA/SVD for initialization
    
    Automatic Fallback
    ------------------
    With ``knn_backend='auto'``, falls back to PyTorch backend if cuVS is not
    available or is not beneficial for the current data. With
    ``knn_backend='cuvs'``, cuVS is required and unavailable/unsupported
    configurations raise instead of falling back.
    
    Parameters
    ----------
    use_cuvs : bool or None, default=None
        Whether to use cuVS for k-NN computation. If None, automatically
        detected based on availability and hardware.
    use_cuml : bool or None, default=None  
        Whether to use cuML for PCA initialization. If None, automatically
        detected based on availability and hardware.
    cuvs_knn_method : {'auto', 'all_neighbors', 'index_search'}, default='auto'
        cuVS graph-construction strategy. ``'auto'`` uses the purpose-built
        all-neighbors API when available unless an explicit legacy
        ``cuvs_index_type`` is selected.
    cuvs_index_type : {'auto', 'ivf_flat', 'ivf_pq', 'cagra', 'flat'}, default='auto'
        Type of legacy cuVS index to build:

        - 'auto': Automatically select based on data characteristics
        - 'ivf_flat': Inverted file index without compression
        - 'ivf_pq': Inverted file index with product quantization
        - 'cagra': Graph-based index for very large datasets
        - 'flat': Brute-force exact search
    cuvs_build_params : dict, optional
        Custom parameters for cuVS index building. Overrides defaults.
    cuvs_search_params : dict, optional  
        Custom parameters for cuVS search. Overrides defaults.
    all_neighbors_algo : {'nn_descent', 'brute_force', 'ivf_pq'}, default='nn_descent'
        Local graph builder used by cuVS all-neighbors.
    all_neighbors_n_clusters : int, default=1
        Paper parameter ``c``. Values greater than one keep input on the host
        and process spatial partitions out-of-core.
    all_neighbors_overlap_factor : int or None, default=None
        Paper spill factor ``s``. ``None`` selects 0 for a single cluster and
        ``min(2, c - 1)`` for an out-of-core build.
    all_neighbors_device_ids : sequence of int or None, default=None
        GPU ids for multi-GPU out-of-core construction.
    all_neighbors_algo_params : dict, optional
        Overrides for the selected local graph builder's parameter object.
    *args, **kwargs
        Additional arguments passed to DiRePyTorch parent class.
        Includes: n_components, n_neighbors, init, max_iter_layout, min_dist,
        spread, cutoff, neg_ratio, verbose, random_state, use_exact_repulsion,
        metric (custom distance function for k-NN computation), knn_backend.
        
    Attributes
    ----------
    use_cuvs : bool
        Whether cuVS backend is enabled and available.
    use_cuml : bool
        Whether cuML backend is enabled and available.
    cuvs_index : object or None
        Built cuVS index for k-NN search.
        
    Examples
    --------
    Basic usage with automatic backend selection::
    
        from dire_rapids import DiReCuVS
        import numpy as np
        
        # Large dataset
        X = np.random.randn(100000, 512)
        
        # Auto-detect cuVS/cuML availability
        reducer = DiReCuVS()
        embedding = reducer.fit_transform(X)
        
    Force cuVS with custom index parameters::
    
        reducer = DiReCuVS(
            use_cuvs=True,
            cuvs_index_type='ivf_pq',
            cuvs_build_params={'n_lists': 2048, 'pq_dim': 64}
        )
        
    Massive dataset processing::
    
        # 10M points, 1000 dimensions
        X = np.random.randn(10_000_000, 1000)
        
        reducer = DiReCuVS(
            use_cuvs=True,
            use_cuml=True,
            cuvs_index_type='cagra',  # Best for very large datasets
            n_neighbors=32
        )
        
        embedding = reducer.fit_transform(X)

    With custom distance metric::

        # Custom expressions use the PyTorch k-NN fallback
        reducer = DiReCuVS(
            metric='(x - y).abs().sum(-1)',  # L1/Manhattan distance
            n_neighbors=32,
        )

        embedding = reducer.fit_transform(X)

    Notes
    -----
    **Requirements:**

    - RAPIDS cuVS 26.06
    - CUDA 12.2--12.9 on Volta or newer, or CUDA 13.0--13.2 on
      Turing or newer
    
    **Legacy Index Selection Guidelines:**

    - < 50K points: 'flat' (exact search)
    - 50K-500K points: 'ivf_flat' 
    - 500K-5M points: 'ivf_pq'
    - > 5M points: 'cagra' (if dimensions <= 500)
    
    **Memory Considerations:**

    - cuVS all-neighbors requires float32 input
    - ``all_neighbors_n_clusters > 1`` keeps input in host memory, but the
      final ``N x k`` graph must still fit on one GPU
    - Larger cluster counts reduce working memory; larger overlap factors
      improve boundary recall at additional compute cost
    """
    
    def __init__(
        self,
        *args,
        use_cuvs=None,  # Auto-detect by default
        use_cuml=None,  # Auto-detect by default
        cuvs_index_type='auto',  # 'auto', 'ivf_flat', 'ivf_pq', 'cagra'
        cuvs_build_params=None,
        cuvs_search_params=None,
        cuvs_knn_method='auto',
        all_neighbors_algo='nn_descent',
        all_neighbors_n_clusters=1,
        all_neighbors_overlap_factor=None,
        all_neighbors_device_ids=None,
        all_neighbors_algo_params=None,
        **kwargs
    ):
        """
        Initialize DiReCuVS with cuVS and cuML backend configuration.
        
        Parameters
        ----------
        *args
            Positional arguments passed to DiRePyTorch parent class.
        use_cuvs : bool or None, default=None
            Whether to use cuVS for k-NN computation:
            - None: Auto-detect based on availability and GPU presence
            - True: Force cuVS usage (raises error if unavailable)
            - False: Disable cuVS, use PyTorch backend
        use_cuml : bool or None, default=None
            Whether to use cuML for PCA initialization:
            - None: Auto-detect based on availability and GPU presence  
            - True: Force cuML usage (raises error if unavailable)
            - False: Disable cuML, use sklearn backend
        cuvs_index_type : {'auto', 'ivf_flat', 'ivf_pq', 'cagra', 'flat'}, default='auto'
            Type of cuVS index to build:

            - 'auto': Automatically select optimal index based on data size/dimensionality
            - 'ivf_flat': Inverted file index without compression (good balance)
            - 'ivf_pq': Inverted file with product quantization (memory efficient)
            - 'cagra': Graph-based index (best for very large datasets)
            - 'flat': Brute-force exact search (small datasets only)
        cuvs_build_params : dict, optional
            Custom parameters for cuVS index building. These override the
            automatically determined parameters. See cuVS documentation for
            index-specific parameters.
        cuvs_search_params : dict, optional
            Custom parameters for cuVS search operations. These override the
            automatically determined parameters. See cuVS documentation for
            index-specific search parameters.
        cuvs_knn_method : {'auto', 'all_neighbors', 'index_search'}, default='auto'
            Select the purpose-built graph API or the legacy build-index and
            self-query implementation.
        all_neighbors_algo : {'nn_descent', 'brute_force', 'ivf_pq'}, default='nn_descent'
            Local graph algorithm for all-neighbors.
        all_neighbors_n_clusters : int, default=1
            Number of spatial partitions (paper parameter ``c``).
        all_neighbors_overlap_factor : int or None, default=None
            Number of partitions per point (paper spill factor ``s``).
        all_neighbors_device_ids : sequence of int or None, default=None
            Device ids used by ``MultiGpuResources`` for a batched build.
        all_neighbors_algo_params : dict, optional
            Local algorithm parameter overrides.
        **kwargs
            Additional keyword arguments passed to DiRePyTorch parent class.
            See DiRePyTorch documentation for available parameters including:
            n_components, n_neighbors, init, max_iter_layout, min_dist, spread,
            cutoff, neg_ratio, verbose, random_state, use_exact_repulsion,
            metric (custom distance function for k-NN computation).
            
        Raises
        ------
        ImportError
            If cuVS or cuML are requested but not available.
        RuntimeError
            If GPU is required but not available.
        """
        super().__init__(*args, **kwargs)

        # Auto-detect cuVS usage
        if use_cuvs is None:
            # Use cuVS if available and we have a GPU
            self.use_cuvs = CUVS_AVAILABLE and self.device.type == 'cuda'
        else:
            self.use_cuvs = use_cuvs and CUVS_AVAILABLE and self.device.type == 'cuda'
        
        if self.use_cuvs:
            self.logger.info("cuVS backend enabled for k-NN computation")
            cp.random.seed(self.random_state)
            self.logger.debug(f"Seeded CuPy random generator: {self.random_state}")
        else:
            if use_cuvs and not CUVS_AVAILABLE:
                self.logger.warning("cuVS requested but not available, falling back to PyTorch")
            elif use_cuvs and self.device.type != 'cuda':
                self.logger.warning("cuVS requested but CUDA is not available, falling back to PyTorch")
        
        # Auto-detect cuML usage for PCA
        if use_cuml is None:
            self.use_cuml = CUML_AVAILABLE and self.device.type == 'cuda'
        else:
            self.use_cuml = use_cuml and CUML_AVAILABLE and self.device.type == 'cuda'
        
        if self.use_cuml:
            self.logger.info("cuML backend enabled for PCA initialization")
        else:
            if use_cuml and not CUML_AVAILABLE:
                self.logger.warning("cuML requested but not available, falling back to sklearn")
            elif use_cuml and self.device.type != 'cuda':
                self.logger.warning("cuML requested but CUDA is not available, falling back to sklearn")
        
        self.cuvs_index_type = cuvs_index_type
        self.cuvs_build_params = cuvs_build_params
        self.cuvs_search_params = cuvs_search_params
        self.cuvs_index = None
        self._n_lists = None  # Store n_lists for search params (older cuVS versions don't expose index.n_lists)

        valid_methods = {'auto', 'all_neighbors', 'index_search'}
        if cuvs_knn_method not in valid_methods:
            raise ValueError(
                f"cuvs_knn_method must be one of {sorted(valid_methods)}, "
                f"got {cuvs_knn_method!r}"
            )
        valid_algos = {'nn_descent', 'brute_force', 'ivf_pq'}
        if all_neighbors_algo not in valid_algos:
            raise ValueError(
                f"all_neighbors_algo must be one of {sorted(valid_algos)}, "
                f"got {all_neighbors_algo!r}"
            )
        if isinstance(all_neighbors_n_clusters, bool) or not isinstance(
                all_neighbors_n_clusters, (int, np.integer)):
            raise TypeError("all_neighbors_n_clusters must be an integer")
        if all_neighbors_n_clusters < 1:
            raise ValueError("all_neighbors_n_clusters must be at least 1")

        if all_neighbors_overlap_factor is None:
            all_neighbors_overlap_factor = (
                0 if all_neighbors_n_clusters == 1
                else min(2, all_neighbors_n_clusters - 1)
            )
        if isinstance(all_neighbors_overlap_factor, bool) or not isinstance(
                all_neighbors_overlap_factor, (int, np.integer)):
            raise TypeError("all_neighbors_overlap_factor must be an integer or None")
        if all_neighbors_n_clusters == 1:
            if all_neighbors_overlap_factor != 0:
                raise ValueError(
                    "all_neighbors_overlap_factor must be 0 when "
                    "all_neighbors_n_clusters is 1"
                )
        elif not 1 <= all_neighbors_overlap_factor < all_neighbors_n_clusters:
            raise ValueError(
                "all_neighbors_overlap_factor must satisfy "
                "1 <= overlap_factor < n_clusters"
            )

        if all_neighbors_device_ids is not None:
            if not isinstance(all_neighbors_device_ids, (list, tuple)):
                raise TypeError("all_neighbors_device_ids must be a list or tuple of integers")
            if not all_neighbors_device_ids:
                raise ValueError("all_neighbors_device_ids cannot be empty")
            if any(
                    isinstance(device_id, bool) or
                    not isinstance(device_id, (int, np.integer)) or
                    device_id < 0
                    for device_id in all_neighbors_device_ids):
                raise ValueError("all_neighbors_device_ids must contain non-negative integers")
            if len(set(all_neighbors_device_ids)) != len(all_neighbors_device_ids):
                raise ValueError("all_neighbors_device_ids must be unique")
            if all_neighbors_n_clusters == 1:
                raise ValueError(
                    "all_neighbors_device_ids requires all_neighbors_n_clusters > 1"
                )
            all_neighbors_device_ids = [int(device_id) for device_id in all_neighbors_device_ids]

        if all_neighbors_algo_params is not None and not isinstance(
                all_neighbors_algo_params, dict):
            raise TypeError("all_neighbors_algo_params must be a dict or None")

        self.cuvs_knn_method = cuvs_knn_method
        self.all_neighbors_algo = all_neighbors_algo
        self.all_neighbors_n_clusters = int(all_neighbors_n_clusters)
        self.all_neighbors_overlap_factor = int(all_neighbors_overlap_factor)
        self.all_neighbors_device_ids = all_neighbors_device_ids
        self.all_neighbors_algo_params = dict(all_neighbors_algo_params or {})
        self._last_cuvs_knn_method = None

    def _select_cuvs_knn_method(self):
        """Resolve the cuVS graph-construction strategy."""
        if self.cuvs_knn_method != 'auto':
            return self.cuvs_knn_method
        if (
                ALL_NEIGHBORS_AVAILABLE and
                self.cuvs_index_type == 'auto' and
                not self.cuvs_build_params and
                not self.cuvs_search_params
        ):
            return 'all_neighbors'
        return 'index_search'

    def _all_neighbors_metric(self):
        """Return the cuVS metric and whether distances need a square root."""
        if self.metric_spec is None:
            return 'sqeuclidean', True
        metric = self.metric_spec.strip().lower()
        if metric in ('euclidean', 'l2'):
            # sqeuclidean is supported by every local all-neighbors builder;
            # convert the returned distances to the DiRe Euclidean contract.
            return 'sqeuclidean', True
        if metric == 'sqeuclidean':
            return 'sqeuclidean', False
        return metric, False

    def _make_all_neighbors_params(self, n_samples, candidate_count, metric):
        """Construct RAPIDS 26.06 all-neighbors and local-algorithm params."""
        algo_params = None
        if self.all_neighbors_algo == 'nn_descent':
            if metric not in {'sqeuclidean', 'l2', 'cosine', 'inner_product'}:
                raise ValueError(
                    f"all_neighbors_algo='nn_descent' does not support metric {metric!r}"
                )
            kwargs = _merged_params(
                {
                    'metric': metric,
                    'graph_degree': candidate_count,
                    'intermediate_graph_degree': candidate_count * 2,
                    'return_distances': True,
                },
                self.all_neighbors_algo_params,
            )
            if kwargs.get('metric') != metric:
                raise ValueError(
                    "all_neighbors_algo_params metric must match the reducer metric"
                )
            if kwargs.get('graph_degree', candidate_count) < candidate_count:
                raise ValueError(
                    "NN-descent graph_degree must be at least n_neighbors + 1"
                )
            if kwargs.get('intermediate_graph_degree', 0) < kwargs['graph_degree']:
                raise ValueError(
                    "NN-descent intermediate_graph_degree must be at least graph_degree"
                )
            algo_params = nn_descent.IndexParams(**kwargs)
        elif self.all_neighbors_algo == 'ivf_pq':
            if metric != 'sqeuclidean':
                raise ValueError(
                    "all_neighbors_algo='ivf_pq' supports only sqeuclidean distance"
                )
            local_rows = int(np.ceil(
                n_samples * max(self.all_neighbors_overlap_factor, 1) /
                self.all_neighbors_n_clusters
            ))
            kwargs = _merged_params(
                {
                    'metric': metric,
                    'n_lists': max(1, min(int(np.sqrt(local_rows)), 1024)),
                    'pq_bits': 8,
                    'pq_dim': 0,
                    'add_data_on_build': True,
                },
                self.all_neighbors_algo_params,
            )
            if kwargs.get('metric') != metric:
                raise ValueError(
                    "all_neighbors_algo_params metric must match the reducer metric"
                )
            algo_params = ivf_pq.IndexParams(**kwargs)
        elif self.all_neighbors_algo_params:
            raise ValueError(
                "all_neighbors_algo_params is not supported with algo='brute_force'"
            )

        params_kwargs = {
            'algo': self.all_neighbors_algo,
            'overlap_factor': self.all_neighbors_overlap_factor,
            'n_clusters': self.all_neighbors_n_clusters,
            'metric': metric,
        }
        if self.all_neighbors_algo == 'nn_descent':
            params_kwargs['nn_descent_params'] = algo_params
        elif self.all_neighbors_algo == 'ivf_pq':
            params_kwargs['ivf_pq_params'] = algo_params
        return all_neighbors.AllNeighborsParams(**params_kwargs)

    def _compute_knn_all_neighbors(self, X):
        """Build the full k-NN graph with cuVS all-neighbors."""
        if not ALL_NEIGHBORS_AVAILABLE:
            raise RuntimeError(
                "cuvs_knn_method='all_neighbors' requires cuVS 25.10 or newer "
                "(RAPIDS 26.06 is recommended)"
            )

        n_samples = X.shape[0]
        candidate_count = self.n_neighbors + 1
        if self.all_neighbors_n_clusters > 1:
            if candidate_count > 1024:
                raise ValueError(
                    "partitioned cuVS all-neighbors supports at most 1024 "
                    "candidates (n_neighbors must be <= 1023)"
                )
            total_assignments = n_samples * self.all_neighbors_overlap_factor
            required_assignments = (
                self.all_neighbors_n_clusters * candidate_count
            )
            if total_assignments < required_assignments:
                raise ValueError(
                    "partitioned cuVS all-neighbors has insufficient average "
                    "cluster capacity for n_neighbors; reduce n_clusters or "
                    "n_neighbors, or increase all_neighbors_overlap_factor"
                )
        metric, take_sqrt = self._all_neighbors_metric()
        params = self._make_all_neighbors_params(
            n_samples, candidate_count, metric
        )

        if self.all_neighbors_n_clusters == 1:
            dataset = cp.asarray(X, dtype=cp.float32, order='C')
            mode = 'in-core'
        else:
            # Host input is the switch that enables cuVS batched/out-of-core
            # graph construction. Passing a device array here is rejected by
            # the API when n_clusters > 1.
            dataset = np.ascontiguousarray(X, dtype=np.float32)
            mode = 'out-of-core'

        if self.all_neighbors_device_ids is None:
            resources = Resources()
        else:
            resources = MultiGpuResources(device_ids=self.all_neighbors_device_ids)

        self.logger.info(
            f"Building k-NN graph with cuVS all-neighbors ({mode}, "
            f"algo={self.all_neighbors_algo}, "
            f"clusters={self.all_neighbors_n_clusters}, "
            f"overlap={self.all_neighbors_overlap_factor})"
        )

        distances_out = cp.empty(
            (n_samples, candidate_count), dtype=cp.float32
        )
        indices, distances = all_neighbors.build(
            dataset,
            candidate_count,
            params,
            distances=distances_out,
            resources=resources,
        )
        resources.sync()

        indices_cp, distances_cp = _remove_self_from_knn_cupy(
            indices, distances, self.n_neighbors
        )
        if take_sqrt:
            cp.maximum(distances_cp, 0.0, out=distances_cp)
            cp.sqrt(distances_cp, out=distances_cp)
        elif metric == 'inner_product':
            cp.negative(distances_cp, out=distances_cp)

        # Preserve the historical NumPy-facing internal contract while also
        # keeping a zero-copy device view for the layout optimizer.
        self._knn_indices = cp.asnumpy(indices_cp)
        self._knn_distances = cp.asnumpy(distances_cp)
        try:
            self._knn_indices_torch = torch.from_dlpack(indices_cp)
        except (RuntimeError, TypeError):
            self._knn_indices_torch = None

        # The host copies and optional DLPack view are now the only live graph
        # state. Release the input, distances, resources, and CuPy cache before
        # PyTorch allocates the layout; the DLPack-backed index allocation stays
        # alive through the Torch tensor.
        del dataset, distances_out, distances, distances_cp, indices, indices_cp
        del resources
        cp.get_default_memory_pool().free_all_blocks()

        self._last_cuvs_knn_method = 'all_neighbors'
        self.logger.info(
            f"k-NN graph computed via cuVS all-neighbors: "
            f"shape {self._knn_indices.shape}"
        )

    def _select_cuvs_index_type(self, n_samples, n_dims, metric='sqeuclidean'):
        """
        Automatically select optimal cuVS index type based on data characteristics.

        This private method uses heuristics to select the most appropriate cuVS index
        type based on dataset size, dimensionality, metric, and performance trade-offs.

        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset.
        n_dims : int
            Number of dimensions/features per sample.
        metric : str, default='sqeuclidean'
            Distance metric to use. CAGRA only supports 'sqeuclidean' and 'inner_product'.

        Returns
        -------
        str
            Selected cuVS index type ('flat', 'ivf_flat', 'ivf_pq', or 'cagra').

        Notes
        -----
        Private method, should not be called directly. Used by _compute_knn().

        Selection Heuristics:
        - **< 50K samples**: 'flat' (exact search)
        - **50K-500K samples or >500D**: 'ivf_flat' (good balance)
        - **500K-5M samples**: 'ivf_pq' (memory efficient)
        - **> 5M samples and ≤500D**: 'cagra' (best performance, if metric is supported)
        - **> 5M samples and >500D**: 'ivf_pq' (high-D fallback)
        - **cosine metric**: Forces IVF method (CAGRA doesn't support cosine)
        """
        if self.cuvs_index_type != 'auto':
            return self.cuvs_index_type

        # Decision tree based on scale and dimensionality
        # For high dimensions (>500), prefer IVF methods over graph-based
        if n_samples < 50000:
            # Small dataset - use flat (IVF with many lists)
            return 'flat'
        if n_samples < 500000 or n_dims > 500:
            # Medium dataset or high-D - IVF without compression
            # IVF-Flat works better than CAGRA for high dimensions
            return 'ivf_flat'
        if n_samples < 5000000:
            # Large dataset - IVF with compression
            return 'ivf_pq'
        # Very large dataset with moderate dimensions
        # CAGRA is best for performance, but only if metric is supported
        if n_dims <= 500 and metric in ('sqeuclidean', 'euclidean', 'inner_product'):
            return 'cagra'
        else:
            # Fallback to IVF-PQ for unsupported metrics (e.g., cosine) or high-D
            return 'ivf_pq'
    
    def _build_cuvs_index(self, X_gpu, index_type, metric='euclidean'):
        """
        Build cuVS index for fast k-NN search.

        This private method constructs the appropriate cuVS index based on the
        specified index type and data characteristics, with optimized parameters
        for each index variant.

        Parameters
        ----------
        X_gpu : cupy.ndarray
            Input data on GPU, shape (n_samples, n_features), dtype float32.
        index_type : str
            Type of index to build ('flat', 'ivf_flat', 'ivf_pq', 'cagra').
        metric : str, default='euclidean'
            Distance metric to use ('euclidean' or 'inner_product').

        Returns
        -------
        cuVS index object or None
            Built cuVS index ready for search operations.
            Returns None for 'flat' type (no index needed).

        Notes
        -----
        Private method, should not be called directly. Used by _compute_knn().

        Index-Specific Optimizations:
        - **IVF-Flat**: Adaptive n_lists based on dataset size and dimensionality
        - **IVF-PQ**: Optimized PQ dimension and quantization parameters
        - **CAGRA**: Graph-based parameters tuned for large datasets

        Raises
        ------
        ValueError
            If unknown index_type is specified.
        """
        n_samples, n_dims = X_gpu.shape

        self.logger.info(f"Building cuVS {index_type} index for {n_samples} points in {n_dims}D with {metric} metric...")

        if index_type == 'flat':
            # Exact search - no index needed
            self.logger.info("Using brute-force search (exact)")
            return None

        if index_type == 'ivf_flat':
            # IVF without compression
            # For high-D data, use more lists for better quantization
            if n_dims > 500:
                # High-D: more lists help with curse of dimensionality
                n_lists = min(int(np.sqrt(n_samples) * 2), 8192)
            else:
                n_lists = min(int(np.sqrt(n_samples)), 4096)

            build_kwargs = _merged_params(
                {
                    'n_lists': n_lists,
                    'metric': metric,
                    'add_data_on_build': True,
                },
                self.cuvs_build_params,
            )
            build_params = ivf_flat.IndexParams(**build_kwargs)

            index = ivf_flat.build(build_params, X_gpu)
            self._n_lists = build_kwargs['n_lists']

            self.logger.info(
                f"Built IVF-Flat index with {self._n_lists} lists for {n_dims}D data"
            )

        elif index_type == 'ivf_pq':
            # IVF with product quantization
            n_lists = min(int(np.sqrt(n_samples)), 8192)
            pq_dim = min(n_dims // 4, 128)  # Reasonable PQ dimension

            build_kwargs = _merged_params(
                {
                    'n_lists': n_lists,
                    'metric': metric,
                    'pq_dim': pq_dim,
                    'pq_bits': 8,
                    'add_data_on_build': True,
                },
                self.cuvs_build_params,
            )
            build_params = ivf_pq.IndexParams(**build_kwargs)

            index = ivf_pq.build(build_params, X_gpu)
            self._n_lists = build_kwargs['n_lists']

            self.logger.info(
                f"Built IVF-PQ index with {self._n_lists} lists, "
                f"PQ dim={build_kwargs['pq_dim']}"
            )

        elif index_type == 'cagra':
            # CAGRA only supports sqeuclidean and inner_product metrics
            if metric == 'euclidean':
                cagra_metric = 'sqeuclidean'
            elif metric == 'sqeuclidean':
                cagra_metric = 'sqeuclidean'
            elif metric == 'inner_product':
                cagra_metric = 'inner_product'
            elif metric == 'cosine':
                raise ValueError(
                    "CAGRA index does not support cosine metric. "
                    "Use IVF-Flat or IVF-PQ index types for cosine distance, "
                    "or normalize your data and use inner_product metric."
                )
            else:
                raise ValueError(
                    f"CAGRA index does not support metric '{metric}'. "
                    f"Valid metrics: ['sqeuclidean', 'inner_product'] "
                    f"(euclidean is automatically converted to sqeuclidean)."
                )

            build_kwargs = _merged_params(
                {
                    'metric': cagra_metric,
                    'graph_degree': 32,
                    'intermediate_graph_degree': 64,
                    'build_algo': 'nn_descent',
                },
                self.cuvs_build_params,
            )
            build_params = cagra.IndexParams(**build_kwargs)

            index = cagra.build(build_params, X_gpu)

            self.logger.info(f"Built CAGRA graph-based index with {cagra_metric} metric")

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        return index
    
    def _search_cuvs(self, index, index_type, X_gpu, k, metric='euclidean'):
        """
        Search cuVS index for k nearest neighbors.

        This private method performs k-NN search using the built cuVS index,
        with optimized search parameters for each index type.

        Parameters
        ----------
        index : cuVS index object or None
            Built cuVS index from _build_cuvs_index().
        index_type : str
            Type of index being searched ('flat', 'ivf_flat', 'ivf_pq', 'cagra').
        X_gpu : cupy.ndarray
            Query data on GPU, shape (n_samples, n_features), dtype float32.
        k : int
            Number of nearest neighbors to find (plus 1 for self).
        metric : str, default='euclidean'
            Distance metric to use ('euclidean' or 'inner_product').

        Returns
        -------
        tuple of cupy.ndarray
            distances : cupy.ndarray of shape (n_samples, k+1)
                Distances to k+1 nearest neighbors (including self).
            indices : cupy.ndarray of shape (n_samples, k+1)
                Indices of k+1 nearest neighbors (including self).

        Notes
        -----
        Private method, should not be called directly. Used by _compute_knn().

        Search Parameters:
        - **IVF methods**: Adaptive n_probes based on index size
        - **CAGRA**: Optimized search width and internal parameters
        - **Flat**: Uses IVF-Flat with high probe count for near-exact results

        Raises
        ------
        ValueError
            If unknown index_type is specified.
        """
        n_samples = X_gpu.shape[0]

        self.logger.info(f"Searching for {k} nearest neighbors using cuVS {index_type}...")

        if index_type == 'flat':
            # For flat/brute force, just use IVF-Flat with many lists for exact search
            # This avoids dtype issues with brute_force module
            n_lists = min(int(np.sqrt(n_samples)), 1024)

            build_kwargs = _merged_params(
                {
                    'n_lists': n_lists,
                    'metric': metric,
                    'add_data_on_build': True,
                },
                self.cuvs_build_params,
            )
            build_params = ivf_flat.IndexParams(**build_kwargs)

            index = ivf_flat.build(build_params, X_gpu)

            # Search with high probe count for near-exact results
            search_params = ivf_flat.SearchParams(**_merged_params(
                {'n_probes': min(build_kwargs['n_lists'], 256)},
                self.cuvs_search_params,
            ))

            distances, indices = ivf_flat.search(
                search_params, index, X_gpu, k+1
            )
            
        elif index_type == 'ivf_flat':
            # IVF search - use stored n_lists (compatible with older cuVS versions)
            n_probes = max(1, min(self._n_lists // 10, 100)) if self._n_lists else 20
            search_params = ivf_flat.SearchParams(**_merged_params(
                {'n_probes': n_probes}, self.cuvs_search_params
            ))
            
            distances, indices = ivf_flat.search(
                search_params, index, X_gpu, k+1
            )
            
        elif index_type == 'ivf_pq':
            # IVF-PQ search - use stored n_lists (compatible with older cuVS versions)
            n_probes = max(1, min(self._n_lists // 10, 200)) if self._n_lists else 20
            search_params = ivf_pq.SearchParams(**_merged_params(
                {'n_probes': n_probes}, self.cuvs_search_params
            ))
            
            distances, indices = ivf_pq.search(
                search_params, index, X_gpu, k+1
            )
            
        elif index_type == 'cagra':
            # CAGRA search
            search_params = cagra.SearchParams(**_merged_params(
                {
                    'max_queries': 0,
                    'itopk_size': min(k * 2, 256),
                    'search_width': 4,
                },
                self.cuvs_search_params,
            ))
            
            distances, indices = cagra.search(
                search_params, index, X_gpu, k+1
            )
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        return distances, indices
    
    def _native_cuvs_metric(self):
        """Return whether the configured metric can run in cuVS."""
        return _is_cuvs_native_metric(self.metric_spec)

    def _cuvs_knn_unavailable_reason(self, n_samples, n_dims):
        """Return a reason cuVS k-NN cannot run for this reducer, or None."""
        del n_samples
        if not CUVS_AVAILABLE:
            return "RAPIDS cuVS is not installed"
        if self.device.type != 'cuda':
            return "cuVS k-NN requires a CUDA device"
        if not self.use_cuvs:
            return "cuVS k-NN is disabled on this reducer"
        if n_dims > 2048:
            return f"cuVS k-NN supports up to 2048 dimensions in this path (got {n_dims})"
        if not self._native_cuvs_metric():
            return "cuVS k-NN only supports named metrics, not custom expressions/callables"
        return None

    def _fallback_to_pytorch_knn(self, X, chunk_size=None, use_fp16=None, reason=None):
        """Fallback from cuVS to a PyTorch k-NN implementation."""
        n_samples, n_dims = X.shape
        if (
            self.knn_backend in ('auto', 'pytorch')
            and _prefer_memory_efficient_for_shape(n_samples, n_dims, self.metric_spec)
        ):
            from .dire_pytorch_memory_efficient import DiRePyTorchMemoryEfficient  # pylint: disable=import-outside-toplevel

            if reason is not None:
                self.logger.info(
                    "Using memory-efficient PyTorch fallback for k-NN "
                    f"because cuVS is unavailable: {reason}"
                )

            fallback = DiRePyTorchMemoryEfficient(
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                init=self.init,
                max_iter_layout=self.max_iter_layout,
                min_dist=self.min_dist,
                spread=self.spread,
                cutoff=self.cutoff,
                n_sample_dirs=self.n_sample_dirs,
                sample_size=self.sample_size,
                neg_ratio=self.neg_ratio,
                verbose=self.verbose,
                random_state=self.random_state,
                use_exact_repulsion=self.use_exact_repulsion,
                metric=self.metric_spec,
                knn_backend='pytorch' if self.knn_backend == 'auto' else self.knn_backend,
                knn_chunk_size=chunk_size,
                memory_fraction=self.knn_memory_fraction,
                knn_memory_fraction=self.knn_memory_fraction,
                knn_broadcast_memory_multiplier=self.knn_broadcast_memory_multiplier,
                normalize=self.normalize,
            )
            fallback.device = self.device
            fallback._compute_knn(X, chunk_size=chunk_size, use_fp16=use_fp16)

            self._knn_indices = fallback._knn_indices
            self._knn_distances = fallback._knn_distances
            self._last_knn_backend = fallback._last_knn_backend
            self._last_knn_chunk_size = fallback._last_knn_chunk_size
            self._last_knn_distance_strategy = fallback._last_knn_distance_strategy
            self._last_knn_reducer = type(fallback).__name__
            return None

        if reason is not None:
            self.logger.info(f"Using PyTorch backend for k-NN ({reason})")
        return super()._compute_knn(X, chunk_size=chunk_size, use_fp16=use_fp16)

    def _compute_knn(self, X, chunk_size=None, use_fp16=None):
        """
        Compute k-NN using cuVS acceleration when available and beneficial.

        This method overrides the parent implementation to use cuVS for k-NN
        computation when it provides performance benefits, automatically falling
        back to PyTorch for cases where cuVS isn't optimal.

        Parameters
        ----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).
        chunk_size : int, optional
            Chunk size for processing (used by fallback PyTorch method). If
            None, the fallback computes a memory-aware chunk size.
        use_fp16 : bool, optional
            Use FP16 precision (used by fallback PyTorch method).
            Note: cuVS requires float32, so FP16 is only used for PyTorch fallback.

        Notes
        -----
        Private method, should not be called directly. Used by fit_transform().

        cuVS Usage Criteria:
        - cuVS backend must be enabled and available
        - Dataset size >= 10,000 samples (cuVS overhead not worth it for smaller datasets)
        - Dimensionality <= 2,048 (cuVS works best for moderate dimensions)
        - Only native named metrics supported (euclidean/l2, sqeuclidean,
          inner_product, cosine)

        If criteria aren't met, falls back to parent PyTorch implementation.

        Side Effects
        ------------
        Sets self._knn_indices and self._knn_distances with computed k-NN graph.
        Cleans up GPU memory after computation.
        """
        n_samples, n_dims = X.shape
        self._knn_indices_torch = None
        self._last_cuvs_knn_method = None

        if self.knn_backend in ('pytorch', 'pykeops'):
            return self._fallback_to_pytorch_knn(
                X,
                chunk_size=chunk_size,
                use_fp16=use_fp16,
                reason=f"knn_backend='{self.knn_backend}' was requested",
            )

        # Check if custom metric expression/callable is specified.
        # cuVS only supports named metrics, not arbitrary tensor expressions.
        unavailable_reason = self._cuvs_knn_unavailable_reason(n_samples, n_dims)
        if self.knn_backend == 'cuvs':
            if unavailable_reason is not None:
                raise RuntimeError(f"knn_backend='cuvs' requested but unavailable: {unavailable_reason}")
        elif unavailable_reason is not None:
            if not self._native_cuvs_metric():
                self.logger.warning(
                    "Custom metric expressions/callables not supported by cuVS. "
                    "Falling back to PyTorch backend for k-NN."
                )
            else:
                self.logger.info(f"Using PyTorch backend for k-NN ({unavailable_reason})")
            return self._fallback_to_pytorch_knn(
                X,
                chunk_size=chunk_size,
                use_fp16=use_fp16,
                reason=unavailable_reason,
            )

        # Decide whether to use cuVS
        use_cuvs_for_this = (
            self.knn_backend == 'cuvs' or
            self.cuvs_knn_method != 'auto' or
            n_samples >= 10000  # cuVS overhead not worth it for small datasets
        )

        if not use_cuvs_for_this:
            # Fall back to PyTorch implementation
            self.logger.info("Using PyTorch backend for k-NN")
            return self._fallback_to_pytorch_knn(
                X,
                chunk_size=chunk_size,
                use_fp16=use_fp16,
                reason="dataset is below the cuVS auto threshold",
            )
        
        # Use cuVS for k-NN
        self._last_knn_backend = 'cuvs'
        self.logger.info(f"Computing {self.n_neighbors}-NN graph using cuVS...")

        cuvs_knn_method = self._select_cuvs_knn_method()
        if cuvs_knn_method == 'all_neighbors':
            return self._compute_knn_all_neighbors(X)
        self._last_cuvs_knn_method = 'index_search'

        # Determine which metric to use for cuVS
        # Default to sqeuclidean (cuVS default), but allow named metrics
        cuvs_metric = 'sqeuclidean'
        if self.metric_spec is not None and isinstance(self.metric_spec, str):
            metric_lower = self.metric_spec.strip().lower()
            if metric_lower in ('euclidean', 'l2'):
                cuvs_metric = 'euclidean'
            elif metric_lower == 'sqeuclidean':
                cuvs_metric = 'sqeuclidean'
            elif metric_lower == 'inner_product':
                cuvs_metric = 'inner_product'
            elif metric_lower == 'cosine':
                # Supported by IVF-Flat, IVF-PQ, and flat index types
                # Note: CAGRA does not support cosine, will be validated in _build_cuvs_index
                cuvs_metric = 'cosine'

        # Convert to CuPy array
        # Note: cuVS requires float32, not float16
        # cuVS also requires C-contiguous (row-major) arrays
        X_gpu = cp.asarray(X, dtype=cp.float32, order='C')

        # Select index type (pass metric to avoid selecting CAGRA for unsupported metrics)
        index_type = self._select_cuvs_index_type(n_samples, n_dims, cuvs_metric)

        # Build index
        if index_type != 'flat':
            self.cuvs_index = self._build_cuvs_index(X_gpu, index_type, cuvs_metric)
        else:
            self.cuvs_index = None
        
        # Search for k-NN
        distances, indices = self._search_cuvs(
            self.cuvs_index, index_type, X_gpu, self.n_neighbors, cuvs_metric
        )
        
        # Convert to NumPy, then remove self. Self is not guaranteed to be the
        # first column when there are duplicate/tied distances.
        indices_cp = cp.asarray(indices)
        distances_cp = cp.asarray(distances)
        self._knn_indices, knn_distances = _remove_self_from_knn(
            cp.asnumpy(indices_cp),
            cp.asnumpy(distances_cp),
            0,
            self.n_neighbors,
        )
        # Match the base reducer's Euclidean-distance contract. Explicit
        # metric='sqeuclidean' intentionally retains squared distances.
        metric_name = (
            self.metric_spec.strip().lower()
            if isinstance(self.metric_spec, str) else None
        )
        if (
                self.metric_spec is None or
                (index_type == 'cagra' and metric_name in ('euclidean', 'l2'))
        ):
            self._knn_distances = np.sqrt(np.maximum(knn_distances, 0.0))
        elif metric_name == 'inner_product':
            self._knn_distances = -knn_distances
        else:
            self._knn_distances = knn_distances
        
        self.logger.info(f"k-NN graph computed: shape {self._knn_indices.shape}")
        
        # Clean up GPU memory
        del X_gpu
        if self.cuvs_index is not None:
            del self.cuvs_index
            self.cuvs_index = None
        cp.get_default_memory_pool().free_all_blocks()
    
    def _initialize_embedding(self, X):
        """
        Initialize embedding using cuML PCA when available, with sklearn fallback.
        
        This method overrides the parent implementation to use GPU-accelerated
        cuML PCA/TruncatedSVD for initialization when available, providing
        significant speedups for high-dimensional data.
        
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
        
        cuML Usage:
        - Uses TruncatedSVD for high-dimensional data (>100 features) for efficiency
        - Uses regular PCA for lower-dimensional data
        - Performs normalization on GPU before converting to PyTorch
        - Uses DLPack for zero-copy GPU tensor transfer
        
        Falls back to parent sklearn-based initialization if:
        - cuML is not available or disabled
        - Initialization method is not 'pca'
        - Any errors occur during cuML processing
        """
        if self.use_cuml and self.init == 'pca':
            self.logger.info("Initializing with cuML PCA (GPU-accelerated)")
            
            # Convert to CuPy array if needed
            if isinstance(X, np.ndarray):
                X_gpu = cp.asarray(X, dtype=cp.float32)
            else:
                X_gpu = X
            
            # Use TruncatedSVD for high-dimensional data (more efficient)
            if X.shape[1] > 100:
                # TruncatedSVD is perfect for high-D to low-D reduction
                pca = cuTruncatedSVD(
                    n_components=self.n_components,
                    random_state=self.random_state
                )
            else:
                # Regular PCA for lower dimensions
                # Note: cuPCA doesn't support random_state parameter
                pca = cuPCA(
                    n_components=self.n_components
                )
            
            # Fit and transform on GPU
            embedding_gpu = pca.fit_transform(X_gpu)
            
            # Convert to PyTorch tensor on GPU
            # cuML returns cupy array, convert to torch
            embedding_cp = cp.asarray(embedding_gpu)
            
            # Normalize on GPU
            embedding_cp -= embedding_cp.mean(axis=0)
            embedding_cp /= embedding_cp.std(axis=0)
            
            # Use DLPack for zero-copy transfer from CuPy to PyTorch.
            # Modern CuPy exposes the Python DLPack protocol directly;
            # ``toDlpack()`` is deprecated.
            embedding_torch = torch.from_dlpack(embedding_cp)
            
            return embedding_torch.to(self.device)
        
        # Fall back to CPU sklearn PCA
        return super()._initialize_embedding(X)
    
    def fit_transform(self, X, y=None):
        """
        Fit the model and transform data with cuVS/cuML acceleration.
        
        This method extends the parent implementation with intelligent backend
        selection and logging to inform users about the acceleration being used.
        
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
        Backend Selection Logic:
        - Uses cuVS for k-NN if dataset is large enough and cuVS is available
        - Uses cuML for PCA initialization if available and init='pca'
        - Falls back to PyTorch implementations automatically
        
        Performance Benefits:
        - cuVS k-NN: 10-100x speedup for large datasets
        - cuML PCA: 5-50x speedup for high-dimensional initialization
        
        Examples
        --------
        Large dataset with cuVS acceleration::
        
            import numpy as np
            from dire_rapids import DiReCuVS
            
            # 500K points, 1000 dimensions  
            X = np.random.randn(500000, 1000)
            
            reducer = DiReCuVS(verbose=True)  # Will log backend selection
            embedding = reducer.fit_transform(X)
            # Output: "Using cuVS-accelerated backend for 500000 points"
        """
        # Log backend being used
        cuvs_unavailable_reason = self._cuvs_knn_unavailable_reason(X.shape[0], X.shape[1])
        if self.use_cuvs and X.shape[0] >= 10000 and cuvs_unavailable_reason is None:
            self.logger.info(f"Using cuVS-accelerated backend for {X.shape[0]} points")
        else:
            if cuvs_unavailable_reason is None:
                self.logger.info(f"Using PyTorch backend for {X.shape[0]} points")
            else:
                self.logger.info(
                    f"Using PyTorch fallback for {X.shape[0]} points "
                    f"({cuvs_unavailable_reason})"
                )
        
        return super().fit_transform(X, y)
