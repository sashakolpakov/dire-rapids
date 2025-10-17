dire\_rapids.atlas\_gpu module
================================

.. automodule:: dire_rapids.atlas_gpu
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

GPU-accelerated implementation of local kNN atlas topology computation using NVIDIA RAPIDS cuVS.

**Status:** ⚠️ Under active development - API may change

**Requirements:** RAPIDS cuVS, giotto-ph for persistence computation

Key Functions
-------------

- ``compute_h0_h1_atlas_gpu``: Main entry point for GPU-accelerated H0/H1 Betti curves
- Utilizes cuVS for fast kNN computation on large datasets
- Falls back to giotto-ph for persistence diagram extraction

Algorithm
---------

1. Use cuVS for GPU-accelerated kNN graph construction
2. Transfer local neighborhoods to CPU for persistence computation
3. Compute persistence diagrams using giotto-ph backend
4. Aggregate Betti curves across all local patches

Performance
-----------

- Scales to millions of points with GPU acceleration
- kNN computation is the primary bottleneck (addressed by cuVS)
- Persistence computation currently on CPU (giotto-ph)

Examples
--------

.. code-block:: python

    from dire_rapids.atlas_gpu import compute_h0_h1_atlas_gpu
    import numpy as np

    # Generate large-scale data
    X = np.random.randn(100_000, 100).astype(np.float32)

    # Compute H0/H1 Betti curves with GPU acceleration
    betti_curves = compute_h0_h1_atlas_gpu(
        X,
        n_neighbors=30,
        n_radii=50,
        use_cuvs=True  # Enable cuVS acceleration
    )

    print(f"Betti curves shape: {betti_curves.shape}")  # (n_radii, 2)
    print(f"Average H0: {betti_curves[:, 0].mean()}")
    print(f"Average H1: {betti_curves[:, 1].mean()}")

Comparison with CPU
-------------------

**Advantages:**
- 10-100x faster kNN computation on large datasets (>100k points)
- Better memory efficiency for high-dimensional data

**Limitations:**
- Requires NVIDIA GPU with CUDA support
- Requires RAPIDS cuVS installation
- Persistence computation still on CPU

See Also
--------

- :mod:`dire_rapids.atlas_cpu`: CPU-only version (no RAPIDS required)
- :mod:`dire_rapids.dire_cuvs`: cuVS-accelerated dimensionality reduction
- ``tests/test_atlas_gpu.py``: GPU-specific tests
