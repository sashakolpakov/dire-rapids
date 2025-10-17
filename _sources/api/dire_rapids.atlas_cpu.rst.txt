dire\_rapids.atlas\_cpu module
================================

.. automodule:: dire_rapids.atlas_cpu
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

CPU implementation of local kNN atlas topology computation using Hodge Laplacian.

**Status:** ⚠️ Under active development - API may change

Key Functions
-------------

- ``compute_h0_h1_atlas_cpu``: Main entry point for computing H0/H1 Betti curves
- ``compute_betti_curve_from_dgm``: Convert persistence diagram to Betti curve
- ``compute_h0_h1_hodge``: Hodge Laplacian-based H0/H1 computation

Algorithm
---------

1. Construct kNN graph for each point's local neighborhood
2. Use combinatorial Nystroem approximation via Hodge Laplacian
3. Extract H0/H1 persistence diagrams from eigenvalue spectrum
4. Aggregate into global Betti curves

Examples
--------

.. code-block:: python

    from dire_rapids.atlas_cpu import compute_h0_h1_atlas_cpu
    import numpy as np

    # Generate sample data
    X = np.random.randn(1000, 50).astype(np.float32)

    # Compute H0/H1 Betti curves
    betti_curves, stats = compute_h0_h1_atlas_cpu(
        X,
        n_neighbors=30,
        n_radii=50
    )

    print(f"Betti curves shape: {betti_curves.shape}")  # (n_radii, 2)
    print(f"H0 at first radius: {betti_curves[0, 0]}")
    print(f"H1 at first radius: {betti_curves[0, 1]}")
    print(f"Radii: {stats['radii'][:5]}")

See Also
--------

- :mod:`dire_rapids.atlas_gpu`: GPU-accelerated version using cuVS
- ``tests/test_atlas_approach.py``: Comprehensive usage examples
- ``tests/test_atlas_scaling.py``: Scaling benchmarks
