"""Tests for the incremental atlas Betti backend."""

import numpy as np

from dire_rapids.betti_curve import compute_betti_curve_fast


def test_incremental_atlas_full_complex_circle():
    """n_steps=1 still returns the full atlas complex Betti numbers."""
    n_samples = 120
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    data = np.column_stack([np.cos(theta), np.sin(theta)]).astype(np.float32)

    result = compute_betti_curve_fast(
        data,
        k_neighbors=15,
        density_threshold=0.7,
        overlap_factor=1.5,
        n_steps=1,
    )

    assert int(result["beta_0"][0]) == 1
    assert int(result["beta_1"][0]) == 1
    assert int(result["n_edges_active"][0]) > 0
    assert int(result["n_triangles_active"][0]) > 0


def test_incremental_atlas_counts_are_monotone_with_filtration():
    """Active simplex counts increase as thresholds increase."""
    rng = np.random.default_rng(42)
    data = rng.normal(size=(80, 4)).astype(np.float32)

    result = compute_betti_curve_fast(
        data,
        k_neighbors=10,
        density_threshold=0.7,
        overlap_factor=1.3,
        n_steps=12,
    )

    order = np.argsort(result["filtration_values"])
    edges = result["n_edges_active"][order]
    triangles = result["n_triangles_active"][order]

    assert np.all(np.diff(edges) >= 0)
    assert np.all(np.diff(triangles) >= 0)
    assert np.all(result["beta_0"] >= 1)
    assert np.all(result["beta_1"] >= 0)
