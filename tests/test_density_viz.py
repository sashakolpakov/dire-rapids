# test_density_viz.py

"""
CPU unit tests for the density / scatter embedding visualization helper.

These exercise ``build_embedding_figure`` directly (no model fitting, no GPU):
the large-point-count density fallback, per-category overlay, auto-switching,
and the key efficiency invariant that the density payload does not grow with
the number of points.
"""

import numpy as np
import pytest

from dire_rapids import build_embedding_figure


def _blobs(n, k=5, dims=2, seed=0):
    """k Gaussian blobs in `dims` dimensions with integer cluster labels."""
    rng = np.random.default_rng(seed)
    per = n // k
    centers = rng.uniform(-10, 10, size=(k, dims))
    pts = np.repeat(centers, per, axis=0) + rng.normal(scale=1.0, size=(per * k, dims))
    labels = np.repeat(np.arange(k), per)
    return pts, labels


def _types(fig):
    return [trace.type for trace in fig.data]


class TestRenderModeSelection:
    """mode / density_threshold control whether we draw scatter or density."""

    def test_auto_scatter_below_threshold(self):
        X, y = _blobs(2_000, k=3)
        fig = build_embedding_figure(X, y, mode="auto", density_threshold=50_000)
        assert _types(fig) == ["scattergl"] or all(t == "scattergl" for t in _types(fig))

    def test_auto_density_above_threshold(self):
        X, y = _blobs(60_000, k=3)
        fig = build_embedding_figure(X, y, mode="auto", density_threshold=50_000,
                                     categorical_labels=True)
        assert all(t == "contour" for t in _types(fig))
        assert "Density" in fig.layout.title.text

    def test_density_mode_forces_density_small_data(self):
        X, y = _blobs(300, k=3)
        fig = build_embedding_figure(X, y, mode="density", categorical_labels=True)
        assert all(t == "contour" for t in _types(fig))

    def test_scatter_mode_forces_scatter_large_data(self):
        X, y = _blobs(60_000, k=3)
        fig = build_embedding_figure(X, y, mode="scatter", density_threshold=50_000)
        assert all(t == "scattergl" for t in _types(fig))

    def test_invalid_mode_raises(self):
        X, _ = _blobs(100)
        with pytest.raises(ValueError):
            build_embedding_figure(X, None, mode="nonsense")

    def test_3d_density_falls_back_to_scatter(self):
        X, _ = _blobs(60_000, k=3, dims=3)
        fig = build_embedding_figure(X, None, n_dims=3, mode="density")
        assert all(t == "scatter3d" for t in _types(fig))

    def test_non_2d_3d_returns_none(self):
        assert build_embedding_figure(np.zeros((10, 5)), None) is None


class TestDensityLabelHandling:
    """Density rendering adapts to label kind."""

    def test_unlabeled_single_count_heatmap(self):
        X, _ = _blobs(60_000)
        fig = build_embedding_figure(X, None, mode="density")
        assert _types(fig) == ["heatmap"]
        assert fig.data[0].colorbar.title.text == "Count"

    def test_continuous_labels_mean_heatmap(self):
        X, _ = _blobs(60_000)
        fig = build_embedding_figure(X, X[:, 0], mode="density", categorical_labels=False)
        assert _types(fig) == ["heatmap"]
        assert fig.data[0].colorbar.title.text == "Mean label"

    def test_categorical_per_category_overlay(self):
        X, y = _blobs(60_000, k=4)
        fig = build_embedding_figure(X, y, mode="density", categorical_labels=True)
        assert _types(fig) == ["contour"] * 4
        assert sorted(t.name for t in fig.data) == ["0", "1", "2", "3"]

    def test_too_many_categories_single_heatmap(self):
        X, _ = _blobs(60_000)
        many = np.random.default_rng(1).integers(0, 30, size=X.shape[0])
        fig = build_embedding_figure(X, many, mode="density", categorical_labels=True)
        assert _types(fig) == ["heatmap"]


class TestEfficiency:
    """The density payload must stay bounded as the point count grows."""

    def test_payload_constant_in_n(self):
        sizes = []
        for n in (50_000, 200_000):
            X, y = _blobs(n, k=5)
            fig = build_embedding_figure(X, y, mode="density", categorical_labels=True)
            sizes.append(len(fig.to_json()))
        # 4x more points must not meaningfully change the shipped payload.
        assert abs(sizes[0] - sizes[1]) / sizes[0] < 0.05

    def test_density_smaller_than_full_scatter_at_scale(self):
        X, y = _blobs(500_000, k=5)
        dense = build_embedding_figure(X, y, mode="density", categorical_labels=True)
        scatter = build_embedding_figure(X, y, mode="scatter", max_points=len(X))
        assert len(dense.to_json()) < len(scatter.to_json())


class TestScatterBackCompat:
    """Scatter path keeps its prior shape."""

    def test_numeric_labels_treated_as_continuous_single_trace(self):
        X, y = _blobs(40, k=2)
        fig = build_embedding_figure(X, y, mode="auto", categorical_labels=False)
        assert _types(fig) == ["scattergl"]

    def test_subsample_caps_scatter_points(self):
        X, y = _blobs(5_000, k=1)
        fig = build_embedding_figure(X, y, mode="scatter", max_points=1_000)
        assert len(fig.data[0].x) == 1_000

    def test_single_point_density_does_not_crash(self):
        fig = build_embedding_figure(np.zeros((1, 2)), None, mode="density")
        assert _types(fig) == ["heatmap"]
