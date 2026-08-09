# test_cpu_basic.py

"""
Basic CPU unit tests for dire-rapids.
These tests run on CPU only and use small datasets for CI/CD.
"""

import pytest
import numpy as np
import torch
from sklearn.datasets import make_blobs, make_swiss_roll

# Import dire-rapids
import dire_rapids.dire_pytorch as dire_pytorch_module
from dire_rapids import DiRePyTorch, create_dire


class TestDiRePyTorchBasic:
    """Basic sanity checks for DiRePyTorch on CPU."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        np.random.seed(42)
        torch.manual_seed(42)
        # Force CPU for tests
        self.device = torch.device('cpu')  # pylint: disable=attribute-defined-outside-init
        
    def test_import(self):
        """Test that the package can be imported."""
        assert DiRePyTorch is not None
        
    def test_initialization(self):
        """Test DiRePyTorch initialization with various parameters."""
        # Default initialization
        model = DiRePyTorch()
        assert model.n_components == 2
        assert model.n_neighbors == 16
        
        # Custom initialization
        model = DiRePyTorch(
            n_components=3,
            n_neighbors=10,
            max_iter_layout=50,
            verbose=False
        )
        assert model.n_components == 3
        assert model.n_neighbors == 10
        assert model.max_iter_layout == 50

    def test_fit_transform_small_data(self):
        """Test fit_transform on a small dataset."""
        # Create small test data
        X, _ = make_blobs(n_samples=100, n_features=10, centers=3, random_state=42)  # _ for labels (sklearn compatibility)
        
        # Fit and transform
        model = DiRePyTorch(n_components=2, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)
        
        # Check output shape
        assert X_embedded.shape == (100, 2)
        assert isinstance(X_embedded, np.ndarray)
        
        # Check that values are finite
        assert np.all(np.isfinite(X_embedded))

    def test_visualize_after_fit(self):
        """Test that visualize imports optional plotting dependencies lazily."""
        X, y = make_blobs(n_samples=20, n_features=5, centers=2, random_state=42)

        model = DiRePyTorch(n_components=2, max_iter_layout=1, verbose=False, random_state=42)
        model.fit_transform(X)
        fig = model.visualize(labels=y, max_points=20)

        assert fig is not None
        assert len(fig.data) == 1
        
    def test_different_n_components(self):
        """Test with different number of components."""
        X, _ = make_blobs(n_samples=50, n_features=10, centers=2, random_state=42)
        
        for n_components in [1, 2, 3]:
            model = DiRePyTorch(n_components=n_components, max_iter_layout=10, verbose=False)
            X_embedded = model.fit_transform(X)
            assert X_embedded.shape == (50, n_components)
            
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with the same random seed.

        Uses Procrustes alignment to account for rotation/reflection invariance
        in dimensionality reduction embeddings.
        """
        from scipy.spatial import procrustes  # pylint: disable=import-outside-toplevel

        X, _ = make_blobs(n_samples=50, n_features=5, centers=2, random_state=42)

        # First run
        model1 = DiRePyTorch(random_state=123, max_iter_layout=10, verbose=False)
        X_embedded1 = model1.fit_transform(X)

        # Second run with same seed
        model2 = DiRePyTorch(random_state=123, max_iter_layout=10, verbose=False)
        X_embedded2 = model2.fit_transform(X)

        # Use Procrustes analysis to align embeddings (accounts for rotation/reflection)
        mtx1, mtx2, disparity = procrustes(X_embedded1, X_embedded2)

        # Disparity should be very small for reproducible results
        assert disparity < 0.01, f"Embeddings not reproducible (disparity={disparity:.6f})"
        
    @pytest.mark.skip(reason="TODO: Add transform() method")
    def test_fit_then_transform(self):
        """Test separate fit and transform methods."""
        X, _ = make_blobs(n_samples=80, n_features=8, centers=2, random_state=42)
        
        model = DiRePyTorch(n_components=2, max_iter_layout=10, verbose=False)
        
        # Fit first
        model.fit(X)
        assert model._layout is not None
        
        # Then transform
        X_embedded = model.transform(X)
        assert X_embedded.shape == (80, 2)
        
        # Compare with fit_transform
        model2 = DiRePyTorch(n_components=2, max_iter_layout=10, verbose=False, 
                            random_state=model.random_state)
        X_embedded2 = model2.fit_transform(X)
        np.testing.assert_array_almost_equal(X_embedded, X_embedded2, decimal=5)
        
    def test_different_initializations(self):
        """Test different initialization methods."""
        X, _ = make_blobs(n_samples=60, n_features=6, centers=2, random_state=42)
        
        # PCA initialization (default)
        model_pca = DiRePyTorch(init='pca', max_iter_layout=10, verbose=False)
        X_pca = model_pca.fit_transform(X)
        assert X_pca.shape == (60, 2)
        
        # Random initialization
        model_random = DiRePyTorch(init='random', max_iter_layout=10, verbose=False)
        X_random = model_random.fit_transform(X)
        assert X_random.shape == (60, 2)
        
        # Check both produce finite values
        assert np.all(np.isfinite(X_pca))
        assert np.all(np.isfinite(X_random))
        
    def test_high_dimensional_data(self):
        """Test with higher dimensional data."""
        # Create high-dimensional data
        X = np.random.randn(50, 100).astype(np.float32)
        
        model = DiRePyTorch(n_components=2, n_neighbors=5, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)
        
        assert X_embedded.shape == (50, 2)
        assert np.all(np.isfinite(X_embedded))
        
    def test_swiss_roll_data(self):
        """Test on Swiss roll dataset (common DR benchmark)."""
        X, _ = make_swiss_roll(n_samples=100, random_state=42)  # _ for color (sklearn compatibility)
        
        model = DiRePyTorch(n_components=2, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)
        
        assert X_embedded.shape == (100, 2)
        assert np.all(np.isfinite(X_embedded))
        
    def test_min_neighbors_validation(self):
        """Test that n_neighbors is validated correctly."""
        X, _ = make_blobs(n_samples=20, n_features=5, centers=2, random_state=42)

        # n_neighbors should be less than n_samples
        model = DiRePyTorch(n_neighbors=15, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)
        assert X_embedded.shape == (20, 2)

        # Should handle case where n_neighbors >= n_samples
        # Should issue warning and adjust n_neighbors to n_samples - 1
        model = DiRePyTorch(n_neighbors=25, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)
        # Should internally adjust n_neighbors to 19 (n_samples - 1)
        assert X_embedded.shape == (20, 2)
        assert model.n_neighbors == 19  # Should be adjusted
        
    @pytest.mark.skip(reason="TODO: Handle edge cases with small datasets")
    def test_single_point(self):
        """Test edge case with single data point."""
        X = np.array([[1.0, 2.0, 3.0]])
        
        model = DiRePyTorch(n_components=2, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)
        
        assert X_embedded.shape == (1, 2)
        assert np.all(np.isfinite(X_embedded))
        
    @pytest.mark.skip(reason="TODO: Handle edge cases with small datasets")
    def test_identical_points(self):
        """Test with identical data points."""
        X = np.ones((10, 5))
        
        model = DiRePyTorch(n_components=2, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)
        
        assert X_embedded.shape == (10, 2)
        assert np.all(np.isfinite(X_embedded))
        
    @pytest.mark.parametrize("n_samples,n_features", [
        (30, 5),
        (50, 10),
        (100, 20),
    ])
    def test_various_data_sizes(self, n_samples, n_features):
        """Test with various data sizes."""
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        
        model = DiRePyTorch(
            n_components=2, 
            n_neighbors=min(10, n_samples - 1),
            max_iter_layout=10, 
            verbose=False
        )
        X_embedded = model.fit_transform(X)
        
        assert X_embedded.shape == (n_samples, 2)
        assert np.all(np.isfinite(X_embedded))
        
    def test_data_types(self):
        """Test with different data types."""
        X_float64 = np.random.randn(50, 10)
        X_float32 = X_float64.astype(np.float32)
        
        model = DiRePyTorch(n_components=2, max_iter_layout=10, verbose=False)
        
        # Should handle both float32 and float64
        X_embedded_64 = model.fit_transform(X_float64)
        assert X_embedded_64.shape == (50, 2)
        
        X_embedded_32 = model.fit_transform(X_float32)
        assert X_embedded_32.shape == (50, 2)
        
    def test_spread_min_dist_parameters(self):
        """Test that spread and min_dist parameters affect the embedding."""
        X, _ = make_blobs(n_samples=50, n_features=10, centers=3, random_state=42)
        
        # Different spread values
        model1 = DiRePyTorch(spread=0.5, min_dist=0.01, max_iter_layout=20, 
                            verbose=False, random_state=42)
        X1 = model1.fit_transform(X)
        
        model2 = DiRePyTorch(spread=2.0, min_dist=0.01, max_iter_layout=20, 
                            verbose=False, random_state=42)
        X2 = model2.fit_transform(X)
        
        # Embeddings should be different
        assert not np.allclose(X1, X2)
        
        # Different min_dist values
        model3 = DiRePyTorch(spread=1.0, min_dist=0.001, max_iter_layout=20, 
                            verbose=False, random_state=42)
        X3 = model3.fit_transform(X)
        
        model4 = DiRePyTorch(spread=1.0, min_dist=0.1, max_iter_layout=20, 
                            verbose=False, random_state=42)
        X4 = model4.fit_transform(X)
        
        # Embeddings should be different
        assert not np.allclose(X3, X4)


class TestDiRePyTorchNormalization:
    """Regression tests for input normalization / fp16 safety."""

    def _cluster_separation(self, emb, labels):
        """Ratio of between-cluster to within-cluster std — a label-aware
        proxy for whether the embedding preserved cluster structure."""
        classes = np.unique(labels)
        centroids = np.stack([emb[labels == c].mean(axis=0) for c in classes])
        within = np.mean([emb[labels == c].std(axis=0).mean() for c in classes])
        between = centroids.std(axis=0).mean()
        return between / max(within, 1e-12)

    def test_raw_scale_high_dim(self):
        """Unnormalized high-D inputs must not collapse.

        Regression test for the fp16-overflow bug: before the fix, high-D data
        at a large scale (e.g. raw [0, 255] pixels) overflowed fp16 squared
        distances inside _compute_knn, silently corrupting the neighbor graph
        and producing a uniform blob. The internal normalization in
        fit_transform and the fp16 safety guard together should prevent this.
        """
        rng = np.random.default_rng(0)
        n_per_cluster = 60
        n_features = 600
        centers = rng.standard_normal((4, n_features)).astype(np.float32) * 3.0
        X = np.concatenate([
            centers[c] + 0.3 * rng.standard_normal((n_per_cluster, n_features)).astype(np.float32)
            for c in range(4)
        ], axis=0)
        y = np.repeat(np.arange(4), n_per_cluster)
        # Push data into the fp16-unsafe regime (raw-pixel-like scale).
        X = (X * 85.0 + 128.0).clip(0, 255).astype(np.float32)

        model = DiRePyTorch(n_components=2, n_neighbors=15, max_iter_layout=64,
                            verbose=False, random_state=0)
        emb = model.fit_transform(X)

        assert np.all(np.isfinite(emb))
        ratio = self._cluster_separation(emb, y)
        # Collapsed embeddings give ratios ~0.01; well-separated give >1.
        assert ratio > 0.5, (
            f"cluster structure collapsed (between/within std ratio = {ratio:.3f}); "
            f"likely an fp16-overflow or normalization regression"
        )

    def test_spectral_init(self):
        """Spectral init produces a valid, finite embedding that separates
        classes on a simple dataset.

        Earlier versions of this test compared ``init='spectral'`` against
        ``init='random'`` with a 0.8 threshold. That assertion is brittle:
        random-projection variance across numpy/scipy/torch wheel versions
        (different for each Python minor) flips the comparison, so CI
        failed on 3.11/3.12 while passing on 3.10 with no real bug. The
        functional property we actually care about is that spectral init
        separates the blobs — any between/within ratio > 1 suffices.
        """
        X, y = make_blobs(n_samples=300, n_features=15, centers=3,
                          cluster_std=0.8, random_state=0)
        r_spec = DiRePyTorch(n_components=2, n_neighbors=10, max_iter_layout=0,
                             init='spectral', verbose=False, random_state=0)
        emb_spec = r_spec.fit_transform(X)
        assert emb_spec.shape == (300, 2)
        assert np.all(np.isfinite(emb_spec))
        ratio_spec = self._cluster_separation(emb_spec, y)
        assert ratio_spec > 1.0, (
            f"spectral init did not separate blobs: "
            f"between/within ratio = {ratio_spec:.2f} (expected > 1)"
        )

    def test_unsupported_topology_preset_is_not_exported(self):
        """A merge or rebase must not resurrect the failed public preset."""
        import dire_rapids

        assert not hasattr(dire_rapids, "TOPOLOGY_TUNED")
        assert "TOPOLOGY_TUNED" not in dire_rapids.__all__
        assert not hasattr(dire_rapids.presets, "TOPOLOGY_TUNED")
        assert "TOPOLOGY_TUNED" not in dire_rapids.presets.__all__

    def test_ripser_tuned_preset_is_narrowly_named_and_frozen(self):
        """The held-out replacement changes only spread from current default."""
        import dire_rapids

        expected = {
            "init": "pca",
            "n_neighbors": 16,
            "spread": 0.8,
            "min_dist": 1e-2,
            "cutoff": 42.0,
            "neg_ratio": 8,
            "max_iter_layout": 128,
        }
        assert dire_rapids.RIPSER_TUNED == expected
        assert dire_rapids.presets.RIPSER_TUNED == expected
        assert "RIPSER_TUNED" in dire_rapids.__all__
        assert dire_rapids.presets.__all__ == ["RIPSER_TUNED"]
        assert not hasattr(dire_rapids, "ATLAS_TUNED")

    def test_frozen_topology_preset_audit_retains_failure_summary(self):
        """The evidence behind preset removal remains a checked fixture."""
        import csv
        from pathlib import Path

        fixture = (
            Path(__file__).with_name("data")
            / "topology_preset_atlas_audit.csv"
        )
        with fixture.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))

        gaps = [float(row["relative_gap"]) for row in rows]
        intervals = [
            (
                float(row["paired_mean_95pct_low"]),
                float(row["paired_mean_95pct_high"]),
            )
            for row in rows
        ]

        assert len(rows) == 12
        assert {int(row["paired_count"]) for row in rows} == {20}
        assert sum(gap > 0 for gap in gaps) == 10
        assert sum(gap > 0.05 for gap in gaps) == 9
        assert sum(low > 0 for low, _ in intervals) == 9
        assert not any(
            gap < -0.05 or high < 0
            for gap, (_, high) in zip(gaps, intervals)
        )

    def test_betti_curve_ripser_on_circle(self):
        """Ripser backend returns correct Betti curve shape and identifies one
        dominant 1-cycle on a noisy circle."""
        try:
            from dire_rapids.betti_curve import compute_betti_curve_ripser
            import ripser  # noqa: F401 — optional dep, skip if missing
        except ImportError:
            pytest.skip("ripser not installed; skipping ripser backend test")
        rng = np.random.default_rng(0)
        t = rng.uniform(0, 2 * np.pi, 300)
        X = np.column_stack([np.cos(t), np.sin(t)]).astype(np.float32)
        X += rng.normal(0, 0.02, X.shape).astype(np.float32)
        result = compute_betti_curve_ripser(X, n_steps=20, maxdim=1)

        # Shape contract matches the other backends
        for key in ('filtration_values', 'beta_0', 'beta_1',
                    'n_edges_active', 'n_triangles_active'):
            assert key in result, f"missing key {key}"
            assert len(result[key]) == 20

        # β_0 must start at n (everyone is their own component)
        assert result['beta_0'][0] == 300
        # ...and drop to 1 by the end (the noisy circle is connected)
        assert result['beta_0'][-1] == 1
        # Expect at least one β_1 bar whose lifetime persists over several
        # filtration steps — the big circular loop.
        assert int(result['beta_1'].max()) >= 1

    def test_betti_curve_selector_defaults_to_atlas(self, monkeypatch):
        """Installing Ripser must not silently change the public default."""
        import dire_rapids.betti_curve as betti_curve

        sentinel = {"backend": "cpu-atlas"}

        def unexpected_ripser(*args, **kwargs):
            raise AssertionError("default selector must not call ripser")

        monkeypatch.setattr(betti_curve, "compute_betti_curve_ripser", unexpected_ripser)
        monkeypatch.setattr(
            betti_curve,
            "compute_betti_curve_fast",
            lambda *args, **kwargs: sentinel,
        )

        result = betti_curve.compute_betti_curve(
            np.zeros((4, 2), dtype=np.float32),
            use_gpu=False,
        )

        assert result is sentinel

    def test_betti_curve_selector_keeps_ripser_as_explicit_option(self, monkeypatch):
        """Callers can still request Ripser without making it the default."""
        import dire_rapids.betti_curve as betti_curve

        sentinel = {"backend": "ripser"}
        monkeypatch.setattr(
            betti_curve,
            "compute_betti_curve_ripser",
            lambda *args, **kwargs: sentinel,
        )

        result = betti_curve.compute_betti_curve(
            np.zeros((4, 2), dtype=np.float32),
            use_gpu=False,
            prefer_ripser=True,
        )

        assert result is sentinel

    def test_normalize_false_preserves_old_behavior(self):
        """normalize=False should leave _data untouched, for back-compat."""
        X = np.full((40, 10), 7.0, dtype=np.float32)
        X += np.random.default_rng(0).standard_normal(X.shape).astype(np.float32)
        model = DiRePyTorch(n_components=2, n_neighbors=5, max_iter_layout=5,
                            verbose=False, normalize=False, random_state=0)
        model.fit_transform(X)
        # With normalize=False, _data is the float32 copy of X (mean ~7, not 0).
        assert abs(model._data.mean() - X.mean()) < 1e-5


class TestKnnBackendSelection:
    """CPU-safe coverage for explicit k-NN backend selection."""

    def test_create_dire_passes_knn_backend_alias(self):
        """Factory backend selection and k-NN backend selection are separate."""
        X = np.random.default_rng(0).standard_normal((12, 4)).astype(np.float32)
        model = create_dire(
            backend='pytorch_cpu',
            knn_backend='torch',
            n_neighbors=2,
            verbose=False,
        )

        assert model.knn_backend == 'pytorch'
        model._compute_knn(X)
        assert model._last_knn_backend == 'pytorch'
        assert model._knn_indices.shape == (12, 2)

    def test_invalid_knn_backend_raises(self):
        """Unknown k-NN engine names fail at construction/factory time."""
        with pytest.raises(ValueError, match="Unknown knn_backend"):
            DiRePyTorch(knn_backend='not-a-backend', verbose=False)

        with pytest.raises(ValueError, match="Unknown knn_backend"):
            create_dire(knn_backend='not-a-backend', verbose=False)

    def test_forced_cuvs_is_strict_on_cpu(self):
        """Manual cuVS selection raises instead of silently falling back."""
        X = np.random.default_rng(1).standard_normal((12, 4)).astype(np.float32)
        model = DiRePyTorch(knn_backend='cuvs', n_neighbors=2, verbose=False)
        model.device = torch.device('cpu')

        with pytest.raises(RuntimeError, match="knn_backend='cuvs' requested"):
            model._compute_knn(X)

        with pytest.raises(RuntimeError, match="knn_backend='cuvs' requested"):
            create_dire(backend='pytorch_cpu', knn_backend='cuvs', verbose=False)

    def test_forced_pykeops_requires_pykeops(self, monkeypatch):
        """Manual PyKeOps selection raises clearly when PyKeOps is unavailable."""
        X = np.random.default_rng(2).standard_normal((12, 4)).astype(np.float32)
        monkeypatch.setattr(dire_pytorch_module, "PYKEOPS_AVAILABLE", False)
        model = DiRePyTorch(knn_backend='pykeops', n_neighbors=2, verbose=False)

        with pytest.raises(RuntimeError, match="PyKeOps is not installed"):
            model._compute_knn(X)

    @pytest.mark.parametrize(
        ("metric", "strategy"),
        [
            ("cosine", "matmul_cosine"),
            ("inner_product", "matmul_inner_product"),
            ("sqeuclidean", "matmul_sqeuclidean"),
        ],
    )
    def test_named_metrics_use_matmul_pytorch_paths(self, metric, strategy):
        """Built-in named metrics must not use broadcast custom metric tensors."""
        X = np.random.default_rng(3).standard_normal((16, 8)).astype(np.float32)
        model = DiRePyTorch(
            metric=metric,
            knn_backend='pytorch',
            n_neighbors=2,
            verbose=False,
        )

        model._compute_knn(X)

        assert model._last_knn_backend == 'pytorch'
        assert model._last_knn_distance_strategy == strategy
        assert model._knn_indices.shape == (16, 2)

    def test_chunk_estimator_accounts_for_broadcast_custom_metrics(self):
        """Mocked high-D sizing should distinguish matmul-safe and broadcast metrics."""
        n_samples = 20_000
        n_dims = 2_049
        available_memory = 1024 ** 3

        cosine_model = DiRePyTorch(metric='cosine', verbose=False)
        custom_model = DiRePyTorch(metric='(x - y).abs().sum(-1)', verbose=False)

        cosine_chunk = cosine_model._compute_auto_knn_chunk_size(
            n_samples,
            n_dims,
            torch.float32,
            use_pykeops=False,
            available_memory=available_memory,
        )
        custom_chunk = custom_model._compute_auto_knn_chunk_size(
            n_samples,
            n_dims,
            torch.float32,
            use_pykeops=False,
            available_memory=available_memory,
        )

        assert not cosine_model._metric_requires_broadcast_tensors(use_pykeops=False)
        assert custom_model._metric_requires_broadcast_tensors(use_pykeops=False)
        assert cosine_model._estimate_knn_memory_per_query(n_samples, n_dims, 4) == n_samples * 4
        assert custom_model._estimate_knn_memory_per_query(n_samples, n_dims, 4) > n_samples * n_dims * 4
        assert cosine_chunk > custom_chunk
        assert custom_chunk < 1000

    def test_high_dimensional_cosine_uses_safe_matmul_chunk(self):
        """Regression smoke for d > 2048 cosine without a broadcast (chunk, n, d) tensor."""
        X = np.random.default_rng(4).standard_normal((24, 2049)).astype(np.float32)
        model = DiRePyTorch(
            metric='cosine',
            knn_backend='pytorch',
            n_neighbors=2,
            verbose=False,
        )

        model._compute_knn(X)

        assert model._last_knn_backend == 'pytorch'
        assert model._last_knn_distance_strategy == 'matmul_cosine'
        assert model._last_knn_chunk_size <= X.shape[0]
        assert model._knn_indices.shape == (24, 2)

    def test_cuvs_high_dimensional_auto_fallback_uses_memory_efficient_path(self):
        """cuVS fallback after d > 2048 should not carry a hard 50000 chunk."""
        from dire_rapids.dire_cuvs import DiReCuVS  # pylint: disable=import-outside-toplevel

        X = np.random.default_rng(5).standard_normal((24, 2049)).astype(np.float32)
        model = DiReCuVS(
            metric='cosine',
            n_neighbors=2,
            use_cuvs=False,
            verbose=False,
        )

        model._compute_knn(X)

        assert model._last_knn_reducer == 'DiRePyTorchMemoryEfficient'
        assert model._last_knn_backend == 'pytorch'
        assert model._last_knn_distance_strategy == 'matmul_cosine'
        assert model._last_knn_chunk_size != 50000
        assert model._last_knn_chunk_size <= X.shape[0]

    def test_auto_factory_prefers_memory_efficient_when_cuvs_metric_unsupported(self, monkeypatch):
        """backend='auto' should avoid cuVS wrapper when metric policy already rules cuVS out."""
        import dire_rapids.dire_cuvs as dire_cuvs_module  # pylint: disable=import-outside-toplevel
        from dire_rapids import DiRePyTorchMemoryEfficient  # pylint: disable=import-outside-toplevel

        monkeypatch.setattr(dire_cuvs_module, "CUVS_AVAILABLE", True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args, **kwargs: "mock cuda")

        model = create_dire(
            backend='auto',
            metric='(x - y).abs().sum(-1)',
            verbose=False,
        )

        assert isinstance(model, DiRePyTorchMemoryEfficient)

    def test_memory_efficient_knn_fraction_uses_knn_specific_knob(self, monkeypatch):
        """The memory-efficient reducer should honor knn_memory_fraction for k-NN chunks."""
        from dire_rapids import DiRePyTorchMemoryEfficient  # pylint: disable=import-outside-toplevel

        model = DiRePyTorchMemoryEfficient(
            memory_fraction=0.50,
            knn_memory_fraction=0.05,
            verbose=False,
        )
        monkeypatch.setattr(model, "_get_available_memory", lambda: 1024 ** 3)

        chunk_size = model._compute_optimal_chunk_size(
            20_000,
            2_049,
            operation_type="knn",
            dtype=torch.float32,
        )

        assert chunk_size < 1000

    def test_forced_cuvs_rejects_custom_metric_at_factory(self, monkeypatch):
        """A forced cuVS k-NN request should fail before it can fall back unsafely."""
        import dire_rapids.dire_cuvs as dire_cuvs_module  # pylint: disable=import-outside-toplevel

        monkeypatch.setattr(dire_cuvs_module, "CUVS_AVAILABLE", True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        with pytest.raises(RuntimeError, match="metric is not a cuVS-native"):
            create_dire(
                backend='auto',
                knn_backend='cuvs',
                metric='(x - y).abs().sum(-1)',
                verbose=False,
            )

    def test_repeated_fit_refreshes_cached_knn_tensor(self):
        """A same-shaped second fit must not optimize against the first graph."""
        first, _ = make_blobs(
            n_samples=24, n_features=4, centers=3, random_state=11
        )
        second, _ = make_blobs(
            n_samples=24, n_features=4, centers=5, random_state=29
        )
        model = DiRePyTorch(
            n_neighbors=3,
            init="random",
            max_iter_layout=1,
            knn_backend="pytorch",
            random_state=7,
            verbose=False,
        )
        model.device = torch.device("cpu")

        model.fit_transform(first)
        first_graph = model._knn_indices.copy()
        model.fit_transform(second)

        assert not np.array_equal(first_graph, model._knn_indices)
        np.testing.assert_array_equal(
            model._knn_indices_torch.cpu().numpy(), model._knn_indices
        )

    def test_vectorized_force_fallback_is_reported(self, monkeypatch):
        """Downstream benchmarks can detect every chunked force fallback."""
        data, _ = make_blobs(
            n_samples=24, n_features=4, centers=3, random_state=23
        )

        def raise_oom(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("forced vectorized-force failure")

        monkeypatch.setattr(
            dire_pytorch_module, "_compute_forces_compiled", raise_oom
        )
        model = DiRePyTorch(
            n_neighbors=3,
            init="random",
            max_iter_layout=2,
            knn_backend="pytorch",
            random_state=7,
            verbose=False,
        )

        embedding = model.fit_transform(data)
        diagnostics = model.get_diagnostics()

        assert embedding.shape == (24, 2)
        assert diagnostics["force_chunked_fallback_used"] is True
        assert diagnostics["force_chunked_fallback_calls"] == 2
        assert model.force_chunked_fallback_calls_ == 2

    def test_memory_efficient_fit_rebuilds_invalidated_knn_tensor(self):
        """The memory-efficient force path accepts a cleared graph cache."""
        data, _ = make_blobs(
            n_samples=24, n_features=4, centers=3, random_state=17
        )
        model = create_dire(
            backend="pytorch_cpu",
            memory_efficient=True,
            n_neighbors=3,
            init="random",
            max_iter_layout=1,
            knn_backend="pytorch",
            random_state=7,
            verbose=False,
        )

        embedding = model.fit_transform(data)

        assert embedding.shape == (24, 2)
        assert np.all(np.isfinite(embedding))
        np.testing.assert_array_equal(
            model._knn_indices_torch.cpu().numpy(), model._knn_indices
        )

class TestDiRePyTorchErrors:
    """Test error handling and edge cases."""
    
    def test_invalid_n_components(self):
        """Test that invalid n_components raises appropriate errors."""
        X = np.random.randn(50, 10)

        # Negative n_components should fail
        with pytest.raises((ValueError, AssertionError)):
            model = DiRePyTorch(n_components=-1)
            model.fit_transform(X)

        # Zero n_components should fail
        with pytest.raises((ValueError, AssertionError)):
            model = DiRePyTorch(n_components=0)
            model.fit_transform(X)

    def test_invalid_n_neighbors(self):
        """Test that invalid n_neighbors raises appropriate errors."""
        X = np.random.randn(50, 10)

        # Negative n_neighbors should fail
        with pytest.raises((ValueError, AssertionError)):
            model = DiRePyTorch(n_neighbors=-1)
            model.fit_transform(X)

        # Zero n_neighbors should fail
        with pytest.raises((ValueError, AssertionError)):
            model = DiRePyTorch(n_neighbors=0)
            model.fit_transform(X)
            
    def test_empty_data(self):
        """Test with empty data."""
        X = np.array([]).reshape(0, 5)
        
        model = DiRePyTorch(n_components=2, verbose=False)
        with pytest.raises((ValueError, RuntimeError, IndexError)):
            model.fit_transform(X)
            
    def test_nan_data(self):
        """Test with NaN values in data."""
        X = np.random.randn(50, 10)
        X[10, 5] = np.nan
        
        model = DiRePyTorch(n_components=2, verbose=False)
        # Should either handle NaNs or raise an error
        try:
            X_embedded = model.fit_transform(X)
            # If it succeeds, check no NaNs in output
            assert not np.any(np.isnan(X_embedded))
        except (ValueError, RuntimeError):
            # Expected behavior - NaNs should cause an error
            pass
            
    def test_inf_data(self):
        """Test with infinite values in data."""
        X = np.random.randn(50, 10)
        X[10, 5] = np.inf
        
        model = DiRePyTorch(n_components=2, verbose=False)
        # Should either handle infs or raise an error
        try:
            X_embedded = model.fit_transform(X)
            # If it succeeds, check no infs in output
            assert not np.any(np.isinf(X_embedded))
        except (ValueError, RuntimeError):
            # Expected behavior - infs should cause an error
            pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU tests skipped on CPU")
class TestDiRePyTorchGPU:
    """GPU-specific tests (skipped in CI if no GPU available)."""
    
    def test_gpu_computation(self):
        """Test that computation works on GPU."""
        X = np.random.randn(100, 10).astype(np.float32)

        model = DiRePyTorch(n_components=2, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)

        assert X_embedded.shape == (100, 2)
        assert np.all(np.isfinite(X_embedded))


class TestDiRePyTorchCustomMetrics:
    """Test custom metric functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        np.random.seed(42)
        torch.manual_seed(42)
        # Force CPU for tests
        self.device = torch.device('cpu')  # pylint: disable=attribute-defined-outside-init

    def test_default_metric_none(self):
        """Test that default metric=None works (uses Euclidean)."""
        X, _ = make_blobs(n_samples=50, n_features=10, centers=3, random_state=42)

        model = DiRePyTorch(metric=None, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)

        assert X_embedded.shape == (50, 2)
        assert np.all(np.isfinite(X_embedded))
        assert model.metric_spec is None
        assert model._metric_fn is None

    def test_euclidean_metric_string(self):
        """Test that metric='euclidean' works (should be same as None)."""
        X, _ = make_blobs(n_samples=50, n_features=10, centers=3, random_state=42)

        model = DiRePyTorch(metric='euclidean', max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)

        assert X_embedded.shape == (50, 2)
        assert np.all(np.isfinite(X_embedded))
        assert model.metric_spec == 'euclidean'
        assert model._metric_fn is None  # Should use fast path

    def test_l2_metric_string(self):
        """Test that metric='l2' works (should be same as euclidean)."""
        X, _ = make_blobs(n_samples=50, n_features=10, centers=3, random_state=42)

        model = DiRePyTorch(metric='l2', max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)

        assert X_embedded.shape == (50, 2)
        assert np.all(np.isfinite(X_embedded))
        assert model.metric_spec == 'l2'
        assert model._metric_fn is None  # Should use fast path

    def test_l1_metric_string(self):
        """Test L1 (Manhattan) distance metric using string expression."""
        X, _ = make_blobs(n_samples=50, n_features=10, centers=3, random_state=42)

        # L1 distance: sum of absolute differences
        l1_expr = "(x - y).abs().sum(-1)"
        model = DiRePyTorch(metric=l1_expr, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)

        assert X_embedded.shape == (50, 2)
        assert np.all(np.isfinite(X_embedded))
        assert model.metric_spec == l1_expr
        assert model._metric_fn is not None
        assert callable(model._metric_fn)

    def test_cosine_metric_string(self):
        """Test cosine distance metric using string expression."""
        X, _ = make_blobs(n_samples=50, n_features=10, centers=3, random_state=42)

        # Cosine distance: 1 - cosine similarity
        # Use (x*y).sum(-1) / (sqrt(sum(x^2)) * sqrt(sum(y^2)))
        # Broadcasting: x is (A,1,D), y is (1,B,D)
        cosine_expr = "1 - (x * y).sum(-1) / (((x ** 2).sum(-1).sqrt() * (y ** 2).sum(-1).sqrt()) + 1e-8)"
        model = DiRePyTorch(metric=cosine_expr, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)

        assert X_embedded.shape == (50, 2)
        assert np.all(np.isfinite(X_embedded))
        assert model.metric_spec == cosine_expr
        assert model._metric_fn is not None
        assert callable(model._metric_fn)

    def test_cosine_metric_alias(self):
        """Test built-in cosine distance alias."""
        X = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        model = DiRePyTorch(metric='cosine', n_neighbors=1, verbose=False)
        model._compute_knn(X)

        assert model._metric_fn is not None
        assert model._knn_indices[0, 0] == 1
        assert model._knn_indices[1, 0] == 0

    @pytest.mark.parametrize("metric", ["cosine", "inner_product"])
    def test_origin_sensitive_metric_normalization_preserves_neighbors(self, metric):
        """Default normalization must not mean-center origin-sensitive metrics."""
        data = np.random.default_rng(13).uniform(1.0, 8.0, size=(16, 4)).astype(
            np.float32
        )
        model = DiRePyTorch(
            metric=metric,
            n_neighbors=3,
            init="random",
            max_iter_layout=1,
            knn_backend="pytorch",
            random_state=5,
            verbose=False,
        )
        model.device = torch.device("cpu")

        model.fit_transform(data)

        scaled = data / np.abs(data).max()
        np.testing.assert_allclose(model._data, scaled, rtol=1e-6, atol=1e-7)
        if metric == "cosine":
            norms = np.linalg.norm(scaled, axis=1)
            pairwise = 1.0 - (scaled @ scaled.T) / np.outer(norms, norms)
        else:
            pairwise = -(scaled @ scaled.T)
        np.fill_diagonal(pairwise, np.inf)
        expected = np.argsort(pairwise, axis=1)[:, :model.n_neighbors]
        np.testing.assert_array_equal(model._knn_indices, expected)

    def test_callable_metric(self):
        """Test custom callable metric function."""
        X, _ = make_blobs(n_samples=50, n_features=10, centers=3, random_state=42)

        # Define custom L1 metric as callable
        def l1_metric(x, y):
            return (x - y).abs().sum(-1)

        model = DiRePyTorch(metric=l1_metric, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)

        assert X_embedded.shape == (50, 2)
        assert np.all(np.isfinite(X_embedded))
        assert model.metric_spec is l1_metric
        assert model._metric_fn is l1_metric

    def test_different_metrics_produce_different_results(self):
        """Test that different metrics produce different embeddings.

        Uses same random seed but different metrics (L2 vs Cosine) to ensure
        differences are due to k-NN graph structure, not random sampling.
        Uses Procrustes alignment to account for rotation/reflection invariance.
        """
        from scipy.spatial import procrustes  # pylint: disable=import-outside-toplevel

        X, _ = make_blobs(n_samples=50, n_features=10, centers=3, random_state=42)

        # L2 (Euclidean) metric - use random_state=42
        model_l2 = DiRePyTorch(metric=None, max_iter_layout=20, verbose=False, random_state=42)
        X_l2 = model_l2.fit_transform(X)

        # Cosine metric - use same random_state=42
        # Different metrics should produce different k-NN graphs even with same seed
        cosine_expr = "1 - (x * y).sum(-1) / (((x ** 2).sum(-1).sqrt() * (y ** 2).sum(-1).sqrt()) + 1e-8)"
        model_cosine = DiRePyTorch(metric=cosine_expr, max_iter_layout=20, verbose=False, random_state=42)
        X_cosine = model_cosine.fit_transform(X)

        # Use Procrustes to align embeddings (accounts for rotation/reflection)
        mtx1, mtx2, disparity = procrustes(X_l2, X_cosine)

        # Different metrics should produce different k-NN graphs, leading to different embeddings
        # If embeddings coincide, disparity would be ~1e-15 (machine precision)
        # For different metrics, disparity should be measurably larger
        assert disparity > 1e-3, f"L2 and Cosine metrics should produce different results (disparity={disparity:.6f})"

        # Both should still be valid embeddings
        assert np.all(np.isfinite(X_l2))
        assert np.all(np.isfinite(X_cosine))

    def test_same_metric_same_seed_produces_same_results(self):
        """Test that same metric and seed produce identical embeddings.

        Uses Procrustes alignment to account for rotation/reflection invariance.
        Disparity should be near machine precision (~1e-15) for identical embeddings.
        """
        from scipy.spatial import procrustes  # pylint: disable=import-outside-toplevel

        X, _ = make_blobs(n_samples=50, n_features=10, centers=3, random_state=42)

        # First run with L2 metric and random_state=42
        model1 = DiRePyTorch(metric=None, max_iter_layout=20, verbose=False, random_state=42)
        X_embed1 = model1.fit_transform(X)

        # Second run with same metric and seed
        model2 = DiRePyTorch(metric=None, max_iter_layout=20, verbose=False, random_state=42)
        X_embed2 = model2.fit_transform(X)

        # Use Procrustes to align embeddings (accounts for rotation/reflection)
        mtx1, mtx2, disparity = procrustes(X_embed1, X_embed2)

        # Same inputs and seeds should produce identical embeddings
        # After Procrustes, disparity should be near machine precision
        assert disparity < 1e-10, f"Same metric and seed should produce identical results (disparity={disparity:.15f})"

        # Both should be valid embeddings
        assert np.all(np.isfinite(X_embed1))
        assert np.all(np.isfinite(X_embed2))

    def test_compile_metric_function_directly(self):
        """Test the _compile_metric function directly."""
        from dire_rapids.dire_pytorch import _compile_metric  # pylint: disable=import-outside-toplevel

        # Test None
        assert _compile_metric(None) is None

        # Test euclidean string
        assert _compile_metric('euclidean') is None
        assert _compile_metric('l2') is None
        assert _compile_metric('  L2  ') is None  # Case insensitive and strips

        # Test custom string
        l1_fn = _compile_metric('(x - y).abs().sum(-1)')
        assert callable(l1_fn)

        # Test callable
        def custom_fn(x, y):
            return x + y
        assert _compile_metric(custom_fn) is custom_fn

        # Test invalid input
        with pytest.raises(ValueError, match="metric must be"):
            _compile_metric(123)  # Invalid type

    def test_metric_function_broadcasting(self):
        """Test that custom metric functions work with proper broadcasting."""
        from dire_rapids.dire_pytorch import _compile_metric  # pylint: disable=import-outside-toplevel

        # Create test tensors with broadcasting shapes (torch already imported at top)
        x = torch.randn(3, 1, 5)  # (A, 1, D)
        y = torch.randn(1, 4, 5)  # (1, B, D)

        # Test L1 metric
        l1_fn = _compile_metric('(x - y).abs().sum(-1)')
        result = l1_fn(x, y)

        # Should broadcast to (A, B) = (3, 4)
        assert result.shape == (3, 4)
        assert torch.all(torch.isfinite(result))

        # Test cosine metric (simplified for broadcasting)
        cosine_fn = _compile_metric('((x - y) ** 2).sum(-1)')  # Use squared euclidean instead for broadcasting test
        result = cosine_fn(x, y)

        assert result.shape == (3, 4)
        assert torch.all(torch.isfinite(result))
