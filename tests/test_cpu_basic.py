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
from dire_rapids import DiRePyTorch


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
        assert disparity > 1e-2, f"L2 and Cosine metrics should produce different results (disparity={disparity:.6f})"

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