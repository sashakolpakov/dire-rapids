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
            
    @pytest.mark.skip(reason="TODO: Fix reproducibility with random seeds")
    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with the same random seed."""
        X, _ = make_blobs(n_samples=50, n_features=5, centers=2, random_state=42)
        
        # First run
        model1 = DiRePyTorch(random_state=123, max_iter_layout=10, verbose=False)
        X_embedded1 = model1.fit_transform(X)
        
        # Second run with same seed
        model2 = DiRePyTorch(random_state=123, max_iter_layout=10, verbose=False)
        X_embedded2 = model2.fit_transform(X)
        
        # Should produce identical results
        np.testing.assert_array_almost_equal(X_embedded1, X_embedded2, decimal=5)
        
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
        
    @pytest.mark.skip(reason="TODO: Handle edge cases with small datasets")
    def test_min_neighbors_validation(self):
        """Test that n_neighbors is validated correctly."""
        X, _ = make_blobs(n_samples=20, n_features=5, centers=2, random_state=42)
        
        # n_neighbors should be less than n_samples
        model = DiRePyTorch(n_neighbors=15, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)
        assert X_embedded.shape == (20, 2)
        
        # Should handle case where n_neighbors >= n_samples
        model = DiRePyTorch(n_neighbors=25, max_iter_layout=10, verbose=False)
        X_embedded = model.fit_transform(X)
        # Should internally adjust n_neighbors
        assert X_embedded.shape == (20, 2)
        
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
    
    @pytest.mark.skip(reason="TODO: Add parameter validation for invalid n_components")
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
            
    @pytest.mark.skip(reason="TODO: Add parameter validation for invalid n_neighbors")
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