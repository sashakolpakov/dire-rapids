#!/usr/bin/env python
"""
Comprehensive test suite for dire-rapids on MNIST 70k dataset.

Tests CPU and GPU implementations of:
- Main DiRe classes (dire_pytorch, dire_pytorch_memory_efficient)
- Metrics (stress, neighbor preservation, topology)
- Atlas implementations (CPU and GPU)

Run from repo root:
    python test_comprehensive.py

Or from Jupyter:
    %run test_comprehensive.py
"""

import sys
import traceback
import time
import numpy as np
import warnings

# Suppress non-critical warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print("=" * 80)
print("COMPREHENSIVE DIRE-RAPIDS TEST SUITE")
print("=" * 80)
print()

# Check environment
print("[Environment Check]")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
except ImportError:
    print("✗ PyTorch not available")
    sys.exit(1)

try:
    import cupy as cp
    print(f"✓ CuPy version: {cp.__version__}")
except ImportError:
    print("✗ CuPy not available")
    cp = None

try:
    import cuml
    print(f"✓ cuML version: {cuml.__version__}")
except ImportError:
    print("✗ cuML not available")
    cuml = None

print()

# Load MNIST dataset
print("[Dataset Loading]")
print("Loading MNIST 70k dataset...")
try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X_full = np.array(mnist.data, dtype=np.float32)
    y_full = np.array(mnist.target, dtype=np.int32)

    # Normalize
    X_full = X_full / 255.0

    print(f"✓ Loaded MNIST: {X_full.shape[0]} samples, {X_full.shape[1]} features")
    print(f"  Data range: [{X_full.min():.3f}, {X_full.max():.3f}]")
    print(f"  Number of classes: {len(np.unique(y_full))}")
except Exception as e:  # pylint: disable=broad-exception-caught
    print(f"✗ Failed to load MNIST: {e}")
    sys.exit(1)

print()

# Test results tracker
test_results = []

def run_test(test_name, test_func, critical=True):
    """Run a single test and track results."""
    global test_results

    print("-" * 80)
    print(f"TEST: {test_name}")
    print("-" * 80)

    try:
        start_time = time.time()
        test_func()
        elapsed = time.time() - start_time

        print(f"✓ PASSED ({elapsed:.2f}s)")
        test_results.append((test_name, "PASS", elapsed, None))
        print()
        return True

    except Exception as e:  # pylint: disable=broad-exception-caught
        elapsed = time.time() - start_time
        error_msg = str(e)

        print(f"✗ FAILED ({elapsed:.2f}s)")
        print(f"Error: {error_msg}")
        print("\nTraceback:")
        traceback.print_exc()

        test_results.append((test_name, "FAIL", elapsed, error_msg))
        print()

        if critical:
            print("=" * 80)
            print("CRITICAL TEST FAILED - STOPPING")
            print("=" * 80)
            print_summary()
            sys.exit(1)

        return False


def print_summary():
    """Print test summary."""
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, status, _, _ in test_results if status == "PASS")
    failed = sum(1 for _, status, _, _ in test_results if status == "FAIL")
    total_time = sum(elapsed for _, _, elapsed, _ in test_results)

    print(f"Total tests: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    print()

    # Detailed results
    for test_name, status, elapsed, error in test_results:
        status_symbol = "✓" if status == "PASS" else "✗"
        print(f"{status_symbol} {test_name:60s} {elapsed:8.2f}s")
        if error:
            print(f"  Error: {error}")

    print("=" * 80)


# =============================================================================
# TEST 1: Import Tests
# =============================================================================

def test_imports():
    """Test that all modules can be imported."""
    print("Importing dire_rapids modules...")

    global DiRe, DiReMemoryEfficient
    global compute_stress, compute_neighbor_score, evaluate_embedding
    global compute_h0_h1_atlas_cpu, compute_h0_h1_atlas_gpu

    from dire_rapids.dire_pytorch import DiRe
    print("  ✓ DiRe")

    from dire_rapids.dire_pytorch_memory_efficient import DiReMemoryEfficient
    print("  ✓ DiReMemoryEfficient")

    from dire_rapids.metrics import (
        compute_stress,
        compute_neighbor_score,
        evaluate_embedding
    )
    print("  ✓ metrics")

    from dire_rapids.atlas_cpu import compute_h0_h1_atlas_cpu
    print("  ✓ atlas_cpu")

    try:
        from dire_rapids.atlas_gpu import compute_h0_h1_atlas_gpu
        print("  ✓ atlas_gpu")
    except ImportError as e:
        print(f"  ⚠ atlas_gpu not available: {e}")
        compute_h0_h1_atlas_gpu = None

    print("All imports successful")


# =============================================================================
# TEST 2: DiRe CPU (PyTorch, no GPU)
# =============================================================================

def test_dire_cpu():
    """Test DiRe on CPU with MNIST subset."""
    print("Testing DiRe on CPU...")

    # Use subset for faster testing
    n_samples = 5000
    X_train = X_full[:n_samples]

    print(f"Training on {n_samples} samples")
    print("Parameters: n_neighbors=15, n_components=2, max_iter=50")

    model = DiRe(
        n_neighbors=15,
        n_components=2,
        max_iter_embedding=50,
        device='cpu',
        verbose=True
    )

    print("\nFitting...")
    layout = model.fit_transform(X_train)

    print(f"\nOutput shape: {layout.shape}")
    print(f"Output range: [{layout.min():.3f}, {layout.max():.3f}]")

    assert layout.shape == (n_samples, 2), f"Wrong output shape: {layout.shape}"
    assert np.isfinite(layout).all(), "Output contains non-finite values"

    print("DiRe CPU test passed")


# =============================================================================
# TEST 3: DiRe GPU (PyTorch with CUDA)
# =============================================================================

def test_dire_gpu():
    """Test DiRe on GPU with MNIST subset."""
    if not torch.cuda.is_available():
        print("⚠ GPU not available, skipping")
        return

    print("Testing DiRe on GPU...")

    n_samples = 5000
    X_train = X_full[:n_samples]

    print(f"Training on {n_samples} samples")
    print("Parameters: n_neighbors=15, n_components=2, max_iter=50")

    model = DiRe(
        n_neighbors=15,
        n_components=2,
        max_iter_embedding=50,
        device='cuda',
        verbose=True
    )

    print("\nFitting...")
    layout = model.fit_transform(X_train)

    print(f"\nOutput shape: {layout.shape}")
    print(f"Output range: [{layout.min():.3f}, {layout.max():.3f}]")

    assert layout.shape == (n_samples, 2), f"Wrong output shape: {layout.shape}"
    assert np.isfinite(layout).all(), "Output contains non-finite values"

    print("DiRe GPU test passed")


# =============================================================================
# TEST 4: DiReMemoryEfficient CPU
# =============================================================================

def test_dire_memory_efficient_cpu():
    """Test DiReMemoryEfficient on CPU with MNIST subset."""
    print("Testing DiReMemoryEfficient on CPU...")

    n_samples = 5000
    X_train = X_full[:n_samples]

    print(f"Training on {n_samples} samples")
    print("Parameters: n_neighbors=15, n_components=2, max_iter=50, chunk_size=1000")

    model = DiReMemoryEfficient(
        n_neighbors=15,
        n_components=2,
        max_iter_embedding=50,
        chunk_size=1000,
        device='cpu',
        verbose=True
    )

    print("\nFitting...")
    layout = model.fit_transform(X_train)

    print(f"\nOutput shape: {layout.shape}")
    print(f"Output range: [{layout.min():.3f}, {layout.max():.3f}]")

    assert layout.shape == (n_samples, 2), f"Wrong output shape: {layout.shape}"
    assert np.isfinite(layout).all(), "Output contains non-finite values"

    print("DiReMemoryEfficient CPU test passed")


# =============================================================================
# TEST 5: DiReMemoryEfficient GPU
# =============================================================================

def test_dire_memory_efficient_gpu():
    """Test DiReMemoryEfficient on GPU with MNIST subset."""
    if not torch.cuda.is_available():
        print("⚠ GPU not available, skipping")
        return

    print("Testing DiReMemoryEfficient on GPU...")

    n_samples = 5000
    X_train = X_full[:n_samples]

    print(f"Training on {n_samples} samples")
    print("Parameters: n_neighbors=15, n_components=2, max_iter=50, chunk_size=1000")

    model = DiReMemoryEfficient(
        n_neighbors=15,
        n_components=2,
        max_iter_embedding=50,
        chunk_size=1000,
        device='cuda',
        verbose=True
    )

    print("\nFitting...")
    layout = model.fit_transform(X_train)

    print(f"\nOutput shape: {layout.shape}")
    print(f"Output range: [{layout.min():.3f}, {layout.max():.3f}]")

    assert layout.shape == (n_samples, 2), f"Wrong output shape: {layout.shape}"
    assert np.isfinite(layout).all(), "Output contains non-finite values"

    print("DiReMemoryEfficient GPU test passed")


# =============================================================================
# TEST 6: Metrics - Stress and Neighbor Preservation (CPU)
# =============================================================================

def test_metrics_cpu():
    """Test metrics computation on CPU."""
    print("Testing metrics on CPU...")

    n_samples = 2000
    X = X_full[:n_samples]

    print(f"Computing embedding for {n_samples} samples...")
    model = DiRe(
        n_neighbors=15,
        n_components=2,
        max_iter_embedding=30,
        device='cpu',
        verbose=False
    )
    layout = model.fit_transform(X)

    print("\nComputing stress...")
    stress = compute_stress(X, layout, n_neighbors=15, use_gpu=False)
    print(f"Stress: {stress:.6f}")

    print("\nComputing neighbor preservation...")
    neighbor_score = compute_neighbor_score(X, layout, n_neighbors=15, use_gpu=False)
    print(f"Neighbor preservation: mean={neighbor_score[0]:.4f}, std={neighbor_score[1]:.4f}")

    assert np.isfinite(stress), "Stress is not finite"
    assert 0.0 <= neighbor_score[0] <= 1.0, "Neighbor score out of range"

    print("Metrics CPU test passed")


# =============================================================================
# TEST 7: Metrics - Stress and Neighbor Preservation (GPU)
# =============================================================================

def test_metrics_gpu():
    """Test metrics computation on GPU."""
    if not (torch.cuda.is_available() and cuml is not None and cp is not None):
        print("⚠ GPU/cuML/CuPy not available, skipping")
        return

    print("Testing metrics on GPU...")

    n_samples = 2000
    X = X_full[:n_samples]

    print(f"Computing embedding for {n_samples} samples...")
    model = DiRe(
        n_neighbors=15,
        n_components=2,
        max_iter_embedding=30,
        device='cuda',
        verbose=False
    )
    layout = model.fit_transform(X)

    print("\nComputing stress...")
    stress = compute_stress(X, layout, n_neighbors=15, use_gpu=True)
    print(f"Stress: {stress:.6f}")

    print("\nComputing neighbor preservation...")
    neighbor_score = compute_neighbor_score(X, layout, n_neighbors=15, use_gpu=True)
    print(f"Neighbor preservation: mean={neighbor_score[0]:.4f}, std={neighbor_score[1]:.4f}")

    assert np.isfinite(stress), "Stress is not finite"
    assert 0.0 <= neighbor_score[0] <= 1.0, "Neighbor score out of range"

    print("Metrics GPU test passed")


# =============================================================================
# TEST 8: Full Metrics Evaluation (CPU)
# =============================================================================

def test_full_metrics_cpu():
    """Test full metrics evaluation on CPU."""
    print("Testing full metrics evaluation on CPU...")

    n_samples = 1000
    X = X_full[:n_samples]
    y = y_full[:n_samples]

    print(f"Computing embedding for {n_samples} samples...")
    model = DiRe(
        n_neighbors=15,
        n_components=2,
        max_iter_embedding=30,
        device='cpu',
        verbose=False
    )
    layout = model.fit_transform(X)

    print("\nEvaluating embedding (distortion + context, no topology)...")
    results = evaluate_embedding(
        X, layout, labels=y,
        n_neighbors=15,
        subsample_threshold=1.0,
        use_gpu=False,
        compute_distortion=True,
        compute_context=True,
        compute_topology=False  # Skip topology for speed
    )

    print("\nResults:")
    if 'local' in results:
        print(f"  Stress: {results['local']['stress']:.6f}")
        print(f"  Neighbor: {results['local']['neighbor']}")

    if 'context' in results:
        print(f"  SVM: {results['context']['svm']}")
        print(f"  kNN: {results['context']['knn']}")

    assert 'local' in results, "Missing local metrics"
    assert 'context' in results, "Missing context metrics"

    print("Full metrics CPU test passed")


# =============================================================================
# TEST 9: Atlas CPU - H0/H1 computation
# =============================================================================

def test_atlas_cpu():
    """Test atlas CPU implementation."""
    print("Testing atlas CPU implementation...")

    # Simple circle dataset
    n_samples = 300
    theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    noise = 0.05
    rng = np.random.RandomState(42)
    x = np.cos(theta) + rng.randn(n_samples) * noise
    y = np.sin(theta) + rng.randn(n_samples) * noise
    data = np.column_stack([x, y]).astype(np.float32)

    print(f"Computing H0/H1 for circle with {n_samples} points...")
    print("Parameters: k_neighbors=20, density_threshold=0.8")

    h0, h1 = compute_h0_h1_atlas_cpu(
        data,
        k_neighbors=20,
        density_threshold=0.8
    )

    beta_0 = len(h0[h0[:, 1] == np.inf])
    beta_1 = len(h1[h1[:, 1] == np.inf])

    print(f"\nResults:")
    print(f"  β₀ (connected components): {beta_0}")
    print(f"  β₁ (loops): {beta_1}")
    print(f"  H0 diagram shape: {h0.shape}")
    print(f"  H1 diagram shape: {h1.shape}")

    # Circle should have 1 component and 1 loop
    assert beta_0 == 1, f"Expected β₀=1 for circle, got {beta_0}"
    assert beta_1 == 1, f"Expected β₁=1 for circle, got {beta_1}"

    print("Atlas CPU test passed")


# =============================================================================
# TEST 10: Atlas GPU - H0/H1 computation
# =============================================================================

def test_atlas_gpu():
    """Test atlas GPU implementation."""
    if not (cp is not None and compute_h0_h1_atlas_gpu is not None):
        print("⚠ CuPy or atlas_gpu not available, skipping")
        return

    print("Testing atlas GPU implementation...")

    # Simple circle dataset
    n_samples = 300
    theta = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
    noise = 0.05
    rng = np.random.RandomState(42)
    x = np.cos(theta) + rng.randn(n_samples) * noise
    y = np.sin(theta) + rng.randn(n_samples) * noise
    data = np.column_stack([x, y]).astype(np.float32)

    print(f"Computing H0/H1 for circle with {n_samples} points...")
    print("Parameters: k_neighbors=20, density_threshold=0.8")

    h0, h1 = compute_h0_h1_atlas_gpu(
        data,
        k_neighbors=20,
        density_threshold=0.8
    )

    beta_0 = len(h0[h0[:, 1] == np.inf])
    beta_1 = len(h1[h1[:, 1] == np.inf])

    print(f"\nResults:")
    print(f"  β₀ (connected components): {beta_0}")
    print(f"  β₁ (loops): {beta_1}")
    print(f"  H0 diagram shape: {h0.shape}")
    print(f"  H1 diagram shape: {h1.shape}")

    # Circle should have 1 component and 1 loop
    assert beta_0 == 1, f"Expected β₀=1 for circle, got {beta_0}"
    assert beta_1 == 1, f"Expected β₁=1 for circle, got {beta_1}"

    print("Atlas GPU test passed")


# =============================================================================
# TEST 11: Large Scale Test - DiRe on Full MNIST 70k (GPU if available)
# =============================================================================

def test_large_scale_mnist():
    """Test DiRe on full MNIST 70k dataset."""
    if not torch.cuda.is_available():
        print("⚠ GPU not available, using CPU (this will be slow)")
        device = 'cpu'
    else:
        device = 'cuda'

    print(f"Testing DiRe on full MNIST 70k dataset ({device.upper()})...")

    n_samples = len(X_full)
    print(f"Training on {n_samples} samples")
    print("Parameters: n_neighbors=15, n_components=2, max_iter=100")

    model = DiRe(
        n_neighbors=15,
        n_components=2,
        max_iter_embedding=100,
        device=device,
        verbose=True
    )

    print("\nFitting (this may take several minutes)...")
    layout = model.fit_transform(X_full)

    print(f"\nOutput shape: {layout.shape}")
    print(f"Output range: [{layout.min():.3f}, {layout.max():.3f}]")

    assert layout.shape == (n_samples, 2), f"Wrong output shape: {layout.shape}"
    assert np.isfinite(layout).all(), "Output contains non-finite values"

    print("Large scale MNIST test passed")


# =============================================================================
# TEST 12: Memory Efficient on Full MNIST 70k (GPU if available)
# =============================================================================

def test_large_scale_memory_efficient():
    """Test DiReMemoryEfficient on full MNIST 70k dataset."""
    if not torch.cuda.is_available():
        print("⚠ GPU not available, using CPU (this will be slow)")
        device = 'cpu'
        chunk_size = 5000
    else:
        device = 'cuda'
        chunk_size = 10000

    print(f"Testing DiReMemoryEfficient on full MNIST 70k ({device.upper()})...")

    n_samples = len(X_full)
    print(f"Training on {n_samples} samples")
    print(f"Parameters: n_neighbors=15, n_components=2, max_iter=100, chunk_size={chunk_size}")

    model = DiReMemoryEfficient(
        n_neighbors=15,
        n_components=2,
        max_iter_embedding=100,
        chunk_size=chunk_size,
        device=device,
        verbose=True
    )

    print("\nFitting (this may take several minutes)...")
    layout = model.fit_transform(X_full)

    print(f"\nOutput shape: {layout.shape}")
    print(f"Output range: [{layout.min():.3f}, {layout.max():.3f}]")

    assert layout.shape == (n_samples, 2), f"Wrong output shape: {layout.shape}"
    assert np.isfinite(layout).all(), "Output contains non-finite values"

    print("Large scale memory efficient test passed")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("STARTING TEST SUITE")
    print("=" * 80)
    print()

    # Run tests in sequence
    run_test("1. Import Tests", test_imports, critical=True)
    run_test("2. DiRe CPU", test_dire_cpu, critical=True)
    run_test("3. DiRe GPU", test_dire_gpu, critical=False)
    run_test("4. DiReMemoryEfficient CPU", test_dire_memory_efficient_cpu, critical=True)
    run_test("5. DiReMemoryEfficient GPU", test_dire_memory_efficient_gpu, critical=False)
    run_test("6. Metrics CPU", test_metrics_cpu, critical=True)
    run_test("7. Metrics GPU", test_metrics_gpu, critical=False)
    run_test("8. Full Metrics Evaluation CPU", test_full_metrics_cpu, critical=True)
    run_test("9. Atlas CPU", test_atlas_cpu, critical=True)
    run_test("10. Atlas GPU", test_atlas_gpu, critical=False)
    run_test("11. Large Scale - DiRe MNIST 70k", test_large_scale_mnist, critical=True)
    run_test("12. Large Scale - Memory Efficient MNIST 70k", test_large_scale_memory_efficient, critical=True)

    # Print final summary
    print_summary()

    # Exit with appropriate code
    failed = sum(1 for _, status, _, _ in test_results if status == "FAIL")
    sys.exit(0 if failed == 0 else 1)
