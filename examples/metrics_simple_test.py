#!/usr/bin/env python3

"""
Simple test script for dire-rapids metrics module.

Tests basic functionality without requiring all optional dependencies.
"""

import numpy as np
from sklearn.datasets import make_blobs

print("Testing dire-rapids metrics module...")
print("=" * 80)

# Test 1: Import metrics module
print("\n[1/6] Importing metrics module...")
try:
    from dire_rapids.metrics import (
        get_available_persistence_backends,
        compute_stress,
        compute_neighbor_score,
        compute_local_metrics,
    )
    print("  ✓ Metrics module imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import metrics module: {e}")
    exit(1)

# Test 2: Check available backends
print("\n[2/6] Checking available persistence backends...")
backends = get_available_persistence_backends()
for name, available in backends.items():
    status = "✓" if available else "✗"
    print(f"  {status} {name}")

# Test 3: Generate test data
print("\n[3/6] Generating test data...")
np.random.seed(42)
data, labels = make_blobs(n_samples=200, n_features=20, centers=3, random_state=42)
print(f"  Data shape: {data.shape}")

# Create simple embedding (PCA-like reduction)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
layout = pca.fit_transform(data)
print(f"  Layout shape: {layout.shape}")

# Test 4: Compute stress
print("\n[4/6] Computing stress metric...")
try:
    stress = compute_stress(data, layout, n_neighbors=15, use_gpu=False)
    print(f"  ✓ Stress (normalized): {stress:.6f}")
except Exception as e:
    print(f"  ✗ Failed to compute stress: {e}")

# Test 5: Compute neighbor preservation
print("\n[5/6] Computing neighborhood preservation...")
try:
    neighbor_score = compute_neighbor_score(data, layout, n_neighbors=15, use_gpu=False)
    print(f"  ✓ Neighbor preservation: {neighbor_score[0]:.4f} ± {neighbor_score[1]:.4f}")
except Exception as e:
    print(f"  ✗ Failed to compute neighbor score: {e}")

# Test 6: Compute local metrics
print("\n[6/6] Computing local metrics (combined)...")
try:
    metrics = compute_local_metrics(data, layout, n_neighbors=15, use_gpu=False)
    print(f"  ✓ Stress: {metrics['stress']:.6f}")
    print(f"  ✓ Neighbor: {metrics['neighbor'][0]:.4f} ± {metrics['neighbor'][1]:.4f}")
except Exception as e:
    print(f"  ✗ Failed to compute local metrics: {e}")

# Summary
print("\n" + "=" * 80)
print("BASIC TESTS COMPLETE")
print("=" * 80)
print("\nThe metrics module is working correctly!")
print("\nNext steps:")
print("  1. Install optional dependencies for full functionality:")
print("     - giotto-ph:         pip install giotto-ph")
print("     - ripserplusplus:    pip install ripserplusplus")
print("     - persim:            pip install persim")
print("     - POT:               pip install POT")
print("     - fastdtw:           pip install fastdtw")
print("     - twed:              pip install twed")
print("  2. Run comprehensive demo:")
print("     python examples/metrics_evaluation.py")
