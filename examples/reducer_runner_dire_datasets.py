#!/usr/bin/env python3

"""
ReducerRunner Demo: DiRe Geometric Datasets

This demonstrates using ReducerRunner with DiRe's geometric datasets:
- dire:disk_uniform - uniformly distributed points in n-dimensional disk
- dire:sphere_uniform - uniformly distributed points on n-dimensional sphere
- dire:ellipsoid_uniform - uniformly distributed points on n-dimensional ellipsoid
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarking'))

from reducer_runner import ReducerRunner, ReducerConfig
from dire_rapids import create_dire

print("=" * 80)
print("DIRE GEOMETRIC DATASETS DEMO")
print("=" * 80)

# Example 1: Disk uniform distribution
print("\n1. DiRe disk_uniform dataset (5D disk, 1000 points)...")
config = ReducerConfig(
    name="DiRe",
    reducer_class=create_dire,
    reducer_kwargs=dict(
        n_components=2,
        n_neighbors=16,
        verbose=False
    ),
    visualize=False
)
runner = ReducerRunner(config=config)

result = runner.run("dire:disk_uniform", dataset_kwargs={"n_features": 5, "n_samples": 1000})
print(f"   Samples: {result['dataset_info']['n_samples']}")
print(f"   Features: {result['dataset_info']['n_features']}")
print(f"   Embedding shape: {result['embedding'].shape}")
print(f"   Time: {result['fit_time_sec']:.3f}s")

# Example 2: Sphere uniform distribution
print("\n2. DiRe sphere_uniform dataset (3D sphere, 500 points)...")
result = runner.run("dire:sphere_uniform", dataset_kwargs={"n_features": 3, "n_samples": 500})
print(f"   Samples: {result['dataset_info']['n_samples']}")
print(f"   Features: {result['dataset_info']['n_features']}")
print(f"   Embedding shape: {result['embedding'].shape}")
print(f"   Time: {result['fit_time_sec']:.3f}s")

# Example 3: Ellipsoid uniform distribution
print("\n3. DiRe ellipsoid_uniform dataset (3D ellipsoid with semi-axes [1, 2, 3])...")
result = runner.run("dire:ellipsoid_uniform", dataset_kwargs={"semi_axes": [1, 2, 3], "n_samples": 800})
print(f"   Samples: {result['dataset_info']['n_samples']}")
print(f"   Features: {result['dataset_info']['n_features']}")
print(f"   Embedding shape: {result['embedding'].shape}")
print(f"   Time: {result['fit_time_sec']:.3f}s")

# Example 4: High-dimensional ellipsoid
print("\n4. DiRe ellipsoid_uniform dataset (10D ellipsoid)...")
result = runner.run("dire:ellipsoid_uniform",
                   dataset_kwargs={"semi_axes": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5], "n_samples": 2000})
print(f"   Samples: {result['dataset_info']['n_samples']}")
print(f"   Features: {result['dataset_info']['n_features']}")
print(f"   Embedding shape: {result['embedding'].shape}")
print(f"   Time: {result['fit_time_sec']:.3f}s")

print("\n" + "=" * 80)
print("DEMO COMPLETE")
print("=" * 80)
print("\nDiRe geometric datasets:")
print("  - dire:disk_uniform     - uniform distribution in n-dimensional disk")
print("  - dire:sphere_uniform   - uniform distribution on n-dimensional sphere")
print("  - dire:ellipsoid_uniform - uniform distribution on n-dimensional ellipsoid")
