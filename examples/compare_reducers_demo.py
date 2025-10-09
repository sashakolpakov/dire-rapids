#!/usr/bin/env python3

"""
Compare Reducers Demo: Compare DiRe, cuML UMAP, and cuML TSNE on various datasets.

This demonstrates using compare_reducers() to evaluate multiple dimensionality
reduction algorithms on quality metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarking'))

from compare_reducers import compare_reducers, print_comparison_summary, ReducerConfig
from dire_rapids import create_dire

print("=" * 80)
print("COMPARE REDUCERS DEMO")
print("=" * 80)

# Example 1: Default reducers (DiRe, cuML UMAP, cuML TSNE if available)
print("\n1. Comparing default reducers on sklearn blobs...")
results = compare_reducers(
    "sklearn:blobs",
    dataset_kwargs={"n_samples": 1000, "n_features": 50, "centers": 5},
    metrics=['distortion', 'context'],
    verbose=True
)
print_comparison_summary(results)

# Example 2: Custom reducer configurations
print("\n\n2. Comparing DiRe with different neighbor settings...")
reducers = [
    ReducerConfig("DiRe-n16", create_dire, {"n_neighbors": 16, "verbose": False}),
    ReducerConfig("DiRe-n32", create_dire, {"n_neighbors": 32, "verbose": False}),
    ReducerConfig("DiRe-n64", create_dire, {"n_neighbors": 64, "verbose": False}),
]

results = compare_reducers(
    "sklearn:circles",
    reducers=reducers,
    dataset_kwargs={"n_samples": 1000, "noise": 0.05, "factor": 0.5},
    metrics=['distortion'],
    verbose=True
)
print_comparison_summary(results)

# Example 3: DiRe geometric datasets
print("\n\n3. Comparing on DiRe sphere dataset...")
try:
    from cuml import UMAP, TSNE
    reducers = [
        ReducerConfig("DiRe", create_dire, {"n_neighbors": 16, "verbose": False}),
        ReducerConfig("UMAP", UMAP, {"n_neighbors": 15, "verbose": False}),
        ReducerConfig("TSNE", TSNE, {"perplexity": 30, "verbose": False}),
    ]
except ImportError:
    reducers = [
        ReducerConfig("DiRe", create_dire, {"n_neighbors": 16, "verbose": False}),
    ]

results = compare_reducers(
    "dire:sphere_uniform",
    reducers=reducers,
    dataset_kwargs={"n_features": 10, "n_samples": 1000},
    metrics=['distortion', 'context'],
    verbose=True
)
print_comparison_summary(results)

print("\n" + "=" * 80)
print("DEMO COMPLETE")
print("=" * 80)
