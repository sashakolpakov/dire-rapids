#!/usr/bin/env python3

"""
ReducerRunner Demo: Running different dimensionality reduction algorithms.

This demonstrates using ReducerRunner with:
- DiRe (create_dire, DiRePyTorch)
- cuML UMAP (if available)
- cuML TSNE (if available)
- sklearn UMAP (if available)
"""

from dire_rapids import ReducerRunner, ReducerConfig, create_dire, DiRePyTorch

print("=" * 80)
print("REDUCER RUNNER DEMO")
print("=" * 80)

# Example 1: DiRe with create_dire factory
print("\n1. Running DiRe with create_dire...")
config = ReducerConfig(
    name="DiRe",
    reducer_class=create_dire,
    reducer_kwargs={
        "memory_efficient": True,
        "n_components": 2,
        "n_neighbors": 16,
        "max_iter_layout": 64,
        "verbose": False
    },
    visualize=False
)
runner = ReducerRunner(config=config)

result = runner.run("blobs", dataset_kwargs={"n_samples": 1000, "n_features": 50, "centers": 5})
print(f"   Samples: {result['dataset_info']['n_samples']}")
print(f"   Features: {result['dataset_info']['n_features']}")
print(f"   Embedding shape: {result['embedding'].shape}")
print(f"   Time: {result['fit_time_sec']:.3f}s")

# Example 2: DiRe with DiRePyTorch class (swiss_roll has continuous labels)
print("\n2. Running DiRe with DiRePyTorch...")
config = ReducerConfig(
    name="DiRePyTorch",
    reducer_class=DiRePyTorch,
    reducer_kwargs={
        "n_components": 2,
        "n_neighbors": 20,
        "verbose": False
    },
    visualize=False,
    categorical_labels=False  # swiss_roll has continuous labels (angle/position)
)
runner = ReducerRunner(config=config)

result = runner.run("swiss_roll", dataset_kwargs={"n_samples": 1000})
print(f"   Samples: {result['dataset_info']['n_samples']}")
print(f"   Embedding shape: {result['embedding'].shape}")
print(f"   Time: {result['fit_time_sec']:.3f}s")

# Example 3: cuML UMAP (if available)
print("\n3. Running cuML UMAP...")
try:
    from cuml import UMAP as cumlUMAP

    config = ReducerConfig(
        name="cuML-UMAP",
        reducer_class=cumlUMAP,
        reducer_kwargs={
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "verbose": False
        },
        visualize=False
    )
    runner = ReducerRunner(config=config)

    result = runner.run("circles", dataset_kwargs={"n_samples": 1000, "noise": 0.05})
    print(f"   Samples: {result['dataset_info']['n_samples']}")
    print(f"   Embedding shape: {result['embedding'].shape}")
    print(f"   Time: {result['fit_time_sec']:.3f}s")
except ImportError:
    print("   cuML UMAP not available (install RAPIDS)")

# Example 4: cuML TSNE (if available)
print("\n4. Running cuML TSNE...")
try:
    from cuml import TSNE as cumlTSNE

    config = ReducerConfig(
        name="cuML-TSNE",
        reducer_class=cumlTSNE,
        reducer_kwargs={
            "n_components": 2,
            "perplexity": 30,
            "verbose": False
        },
        visualize=False
    )
    runner = ReducerRunner(config=config)

    result = runner.run("moons", dataset_kwargs={"n_samples": 1000, "noise": 0.1})
    print(f"   Samples: {result['dataset_info']['n_samples']}")
    print(f"   Embedding shape: {result['embedding'].shape}")
    print(f"   Time: {result['fit_time_sec']:.3f}s")
except ImportError:
    print("   cuML TSNE not available (install RAPIDS)")

# Example 5: sklearn UMAP (if available)
print("\n5. Running sklearn-compatible UMAP...")
try:
    from umap import UMAP

    config = ReducerConfig(
        name="UMAP",
        reducer_class=UMAP,
        reducer_kwargs={
            "n_components": 2,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "verbose": False
        },
        visualize=False
    )
    runner = ReducerRunner(config=config)

    result = runner.run("digits")
    print(f"   Samples: {result['dataset_info']['n_samples']}")
    print(f"   Embedding shape: {result['embedding'].shape}")
    print(f"   Time: {result['fit_time_sec']:.3f}s")
except ImportError:
    print("   umap-learn not available (pip install umap-learn)")

# Example 6: Using different data sources
print("\n6. Testing different data sources...")

# sklearn dataset
print("\n   a) sklearn dataset (iris):")
config = ReducerConfig(
    name="DiRe",
    reducer_class=create_dire,
    reducer_kwargs={"n_components": 2, "verbose": False},
    visualize=False
)
runner = ReducerRunner(config=config)
result = runner.run("sklearn:iris")
print(f"      Loaded: {result['dataset_info']['n_samples']} samples, "
      f"{result['dataset_info']['n_features']} features")


print("\n" + "=" * 80)
print("DEMO COMPLETE")
print("=" * 80)
print("\nReducerRunner supports:")
print("  - DiRe: create_dire, DiRePyTorch, DiRePyTorchMemoryEfficient, DiReCuVS")
print("  - cuML: UMAP, TSNE")
print("  - Any sklearn TransformerMixin-compatible class")
