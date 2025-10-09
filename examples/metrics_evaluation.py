#!/usr/bin/env python3

"""
Example: Comprehensive evaluation of dimensionality reduction quality using dire-rapids metrics.

This script demonstrates:
1. Using multiple persistence backends (giotto-ph, ripser++, ripser)
2. Computing distortion metrics (stress, neighborhood preservation)
3. Computing context metrics (SVM, kNN classification accuracy)
4. Computing topological metrics (persistence diagrams, Betti curves, distances)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_swiss_roll

# Import dire-rapids
try:
    from dire_rapids import DiRePyTorch, DiReCuVS
    HAS_DIRE = True
except ImportError:
    print("dire-rapids not installed. Install it first.")
    HAS_DIRE = False
    exit(1)

# Import metrics module
try:
    from dire_rapids.metrics import (
        evaluate_embedding,
        get_available_persistence_backends,
        set_persistence_backend,
        compute_local_metrics,
        compute_context_measures,
        compute_global_metrics
    )
    HAS_METRICS = True
except ImportError:
    print("Metrics module not available")
    HAS_METRICS = False
    exit(1)


def generate_test_data(n_samples=1024, dataset_type='classification'):
    """
    Generate test datasets.

    Parameters
    ----------
    n_samples : int
        Number of samples
    dataset_type : str
        'classification' or 'manifold'

    Returns
    -------
    tuple : (data, labels)
    """
    if dataset_type == 'classification':
        # High-dimensional classification data
        data, labels = make_classification(
            n_samples=n_samples,
            n_features=50,
            n_informative=30,
            n_redundant=10,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42
        )
    elif dataset_type == 'manifold':
        # Swiss roll manifold
        data, labels = make_swiss_roll(
            n_samples=n_samples,
            noise=0.05,
            random_state=42
        )
        # Convert continuous labels to discrete
        labels = (labels * 3 / labels.max()).astype(int)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return data, labels


def demo_persistence_backends():
    """
    Demonstrate available persistence backends and their usage.
    """
    print("=" * 80)
    print("PERSISTENCE BACKENDS DEMO")
    print("=" * 80)

    # Check available backends
    backends = get_available_persistence_backends()
    print("\nAvailable persistence backends:")
    for name, available in backends.items():
        status = "✓ Available" if available else "✗ Not installed"
        print(f"  {name:15s} {status}")

    # Recommend installation
    print("\nRecommended installation priority:")
    print("  1. giotto-ph     (fastest CPU, multi-threaded)")
    print("     pip install giotto-ph")
    print("  2. ripser++      (GPU-accelerated)")
    print("     pip install ripserplusplus")
    print("  3. ripser        (CPU fallback)")
    print("     pip install ripser")

    # Try setting backends
    print("\nTesting backend selection:")
    for backend_name in ['giotto-ph', 'ripser++', 'ripser']:
        try:
            set_persistence_backend(backend_name)
            print(f"  ✓ Successfully set backend to: {backend_name}")
        except ValueError as e:
            print(f"  ✗ {e}")

    # Reset to auto
    set_persistence_backend(None)
    print("\n  Reset to automatic backend selection")


def demo_local_metrics():
    """
    Demonstrate distortion metrics (stress, neighborhood preservation).
    """
    print("\n" + "=" * 80)
    print("LOCAL METRICS DEMO (Distortion)")
    print("=" * 80)

    # Generate data
    print("\nGenerating test data...")
    data, labels = generate_test_data(n_samples=1000, dataset_type='manifold')
    print(f"Data shape: {data.shape}")

    # Create embedding
    print("\nComputing DIRE embedding...")
    reducer = DiRePyTorch(n_components=2, n_neighbors=15, verbose=False)
    layout = reducer.fit_transform(data)

    # Compute local metrics
    print("\nComputing local metrics (stress, neighborhood preservation)...")
    local_results = compute_local_metrics(
        data, layout, n_neighbors=15, use_gpu=True
    )

    print("\nResults:")
    print(f"  Stress (normalized):      {local_results['stress']:.6f}")
    print(f"  Neighbor preservation:    {local_results['neighbor'][0]:.4f} ± {local_results['neighbor'][1]:.4f}")

    if 'note' in local_results:
        print(f"  Note: {local_results['note']}")

    return data, layout, labels


def demo_context_metrics(data, layout, labels):
    """
    Demonstrate context preservation metrics (SVM, kNN).
    """
    print("\n" + "=" * 80)
    print("CONTEXT METRICS DEMO (Classification Accuracy)")
    print("=" * 80)

    # Compute context metrics
    print("\nComputing context preservation metrics (SVM, kNN)...")
    context_results = compute_context_measures(
        data, layout, labels,
        subsample_threshold=0.8,
        n_neighbors=15,
        random_state=42,
        use_gpu=True
    )

    print("\nSVM Classification Results:")
    svm_scores = context_results['svm']
    print(f"  Accuracy (high-dim):      {svm_scores[0]:.4f}")
    print(f"  Accuracy (low-dim):       {svm_scores[1]:.4f}")
    print(f"  Log ratio:                {svm_scores[2]:.4f}")

    print("\nkNN Classification Results:")
    knn_scores = context_results['knn']
    print(f"  Accuracy (high-dim):      {knn_scores[0]:.4f}")
    print(f"  Accuracy (low-dim):       {knn_scores[1]:.4f}")
    print(f"  Log ratio:                {knn_scores[2]:.4f}")


def demo_topological_metrics(data, layout):
    """
    Demonstrate topological metrics (persistence diagrams, Betti curves).
    """
    print("\n" + "=" * 80)
    print("TOPOLOGICAL METRICS DEMO (Persistence Homology)")
    print("=" * 80)

    # Compute topological metrics
    print("\nComputing topological metrics (persistence diagrams, Betti curves)...")
    print("This may take a while depending on data size and backend...")

    topo_results = compute_global_metrics(
        data, layout,
        dimension=1,
        subsample_threshold=0.3,  # Use subset for speed
        random_state=42,
        n_steps=100,
        metrics_only=False,  # Get diagrams and Betti curves too
        backend=None,  # Auto-select
        n_threads=-1,  # Use all cores for giotto-ph
        collapse_edges=True
    )

    backend_used = topo_results.get('backend', 'unknown')
    print(f"\nUsed backend: {backend_used}")

    metrics = topo_results['metrics']
    print("\nTopological Distance Metrics:")
    for dim in range(len(metrics['dtw'])):
        print(f"\n  Dimension {dim}:")
        print(f"    DTW (Betti curves):       {metrics['dtw'][dim]:.6f}")
        print(f"    TWED (Betti curves):      {metrics['twed'][dim]:.6f}")
        print(f"    EMD (Betti curves):       {metrics['emd'][dim]:.6f}")
        print(f"    Wasserstein (diagrams):   {metrics['wass'][dim]:.6f}")
        print(f"    Bottleneck (diagrams):    {metrics['bott'][dim]:.6f}")

    # Optionally visualize Betti curves
    if 'bettis' in topo_results:
        plot_betti_curves(topo_results['bettis'])

    return topo_results


def plot_betti_curves(betti_curves):
    """
    Plot Betti curves for high-dim and low-dim data.
    """
    print("\nPlotting Betti curves...")

    n_dims = len(betti_curves['data'])
    fig, axes = plt.subplots(1, n_dims, figsize=(6*n_dims, 4))

    if n_dims == 1:
        axes = [axes]

    for dim in range(n_dims):
        ax = axes[dim]

        # High-dimensional Betti curve
        axis_x_hd, axis_y_hd = betti_curves['data'][dim]
        ax.plot(axis_x_hd, axis_y_hd, 'b-', label='High-dim', linewidth=2)

        # Low-dimensional Betti curve
        axis_x_ld, axis_y_ld = betti_curves['layout'][dim]
        ax.plot(axis_x_ld, axis_y_ld, 'r--', label='Low-dim', linewidth=2)

        ax.set_xlabel('Filtration value')
        ax.set_ylabel('Betti number')
        ax.set_title(f'Betti Curve (Dimension {dim})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('betti_curves.png', dpi=150, bbox_inches='tight')
    print("  Saved to: betti_curves.png")
    plt.close()


def demo_comprehensive_evaluation():
    """
    Demonstrate comprehensive evaluation using evaluate_embedding().
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION DEMO")
    print("=" * 80)

    # Generate data
    print("\nGenerating test data...")
    data, labels = generate_test_data(n_samples=2048, dataset_type='classification')
    print(f"Data shape: {data.shape}")

    # Create embedding
    print("\nComputing DIRE embedding...")
    reducer = DiRePyTorch(n_components=2, n_neighbors=16, verbose=False)
    layout = reducer.fit_transform(data)

    # Comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    results = evaluate_embedding(
        data, layout, labels,
        n_neighbors=16,
        subsample_threshold=0.5,
        max_homology_dim=1,
        random_state=42,
        use_gpu=True,
        persistence_backend=None,  # Auto-select
        n_threads=-1
    )

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    if 'local' in results:
        print("\nLocal Metrics (Distortion):")
        print(f"  Stress:                   {results['local']['stress']:.6f}")
        print(f"  Neighbor preservation:    {results['local']['neighbor'][0]:.4f} ± {results['local']['neighbor'][1]:.4f}")

    if 'context' in results:
        print("\nContext Metrics (Classification):")
        print(f"  SVM accuracy (high-dim):  {results['context']['svm'][0]:.4f}")
        print(f"  SVM accuracy (low-dim):   {results['context']['svm'][1]:.4f}")
        print(f"  kNN accuracy (high-dim):  {results['context']['knn'][0]:.4f}")
        print(f"  kNN accuracy (low-dim):   {results['context']['knn'][1]:.4f}")

    if 'topology' in results:
        backend = results['topology'].get('backend', 'unknown')
        print(f"\nTopological Metrics (using {backend}):")
        metrics = results['topology']['metrics']
        for dim in range(len(metrics['dtw'])):
            print(f"  Dimension {dim}:")
            print(f"    Wasserstein:            {metrics['wass'][dim]:.6f}")
            print(f"    Bottleneck:             {metrics['bott'][dim]:.6f}")

    return results


def main():
    """
    Run all demos.
    """
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  DIRE-RAPIDS METRICS MODULE - COMPREHENSIVE DEMO".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")

    # Demo 1: Check available backends
    demo_persistence_backends()

    # Demo 2: Local metrics (distortion)
    data, layout, labels = demo_local_metrics()

    # Demo 3: Context metrics (classification)
    demo_context_metrics(data, layout, labels)

    # Demo 4: Topological metrics (persistence)
    try:
        demo_topological_metrics(data, layout)
    except RuntimeError as e:
        print(f"\nSkipping topological metrics: {e}")
        print("Install at least one persistence backend (giotto-ph, ripser++, or ripser)")

    # Demo 5: Comprehensive evaluation
    print("\n")
    demo_comprehensive_evaluation()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  - dire_rapids.metrics module documentation")
    print("  - examples in examples/ directory")


if __name__ == "__main__":
    main()
