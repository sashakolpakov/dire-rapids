#!/usr/bin/env python3

"""
Compare multiple dimensionality reduction algorithms using ReducerRunner and metrics.

This function provides a unified interface to compare DiRe, cuML UMAP, cuML TSNE,
and other sklearn-compatible reducers on various quality metrics.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from dire_rapids import ReducerRunner, ReducerConfig


def compare_reducers(
    dataset: str,
    reducers: Optional[List[ReducerConfig]] = None,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    metrics: Optional[List[str]] = None,
    subsample_threshold: float = 0.5,
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple dimensionality reduction algorithms on a dataset.

    Parameters
    ----------
    dataset : str
        Dataset selector (sklearn:name, openml:name, cytof:name, dire:name, file:path)
    reducers : list of ReducerConfig, optional
        List of reducer configurations to compare. If None, uses default set.
    dataset_kwargs : dict, optional
        Arguments for dataset loader
    metrics : list of str, optional
        Metrics to compute: 'distortion', 'context', 'topology'. Default: all.
    subsample_threshold : float
        Subsampling probability for all metrics (must be between 0.0 and 1.0, default 0.5).
        Lower values reduce RAM usage. For topological metrics with ripser, try 0.1-0.2 for large datasets.
    verbose : bool
        Print progress information

    Returns
    -------
    dict
        Results for each reducer containing:
        - embedding: reduced data
        - labels: data labels
        - fit_time_sec: time taken for fit_transform
        - metrics: computed quality metrics
        - dataset_info: dataset metadata (only for first reducer)

    Examples
    --------
    >>> # Compare default reducers on sklearn blobs
    >>> results = compare_reducers("sklearn:blobs",
    ...                           dataset_kwargs={"n_samples": 1000, "n_features": 50})

    >>> # Compare specific reducers with custom metrics and subsampling
    >>> from dire_rapids import create_dire
    >>> from cuml import UMAP
    >>> reducers = [
    ...     ReducerConfig("DiRe", create_dire, {"n_neighbors": 16}, visualize=False),
    ...     ReducerConfig("UMAP", UMAP, {"n_neighbors": 15}, visualize=True)
    ... ]
    >>> results = compare_reducers("digits", reducers=reducers,
    ...                           metrics=['distortion', 'context'],
    ...                           subsample_threshold=0.2)
    """
    # Default reducers if none provided
    if reducers is None:
        reducers = _get_default_reducers()

    # Default metrics if none provided
    if metrics is None:
        metrics = ['distortion', 'context', 'topology']

    # Validate metrics
    valid_metrics = {'distortion', 'context', 'topology'}
    invalid = set(metrics) - valid_metrics
    if invalid:
        raise ValueError(f"Invalid metrics: {invalid}. Valid: {valid_metrics}")

    # Validate subsample_threshold
    if not 0.0 <= subsample_threshold <= 1.0:
        raise ValueError(f"subsample_threshold must be between 0.0 and 1.0, got {subsample_threshold}")

    # Check if metrics module is available
    try:
        from dire_rapids.metrics import evaluate_embedding
        has_metrics = True
    except ImportError:
        if verbose:
            print("Warning: dire_rapids.metrics not available. Skipping metric computation.")
        has_metrics = False

    results = {}
    X_orig = None
    y_orig = None

    for i, config in enumerate(reducers):
        if verbose:
            print(f"\n{'='*80}")
            print(f"Running {config.name} ({i+1}/{len(reducers)})...")
            print(f"{'='*80}")

        # Create runner with ReducerConfig
        runner = ReducerRunner(config=config)

        # Run reducer
        try:
            result = runner.run(dataset, dataset_kwargs=dataset_kwargs)

            # Store original data from first reducer
            if X_orig is None:
                # Load data again to get original
                from dire_rapids.utils import _parse_selector, _load_sklearn_any, _load_file, _load_cytof, _load_dire_dataset, _coerce_Xy
                scheme, name = _parse_selector(dataset)
                dkwargs = dataset_kwargs or {}

                if scheme == "sklearn":
                    X_orig, y_orig = _load_sklearn_any(name, **dkwargs)
                elif scheme == "file":
                    X_orig, y_orig = _load_file(name, **dkwargs)
                elif scheme == "openml":
                    from sklearn.datasets import fetch_openml
                    try:
                        data_id = int(str(name))
                        ds = fetch_openml(data_id=data_id, return_X_y=True, **dkwargs)
                    except Exception:
                        ds = fetch_openml(name=name, return_X_y=True, **dkwargs)
                    X_orig, y_orig = _coerce_Xy(ds[0], ds[1])
                elif scheme == "cytof":
                    X_orig, y_orig = _load_cytof(name, **dkwargs)
                elif scheme == "dire":
                    X_orig, y_orig = _load_dire_dataset(name, **dkwargs)
                else:
                    raise ValueError(f"Unsupported scheme '{scheme}'")

                # Convert to numpy arrays (for non-openml schemes that didn't use _coerce_Xy)
                if scheme != "openml":
                    if hasattr(X_orig, 'toarray'):
                        X_orig = X_orig.toarray()
                    X_orig = np.asarray(X_orig, dtype=np.float32)
                    if y_orig is not None:
                        y_orig = np.asarray(y_orig)

            # Compute metrics
            result['metrics'] = {}
            if has_metrics and X_orig is not None:
                if verbose:
                    print(f"Computing metrics for {config.name}...")
                    print(f"  Using subsample_threshold={subsample_threshold}")

                metric_results = evaluate_embedding(
                    X_orig,
                    result['embedding'],
                    y_orig,
                    subsample_threshold=subsample_threshold,
                    compute_distortion='distortion' in metrics,
                    compute_context='context' in metrics,
                    compute_topology='topology' in metrics,
                    verbose=verbose
                )
                result['metrics'] = metric_results

            # Store results
            results[config.name] = result

            if verbose:
                print(f"\n{config.name} Results:")
                print(f"  Fit time: {result['fit_time_sec']:.3f}s")
                if result['metrics']:
                    _print_metrics(result['metrics'])

        except Exception as e:
            if verbose:
                print(f"  ERROR: {config.name} failed - {e}")
            results[config.name] = {"error": str(e)}

    return results


def _get_default_reducers() -> List[ReducerConfig]:
    """Get default set of reducers to compare."""
    reducers = []

    # DiRe
    try:
        from dire_rapids import create_dire
        reducers.append(ReducerConfig(
            name="DiRe",
            reducer_class=create_dire,
            reducer_kwargs={"n_components": 2, "n_neighbors": 16, "verbose": False},
            visualize=False
        ))
    except ImportError:
        pass

    # cuML UMAP
    try:
        from cuml import UMAP
        reducers.append(ReducerConfig(
            name="cuML-UMAP",
            reducer_class=UMAP,
            reducer_kwargs={"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "verbose": False},
            visualize=False
        ))
    except ImportError:
        pass

    # cuML TSNE
    try:
        from cuml import TSNE
        reducers.append(ReducerConfig(
            name="cuML-TSNE",
            reducer_class=TSNE,
            reducer_kwargs={"n_components": 2, "perplexity": 30, "verbose": False},
            visualize=False
        ))
    except ImportError:
        pass

    if not reducers:
        raise RuntimeError("No reducers available. Install dire-rapids and/or cuML.")

    return reducers


def _print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty print metrics results."""
    if 'local' in metrics:
        print(f"  Distortion metrics:")
        if 'stress' in metrics['local']:
            print(f"    Stress: {metrics['local']['stress']:.4f}")
        if 'neighbor_score' in metrics['local']:
            print(f"    Neighbor score: {metrics['local']['neighbor_score']:.4f}")

    if 'context' in metrics:
        print(f"  Context metrics:")
        if 'svm' in metrics['context']:
            acc_hd, acc_ld, score = metrics['context']['svm']
            print(f"    SVM accuracy: HD={acc_hd:.4f}, LD={acc_ld:.4f}, score={score:.4f}")
        if 'knn' in metrics['context']:
            acc_hd, acc_ld, score = metrics['context']['knn']
            print(f"    kNN accuracy: HD={acc_hd:.4f}, LD={acc_ld:.4f}, score={score:.4f}")

    if 'topology' in metrics:
        print(f"  Topology metrics:")
        if 'metrics' in metrics['topology']:
            topo = metrics['topology']['metrics']
            if 'wass' in topo:
                print(f"    Wasserstein: {topo['wass'][0]:.6f}")
            if 'bottleneck' in topo:
                print(f"    Bottleneck: {topo['bottleneck'][0]:.6f}")


def print_comparison_summary(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a summary comparison table of all reducers."""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    # Print table header
    print(f"{'Reducer':<15} {'Time(s)':<10} {'Stress':<10} {'SVM Score':<12} {'kNN Score':<12} {'Wasserstein':<12}")
    print("-" * 95)

    for name, result in results.items():
        if 'error' in result:
            print(f"{name:<15} ERROR: {result['error']}")
            continue

        time_str = f"{result['fit_time_sec']:.3f}"

        stress_str = "-"
        if 'local' in result.get('metrics', {}) and 'stress' in result['metrics']['local']:
            stress_str = f"{result['metrics']['local']['stress']:.4f}"

        svm_str = "-"
        if 'context' in result.get('metrics', {}) and 'svm' in result['metrics']['context']:
            svm_str = f"{result['metrics']['context']['svm'][2]:.4f}"  # Context score (log ratio)

        knn_str = "-"
        if 'context' in result.get('metrics', {}) and 'knn' in result['metrics']['context']:
            knn_str = f"{result['metrics']['context']['knn'][2]:.4f}"  # Context score (log ratio)

        wass_str = "-"
        if 'topology' in result.get('metrics', {}):
            if 'metrics' in result['metrics']['topology'] and 'wass' in result['metrics']['topology']['metrics']:
                wass_str = f"{result['metrics']['topology']['metrics']['wass'][0]:.6f}"

        print(f"{name:<15} {time_str:<10} {stress_str:<10} {svm_str:<12} {knn_str:<12} {wass_str:<12}")
