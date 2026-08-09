"""Validated public hyperparameter presets for DiRePyTorch.

The former ``TOPOLOGY_TUNED`` configuration remains withdrawn. The canonical
``ATLAS_TUNED`` and ``RIPSER_TUNED`` presets are evaluator-specific public
names backed by a crossed search and six-dataset, 20-seed held-out validation.
An additional quality-gated Atlas refinement resolved their distinct layout
budgets: Atlas uses 96 layout iterations and Ripser uses 128.
"""

_SHARED_TOPOLOGY_PARAMETERS = {
    "init": "pca",
    "n_neighbors": 16,
    "spread": 0.8,
    "min_dist": 1e-2,
    "cutoff": 42.0,
    "neg_ratio": 8,
}

ATLAS_TUNED = {**_SHARED_TOPOLOGY_PARAMETERS, "max_iter_layout": 96}
RIPSER_TUNED = {**_SHARED_TOPOLOGY_PARAMETERS, "max_iter_layout": 128}

__all__ = ["ATLAS_TUNED", "RIPSER_TUNED"]
