"""Validated public hyperparameter presets for DiRePyTorch.

The former ``TOPOLOGY_TUNED`` configuration remains withdrawn. The canonical
``ATLAS_TUNED`` and ``RIPSER_TUNED`` presets are evaluator-specific public
names backed by a crossed search and six-dataset, 20-seed held-out validation.
The current validation converged on the same parameters for both evaluators;
the separate names keep their objective contracts explicit if later evidence
causes them to diverge.
"""

_CROSSED_TOPOLOGY_PARAMETERS = {
    "init": "pca",
    "n_neighbors": 16,
    "spread": 0.8,
    "min_dist": 1e-2,
    "cutoff": 42.0,
    "neg_ratio": 8,
    "max_iter_layout": 128,
}

ATLAS_TUNED = dict(_CROSSED_TOPOLOGY_PARAMETERS)
RIPSER_TUNED = dict(_CROSSED_TOPOLOGY_PARAMETERS)

__all__ = ["ATLAS_TUNED", "RIPSER_TUNED"]
