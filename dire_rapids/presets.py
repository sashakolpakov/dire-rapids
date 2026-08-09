"""Validated public hyperparameter presets for DiRePyTorch.

The former ``TOPOLOGY_TUNED`` configuration remains withdrawn. The narrower
``RIPSER_TUNED`` replacement was selected with crossed Atlas/Ripser scoring on
four datasets and confirmed with both evaluators over six untouched datasets
and 20 paired layout seeds. Its name intentionally describes the selection
objective rather than promising a universal topology improvement.
"""

RIPSER_TUNED = {
    "init": "pca",
    "n_neighbors": 16,
    "spread": 0.8,
    "min_dist": 1e-2,
    "cutoff": 42.0,
    "neg_ratio": 8,
    "max_iter_layout": 128,
}

__all__ = ["RIPSER_TUNED"]
