"""Recommended hyperparameter presets for DiRePyTorch.

Presets are named hyperparameter dictionaries distilled from multi-objective
Optuna studies on OpenML benchmarks.  Use with ``DiRePyTorch(**preset)``.
"""

# ---------------------------------------------------------------------------
# TOPOLOGY_TUNED — discovered 2026-04-22/23 via NSGA-II Pareto studies on
# MNIST, Fashion-MNIST, isolet, and covertype with two objectives: maximise
# kNN classification accuracy in 2D, minimise topology error on stress-test
# manifolds (figure-8 at σ=0.2, standard torus at σ=0.05).
#
# This config sits near the Pareto knee across those datasets: it beats the
# default DiRe PCA/spectral settings on kNN and recovers β₁ almost exactly on
# the manifold stress tests.  On covertype, Fashion-MNIST, and isolet it
# strictly Pareto-dominates cuML-UMAP on both axes simultaneously.
#
# Key deviations from defaults:
#   spread             1.0 →  3.6   (more important than any other single knob)
#   max_iter_layout    128 →  150   (modest but consistent across studies)
#   n_neighbors         16 →   15   (kept essentially default)
#   init              'pca' → 'spectral'
# ---------------------------------------------------------------------------

TOPOLOGY_TUNED = {
    "init": "spectral",
    "n_neighbors": 15,
    "spread": 3.6,
    "min_dist": 1e-2,
    "cutoff": 42.0,
    "neg_ratio": 8,
    "max_iter_layout": 150,
}
