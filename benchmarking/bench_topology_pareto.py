#!/usr/bin/env python3
"""
Benchmark DiRe vs cuML UMAP via multi-objective Pareto search.

For each dataset, runs an NSGA-II search over DiRe hyperparameters with two
objectives:

  maximise  kNN classification accuracy in 2D
  minimise  topology error on stress-test manifolds
            = |β₁(2D embedding of noisy figure-8 at σ=0.2) − 2|
            + |β₁(2D embedding of noisy torus at σ=0.05) − 2|

The topology-error axis penalises embeddings that invent or destroy homology
on a fixed pair of manifolds with known first Betti number — a direct
measurement of topological faithfulness that is much less gameable than
neighborhood-preservation metrics (which reward noise memorisation).

Each Pareto front is compared against cuML UMAP and default DiRe. Datasets
where a Pareto trial strictly dominates cuML UMAP on BOTH axes are the
strongest single-dataset claims.

Requires:
    pip install ripser optuna cuml-cu13 (or compatible rapids build)

Usage:
    python bench_topology_pareto.py [--datasets covertype,mnist,...] [--n-trials 150]
"""

import argparse
import gc
import json
import sys
import time
import warnings

try:
    import cugraph  # noqa: F401  # import-order gotcha vs torch on rapids-26.04+
except Exception:
    pass

import numpy as np
import torch
import optuna
from optuna.samplers import NSGAIISampler

from ripser import ripser
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from dire_rapids import DiRePyTorch


try:
    from cuml import UMAP as cuUMAP  # pylint: disable=import-error
    HAS_CUML = True
except Exception:
    HAS_CUML = False


# Default benchmark mix. Each entry is (display_name, openml_id, optional subsample N).
DEFAULT_DATASETS = [
    ("mfeat-factors",  12,     None),
    ("satimage",       182,    None),
    ("pendigits",      32,     None),
    ("isolet",         300,    None),
    ("HAR",            1478,   None),
    ("letter",         6,      None),
    ("magic",          1120,   None),
    ("MNIST",          554,    None),
    ("Fashion-MNIST",  40996,  None),
    ("connect-4",      40668,  None),
    ("covertype",      1596,   None),
]


# ---------------------------------------------------------------------------
# Topology stress tests
# ---------------------------------------------------------------------------

def _sample_figure8(n, noise, seed):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([np.sin(t), np.sin(t) * np.cos(t)])
    return (X + rng.normal(0, noise, X.shape)).astype(np.float32)


def _sample_torus(n, noise, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([
        (2 + np.cos(theta)) * np.cos(phi),
        (2 + np.cos(theta)) * np.sin(phi),
        np.sin(theta),
    ])
    return (X + rng.normal(0, noise, X.shape)).astype(np.float32)


def _significant_bars(dgm, thresh_frac=0.3):
    """Count persistence-diagram bars whose persistence is at least
    ``thresh_frac`` of the largest finite bar in that dimension."""
    if len(dgm) == 0:
        return 0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return 0
    pers = finite[:, 1] - finite[:, 0]
    if pers.max() <= 0:
        return 0
    return int((pers >= thresh_frac * pers.max()).sum())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _load_openml(oml_id, subsample=None, rng_seed=42):
    d = fetch_openml(data_id=oml_id, as_frame=False, parser="auto")
    X = np.asarray(d.data, dtype=np.float32)
    y = np.asarray(d.target)
    try:
        y = y.astype(int)
    except Exception:
        y = LabelEncoder().fit_transform(y)
    # Drop rows containing NaN/inf (some OpenML datasets ship them).
    mask = np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    if subsample and X.shape[0] > subsample:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(X.shape[0], subsample, replace=False)
        X, y = X[idx], y[idx]
    return X, y


def _fit_dire(params, X):
    r = DiRePyTorch(n_components=2, verbose=False, random_state=0, **params)
    return np.asarray(r.fit_transform(X), dtype=np.float32)


def _fit_umap(X):
    u = cuUMAP(n_components=2, n_neighbors=16, random_state=0, verbose=False)
    return np.asarray(u.fit_transform(X), dtype=np.float32)


def _evaluate(params, X, y, X_fig8, X_torus, tr_idx, te_idx):
    emb = _fit_dire(params, X)
    if not np.all(np.isfinite(emb)):
        return None
    knn = KNeighborsClassifier(n_neighbors=15).fit(
        emb[tr_idx], y[tr_idx]).score(emb[te_idx], y[te_idx])
    emb_f = _fit_dire(params, X_fig8)
    emb_t = _fit_dire(params, X_torus)
    b1f = _significant_bars(ripser(emb_f, maxdim=1)['dgms'][1])
    b1t = _significant_bars(ripser(emb_t, maxdim=1)['dgms'][1])
    return float(knn), abs(b1f - 2) + abs(b1t - 2), int(b1f), int(b1t)


def _evaluate_umap(X, y, X_fig8, X_torus, tr_idx, te_idx):
    emb = _fit_umap(X)
    knn = KNeighborsClassifier(n_neighbors=15).fit(
        emb[tr_idx], y[tr_idx]).score(emb[te_idx], y[te_idx])
    emb_f = _fit_umap(X_fig8)
    emb_t = _fit_umap(X_torus)
    b1f = _significant_bars(ripser(emb_f, maxdim=1)['dgms'][1])
    b1t = _significant_bars(ripser(emb_t, maxdim=1)['dgms'][1])
    return float(knn), abs(b1f - 2) + abs(b1t - 2), int(b1f), int(b1t)


def _run_one_dataset(name, oml_id, subsample, n_trials, X_fig8, X_torus):
    print(f"\n{'='*72}\n=== {name} (openml {oml_id}) ===", flush=True)
    try:
        X, y = _load_openml(oml_id, subsample=subsample)
    except Exception as e:
        print(f"  load failed: {e}", flush=True)
        return None
    print(f"  shape: {X.shape}  classes: {len(np.unique(y))}", flush=True)
    tr_idx, te_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, random_state=0)

    # Baselines: default DiRe (pca + spectral) and cuML UMAP.
    baselines = {}
    for tag, setup in [
        ("DiRe-pca-default",  dict(init="pca",      n_neighbors=16,
                                   max_iter_layout=128, min_dist=1e-2,
                                   spread=1.0, cutoff=42.0, neg_ratio=8)),
        ("DiRe-spec-default", dict(init="spectral", n_neighbors=16,
                                   max_iter_layout=128, min_dist=1e-2,
                                   spread=1.0, cutoff=42.0, neg_ratio=8)),
    ]:
        res = _evaluate(setup, X, y, X_fig8, X_torus, tr_idx, te_idx)
        if res is None:
            continue
        baselines[tag] = res
        print(f"  {tag:<20}  kNN={res[0]:.4f}  topo_err={res[1]}  "
              f"β₁(fig-8={res[2]}, T²={res[3]})", flush=True)
    if HAS_CUML:
        try:
            res = _evaluate_umap(X, y, X_fig8, X_torus, tr_idx, te_idx)
            baselines["cuML-UMAP"] = res
            print(f"  {'cuML-UMAP':<20}  kNN={res[0]:.4f}  topo_err={res[1]}  "
                  f"β₁(fig-8={res[2]}, T²={res[3]})", flush=True)
        except Exception as e:
            print(f"  cuML-UMAP failed: {e}", flush=True)

    # NSGA-II search over hyperparameters (kNN up, topo_err down).
    def objective(trial):
        init = trial.suggest_categorical(
            "init", ["pca", "spectral", "diffusion", "jl"])
        params = dict(
            init=init,
            n_neighbors=trial.suggest_int("n_neighbors", 8, 48),
            cutoff=trial.suggest_float("cutoff", 2.0, 42.0),
            spread=trial.suggest_float("spread", 0.5, 4.0),
            min_dist=trial.suggest_float("min_dist", 1e-4, 1e-1, log=True),
            neg_ratio=trial.suggest_int("neg_ratio", 2, 32),
            max_iter_layout=trial.suggest_int("max_iter_layout", 64, 256),
        )
        if init == "diffusion":
            params["diffusion_time"] = trial.suggest_float(
                "diffusion_time", 0.0, 4.0)
        gc.collect()
        torch.cuda.empty_cache()
        try:
            res = _evaluate(params, X, y, X_fig8, X_torus, tr_idx, te_idx)
        except Exception:
            raise optuna.TrialPruned()
        if res is None:
            raise optuna.TrialPruned()
        trial.set_user_attr("b1_fig8", res[2])
        trial.set_user_attr("b1_T2", res[3])
        return res[0], res[1]

    print(f"  -- NSGA-II, {n_trials} trials --", flush=True)
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=NSGAIISampler(seed=0, population_size=25),
    )
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.1f}s", flush=True)

    pareto = study.best_trials
    print(f"  -- Pareto front: {len(pareto)} trials --", flush=True)
    for t in sorted(pareto, key=lambda x: -x.values[0]):
        print(f"    kNN={t.values[0]:.4f}  topo_err={t.values[1]:.0f}  "
              f"β₁(fig-8={t.user_attrs.get('b1_fig8')}, "
              f"T²={t.user_attrs.get('b1_T2')})  "
              f"{t.params.get('init')} k={t.params.get('n_neighbors')} "
              f"spread={t.params.get('spread'):.2f} "
              f"iter={t.params.get('max_iter_layout')}",
              flush=True)

    print("  -- dominance vs baselines --", flush=True)
    for bname, bvals in baselines.items():
        knn, tpe = bvals[0], bvals[1]
        ndom = sum(1 for t in pareto
                   if t.values[0] > knn and t.values[1] < tpe)
        print(f"    {bname:<20}  dominated by {ndom}/{len(pareto)} Pareto trials",
              flush=True)

    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "baselines": {k: list(v) for k, v in baselines.items()},
        "pareto": [
            {
                "values": list(t.values),
                "params": t.params,
                "user_attrs": dict(t.user_attrs),
            }
            for t in pareto
        ],
        "elapsed_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=None,
                        help="comma-separated display names to run (default: all)")
    parser.add_argument("--n-trials", type=int, default=150,
                        help="Optuna trials per dataset")
    parser.add_argument("--output", default="topology_pareto_results.json",
                        help="JSON output path")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sys.stdout.reconfigure(line_buffering=True)

    datasets = DEFAULT_DATASETS
    if args.datasets:
        wanted = set(s.strip() for s in args.datasets.split(","))
        datasets = [d for d in datasets if d[0] in wanted]
        if not datasets:
            print(f"No matching datasets in --datasets {args.datasets!r}", file=sys.stderr)
            return 1

    # Stress-test manifolds are sampled once and held fixed across all datasets
    # and trials, so the topology axis is comparable between runs.
    X_fig8 = _sample_figure8(1000, noise=0.2, seed=0)
    X_torus = _sample_torus(1000, noise=0.05, seed=0)

    all_results = {}
    for name, oml_id, sub in datasets:
        res = _run_one_dataset(name, oml_id, sub, args.n_trials, X_fig8, X_torus)
        if res is not None:
            all_results[name] = res
        # Incremental save in case of crash
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Aggregate summary: for each dataset, how many Pareto trials dominate UMAP?
    print(f"\n{'='*72}\nSUMMARY — Pareto dominance of cuML-UMAP")
    print(f"{'='*72}")
    print(f"{'Dataset':<18}{'|Pareto|':>10}{'#domUMAP':>10}  top-kNN / knee / top-topo")
    for name, r in all_results.items():
        if "baselines" not in r:
            continue
        umap = r["baselines"].get("cuML-UMAP")
        pareto = r["pareto"]
        if umap and pareto:
            ndom = sum(1 for p in pareto
                       if p["values"][0] > umap[0] and p["values"][1] < umap[1])
            vals = sorted(pareto, key=lambda p: -p["values"][0])
            best = vals[0]["values"]
            topo = min(pareto, key=lambda p: p["values"][1])["values"]
            print(f"{name:<18}{len(pareto):>10}{ndom:>10}  "
                  f"(kNN {best[0]:.3f}, topo {best[1]:.0f}) / "
                  f"(kNN {topo[0]:.3f}, topo {topo[1]:.0f})")

    print(f"\nResults saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
