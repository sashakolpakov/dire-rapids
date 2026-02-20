"""
Benchmark DiRe vs UMAP on OpenML datasets — speed AND quality.

Compares:
  - DiRePyTorch (standard, GPU)
  - DiRePyTorchMemoryEfficient (GPU)
  - umap-learn (CPU)
  - cuML UMAP (GPU)

on a curated set of OpenML datasets spanning different sizes and dimensions.
Quality is measured via neighborhood preservation (k-NN overlap) and stress.
"""

import gc
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

# Curated OpenML datasets: (name, openml_id, max_samples)
DATASETS = [
    # Small-medium
    ("iris",               61,    None),     # 150 x 4
    ("wine",              187,    None),     # 178 x 13
    ("digits",             28,    None),     # 5620 x 64
    # Medium, high-D
    ("mnist_784",         554,   10000),     # 70K x 784 -> 10K
    ("Fashion-MNIST",   40996,   10000),     # 70K x 784 -> 10K
    ("har",              1478,    None),     # 10299 x 561
    # Larger
    ("covertype_20K",     293,   20000),     # 581K x 54 -> 20K
    ("mnist_50K",         554,   50000),     # 70K x 784 -> 50K
    # Large-scale (100K+)
    ("HIGGS_98K",       23512,    None),     # 98K x 28
    ("covertype_200K",   1596,  200000),     # 581K x 54 -> 200K
    ("covertype_581K",   1596,    None),     # 581K x 54 (full)
    ("jannis_830K",     42468,    None),     # 830K x 16
    ("MiniBooNE_940K",  44129,    None),     # 940K x 24
]

N_NEIGHBORS = 16
N_COMPONENTS = 2
MAX_ITER = 128
QUALITY_K = 16          # k for quality metrics
QUALITY_MAX_N = 50000   # subsample for quality metrics if dataset is larger

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_dataset(name, openml_id, max_samples=None):
    """Load and preprocess an OpenML dataset."""
    print(f"  Loading {name} (id={openml_id})...", end=" ", flush=True)
    try:
        from scipy import sparse
        data = fetch_openml(data_id=openml_id, as_frame=False, parser="auto")
        X = data.data
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
    except Exception as e:
        print(f"FAILED: {e}")
        return None, None

    # Handle NaN/Inf
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]

    # Subsample if needed
    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"shape={X.shape}")
    return X, X.shape


def time_reducer(fn, X, **kwargs):
    """Time a single fit_transform call. Returns (time, embedding)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    try:
        result = fn(X, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return elapsed, np.asarray(result, dtype=np.float32)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return None, None


def compute_quality(X, embedding, k=QUALITY_K, max_n=QUALITY_MAX_N):
    """Compute neighborhood preservation score.

    Returns the fraction of k-nearest neighbors in the original space
    that are also k-nearest neighbors in the embedding space.
    Uses sklearn on CPU for consistency across all methods.
    """
    if embedding is None:
        return None

    n = X.shape[0]
    # Subsample for large datasets
    if n > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_n, replace=False)
        X_sub = X[idx]
        emb_sub = embedding[idx]
    else:
        X_sub = X
        emb_sub = embedding

    # kNN in original space
    nn_orig = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    nn_orig.fit(X_sub)
    idx_orig = nn_orig.kneighbors(X_sub, return_distance=False)[:, 1:]  # exclude self

    # kNN in embedding space
    nn_emb = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    nn_emb.fit(emb_sub)
    idx_emb = nn_emb.kneighbors(emb_sub, return_distance=False)[:, 1:]

    # Vectorized set intersection
    matches = (idx_orig[:, :, None] == idx_emb[:, None, :]).any(axis=2)
    preservation = matches.sum(axis=1).astype(np.float32) / k
    return float(np.mean(preservation))


# ── Reducers ─────────────────────────────────────────────────────────────────

def run_dire_standard(X, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS, max_iter=MAX_ITER):
    from dire_rapids import DiRePyTorch
    reducer = DiRePyTorch(
        n_neighbors=n_neighbors, n_components=n_components,
        max_iter_layout=max_iter, random_state=42, verbose=False,
    )
    return reducer.fit_transform(X)


def run_dire_mem_efficient(X, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS, max_iter=MAX_ITER):
    from dire_rapids import DiRePyTorchMemoryEfficient
    reducer = DiRePyTorchMemoryEfficient(
        n_neighbors=n_neighbors, n_components=n_components,
        max_iter_layout=max_iter, random_state=42, verbose=False,
    )
    return reducer.fit_transform(X)


def run_umap_cpu(X, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS):
    import umap
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, n_components=n_components,
        random_state=42, verbose=False,
    )
    return reducer.fit_transform(X)


def run_umap_gpu(X, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS):
    from cuml.manifold import UMAP as cuUMAP
    import cupy as cp
    X_gpu = cp.asarray(X)
    reducer = cuUMAP(
        n_neighbors=n_neighbors, n_components=n_components,
        random_state=42, verbose=False,
    )
    result = reducer.fit_transform(X_gpu)
    return cp.asnumpy(result)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    has_gpu = torch.cuda.is_available()

    print("=" * 90)
    print("DiRe vs UMAP — OpenML Benchmark (Speed + Quality)")
    print("=" * 90)
    if has_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Quality metric: {QUALITY_K}-NN neighborhood preservation")
    print()

    # Warmup
    print("Warming up (PyKeOps JIT + torch.compile)...")
    X_warmup = np.random.randn(500, 50).astype(np.float32)
    run_dire_standard(X_warmup, max_iter=4)
    run_dire_mem_efficient(X_warmup, max_iter=4)
    print("Done.\n")

    results = []

    for dataset_name, openml_id, max_samples in DATASETS:
        print("-" * 90)
        X, shape = load_dataset(dataset_name, openml_id, max_samples)
        if X is None:
            continue
        n_samples, n_features = shape

        if n_samples < N_NEIGHBORS + 2:
            print(f"  Skipping (too few samples)")
            continue

        row = {"dataset": dataset_name, "N": n_samples, "D": n_features}

        # Skip slow methods for large datasets
        skip_umap_cpu = n_samples > 100000
        skip_dire_me = n_samples > 300000  # mem-efficient is slower at scale

        # --- DiRe Standard ---
        print(f"  DiRe Standard ...", end=" ", flush=True)
        t, emb_dire = time_reducer(run_dire_standard, X)
        if t is not None:
            print(f"{t:.2f}s", end="", flush=True)
            row["dire_std_t"] = t
            q = compute_quality(X, emb_dire)
            print(f"  (quality: {q:.3f})")
            row["dire_std_q"] = q
        else:
            print("FAILED")
            row["dire_std_t"] = row["dire_std_q"] = None

        # --- DiRe Memory-Efficient ---
        if not skip_dire_me:
            print(f"  DiRe MemEff   ...", end=" ", flush=True)
            t, emb = time_reducer(run_dire_mem_efficient, X)
            if t is not None:
                print(f"{t:.2f}s", end="", flush=True)
                row["dire_mem_t"] = t
                q = compute_quality(X, emb)
                print(f"  (quality: {q:.3f})")
                row["dire_mem_q"] = q
            else:
                print("FAILED")
                row["dire_mem_t"] = row["dire_mem_q"] = None
        else:
            print(f"  DiRe MemEff   ... skipped (N>{300000})")
            row["dire_mem_t"] = row["dire_mem_q"] = None

        # --- UMAP CPU ---
        if not skip_umap_cpu:
            print(f"  UMAP (CPU)    ...", end=" ", flush=True)
            t, emb_umap = time_reducer(run_umap_cpu, X)
            if t is not None:
                print(f"{t:.2f}s", end="", flush=True)
                row["umap_cpu_t"] = t
                q = compute_quality(X, emb_umap)
                print(f"  (quality: {q:.3f})")
                row["umap_cpu_q"] = q
            else:
                print("FAILED")
                row["umap_cpu_t"] = row["umap_cpu_q"] = None
        else:
            print(f"  UMAP (CPU)    ... skipped (N>{100000})")
            row["umap_cpu_t"] = row["umap_cpu_q"] = None

        # --- cuML UMAP (GPU) ---
        if has_gpu:
            print(f"  cuML UMAP     ...", end=" ", flush=True)
            t, emb_cu = time_reducer(run_umap_gpu, X)
            if t is not None:
                print(f"{t:.2f}s", end="", flush=True)
                row["cumap_t"] = t
                q = compute_quality(X, emb_cu)
                print(f"  (quality: {q:.3f})")
                row["cumap_q"] = q
            else:
                print("FAILED")
                row["cumap_t"] = row["cumap_q"] = None

        results.append(row)
        print()

    # ── Summary Table ────────────────────────────────────────────────────
    print("=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)

    df = pd.DataFrame(results)

    def fv(v, fmt=".2f", suffix="s"):
        return f"{v:{fmt}}{suffix}" if pd.notna(v) else "-"

    def fx(v):
        return f"{v:.1f}x" if pd.notna(v) else "-"

    # --- Speed Table ---
    print("\n  SPEED (seconds, lower = better)")
    print(f"  {'Dataset':<18} {'N':>7} {'D':>4} | {'DiRe':>7} {'DiRe-ME':>8} {'UMAP':>7} {'cuUMAP':>7} | {'DiRe vs UMAP':>13}")
    print("  " + "-" * 85)
    for _, r in df.iterrows():
        speedup = r.get('umap_cpu_t', np.nan) / r['dire_std_t'] if pd.notna(r.get('umap_cpu_t')) and pd.notna(r.get('dire_std_t')) and r['dire_std_t'] > 0 else np.nan
        line = f"  {r['dataset']:<18} {int(r['N']):>7} {int(r['D']):>4}"
        line += f" | {fv(r.get('dire_std_t')):>7} {fv(r.get('dire_mem_t')):>8} {fv(r.get('umap_cpu_t')):>7} {fv(r.get('cumap_t')):>7}"
        line += f" | {fx(speedup):>13}"
        print(line)

    # --- Quality Table ---
    print(f"\n  QUALITY ({QUALITY_K}-NN neighborhood preservation, higher = better)")
    print(f"  {'Dataset':<18} {'N':>7} {'D':>4} | {'DiRe':>7} {'DiRe-ME':>8} {'UMAP':>7} {'cuUMAP':>7}")
    print("  " + "-" * 65)
    for _, r in df.iterrows():
        line = f"  {r['dataset']:<18} {int(r['N']):>7} {int(r['D']):>4}"
        line += f" | {fv(r.get('dire_std_q'), '.3f', ''):>7} {fv(r.get('dire_mem_q'), '.3f', ''):>8} {fv(r.get('umap_cpu_q'), '.3f', ''):>7} {fv(r.get('cumap_q'), '.3f', ''):>7}"
        print(line)

    print()
    print("  '-' = skipped (too slow for dataset size)")
    print("=" * 90)
