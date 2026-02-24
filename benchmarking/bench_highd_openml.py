"""
Benchmark DiRe vs cuML UMAP on high-dimensional OpenML datasets.

These datasets have very large feature counts (50K-200K) where
dimensionality reduction is most needed. They complement the
medium-dimensional benchmarks in bench_openml.py.

Datasets:
  - ovarianTumour (1086): 283 x 54,621  -- Gene expression, mass spectrometry
  - Dorothea (4137): 1,150 x 100,000    -- NIPS 2003 drug discovery, sparse
  - Flora (42708): 15,000 x 200,000     -- ChaLearn AutoML 2014, sparse

Note: OpenML's features API times out on these high-D datasets, so we
download ARFF/sparse_ARFF files directly and parse them ourselves.
"""

import gc
import json
import os
import platform
import sys
import tempfile
import time
import urllib.request
import warnings
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

# (name, openml_id, download_url, format, max_samples)
DATASETS = [
    ("ovarianTumour", 1086,
     "https://openml.org/data/v1/download/1390178/ovarianTumour.arff",
     "arff", None),
    ("Dorothea", 4137,
     "https://openml.org/data/v1/download/1681112/Dorothea.sparse_arff",
     "sparse_arff", None),
    ("Flora", 42708,
     "https://openml.org/data/v1/download/22044609/Flora.sparse_arff",
     "sparse_arff", None),
]

N_NEIGHBORS = 16
N_COMPONENTS = 2
MAX_ITER = 128
QUALITY_K = 16
QUALITY_MAX_N = 50000

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CACHE_DIR = os.path.join(tempfile.gettempdir(), "openml_cache")

# ── Data Loading ─────────────────────────────────────────────────────────────

def _download_cached(url, name, fmt):
    """Download a file, caching in /tmp."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    local_path = os.path.join(CACHE_DIR, f"{name}.{fmt}")
    if os.path.exists(local_path):
        print(f"cached, ", end="", flush=True)
        return local_path
    print(f"downloading...", end=" ", flush=True)
    urllib.request.urlretrieve(url, local_path)
    mb = os.path.getsize(local_path) / (1024 ** 2)
    print(f"{mb:.1f}MB, ", end="", flush=True)
    return local_path


def _load_arff(path):
    """Load a regular ARFF file via scipy, return numeric features as float32."""
    from scipy.io import arff
    import pandas as pd

    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].values.astype(np.float32)


def _load_sparse_arff(path):
    """Parse a sparse ARFF file, return dense float32 array of numeric features."""
    n_attrs = 0
    data_started = False
    rows = []
    target_col = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            upper = line.upper()
            if upper.startswith("@ATTRIBUTE"):
                parts = line.split()
                if len(parts) >= 2:
                    attr_name = parts[1].strip("'\"")
                    # Detect target/class column
                    if attr_name.lower() in ("class", "target", "label"):
                        target_col = n_attrs
                n_attrs += 1
            elif upper.startswith("@DATA"):
                data_started = True
            elif data_started and line.startswith("{"):
                content = line[1:line.rindex("}")].strip()
                row_data = {}
                if content:
                    for pair in content.split(","):
                        pair = pair.strip()
                        if not pair:
                            continue
                        parts = pair.split(None, 1)
                        if len(parts) == 2:
                            try:
                                row_data[int(parts[0])] = float(parts[1])
                            except ValueError:
                                pass  # skip categorical values
                rows.append(row_data)

    # Build dense matrix excluding target column
    feature_cols = [i for i in range(n_attrs) if i != target_col]
    col_map = {old: new for new, old in enumerate(feature_cols)}
    n_features = len(feature_cols)

    X = np.zeros((len(rows), n_features), dtype=np.float32)
    for i, row_data in enumerate(rows):
        for col, val in row_data.items():
            if col in col_map:
                X[i, col_map[col]] = val

    return X


def load_dataset(name, openml_id, url, fmt, max_samples=None):
    """Load and preprocess an OpenML dataset via direct download."""
    print(f"  Loading {name} (id={openml_id})... ", end="", flush=True)
    try:
        t0 = time.perf_counter()
        path = _download_cached(url, name, fmt)
        if fmt == "arff":
            X = _load_arff(path)
        else:
            X = _load_sparse_arff(path)
        load_time = time.perf_counter() - t0
    except Exception as e:
        print(f"FAILED: {e}")
        return None, None

    # Handle NaN/Inf
    mask = np.isfinite(X).all(axis=1)
    n_dropped = X.shape[0] - mask.sum()
    if n_dropped > 0:
        print(f"(dropped {n_dropped} NaN/Inf rows) ", end="", flush=True)
    X = X[mask]

    # Subsample if needed
    if max_samples is not None and X.shape[0] > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    nnz = np.count_nonzero(X)
    sparsity = 1.0 - nnz / (X.shape[0] * X.shape[1])
    mem_gb = X.nbytes / (1024 ** 3)
    print(f"shape={X.shape}, sparsity={sparsity:.2%}, "
          f"mem={mem_gb:.2f}GB, loaded in {load_time:.1f}s")
    return X, X.shape


# ── Timing & Quality ─────────────────────────────────────────────────────────

def time_reducer(fn, X, **kwargs):
    """Time a single fit_transform call with CUDA sync."""
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
        err_type = type(e).__name__
        if "out of memory" in str(e).lower() or "OOM" in str(e):
            print(f"OOM ({elapsed:.1f}s)", end="", flush=True)
            return "OOM", None
        print(f"ERROR: {err_type}: {e} ({elapsed:.1f}s)", end="", flush=True)
        return f"ERROR: {err_type}: {e}", None


def compute_quality(X, embedding, k=QUALITY_K, max_n=QUALITY_MAX_N):
    """k-NN neighborhood preservation between original and embedding space."""
    if embedding is None:
        return None

    n = X.shape[0]
    if n <= k + 1:
        return None

    if n > max_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_n, replace=False)
        X_sub, emb_sub = X[idx], embedding[idx]
    else:
        X_sub, emb_sub = X, embedding

    print(f"    Computing {k}-NN quality (N={X_sub.shape[0]}, D={X_sub.shape[1]})...",
          end=" ", flush=True)
    t0 = time.perf_counter()

    nn_orig = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn_orig.fit(X_sub)
    idx_orig = nn_orig.kneighbors(X_sub, return_distance=False)[:, 1:]

    nn_emb = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn_emb.fit(emb_sub)
    idx_emb = nn_emb.kneighbors(emb_sub, return_distance=False)[:, 1:]

    matches = (idx_orig[:, :, None] == idx_emb[:, None, :]).any(axis=2)
    preservation = matches.sum(axis=1).astype(np.float32) / k
    score = float(np.mean(preservation))

    elapsed = time.perf_counter() - t0
    print(f"{score:.3f} ({elapsed:.1f}s)")
    return score


# ── Reducers ─────────────────────────────────────────────────────────────────

def run_dire(X, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS, max_iter=MAX_ITER):
    from dire_rapids import DiRePyTorch
    reducer = DiRePyTorch(
        n_neighbors=n_neighbors, n_components=n_components,
        max_iter_layout=max_iter, random_state=42, verbose=False,
    )
    return reducer.fit_transform(X)


def run_cuml_umap(X, n_neighbors=N_NEIGHBORS, n_components=N_COMPONENTS):
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

def build_metadata():
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "n_neighbors": N_NEIGHBORS,
        "n_components": N_COMPONENTS,
        "max_iter": MAX_ITER,
        "quality_k": QUALITY_K,
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    try:
        import cuml
        meta["cuml_version"] = cuml.__version__
    except Exception:
        pass
    return meta


def format_val(v, fmt=".2f", suffix="s"):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v[:12]
    return f"{v:{fmt}}{suffix}"


def run_benchmarks():
    has_gpu = torch.cuda.is_available()

    print("=" * 80)
    print("DiRe vs cuML UMAP — High-Dimensional OpenML Benchmark")
    print("=" * 80)
    if has_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Parameters: n_neighbors={N_NEIGHBORS}, n_components={N_COMPONENTS}, "
          f"max_iter={MAX_ITER}")
    print(f"Quality: {QUALITY_K}-NN neighborhood preservation")
    print()

    # Warmup torch.compile + cuML JIT
    print("Warming up (torch.compile + cuML JIT)...")
    X_warmup = np.random.randn(500, 50).astype(np.float32)
    run_dire(X_warmup, max_iter=4)
    if has_gpu:
        run_cuml_umap(X_warmup)
    print("Done.\n")

    results = []
    metadata = build_metadata()

    for dataset_name, openml_id, url, fmt, max_samples in DATASETS:
        print("-" * 80)
        X, shape = load_dataset(dataset_name, openml_id, url, fmt, max_samples)
        if X is None:
            continue
        n_samples, n_features = shape

        if n_samples < N_NEIGHBORS + 2:
            print(f"  Skipping (too few samples: {n_samples})")
            continue

        row = {"dataset": dataset_name, "N": n_samples, "D": n_features}

        # --- DiRe ---
        print(f"  DiRe          ...", end=" ", flush=True)
        t, emb_dire = time_reducer(run_dire, X)
        if isinstance(t, (int, float)):
            print(f"{t:.2f}s")
            row["dire_time"] = round(t, 3)
            row["dire_quality"] = compute_quality(X, emb_dire)
        else:
            print()
            row["dire_time"] = t
            row["dire_quality"] = None

        # --- cuML UMAP ---
        if has_gpu:
            print(f"  cuML UMAP     ...", end=" ", flush=True)
            t, emb_cu = time_reducer(run_cuml_umap, X)
            if isinstance(t, (int, float)):
                print(f"{t:.2f}s")
                row["cuml_time"] = round(t, 3)
                row["cuml_quality"] = compute_quality(X, emb_cu)
            else:
                print()
                row["cuml_time"] = t
                row["cuml_quality"] = None

        # Speedup
        dt = row.get("dire_time")
        ct = row.get("cuml_time")
        if isinstance(dt, (int, float)) and isinstance(ct, (int, float)) and dt > 0:
            row["speedup_vs_cuml"] = round(ct / dt, 2)
        else:
            row["speedup_vs_cuml"] = None

        results.append(row)
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 80)
    print("RESULTS SUMMARY — High-Dimensional Datasets")
    print("=" * 80)

    header = (f"  {'Dataset':<16} {'N':>6} {'D':>8} | "
              f"{'DiRe':>8} {'cuUMAP':>8} {'Speedup':>8} | "
              f"{'DiRe Q':>7} {'cuML Q':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    summary_lines = [header, "  " + "-" * (len(header) - 2)]
    for r in results:
        dt = format_val(r.get("dire_time"))
        ct = format_val(r.get("cuml_time"))
        sp = format_val(r.get("speedup_vs_cuml"), ".1f", "x")
        dq = format_val(r.get("dire_quality"), ".3f", "")
        cq = format_val(r.get("cuml_quality"), ".3f", "")
        line = (f"  {r['dataset']:<16} {r['N']:>6} {r['D']:>8} | "
                f"{dt:>8} {ct:>8} {sp:>8} | "
                f"{dq:>7} {cq:>7}")
        print(line)
        summary_lines.append(line)

    print("=" * 80)

    # ── Save results ─────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)

    json_path = os.path.join(RESULTS_DIR, "highd_openml_results.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": results}, f, indent=2)
    print(f"\nResults saved to {json_path}")

    txt_path = os.path.join(RESULTS_DIR, "highd_openml_summary.txt")
    with open(txt_path, "w") as f:
        f.write("DiRe vs cuML UMAP — High-Dimensional OpenML Benchmark\n")
        f.write(f"Date: {metadata['timestamp']}\n")
        if "gpu" in metadata:
            f.write(f"GPU: {metadata['gpu']}\n")
        f.write(f"PyTorch: {metadata['pytorch']}\n")
        f.write(f"Parameters: n_neighbors={N_NEIGHBORS}, n_components={N_COMPONENTS}, "
                f"max_iter={MAX_ITER}\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\n")
    print(f"Summary saved to {txt_path}")


if __name__ == "__main__":
    run_benchmarks()
