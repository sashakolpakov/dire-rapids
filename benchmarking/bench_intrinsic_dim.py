"""
Intrinsic dimensionality detection via n_components sweep.

Reduces data to dimensions 1, 2, ..., k and plots quality metrics
(stress, neighbor preservation) vs. target dimension. An elbow at the
intrinsic dimension indicates where structure is fully captured.

Validated on synthetic manifolds with known intrinsic dimension, then
tested on real datasets (digits, MNIST subset) with estimated intrinsic
dimensions.
"""

import gc
import json
import os
import platform
import sys
import time
import warnings
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits, make_swiss_roll

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

N_NEIGHBORS = 16
MAX_ITER = 128
RANDOM_STATE = 42
AMBIENT_DIM = 100  # embed all synthetic data into R^D

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── Synthetic manifold generators ────────────────────────────────────────────

def _random_embedding_matrix(d_intrinsic, D_ambient, rng):
    """Random orthogonal D_ambient x d_intrinsic matrix via QR."""
    A = rng.standard_normal((D_ambient, d_intrinsic))
    Q, _ = np.linalg.qr(A)
    return Q[:, :d_intrinsic]


def _embed_in_ambient(X_intrinsic, D_ambient, rng, noise_scale=0.01):
    """Embed intrinsic data into ambient space via random orthogonal rotation + noise."""
    d = X_intrinsic.shape[1]
    Q = _random_embedding_matrix(d, D_ambient, rng)
    X = X_intrinsic @ Q.T
    X += rng.standard_normal(X.shape) * noise_scale
    return X.astype(np.float32)


def make_circle(n=1000, rng=None):
    """S^1 embedded in R^2."""
    theta = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    return X


def make_sphere(n=2000, rng=None):
    """S^2 via rejection sampling of unit vectors in R^3."""
    points = []
    while len(points) < n:
        v = rng.standard_normal((n * 2, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        points.append(v)
    return np.vstack(points)[:n]


def make_flat_torus(n=2000, rng=None):
    """T^2 = S^1 x S^1 embedded in R^4."""
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([np.cos(theta), np.sin(theta),
                         np.cos(phi), np.sin(phi)])
    return X


def make_triple_circle(n=3000, rng=None):
    """S^1 x S^1 x S^1 embedded in R^6."""
    theta1 = rng.uniform(0, 2 * np.pi, n)
    theta2 = rng.uniform(0, 2 * np.pi, n)
    theta3 = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([np.cos(theta1), np.sin(theta1),
                         np.cos(theta2), np.sin(theta2),
                         np.cos(theta3), np.sin(theta3)])
    return X


def make_gaussian(n=3000, d=5, rng=None):
    """Gaussian blob in R^d."""
    return rng.standard_normal((n, d)).astype(np.float32)


def _render_disk(frame, cx, cy, r, res):
    """Draw a filled disk into a (res, res) frame using antialiased mask."""
    yy, xx = np.mgrid[:res, :res]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    # Smooth edge: 1 inside, 0 outside, antialiased over 1px
    mask = np.clip(r - dist + 0.5, 0, 1)
    np.maximum(frame, mask, out=frame)


def make_ball_parabola(n=1000, res=64, rng=None):
    """Video frames of a ball in parabolic flight.

    One degree of freedom (time), so intrinsic d=1. Each frame is a
    res×res grayscale image flattened to a res²-dimensional vector.

    The ball follows x(t) = v₀t, y(t) = h - ½gt² with uniform t
    samples, producing a 1D manifold in R^(res²).
    """
    t = rng.uniform(0, 1, n)
    # Parabola: ball travels left-to-right with an arc
    ball_r = 3.0
    margin = ball_r + 2
    cx = margin + t * (res - 2 * margin)
    # Peak at t=0.5, touching top ~10px from edge
    cy = 10 + 4 * (res - 20) * (t - 0.5)**2

    frames = np.zeros((n, res, res), dtype=np.float32)
    for i in range(n):
        _render_disk(frames[i], cx[i], cy[i], ball_r, res)

    return frames.reshape(n, -1)  # (n, res²) — no ambient embedding needed


def make_ball_parabola_size(n=1000, res=64, rng=None):
    """Video frames of a ball in parabolic flight with varying size.

    Two degrees of freedom: time (position along parabola) and radius
    (varies independently), so intrinsic d=2.
    """
    t = rng.uniform(0, 1, n)
    s = rng.uniform(0, 1, n)  # independent size parameter

    margin_max = 7  # max radius + buffer
    cx = margin_max + t * (res - 2 * margin_max)
    cy = 10 + 4 * (res - 20) * (t - 0.5)**2
    ball_r = 2.0 + 4.0 * s  # radius from 2 to 6

    frames = np.zeros((n, res, res), dtype=np.float32)
    for i in range(n):
        _render_disk(frames[i], cx[i], cy[i], ball_r[i], res)

    return frames.reshape(n, -1)


def make_two_balls(n=1500, res=64, rng=None):
    """Video frames of two balls on independent linear trajectories.

    Two degrees of freedom (each ball's position along its path),
    so intrinsic d=2.
    """
    t1 = rng.uniform(0, 1, n)
    t2 = rng.uniform(0, 1, n)
    ball_r = 3.0

    # Ball 1: horizontal across upper half
    cx1 = 5 + t1 * (res - 10)
    cy1 = np.full(n, res * 0.25)

    # Ball 2: horizontal across lower half
    cx2 = 5 + t2 * (res - 10)
    cy2 = np.full(n, res * 0.75)

    frames = np.zeros((n, res, res), dtype=np.float32)
    for i in range(n):
        _render_disk(frames[i], cx1[i], cy1[i], ball_r, res)
        _render_disk(frames[i], cx2[i], cy2[i], ball_r, res)

    return frames.reshape(n, -1)


SYNTHETIC_DATASETS = [
    ("circle_S1",    1, make_circle,       {"n": 1000}),
    ("swiss_roll",   2, None,              {"n": 2000}),   # special case: sklearn
    ("flat_torus",   2, make_flat_torus,   {"n": 2000}),
    ("sphere_S2",    2, make_sphere,       {"n": 2000}),
    ("triple_S1",    3, make_triple_circle, {"n": 3000}),
    ("gaussian_d5",  5, make_gaussian,     {"n": 3000, "d": 5}),
    ("gaussian_d10", 10, make_gaussian,    {"n": 3000, "d": 10}),
    ("ball_flight",  1, make_ball_parabola, {"n": 1000, "res": 64}),
    ("ball_flight_size", 2, make_ball_parabola_size, {"n": 1000, "res": 64}),
    ("two_balls",    2, make_two_balls,    {"n": 1500, "res": 64}),
]


VIDEO_GENERATORS = {make_ball_parabola, make_ball_parabola_size, make_two_balls}


def generate_synthetic(name, true_d, gen_fn, kwargs, D_ambient=AMBIENT_DIM):
    """Generate a synthetic dataset embedded in high-dimensional ambient space."""
    rng = np.random.default_rng(RANDOM_STATE)

    if name == "swiss_roll":
        X_intrinsic, _ = make_swiss_roll(n_samples=kwargs["n"], noise=0.5,
                                          random_state=RANDOM_STATE)
        # swiss_roll returns 3D but intrinsic dim is 2; use the 3D embedding
        X = _embed_in_ambient(X_intrinsic, D_ambient, rng)
    elif gen_fn in VIDEO_GENERATORS:
        # Video frame generators already produce high-D data (res² pixels)
        X = gen_fn(rng=rng, **kwargs)
    else:
        X_intrinsic = gen_fn(rng=rng, **kwargs)
        X = _embed_in_ambient(X_intrinsic, D_ambient, rng)

    return X


# ── Real datasets ────────────────────────────────────────────────────────────

REAL_DATASETS = [
    ("digits",  8,  None),   # estimated intrinsic dim ~8-10
    ("MNIST_10K", 12, None), # estimated intrinsic dim ~12-14
]


def load_real_dataset(name):
    """Load a real dataset, return (X, estimated_d)."""
    if name == "digits":
        data = load_digits()
        X = data.data.astype(np.float32)
        return X, 8

    elif name == "MNIST_10K":
        try:
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml("mnist_784", version=1, as_frame=False,
                                 parser="liac-arff")
            X = mnist.data.astype(np.float32)
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(X.shape[0], 10000, replace=False)
            X = X[idx]
            return X, 12
        except Exception as e:
            print(f"    Could not load MNIST: {e}")
            return None, None

    return None, None


# ── DiRe runner ──────────────────────────────────────────────────────────────

def run_dire(X, n_components, n_neighbors=N_NEIGHBORS, max_iter=MAX_ITER):
    """Run DiRe reduction and return embedding."""
    from dire_rapids import DiRePyTorch
    reducer = DiRePyTorch(
        n_neighbors=n_neighbors,
        n_components=n_components,
        max_iter_layout=max_iter,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    return reducer.fit_transform(X)


def time_dire(X, n_components):
    """Time a single DiRe run with CUDA sync."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    try:
        result = run_dire(X, n_components)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        return elapsed, np.asarray(result, dtype=np.float32)
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"ERROR: {e} ({elapsed:.1f}s)")
        return None, None


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(X, embedding, n_neighbors=N_NEIGHBORS):
    """Compute stress and neighbor preservation score using sklearn CPU kNN.

    We bypass dire_rapids.metrics here because (a) cuML's NVRTC compilation
    can fail in some environments and (b) the CPU fallback in metrics.py
    doesn't import sklearn when cuML is available. For N<10K, sklearn is fast
    enough.
    """
    from sklearn.neighbors import NearestNeighbors

    X_np = np.asarray(X, dtype=np.float32)
    emb_np = np.asarray(embedding, dtype=np.float32)
    k = n_neighbors

    # kNN in original space
    nn_orig = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn_orig.fit(X_np)
    dist_orig, idx_orig = nn_orig.kneighbors(X_np)
    dist_orig = dist_orig[:, 1:]  # drop self
    idx_orig = idx_orig[:, 1:]

    # kNN in embedding space
    nn_emb = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn_emb.fit(emb_np)
    dist_emb_full, idx_emb = nn_emb.kneighbors(emb_np)
    idx_emb = idx_emb[:, 1:]

    # Stress: ratio of original to embedding distances for neighbor pairs
    neighbor_coords = emb_np[idx_orig]  # (N, k, d)
    point_coords = emb_np[:, None, :]   # (N, 1, d)
    dist_emb = np.linalg.norm(neighbor_coords - point_coords, axis=2)
    eps = 1e-6
    ratios = np.abs(dist_orig / np.maximum(dist_emb, eps) - 1.0)
    stress_mean = float(np.mean(ratios))
    stress_std = float(np.std(ratios))
    stress = 0.0 if stress_mean < eps else stress_std / stress_mean

    # Neighbor preservation: fraction of k-NN preserved
    matches = (idx_orig[:, :, None] == idx_emb[:, None, :]).any(axis=2)
    preservation = matches.sum(axis=1).astype(np.float32) / k
    neighbor_score = float(np.mean(preservation))

    return stress, neighbor_score


def compute_pca_variance(X, max_components):
    """Compute cumulative explained variance ratio via PCA."""
    X_t = torch.from_numpy(X).float()
    if torch.cuda.is_available():
        X_t = X_t.cuda()

    # Center the data
    X_t = X_t - X_t.mean(dim=0, keepdim=True)
    q = min(max_components, X_t.shape[0] - 1, X_t.shape[1])
    _, S, _ = torch.pca_lowrank(X_t, q=q)
    var = (S ** 2) / (X_t.shape[0] - 1)
    total_var = var.sum()
    cumulative = torch.cumsum(var, dim=0) / total_var
    return cumulative.cpu().numpy()


# ── Elbow detection ──────────────────────────────────────────────────────────

def detect_elbow(values, increasing=True):
    """Detect elbow via the Kneedle method (max distance from baseline).

    Normalize the curve to [0,1] x [0,1], then find the point with
    maximum difference from the diagonal baseline. For a concave-up
    increasing curve (rises steeply then plateaus), this finds the
    transition point.

    Returns 1-indexed dimension.
    """
    if len(values) < 3:
        return 1
    v = np.array(values, dtype=np.float64)

    # Normalize x to [0, 1]
    x = np.arange(len(v), dtype=np.float64)
    x_norm = x / (len(v) - 1)

    # Normalize y to [0, 1]
    y_min, y_max = v.min(), v.max()
    if abs(y_max - y_min) < 1e-12:
        return 1

    if increasing:
        # Normalize so curve goes from low to high
        y_norm = (v - y_min) / (y_max - y_min)
    else:
        # Flip decreasing curve: high stress at low d → treat as increasing
        y_norm = (y_max - v) / (y_max - y_min)

    # The knee is where y_norm is farthest above the diagonal
    # (the curve bows upward for a concave-up shape)
    diff = y_norm - x_norm
    idx = np.argmax(diff)
    return int(idx + 1)  # 1-indexed dimension


def detect_pca_elbow(cumvar, threshold=0.95):
    """Find the dimension where cumulative variance exceeds threshold."""
    above = np.where(cumvar >= threshold)[0]
    if len(above) > 0:
        return int(above[0]) + 1
    return len(cumvar)


# ── Main sweep ───────────────────────────────────────────────────────────────

def sweep_dataset(name, X, true_d, max_dim):
    """Sweep n_components for a single dataset, return metrics dict."""
    dims = list(range(1, max_dim + 1))
    stresses = []
    neighbor_scores = []
    times = []

    print(f"\n  Sweeping n_components = 1..{max_dim}")
    for d in dims:
        print(f"    d={d:>2}: ", end="", flush=True)
        elapsed, emb = time_dire(X, d)
        if emb is None:
            stresses.append(None)
            neighbor_scores.append(None)
            times.append(None)
            continue

        stress, nscore = compute_metrics(X, emb)
        stresses.append(float(stress))
        neighbor_scores.append(float(nscore))
        times.append(round(elapsed, 3))
        print(f"stress={stress:.4f}  neighbor={nscore:.4f}  time={elapsed:.2f}s")

    # PCA baseline
    print(f"    PCA variance...", end=" ", flush=True)
    cumvar = compute_pca_variance(X, max_dim)
    pca_dim = detect_pca_elbow(cumvar)
    print(f"95% at d={pca_dim}")

    # Elbow detection (skip None values)
    valid_stresses = [s for s in stresses if s is not None]
    valid_neighbors = [n for n in neighbor_scores if n is not None]

    det_stress = detect_elbow(valid_stresses, increasing=False) if len(valid_stresses) >= 3 else None
    det_neighbor = detect_elbow(valid_neighbors, increasing=True) if len(valid_neighbors) >= 3 else None

    return {
        "name": name,
        "true_d": true_d,
        "N": X.shape[0],
        "D": X.shape[1],
        "dims": dims,
        "stress": stresses,
        "neighbor_score": neighbor_scores,
        "time": times,
        "pca_cumvar": cumvar.tolist(),
        "detected_stress": det_stress,
        "detected_neighbor": det_neighbor,
        "detected_pca": pca_dim,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(all_results, output_path):
    """Generate a multi-panel figure: one row per dataset, three columns."""
    n_datasets = len(all_results)
    fig, axes = plt.subplots(n_datasets, 3, figsize=(15, 4 * n_datasets),
                              squeeze=False)

    for i, res in enumerate(all_results):
        dims = res["dims"]
        true_d = res["true_d"]

        # Column 1: Stress vs dimension
        ax = axes[i, 0]
        valid = [(d, s) for d, s in zip(dims, res["stress"]) if s is not None]
        if valid:
            ds, ss = zip(*valid)
            ax.plot(ds, ss, "o-", color="tab:red", linewidth=2, markersize=5)
        if res["detected_stress"] is not None:
            ax.axvline(res["detected_stress"], color="tab:red", linestyle=":",
                       alpha=0.7, label=f"detected={res['detected_stress']}")
        ax.axvline(true_d, color="black", linestyle="--", alpha=0.5,
                   label=f"true d={true_d}")
        ax.set_xlabel("n_components")
        ax.set_ylabel("Stress (lower = better)")
        ax.set_title(f"{res['name']}  (N={res['N']}, D={res['D']})")
        ax.legend(fontsize=8)
        ax.set_xticks(dims)

        # Column 2: Neighbor score vs dimension
        ax = axes[i, 1]
        valid = [(d, n) for d, n in zip(dims, res["neighbor_score"]) if n is not None]
        if valid:
            ds, ns = zip(*valid)
            ax.plot(ds, ns, "o-", color="tab:blue", linewidth=2, markersize=5)
        if res["detected_neighbor"] is not None:
            ax.axvline(res["detected_neighbor"], color="tab:blue", linestyle=":",
                       alpha=0.7, label=f"detected={res['detected_neighbor']}")
        ax.axvline(true_d, color="black", linestyle="--", alpha=0.5,
                   label=f"true d={true_d}")
        ax.set_xlabel("n_components")
        ax.set_ylabel("Neighbor score (higher = better)")
        ax.set_title(f"{res['name']}")
        ax.legend(fontsize=8)
        ax.set_xticks(dims)

        # Column 3: PCA cumulative variance
        ax = axes[i, 2]
        pca_dims = list(range(1, len(res["pca_cumvar"]) + 1))
        ax.plot(pca_dims, res["pca_cumvar"], "o-", color="tab:green",
                linewidth=2, markersize=5)
        ax.axhline(0.95, color="gray", linestyle=":", alpha=0.5, label="95%")
        if res["detected_pca"] is not None:
            ax.axvline(res["detected_pca"], color="tab:green", linestyle=":",
                       alpha=0.7, label=f"PCA d={res['detected_pca']}")
        ax.axvline(true_d, color="black", linestyle="--", alpha=0.5,
                   label=f"true d={true_d}")
        ax.set_xlabel("n_components")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_title(f"{res['name']} — PCA baseline")
        ax.legend(fontsize=8)
        ax.set_xticks(pca_dims)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {output_path}")


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary(all_results):
    """Print detection accuracy table."""
    header = (f"{'Dataset':<16} {'true_d':>6} "
              f"{'stress':>8} {'neighbor':>10} {'PCA':>6}")
    sep = "-" * len(header)

    print("\n" + "=" * 60)
    print("INTRINSIC DIMENSION DETECTION SUMMARY")
    print("=" * 60)
    print(header)
    print(sep)

    lines = [header, sep]
    for r in all_results:
        ds = str(r["detected_stress"]) if r["detected_stress"] is not None else "-"
        dn = str(r["detected_neighbor"]) if r["detected_neighbor"] is not None else "-"
        dp = str(r["detected_pca"]) if r["detected_pca"] is not None else "-"
        line = f"{r['name']:<16} {r['true_d']:>6} {ds:>8} {dn:>10} {dp:>6}"
        print(line)
        lines.append(line)

    print("=" * 60)
    return lines


# ── Main ─────────────────────────────────────────────────────────────────────

def build_metadata():
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "n_neighbors": N_NEIGHBORS,
        "max_iter": MAX_ITER,
        "random_state": RANDOM_STATE,
        "ambient_dim": AMBIENT_DIM,
    }
    if torch.cuda.is_available():
        meta["gpu"] = torch.cuda.get_device_name(0)
        meta["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    return meta


def main():
    print("=" * 70)
    print("Intrinsic Dimensionality Detection via n_components Sweep")
    print("=" * 70)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Parameters: n_neighbors={N_NEIGHBORS}, max_iter={MAX_ITER}")
    print()

    # Warmup torch.compile
    print("Warming up torch.compile...")
    X_warmup = np.random.randn(500, 50).astype(np.float32)
    run_dire(X_warmup, n_components=2, max_iter=4)
    print("Done.\n")

    all_results = []

    # ── Synthetic datasets ────────────────────────────────────────────────
    print("=" * 70)
    print("SYNTHETIC MANIFOLDS (known intrinsic dimension)")
    print("=" * 70)

    for name, true_d, gen_fn, kwargs in SYNTHETIC_DATASETS:
        max_dim = min(2 * true_d + 5, 25)
        print(f"\n{'─' * 60}")
        print(f"  {name}: intrinsic d={true_d}, sweep 1..{max_dim}")

        X = generate_synthetic(name, true_d, gen_fn, kwargs)
        print(f"  Shape: {X.shape}")

        res = sweep_dataset(name, X, true_d, max_dim)
        all_results.append(res)

    # ── Real datasets ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REAL DATASETS (estimated intrinsic dimension)")
    print("=" * 70)

    for name, est_d, _ in REAL_DATASETS:
        max_dim = 25
        print(f"\n{'─' * 60}")
        print(f"  {name}: estimated d~{est_d}, sweep 1..{max_dim}")

        X, est_d = load_real_dataset(name)
        if X is None:
            print(f"  Skipping {name}")
            continue
        print(f"  Shape: {X.shape}")

        res = sweep_dataset(name, X, est_d, max_dim)
        all_results.append(res)

    # ── Summary and output ────────────────────────────────────────────────
    summary_lines = print_summary(all_results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metadata = build_metadata()

    # JSON results (convert numpy types for serialization)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json_path = os.path.join(RESULTS_DIR, "intrinsic_dim_results.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "results": all_results}, f,
                  indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {json_path}")

    # Summary text
    txt_path = os.path.join(RESULTS_DIR, "intrinsic_dim_summary.txt")
    with open(txt_path, "w") as f:
        f.write("Intrinsic Dimensionality Detection via n_components Sweep\n")
        f.write(f"Date: {metadata['timestamp']}\n")
        if "gpu" in metadata:
            f.write(f"GPU: {metadata['gpu']}\n")
        f.write(f"PyTorch: {metadata['pytorch']}\n")
        f.write(f"Parameters: n_neighbors={N_NEIGHBORS}, max_iter={MAX_ITER}\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write("\n")
    print(f"Summary saved to {txt_path}")

    # Plot
    plot_path = os.path.join(RESULTS_DIR, "intrinsic_dim_plots.png")
    plot_results(all_results, plot_path)


if __name__ == "__main__":
    main()
