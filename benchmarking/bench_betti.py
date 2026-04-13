"""
Benchmark Betti curve computation: correctness and speed.

Three backends:
  - CPU (eigsh): sparse eigenvalue reference, fast at small N but approximate
  - Fast (UF+rank): exact union-find β₀ + GPU SVD rank β₁, slower but exact
  - GPU (CuPy): CuPy-accelerated eigsh

Part 1: Correctness — all backends on small synthetic datasets with known topology
Part 2: Speed scaling — circle with varying N (each backend runs at its tractable sizes)
Part 3: Topological preservation — Betti curve DTW for DiRe vs cuML UMAP embeddings
"""

import gc
import sys
import time
import warnings

import numpy as np
from fastdtw import fastdtw

warnings.filterwarnings("ignore")


# ── Synthetic datasets with known topology ──────────────────────────────────

def make_circle(n=1000, noise=0.02, seed=42):
    """S¹: β₀=1, β₁=1"""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    X = np.column_stack([np.cos(theta), np.sin(theta)])
    X += rng.normal(0, noise, X.shape)
    return X.astype(np.float32), "circle (S¹)", {"β₀": 1, "β₁": 1}


def make_sphere(n=1000, noise=0.02, seed=42):
    """S² in R³: β₀=1, β₁=0"""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 3)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X += rng.normal(0, noise, X.shape).astype(np.float32)
    return X, "sphere (S²)", {"β₀": 1, "β₁": 0}


def make_torus(n=2000, noise=0.02, R=2.0, r=1.0, seed=42):
    """T² in R³: β₀=1, β₁=2"""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    X = np.column_stack([x, y, z]).astype(np.float32)
    X += rng.normal(0, noise, X.shape).astype(np.float32)
    return X, "torus (T²)", {"β₀": 1, "β₁": 2}


def make_linked_rings(n=1000, noise=0.02, seed=42):
    """Two linked circles in R³: β₀=2, β₁=2"""
    rng = np.random.default_rng(seed)
    half = n // 2
    theta1 = rng.uniform(0, 2 * np.pi, half)
    ring1 = np.column_stack([np.cos(theta1), np.sin(theta1), np.zeros(half)])
    theta2 = rng.uniform(0, 2 * np.pi, n - half)
    ring2 = np.column_stack([
        1.0 + np.cos(theta2),
        np.zeros(n - half),
        np.sin(theta2)
    ])
    X = np.vstack([ring1, ring2]).astype(np.float32)
    X += rng.normal(0, noise, X.shape).astype(np.float32)
    return X, "linked rings", {"β₀": 2, "β₁": 2}


def make_blobs(n=500, k=5, dim=10, seed=42):
    """k well-separated blobs: β₀=k, β₁=0"""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-20, 20, (k, dim))
    labels = rng.integers(0, k, n)
    X = centers[labels] + rng.normal(0, 0.3, (n, dim))
    return X.astype(np.float32), f"{k} blobs (R^{dim})", {"β₀": k, "β₁": 0}


# ── Helpers ─────────────────────────────────────────────────────────────────

def time_betti(fn, data, **kwargs):
    """Time a Betti curve computation. Returns (time, result)."""
    gc.collect()
    t0 = time.perf_counter()
    try:
        result = fn(data, **kwargs)
        elapsed = time.perf_counter() - t0
        return elapsed, result
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return None, str(e)


def extract_betti_at_full(result):
    """Extract β₀, β₁ at the full complex (first filtration step = all edges)."""
    if isinstance(result, str):
        return None, None
    return int(result['beta_0'][0]), int(result['beta_1'][0])


def dtw_betti(curve_orig, curve_emb):
    """DTW distance between two Betti curves (1D sequences)."""
    if curve_orig is None or curve_emb is None:
        return None
    c1 = np.array(curve_orig, dtype=np.float64).reshape(-1, 1)
    c2 = np.array(curve_emb, dtype=np.float64).reshape(-1, 1)
    dist, _ = fastdtw(c1, c2, radius=5)
    return float(dist)


def fmt_time(t):
    return f"{t:.2f}s" if t is not None else "-"


def fmt_betti(b0, b1):
    return f"β₀={b0},β₁={b1}" if b0 is not None else "FAILED"


def log(msg="", **kwargs):
    print(msg, **kwargs)
    sys.stdout.flush()


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("=" * 100)
    log("Betti Curve Benchmark: Correctness + Speed")
    log("=" * 100)

    has_gpu = False
    try:
        import cupy as cp
        has_gpu = True
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        log(f"GPU: {gpu_name}")
    except ImportError:
        log("GPU: Not available (CuPy not installed)")
    log()

    from dire_rapids.betti_curve import compute_betti_curve_cpu, compute_betti_curve_fast
    if has_gpu:
        from dire_rapids.betti_curve import compute_betti_curve_gpu

    # ── Part 1: Correctness ───────────────────────────────────────────
    log("-" * 100)
    log("PART 1: Correctness — Betti numbers on synthetic datasets (full complex)")
    log("  Small N, k=15 to keep eigsh tractable. Focus: do all backends agree?")
    log("-" * 100)

    datasets = [
        make_circle(n=200),
        make_sphere(n=200),
        make_torus(n=300),
        make_linked_rings(n=200),
        make_blobs(n=200, k=5, dim=10),
        make_blobs(n=200, k=3, dim=5),
    ]

    betti_params = dict(k_neighbors=15, density_threshold=0.8, overlap_factor=1.5, n_steps=10)

    hdr = f"  {'Dataset':<22} {'N':>5} {'D':>3} | {'Expected':>10} | {'CPU (eigsh)':>12} {'time':>7} | {'Fast (UF+rank)':>14} {'time':>7}"
    if has_gpu:
        hdr += f" | {'GPU (CuPy)':>12} {'time':>7}"
    hdr += " | Match?"
    log(f"\n{hdr}")
    log("  " + "-" * (len(hdr) - 2))

    for data, name, expected in datasets:
        n, d = data.shape
        exp_str = f"β₀={expected['β₀']},β₁={expected['β₁']}"

        # CPU (eigsh reference)
        log(f"  {name} (cpu)...", end=" ")
        t_cpu, res_cpu = time_betti(compute_betti_curve_cpu, data, **betti_params)
        b0_cpu, b1_cpu = extract_betti_at_full(res_cpu)

        # Fast (union-find + GPU SVD rank)
        log("(fast)...", end=" ")
        t_fast, res_fast = time_betti(compute_betti_curve_fast, data, **betti_params)
        b0_fast, b1_fast = extract_betti_at_full(res_fast)

        line = f"\r  {name:<22} {n:>5} {d:>3} | {exp_str:>10} | {fmt_betti(b0_cpu, b1_cpu):>12} {fmt_time(t_cpu):>7} | {fmt_betti(b0_fast, b1_fast):>14} {fmt_time(t_fast):>7}"

        if has_gpu:
            log("(gpu)...", end=" ")
            t_gpu, res_gpu = time_betti(compute_betti_curve_gpu, data, **betti_params)
            b0_gpu, b1_gpu = extract_betti_at_full(res_gpu)
            line += f" | {fmt_betti(b0_gpu, b1_gpu):>12} {fmt_time(t_gpu):>7}"
            all_match = (b0_cpu == b0_fast == b0_gpu) and (b1_cpu == b1_fast == b1_gpu)
        else:
            all_match = (b0_cpu == b0_fast) and (b1_cpu == b1_fast)

        line += f" | {'YES' if all_match else 'NO':>6}"
        log(line)

    # ── Part 2: Scaling ─────────────────────────────────────────────────
    log()
    log("-" * 100)
    log("PART 2: Speed scaling (circle, k=15, n_steps=10)")
    log("  Note: 'fast' uses dense SVD for rank(B2) — O(E^3) per step.")
    log("  At small N, eigsh is faster. The rank method's advantage is exactness, not speed.")
    log("-" * 100)

    # CPU eigsh scales well at small N; fast/GPU only at sizes where eigsh becomes unreliable
    cpu_sizes = [100, 200, 500]
    fast_sizes = [100, 200, 500]
    betti_params_scale = dict(k_neighbors=15, density_threshold=0.8, overlap_factor=1.5, n_steps=10)

    log(f"\n  CPU (eigsh):")
    for sz in cpu_sizes:
        log(f"    N={sz}...", end=" ")
        data, _, _ = make_circle(n=sz)
        t, _ = time_betti(compute_betti_curve_cpu, data, **betti_params_scale)
        log(f"\r    N={sz:>5}  {fmt_time(t):>8}")

    log(f"\n  Fast (union-find + rank):")
    for sz in fast_sizes:
        log(f"    N={sz}...", end=" ")
        data, _, _ = make_circle(n=sz)
        t, _ = time_betti(compute_betti_curve_fast, data, **betti_params_scale)
        log(f"\r    N={sz:>5}  {fmt_time(t):>8}")

    if has_gpu:
        log(f"\n  GPU (CuPy eigsh):")
        gpu_sizes = [100, 200, 500]
        for sz in gpu_sizes:
            log(f"    N={sz}...", end=" ")
            data, _, _ = make_circle(n=sz)
            t, _ = time_betti(compute_betti_curve_gpu, data, **betti_params_scale)
            log(f"\r    N={sz:>5}  {fmt_time(t):>8}")

    # ── Part 3: Topological preservation — DiRe vs cuML UMAP ──────────
    if has_gpu:
        log()
        log("-" * 100)
        log("PART 3: Topological preservation — DiRe vs cuML UMAP")
        log("  Betti curve DTW distance: ORIGINAL vs EMBEDDING (lower = better)")
        log("  Using CPU eigsh for Betti (faster at these sizes)")
        log("-" * 100)

        from dire_rapids import DiRePyTorch
        try:
            from cuml.manifold import UMAP as cuUMAP
            import cupy as cp
            has_cuml = True
        except ImportError:
            has_cuml = False
            log("  cuML UMAP not available, skipping.")

        if has_cuml:
            topo_datasets = [
                make_circle(n=300),
                make_torus(n=500),
                make_linked_rings(n=300),
                make_blobs(n=300, k=5, dim=10),
            ]

            qparams = dict(k_neighbors=15, density_threshold=0.8,
                           overlap_factor=1.5, n_steps=15)

            log(f"\n  {'Dataset':<22} | {'Expected':>10} | {'Orig β₀,β₁':>12} | {'DiRe DTW β₀':>12} {'DiRe DTW β₁':>12} | {'cuML DTW β₀':>12} {'cuML DTW β₁':>12}")
            log("  " + "-" * 105)

            for data, name, expected in topo_datasets:
                log(f"  {name}...", end=" ")
                exp_str = f"β₀={expected['β₀']},β₁={expected['β₁']}"

                # Betti curve on ORIGINAL data
                _, res_orig = time_betti(compute_betti_curve_cpu, data, **qparams)
                if isinstance(res_orig, str):
                    log(f"\r  {name:<22} | original Betti curve FAILED: {res_orig}")
                    continue
                b0_orig, b1_orig = extract_betti_at_full(res_orig)
                orig_str = f"β₀={b0_orig},β₁={b1_orig}"

                # DiRe embedding
                dire_dtw0, dire_dtw1 = None, None
                try:
                    reducer = DiRePyTorch(
                        n_neighbors=16, n_components=2,
                        max_iter_layout=128, random_state=42, verbose=False,
                    )
                    emb_dire = np.asarray(reducer.fit_transform(data), dtype=np.float32)
                    _, res_dire = time_betti(compute_betti_curve_cpu, emb_dire, **qparams)
                    if not isinstance(res_dire, str):
                        dire_dtw0 = dtw_betti(res_orig['beta_0'], res_dire['beta_0'])
                        dire_dtw1 = dtw_betti(res_orig['beta_1'], res_dire['beta_1'])
                except Exception:
                    pass

                # cuML UMAP embedding
                cuml_dtw0, cuml_dtw1 = None, None
                try:
                    X_gpu = cp.asarray(data)
                    umap_r = cuUMAP(n_neighbors=16, n_components=2, random_state=42, verbose=False)
                    emb_umap = cp.asnumpy(umap_r.fit_transform(X_gpu)).astype(np.float32)
                    _, res_umap = time_betti(compute_betti_curve_cpu, emb_umap, **qparams)
                    if not isinstance(res_umap, str):
                        cuml_dtw0 = dtw_betti(res_orig['beta_0'], res_umap['beta_0'])
                        cuml_dtw1 = dtw_betti(res_orig['beta_1'], res_umap['beta_1'])
                except Exception:
                    pass

                def fv(v):
                    return f"{v:.1f}" if v is not None else "-"

                def winner(d, c):
                    if d is None or c is None:
                        return "", ""
                    if d < c:
                        return " *", ""
                    elif c < d:
                        return "", " *"
                    return "", ""

                w_d0, w_c0 = winner(dire_dtw0, cuml_dtw0)
                w_d1, w_c1 = winner(dire_dtw1, cuml_dtw1)

                log(f"\r  {name:<22} | {exp_str:>10} | {orig_str:>12} | {fv(dire_dtw0):>10}{w_d0:2} {fv(dire_dtw1):>10}{w_d1:2} | {fv(cuml_dtw0):>10}{w_c0:2} {fv(cuml_dtw1):>10}{w_c1:2}")

            log()
            log("  DTW = Dynamic Time Warping distance between Betti curves")
            log("  Lower DTW = embedding better preserves the topology of the original data")
            log("  * = winner for that metric")

    log()
    log("=" * 100)
    log("Done.")
