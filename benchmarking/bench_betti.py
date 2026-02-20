"""
Benchmark Betti curve computation: CPU (scipy) vs GPU (CuPy/cuVS).

Part 1: Speed — CPU vs GPU on synthetic datasets
Part 2: Quality — Betti curve DTW distance for DiRe vs cuML UMAP embeddings
        (how well does each method preserve topology?)
"""

import gc
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
    # Filtration goes from 100th percentile (index 0, all edges) to 0th (last, fewest edges)
    return int(result['beta_0'][0]), int(result['beta_1'][0])


def dtw_betti(curve_orig, curve_emb):
    """DTW distance between two Betti curves (1D sequences)."""
    if curve_orig is None or curve_emb is None:
        return None
    c1 = np.array(curve_orig, dtype=np.float64).reshape(-1, 1)
    c2 = np.array(curve_emb, dtype=np.float64).reshape(-1, 1)
    dist, _ = fastdtw(c1, c2, radius=5)
    return float(dist)


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 90)
    print("Betti Curve Benchmark: Speed + Topological Quality")
    print("=" * 90)

    has_gpu = False
    try:
        import cupy as cp
        has_gpu = True
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        print(f"GPU: {gpu_name}")
    except ImportError:
        print("GPU: Not available (CuPy not installed)")
    print()

    from dire_rapids.betti_curve import compute_betti_curve_cpu
    if has_gpu:
        from dire_rapids.betti_curve import compute_betti_curve_gpu

    # ── Part 1: Correctness + Speed (CPU vs GPU) ───────────────────────
    print("-" * 90)
    print("PART 1: Betti numbers on synthetic datasets (full complex)")
    print("-" * 90)

    datasets = [
        make_circle(n=500),
        make_sphere(n=800),
        make_torus(n=1000),
        make_linked_rings(n=800),
        make_blobs(n=500, k=5, dim=10),
        make_blobs(n=500, k=3, dim=5),
    ]

    betti_params = dict(k_neighbors=20, density_threshold=0.8, overlap_factor=1.5, n_steps=30)

    print(f"\n  {'Dataset':<22} {'N':>5} {'D':>3} | {'Expected':>10} | {'CPU β₀,β₁':>10} {'CPU time':>9}", end="")
    if has_gpu:
        print(f" | {'GPU β₀,β₁':>10} {'GPU time':>9} {'Speedup':>8}", end="")
    print()
    print("  " + "-" * (75 + (30 if has_gpu else 0)))

    for data, name, expected in datasets:
        n, d = data.shape
        exp_str = f"β₀={expected['β₀']},β₁={expected['β₁']}"

        # CPU
        t_cpu, res_cpu = time_betti(compute_betti_curve_cpu, data, **betti_params)
        b0_cpu, b1_cpu = extract_betti_at_full(res_cpu)
        cpu_str = f"β₀={b0_cpu},β₁={b1_cpu}" if b0_cpu is not None else "FAILED"
        cpu_t_str = f"{t_cpu:.2f}s" if t_cpu is not None else "-"

        line = f"  {name:<22} {n:>5} {d:>3} | {exp_str:>10} | {cpu_str:>10} {cpu_t_str:>9}"

        if has_gpu:
            t_gpu, res_gpu = time_betti(compute_betti_curve_gpu, data, **betti_params)
            b0_gpu, b1_gpu = extract_betti_at_full(res_gpu)
            gpu_str = f"β₀={b0_gpu},β₁={b1_gpu}" if b0_gpu is not None else "FAILED"
            gpu_t_str = f"{t_gpu:.2f}s" if t_gpu is not None else "-"
            if t_cpu and t_gpu:
                speedup = t_cpu / t_gpu
                line += f" | {gpu_str:>10} {gpu_t_str:>9} {speedup:>7.1f}x"
            else:
                line += f" | {gpu_str:>10} {gpu_t_str:>9} {'  -':>8}"

        print(line)

    # ── Part 2: Scaling ─────────────────────────────────────────────────
    print()
    print("-" * 90)
    print("PART 2: Scaling benchmark (circle, varying N)")
    print("-" * 90)

    sizes = [500, 1000, 2000, 3000, 5000]
    betti_params_scale = dict(k_neighbors=20, density_threshold=0.8, overlap_factor=1.5, n_steps=20)

    print(f"\n  {'N':>6} | {'CPU time':>10}", end="")
    if has_gpu:
        print(f" | {'GPU time':>10} | {'Speedup':>8}", end="")
    print()
    print("  " + "-" * (20 + (25 if has_gpu else 0)))

    for sz in sizes:
        data, _, _ = make_circle(n=sz)
        t_cpu, _ = time_betti(compute_betti_curve_cpu, data, **betti_params_scale)
        cpu_t_str = f"{t_cpu:.2f}s" if t_cpu else "-"
        line = f"  {sz:>6} | {cpu_t_str:>10}"

        if has_gpu:
            t_gpu, _ = time_betti(compute_betti_curve_gpu, data, **betti_params_scale)
            gpu_t_str = f"{t_gpu:.2f}s" if t_gpu else "-"
            if t_cpu and t_gpu:
                speedup = t_cpu / t_gpu
                line += f" | {gpu_t_str:>10} | {speedup:>7.1f}x"
            else:
                line += f" | {gpu_t_str:>10} | {'  -':>8}"

        print(line)

    # ── Part 3: Topological preservation — DiRe vs cuML UMAP ──────────
    if has_gpu:
        print()
        print("-" * 90)
        print("PART 3: Topological preservation — DiRe vs cuML UMAP")
        print("  Compare Betti curves of ORIGINAL data vs EMBEDDING (DTW distance, lower = better)")
        print("-" * 90)

        from dire_rapids import DiRePyTorch
        try:
            from cuml.manifold import UMAP as cuUMAP
            import cupy as cp
            has_cuml = True
        except ImportError:
            has_cuml = False
            print("  cuML UMAP not available, skipping.")

        if has_cuml:
            topo_datasets = [
                make_circle(n=800),
                make_torus(n=1500),
                make_linked_rings(n=1000),
                make_blobs(n=600, k=5, dim=10),
            ]

            # Betti curve params for quality comparison
            qparams = dict(k_neighbors=20, density_threshold=0.8,
                           overlap_factor=1.5, n_steps=30)

            print(f"\n  {'Dataset':<22} | {'Expected':>10} | {'Orig β₀,β₁':>12} | {'DiRe DTW β₀':>12} {'DiRe DTW β₁':>12} | {'cuML DTW β₀':>12} {'cuML DTW β₁':>12}")
            print("  " + "-" * 105)

            for data, name, expected in topo_datasets:
                exp_str = f"β₀={expected['β₀']},β₁={expected['β₁']}"

                # Betti curve on ORIGINAL data
                _, res_orig = time_betti(compute_betti_curve_cpu, data, **qparams)
                if isinstance(res_orig, str):
                    print(f"  {name:<22} | original Betti curve FAILED: {res_orig}")
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
                except Exception as e:
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
                except Exception as e:
                    pass

                def fv(v):
                    return f"{v:.1f}" if v is not None else "-"

                dire_d0 = fv(dire_dtw0)
                dire_d1 = fv(dire_dtw1)
                cuml_d0 = fv(cuml_dtw0)
                cuml_d1 = fv(cuml_dtw1)

                # Highlight winner
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

                print(f"  {name:<22} | {exp_str:>10} | {orig_str:>12} | {dire_d0:>10}{w_d0:2} {dire_d1:>10}{w_d1:2} | {cuml_d0:>10}{w_c0:2} {cuml_d1:>10}{w_c1:2}")

            print()
            print("  DTW = Dynamic Time Warping distance between Betti curves")
            print("  Lower DTW = embedding better preserves the topology of the original data")
            print("  * = winner for that metric")

    print()
    print("=" * 90)
    print("Done.")
