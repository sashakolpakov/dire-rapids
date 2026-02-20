"""Benchmark optimizations: current code vs baseline (pre-optimization) behavior."""

import time
import numpy as np
import torch

# ── Helpers ──────────────────────────────────────────────────────────────────

def time_fn(fn, *args, warmup=1, repeats=3, **kwargs):
    """Time a function with warmup and multiple repeats."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return result, times


def fmt(times):
    """Format timing results."""
    mean = np.mean(times)
    std = np.std(times)
    return f"{mean:.4f}s +/- {std:.4f}s"


# ── Data Generation ─────────────────────────────────────────────────────────

def make_data(n_samples, n_dims, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_dims)).astype(np.float32)


# ── Benchmark 1: Full fit_transform ─────────────────────────────────────────

def bench_fit_transform(X, n_neighbors=16, max_iter=64):
    from dire_rapids import DiRePyTorch
    reducer = DiRePyTorch(
        n_neighbors=n_neighbors,
        n_components=2,
        max_iter_layout=max_iter,
        random_state=42,
        verbose=False,
    )
    return reducer.fit_transform(X)


def bench_fit_transform_mem_efficient(X, n_neighbors=16, max_iter=64):
    from dire_rapids import DiRePyTorchMemoryEfficient
    reducer = DiRePyTorchMemoryEfficient(
        n_neighbors=n_neighbors,
        n_components=2,
        max_iter_layout=max_iter,
        random_state=42,
        verbose=False,
    )
    return reducer.fit_transform(X)


# ── Benchmark 2: Force computation kernel (isolated) ────────────────────────

def bench_force_kernels(n_samples=5000, n_neighbors=16, n_dims=2, device='cpu'):
    """Compare compiled vs eager force kernels directly."""
    from dire_rapids.dire_pytorch import (
        _attraction_forces_kernel,
        _attraction_forces_compiled,
        _repulsion_forces_kernel,
        _repulsion_forces_compiled,
    )

    torch.manual_seed(42)
    dev = torch.device(device)
    positions = torch.randn(n_samples, n_dims, device=dev)
    knn_indices = torch.randint(0, n_samples, (n_samples, n_neighbors), device=dev, dtype=torch.long)
    neg_positions = torch.randn(n_samples, n_neighbors * 5, n_dims, device=dev)

    a_val, b_exp, cutoff = 1.0, 1.0, 4.0

    warmup = 3 if device == 'cuda' else 2
    repeats = 10 if device == 'cuda' else 5

    # Attraction - eager
    _, t_att_eager = time_fn(
        _attraction_forces_kernel, positions, positions, knn_indices, a_val, b_exp,
        warmup=warmup, repeats=repeats
    )

    # Attraction - compiled
    _, t_att_compiled = time_fn(
        _attraction_forces_compiled, positions, positions, knn_indices, a_val, b_exp,
        warmup=warmup, repeats=repeats
    )

    # Repulsion - eager
    _, t_rep_eager = time_fn(
        _repulsion_forces_kernel, positions, neg_positions, a_val, b_exp, cutoff,
        warmup=warmup, repeats=repeats
    )

    # Repulsion - compiled
    _, t_rep_compiled = time_fn(
        _repulsion_forces_compiled, positions, neg_positions, a_val, b_exp, cutoff,
        warmup=warmup, repeats=repeats
    )

    return t_att_eager, t_att_compiled, t_rep_eager, t_rep_compiled


# ── Benchmark 3: Neighbor score (vectorized vs loop) ────────────────────────

def bench_neighbor_score_loop(indices_data, indices_layout):
    """Old implementation with Python loop."""
    n_samples = indices_data.shape[0]
    k = indices_data.shape[1]
    scores = np.empty(n_samples, dtype=np.float32)
    for i in range(n_samples):
        scores[i] = np.isin(indices_data[i], indices_layout[i]).sum() / k
    return float(np.mean(scores)), float(np.std(scores))


def bench_neighbor_score_vectorized(indices_data, indices_layout):
    """New implementation with broadcasting."""
    k = indices_data.shape[1]
    matches = (indices_data[:, :, None] == indices_layout[:, None, :]).any(axis=2)
    scores = matches.sum(axis=1).astype(np.float32) / k
    return float(np.mean(scores)), float(np.std(scores))


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("DiRe-Rapids Performance Benchmark")
    print("=" * 70)

    has_gpu = torch.cuda.is_available()
    if has_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("GPU: Not available (CPU only)")
    print(f"PyTorch: {torch.__version__}")
    print()

    # ── 1. Neighbor Score ────────────────────────────────────────────────
    print("-" * 70)
    print("Benchmark 1: compute_neighbor_score (loop vs vectorized)")
    print("-" * 70)

    for n in [1000, 5000, 10000]:
        k = 16
        rng = np.random.default_rng(42)
        idx_a = rng.integers(0, n, size=(n, k))
        idx_b = rng.integers(0, n, size=(n, k))

        _, t_loop = time_fn(bench_neighbor_score_loop, idx_a, idx_b, warmup=1, repeats=3)
        _, t_vec = time_fn(bench_neighbor_score_vectorized, idx_a, idx_b, warmup=1, repeats=3)

        speedup = np.mean(t_loop) / np.mean(t_vec)
        print(f"  N={n:>6}, k={k}: loop={fmt(t_loop)}  vec={fmt(t_vec)}  speedup={speedup:.1f}x")

    print()

    # ── 2. Force Kernels (isolated, after warmup) ───────────────────────
    print("-" * 70)
    print("Benchmark 2: Force kernels (eager vs torch.compile, post-warmup)")
    print("-" * 70)

    devices = ['cuda'] if has_gpu else ['cpu']
    for device_str in devices:
        for n in [2000, 5000, 10000]:
            t_ae, t_ac, t_re, t_rc = bench_force_kernels(
                n_samples=n, n_neighbors=16, n_dims=2, device=device_str
            )
            att_speedup = np.mean(t_ae) / np.mean(t_ac)
            rep_speedup = np.mean(t_re) / np.mean(t_rc)
            print(f"  [{device_str:>4}] N={n:>5}: attraction eager={fmt(t_ae)} compiled={fmt(t_ac)} ({att_speedup:.2f}x)")
            print(f"  [{device_str:>4}] N={n:>5}: repulsion  eager={fmt(t_re)} compiled={fmt(t_rc)} ({rep_speedup:.2f}x)")
        print()

    # ── 3. Full fit_transform ────────────────────────────────────────────
    print("-" * 70)
    print("Benchmark 3: Full fit_transform (end-to-end)")
    print("-" * 70)

    # Warmup run to pay PyKeOps JIT + torch.compile cost once
    print("  Warming up (compiling PyKeOps kernels, torch graphs)...")
    X_warmup = make_data(500, 50)
    bench_fit_transform(X_warmup, 16, 8)
    bench_fit_transform_mem_efficient(X_warmup, 16, 8)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("  Warmup done.\n")

    configs = [
        (2000, 50, 16, 64),
        (5000, 50, 16, 64),
        (10000, 50, 16, 128),
    ]
    if has_gpu:
        configs.append((10000, 100, 16, 128))
        configs.append((50000, 100, 16, 128))

    for n, d, k, iters in configs:
        X = make_data(n, d)

        # Standard backend
        _, t_std = time_fn(
            bench_fit_transform, X, k, iters,
            warmup=0, repeats=1
        )
        # Memory-efficient backend
        _, t_mem = time_fn(
            bench_fit_transform_mem_efficient, X, k, iters,
            warmup=0, repeats=1
        )
        dev = "cuda" if has_gpu else "cpu"
        print(f"  [{dev}] N={n}, D={d}, k={k}, iters={iters}")
        print(f"    Standard:         {fmt(t_std)}")
        print(f"    Memory-efficient: {fmt(t_mem)}")

    print()
    print("=" * 70)
    print("Done.")
