# DiRe-Rapids TODO

## Bugs

- [x] **metrics.py CPU fallback broken when cuML available** — FIXED
  sklearn imports were conditional on `HAS_CUML=False`; moved to unconditional.

## Performance

- [x] **Force OOM at ~940K points** — RESOLVED (no longer reproduces)
  Was a pre-cuVS issue: old kNN path materialized huge distance matrices.
  With cuVS for kNN and bf16 force kernel, 1.16M points runs in 11s / 2.2GB.

## Benchmarks

- [ ] **Re-run Betti benchmark with fast implementation**
  The union-find + GPU SVD rank path (`compute_betti_curve_fast`) is complete
  but the benchmark (`benchmarking/bench_betti.py`) hasn't been re-run with
  current code to get updated timings.

## Release / Packaging

- [x] **PyPI publishing prep** — DONE
  - PyKeOps was already optional
  - Removed dead imports: pandas, plotly from dire_pytorch.py; pandas from utils.py
  - Removed 5 unnecessary core deps (pandas, plotly, websockets, psutil, fastdtw)
  - Moved fastdtw+psutil to `[metrics]` extra, pandas+plotly to `[viz]` extra
  - Python version bumped to >=3.10 (torch.compile requires it), removed 3.9 classifier
  - Core deps now: numpy, torch, loguru, tqdm, scipy, scikit-learn

## Paper

- [ ] **Paper revision**
  Referee comments pending. New results to incorporate:
  - DiRe beats cuML UMAP on covertype (3.1s vs 32s) and jannis (5.5s vs 9s)
  - cuML spectral init silently changed behavior in RAPIDS 26.02 (relative norm diff 1.26)
  - High-D benchmarks (Flora 3.4x speedup)
  - Intrinsic dimension detection results

## Performance (future)

- [ ] **Speed up Betti curve computation — the bottleneck is boundary-matrix
  rank reduction, NOT triangle enumeration**
  Diagnosed 2026-04-22 while the earlier "cuGraph triangle_count" idea was
  on the todo list. Three implementations exist in `betti_curve.py`:
    * `compute_betti_curve_cpu` — eigsh on L₀, L₁. Hangs beyond ~N=2K because
      `which='SM'` forces ARPACK shift-invert (sparse LU on L), same fill-in
      catastrophe as the pre-LOBPCG spectral init.
    * `compute_betti_curve_fast` / `_gpu` — union-find for β₀ (genuinely fast,
      O(E α)) + dense SVD for rank(B₂). **At n=300 the dense SVD is already
      13× slower than eigsh**, and it times out at n=1000. The "fast" name
      is misleading.
  Triangle counting is not the problem: `A ⊙ A²` via cupyx gives triangle
  counts in sub-second at 500K nodes (verified during the Forman experiment).
  Empirical test showed `rank(B₂) = T_active` (the β₂=0 shortcut that
  would skip SVD) **fails badly** — β₂ of the 2-complex is on the order of
  tens of thousands on n=300 data (many closed 2-surfaces in dense kNN
  complexes). So we need the actual rank.
  Real options:
    (1) Depend on `ripser.py` (C++ persistent homology). Drop the `eigsh` /
        dense-SVD paths; it's much faster at all scales and gives full
        persistence diagrams from which Betti curves fall out.
    (2) Hand-roll incremental column reduction over F₂: per filtration step,
        each new edge/triangle changes rank by 0 or 1. Maintain a reduced
        basis bitset; O(E) per step, O(E × n_steps) total.
    (3) β₀-only mode: union-find on GPU via cupyx.csgraph.connected_components.
        Trivially fast (ms at MNIST scale). Useful right now as an Optuna /
        RL-harness objective that captures cluster-preservation signal
        without the β₁ cost.
  Prefer (3) for optimization objectives and (1) for ablation figures.

## Research Ideas

- [ ] **Deepfake detection via intrinsic dimension**
  Hypothesis: AI-generated video has lower intrinsic dimension than real video
  (generator's latent bottleneck). Use DiRe dimension sweep on matched real vs
  fake frame sequences. Geometry-based detector, not pixel artifacts.
