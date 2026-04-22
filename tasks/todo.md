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

- [ ] **Speed up Betti curve computation via cuGraph triangle finding**
  `compute_betti_curve_cpu` uses scipy eigsh on the sparse Laplacian of the
  kNN complex and becomes intractable beyond ~N=2K (e.g. MNIST n=3000 ran for
  6+ hours with no progress). cuGraph exposes a GPU `triangle_count` primitive
  (part of the RAPIDS stack we already depend on) — per-vertex and per-edge
  triangle counts are a common building block for the B2 rank computation and
  the connected-components β₀ estimate. Replacing the CPU path with a
  cuGraph-backed triangle enumerator would let us use Betti DTW as an objective
  during hyperparameter tuning and in production ablations on real datasets.
  Currently deferred because the CPU Betti is the blocker, not the DR pipeline.

## Research Ideas

- [ ] **Deepfake detection via intrinsic dimension**
  Hypothesis: AI-generated video has lower intrinsic dimension than real video
  (generator's latent bottleneck). Use DiRe dimension sweep on matched real vs
  fake frame sequences. Geometry-based detector, not pixel artifacts.
