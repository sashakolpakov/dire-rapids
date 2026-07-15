# Lessons Learned

## Environment
- Must use `python -m pip` (not bare `pip`) in the rapids conda env.
- Run commands via the pinned environment: `conda run --no-capture-output -n rapids-26.06 ...`.
- With RAPIDS 26.06/CUDA 13 and pip PyTorch, preload the conda copy of nvJitLink: `LD_PRELOAD=$CONDA_PREFIX/lib/libnvJitLink.so.13`. `LD_LIBRARY_PATH` alone does not make imports deterministic once PyTorch's copy is loaded.
- Smoke-test `torch` then `cuml` and `cuml` then `torch` in separate Python processes; one process cannot detect order-dependent dynamic-loader state reliably.
- torch.compile cache invalidation causes slow first run after code changes.

## Numerical
- B2 rank computation requires float64 — float32 perturbs zero singular values above the default threshold.
- GPU PCA must use `torch.pca_lowrank`, not `torch.linalg.svd` (the latter fails at D > ~50K).
- Betti β₁ is sensitive to `k_neighbors`; need k ≥ 20 for good manifold triangulation.
- **fp16 squared-distance overflow** silently corrupts kNN. `torch.cdist` computes sum-of-squares before sqrt, so any dataset where `n_dims * x_abs_max^2 > 65504` (e.g. raw [0,255] pixels in 784-D: $784 \cdot 255^2 \approx 5 \times 10^7$) returns all-`inf` distances. Output kNN has 0% overlap with fp32 reference; the embedding collapses to a uniform blob at origin. Always guard fp16 with a scale check *or* normalize inputs before the auto-fp16 switch. (Fixed 2026-04-20 in `dire_pytorch.py`: `normalize=True` default + fp16 safety guard in `_compute_knn`.)
- Do not debug an algorithm on unnormalized data and conclude the algorithm is broken. Always verify the kNN graph is healthy first (overlap with fp32 reference, no `inf`/`nan` distances).

## API Gotchas
- OpenML features API is broken for datasets with D > 50K; must download ARFF files directly.
- cuML spectral init changed behavior silently in RAPIDS 26.02 (was ~0 norm diff, now 1.26).

## Architecture
- Chunked force computation had 10x overhead vs unchunked — always prefer unchunked unless OOM.
- cuVS IVF-Flat recall is dataset- and tuning-dependent. On 100K×64 Gaussian data with RAPIDS 26.06 and the current automatic parameters, IVF-Flat measured 51.1% recall against exact kNN, while all-neighbors NN-descent measured 98.9%; do not assume the historical 97–99% figure without benchmarking the target data.
- kNN was the bottleneck (96-98% of total time at large N) before cuVS integration.
- RAPIDS 26.06 all-neighbors avoids staging the whole dataset in a single-GPU index. Host-backed/out-of-core execution requires more than one cluster; keep the index-and-search path for explicit legacy index types.
