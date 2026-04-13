# Lessons Learned

## Environment
- Must use `python -m pip` (not bare `pip`) in the rapids conda env.
- Run commands via: `conda run --no-capture-output -n rapids-26.02 bash -c 'LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH ...'`
- torch.compile cache invalidation causes slow first run after code changes.

## Numerical
- B2 rank computation requires float64 — float32 perturbs zero singular values above the default threshold.
- GPU PCA must use `torch.pca_lowrank`, not `torch.linalg.svd` (the latter fails at D > ~50K).
- Betti β₁ is sensitive to `k_neighbors`; need k ≥ 20 for good manifold triangulation.

## API Gotchas
- OpenML features API is broken for datasets with D > 50K; must download ARFF files directly.
- cuML spectral init changed behavior silently in RAPIDS 26.02 (was ~0 norm diff, now 1.26).

## Architecture
- Chunked force computation had 10x overhead vs unchunked — always prefer unchunked unless OOM.
- cuVS IVF-Flat gives 97-99% recall vs exact kNN; embedding quality is identical.
- kNN was the bottleneck (96-98% of total time at large N) before cuVS integration.
