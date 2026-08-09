# Frozen regression evidence

`topology_preset_atlas_audit.csv` is the 12-row `dire_topology` projection of
the paired six-dataset Atlas audit archived at
`sashakolpakov/homological-stability-repro@cd7876ac9e72caa786923c0dfa32c490c3c9a958`.

Source file:
`generated/revision3/revision3-small-atlas-topology-paired-effects.csv`

Source SHA-256:
`f6c05d21035f8b21c4f606cea28ca2b9d8f666d62ef764831c4256d377dcb2b6`

The projection retains every paired count, mean, gap, interval, relative gap,
and interpretation needed to reproduce the preset comparison in issue #14.
Positive gaps mean the former preset has worse discrepancy than default DiRe.

`topology_umap_tsne_atlas_baselines.json` retains the full per-seed Atlas
topology distributions for DiRe, CPU/GPU UMAP, and CPU/GPU t-SNE from the
archived JSON logs in
`sashakolpakov/homological-stability-repro@8bd06c53c0b74f37b17991df335ca056136411dc`.
The source manifest and all six logs are SHA-256 pinned by
`benchmarking/extract_topology_baselines.py`. Candidate preset validation uses
the lowest UMAP/t-SNE mean in each dataset/homology cell as its no-rerun
threshold; the retained values remain available for interval calculations.
Its `ripser_canonical` section is a no-fit seed-42 screen: it evaluates the
archived canonical embeddings against the same frozen original-data subsets
with the hash-pinned `293b622` evaluator. It is intentionally not treated as a
repeat distribution; a competitive DiRe candidate triggers only the strongest
missing Ripser comparator rerun.

`topology_preset_tuning_manifest.json` records the OpenML IDs, shapes, label
counts, preprocessing versions, and file/array hashes for the four disjoint
tuning datasets used by the crossed search. The 22 MB arrays are retained with
the run artifacts rather than committed to Git.
