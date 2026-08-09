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
repeat distribution. A repeated Ripser comparator claim would require rerunning
the strongest missing methods in their exact archived RAPIDS 26.02/Python 3.10
environment; the issue-14 result makes no such repeated-comparator claim.

`topology_preset_tuning_manifest.json` records the OpenML IDs, shapes, label
counts, preprocessing versions, and file/array hashes for the four disjoint
tuning datasets used by the crossed search. The 22 MB arrays are retained with
the run artifacts rather than committed to Git.

`topology_historical_h100_audit.tar.gz` is the completed 960-record H100 audit
archive (SHA-256
`50e548e9b93beebb54e06e023889595a6304b2c3e5ba9746cde574bdc5dfa812`).
It contains four 240-record JSONL files and manifests, all 120 shared Atlas and
Ripser reference curves, the frozen-dataset manifest, and the three summaries;
only the large `.npy` arrays and source downloads are omitted because their
canonical hashes and reconstruction contract are already retained. The three
summary files are also unpacked under `topology_historical_h100/` for ordinary
diff review. The predeclared result found no material post-`9117dc4`
implementation change in any Atlas cell.

`topology_preset_search_h100_audit.tar.gz` is the completed 556-record crossed
search and validation archive (SHA-256
`ca101bd25db6e24a438718544b5f4a4f799ab4b1f9e5f44ba88cb27f1cc813eb`).
It contains the 272-record broad search, 152-record guarded local refinement,
12-record seed-42 held-out screen, 120-record/20-seed confirmation, all four
manifests and summaries, eight tuning reference curves, and the frozen tuning
manifest. The 22 MB tuning arrays are omitted but their file and array hashes
remain enforced. The compact conclusion is also unpacked under
`topology_preset_search_h100/`: only `spread=0.8` transferred. It independently
improved both evaluators on the held-out suite, supporting canonical
`ATLAS_TUNED` and `RIPSER_TUNED` objective names with currently coincident
parameters.
