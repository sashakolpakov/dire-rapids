# PNAS/arXiv submission snapshot

`main.tex`, `main.pdf`, and `dire-pnas-arxiv.zip` capture the manuscript as
compiled on April 26, 2026. The archive is internally consistent with the
checked-in source and records experiments run with `dire-rapids` 0.3.0,
RAPIDS 26.04, and the historical `TOPOLOGY_TUNED` preset.

That preset is not part of the current public API. A subsequent paired,
held-out Atlas audit found worse topology discrepancy in 10 of 12
dataset-by-homology comparisons, so the development line withdrew the preset
instead of presenting it as a generally topology-improving default. The
[frozen audit projection and provenance](../../tests/data/) are checked in with
the regression tests.

[The PR 12 H100 measurements](https://github.com/sashakolpakov/dire-rapids/pull/12)
answer a separate question: they compare the explicit cuVS all-neighbors graph
builder with the released index-and-search policy. They use different
hardware, software, and timing boundaries from the GH200 results in this
manuscript and must not be substituted for the paper's numbers. Those
measurements support keeping all-neighbors explicit opt-in; they do not
revalidate the withdrawn topology preset.

Treat the PDF and ZIP as submission artifacts rather than current API
documentation. Use the repository [README](../../README.md) and
[changelog](../../CHANGELOG.md) for supported defaults and current benchmark
caveats.
