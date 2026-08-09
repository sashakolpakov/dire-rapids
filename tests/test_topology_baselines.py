"""Contract tests for retained issue #14 UMAP/t-SNE baseline evidence."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = ROOT / "benchmarking" / "extract_topology_baselines.py"
SPECIFICATION = importlib.util.spec_from_file_location(
    "extract_topology_baselines", EXTRACTOR
)
assert SPECIFICATION is not None and SPECIFICATION.loader is not None
extractor = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(extractor)


@pytest.mark.cpu
def test_retained_baselines_are_complete_and_best_thresholds_are_derived():
    fixture = json.loads(
        (ROOT / "tests" / "data" / "topology_umap_tsne_atlas_baselines.json")
        .read_text(encoding="utf-8")
    )

    assert fixture["schema_version"] == extractor.SCHEMA_VERSION
    assert fixture["source"]["revision"] == extractor.SOURCE_REVISION
    assert set(fixture["datasets"]) == set(extractor.DATASETS)
    assert set(fixture["ripser_canonical"]["datasets"]) == set(
        extractor.DATASETS
    )
    for dataset in extractor.DATASETS:
        record = fixture["datasets"][dataset]
        assert set(record["methods"]) == set(extractor.METHODS)
        for method in extractor.METHODS:
            method_record = record["methods"][method]
            assert len(method_record["seeds"]) in (10, 20)
            for metric in extractor.TOPOLOGY_METRICS:
                summary = method_record["atlas"][metric]
                assert summary["n"] == len(summary["values"])
        for metric in extractor.TOPOLOGY_METRICS:
            means = {
                method: record["methods"][method]["atlas"][metric]["mean"]
                for method in extractor.COMPARATOR_METHODS
            }
            expected = min(means, key=means.get)
            assert record["thresholds"][metric] == {
                "best_method": expected,
                "best_mean": means[expected],
            }

            ripser_record = fixture["ripser_canonical"]["datasets"][dataset]
            ripser_values = {
                method: ripser_record["methods"][method][metric]
                for method in extractor.COMPARATOR_METHODS
            }
            ripser_best = min(ripser_values, key=ripser_values.get)
            assert ripser_record["thresholds"][metric] == {
                "best_method": ripser_best,
                "best_value": ripser_values[ripser_best],
            }


@pytest.mark.cpu
def test_extractor_rejects_changed_archived_payload(tmp_path):
    for name in extractor.SOURCE_FILE_SHA256:
        (tmp_path / name).write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source hash mismatch"):
        extractor.extract_baselines(tmp_path)
