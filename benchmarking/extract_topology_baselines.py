#!/usr/bin/env python3
"""Extract the already-measured UMAP/t-SNE Atlas baselines for issue #14.

The source logs are the self-contained archived JSON payload in
``sashakolpakov/homological-stability-repro``.  This extractor deliberately
retains the per-seed topology values so preset validation can compare against
the existing distributions without fitting UMAP or t-SNE again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


SCHEMA_VERSION = 1
SOURCE_REPOSITORY = "https://github.com/sashakolpakov/homological-stability-repro"
SOURCE_REVISION = "8bd06c53c0b74f37b17991df335ca056136411dc"
SOURCE_FILE_SHA256 = {
    "manifest.json": "d016db20025f26d96dd1f4a2256b3accf7705e3f52d1517ca6d11da912281f3a",
    "blobs.json": "21ce6bdf19ef0fdba914dbceaf23c60ab374cef582ac7c0d3b1fd11346f8cfb0",
    "disk.json": "2ff7297536bd0c45c900329aa7e3c09e96209c778ba307e7a7f0232bf8d6a0b5",
    "moons.json": "9fcf15d0ba9cc92c843646a3bcea04cb249de270f8df3d108e053207aeba487b",
    "mnist.json": "b2e86c905105a53c0ddd594476ebf5659fe44203b926ac4a4ab6c3f8d1d5cb36",
    "levine13.json": "921264e065c41d2062a734dbe0d4ff12d0112ab6a7b9dd591b9165cc939b4a6d",
    "levine32.json": "d1fe42ad485a60760cbdb7661ec580ec03bd65d87fc6d02f78d26344e10b6296",
}
DATASETS = ("blobs", "disk", "moons", "mnist", "levine13", "levine32")
METHODS = ("dire", "cuml_tsne", "cuml_umap", "opentsne", "umap")
COMPARATOR_METHODS = ("cuml_tsne", "cuml_umap", "opentsne", "umap")
TOPOLOGY_METRICS = ("dtw_beta0", "dtw_beta1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def numeric_summary(values) -> dict | None:
    retained = [float(value) for value in values if value is not None]
    if not retained:
        return None
    return {
        "n": len(retained),
        "mean": float(statistics.fmean(retained)),
        "sample_std": (
            float(statistics.stdev(retained)) if len(retained) > 1 else 0.0
        ),
        "min": min(retained),
        "max": max(retained),
        "values": retained,
    }


def compute_ripser_isolated(evaluator_source: Path, data: np.ndarray) -> dict:
    """Evaluate one point cloud in a process that releases Ripser native memory."""
    with TemporaryDirectory(prefix="dire-ripser-") as temporary_root:
        temporary = Path(temporary_root)
        input_path = temporary / "input.npy"
        output_path = temporary / "curve.json"
        with input_path.open("wb") as stream:
            np.save(stream, np.asarray(data, dtype=np.float32), allow_pickle=False)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "_ripser-worker",
                str(evaluator_source),
                str(input_path),
                str(output_path),
            ],
            check=True,
        )
        return json.loads(output_path.read_text(encoding="utf-8"))


def ripser_worker(evaluator_source: Path, input_path: Path, output_path: Path) -> None:
    """Internal worker entry point; parent validates all source identities."""
    import bench_topology_historical as historical

    evaluator = historical.load_fixed_evaluator(evaluator_source)
    data = np.load(input_path, allow_pickle=False)
    curve = evaluator.compute_betti_curve_ripser(data, n_steps=100, maxdim=1)
    historical.write_json(output_path, curve)


def extract_baselines(root: Path) -> dict:
    """Validate the pinned archive and return its reusable topology evidence."""
    for name, expected in SOURCE_FILE_SHA256.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned benchmark log: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"source hash mismatch for {name}: {actual} != {expected}")

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise RuntimeError("expected archived JSON log schema version 2")
    if tuple(manifest.get("datasets", ())) != DATASETS:
        raise RuntimeError("archived dataset order does not match the pinned suite")
    if tuple(manifest.get("methods", ())) != METHODS:
        raise RuntimeError("archived method order does not match the pinned suite")

    output = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "relative_root": "data/archived/json_logs",
            "file_sha256": SOURCE_FILE_SHA256,
            "source_bundle": manifest["source_bundle"],
            "source_bundle_archive_sha256": manifest[
                "source_bundle_archive_sha256"
            ],
        },
        "contract": {
            "comparator_methods": list(COMPARATOR_METHODS),
            "topology_backend": "atlas",
            "topology_backend_detail": "direct GPU rank-based local-kNN atlas",
            "topology_prefer_ripser": False,
            "topology_parameters": {
                "k_neighbors": 20,
                "density_threshold": 0.8,
                "overlap_factor": 1.5,
                "n_steps": 100,
            },
            "threshold_rule": (
                "lowest archived comparator mean for each dataset and homology dimension"
            ),
        },
        "datasets": {},
    }

    for dataset in DATASETS:
        payload = json.loads((root / f"{dataset}.json").read_text(encoding="utf-8"))
        if payload.get("schema_version") != 2 or payload.get("dataset_name") != dataset:
            raise RuntimeError(f"invalid dataset identity in {dataset}.json")
        if tuple(payload.get("methods", {}).keys()) != METHODS:
            raise RuntimeError(f"method order mismatch in {dataset}.json")

        dataset_record = {
            "input": payload["dataset"],
            "methods": {},
            "thresholds": {},
        }
        for method in METHODS:
            method_payload = payload["methods"][method]
            repeats = method_payload.get("repeats", [])
            if not repeats:
                raise RuntimeError(f"{dataset}/{method} has no retained repeats")
            seeds = [int(repeat["seed"]) for repeat in repeats]
            if len(seeds) != len(set(seeds)):
                raise RuntimeError(f"{dataset}/{method} contains duplicate seeds")

            topology = {}
            for metric in TOPOLOGY_METRICS:
                values = []
                for repeat in repeats:
                    topology_record = repeat["metrics"]["topology"]
                    if topology_record.get("backend") != "atlas":
                        raise RuntimeError(f"{dataset}/{method} did not use Atlas")
                    if topology_record.get("prefer_ripser") is not False:
                        raise RuntimeError(f"{dataset}/{method} allowed Ripser fallback")
                    parameters = topology_record.get("parameters", {})
                    expected_parameters = output["contract"]["topology_parameters"]
                    if any(
                        parameters.get(name) != expected
                        for name, expected in expected_parameters.items()
                    ):
                        raise RuntimeError(
                            f"{dataset}/{method} topology parameters changed"
                        )
                    values.append(topology_record["metrics"][metric])
                topology[metric] = numeric_summary(values)

            method_record = {
                "seeds": seeds,
                "atlas": topology,
                "knn_accuracy": numeric_summary(
                    repeat["metrics"]
                    .get("context", {})
                    .get("knn", (None, None))[1]
                    for repeat in repeats
                ),
                "canonical_seed": int(method_payload["seed"]),
                "canonical_embedding_float32_sha256": array_sha256(
                    method_payload["embedding"]
                ),
            }
            if method_record["canonical_seed"] != seeds[0]:
                raise RuntimeError(f"{dataset}/{method} canonical seed is not repeat zero")
            dataset_record["methods"][method] = method_record

        for metric in TOPOLOGY_METRICS:
            best_method = min(
                COMPARATOR_METHODS,
                key=lambda name: dataset_record["methods"][name]["atlas"][metric][
                    "mean"
                ],
            )
            dataset_record["thresholds"][metric] = {
                "best_method": best_method,
                "best_mean": dataset_record["methods"][best_method]["atlas"][metric][
                    "mean"
                ],
            }
        output["datasets"][dataset] = dataset_record

    return output


def add_ripser_canonical(
    output: dict,
    logs_root: Path,
    frozen_datasets: Path,
    evaluator_source: Path,
    reference_cache: Path,
) -> None:
    """Add a no-fit Ripser screen from each retained seed-42 embedding."""
    import bench_topology_historical as historical

    dataset_manifest, datasets = historical.load_dataset_manifest(frozen_datasets)
    evaluator = historical.load_fixed_evaluator(evaluator_source)
    ripser_output = {
        "scope": (
            "seed-42 screening only; rerun the winning baseline across seeds "
            "before making a repeated-measures Ripser claim"
        ),
        "evaluator_sha256": historical.EVALUATOR_SHA256,
        "frozen_dataset_manifest_sha256": historical.sha256_file(
            frozen_datasets / "manifest.json"
        ),
        "datasets": {},
    }

    for dataset in DATASETS:
        payload = json.loads(
            (logs_root / f"{dataset}.json").read_text(encoding="utf-8")
        )
        first_repeat = payload["methods"]["dire"]["repeats"][0]
        if int(first_repeat["seed"]) != 42:
            raise RuntimeError(f"{dataset} repeat zero is not seed 42")
        indices = np.asarray(
            first_repeat["topology_sample"]["indices"], dtype=np.int64
        )
        expected_indices = historical.topology_subset_indices(len(datasets[dataset]), 42)
        if not np.array_equal(indices, expected_indices):
            raise RuntimeError(f"{dataset} archived topology subset changed")
        if historical.subset_sha256(indices) != first_repeat["topology_sample"][
            "indices_sha256"
        ]:
            raise RuntimeError(f"{dataset} archived topology subset hash changed")

        _atlas_reference, reference, reference_hashes = (
            historical.load_or_compute_reference_curves(
                evaluator,
                datasets[dataset],
                indices,
                dataset,
                dataset_manifest["datasets"][dataset]["array_sha256"],
                42,
                reference_cache,
            )
        )
        dataset_output = {
            "seed": 42,
            "subset_seed": int(first_repeat["topology_sample"]["subset_seed"]),
            "subset_indices_sha256": historical.subset_sha256(indices),
            "dataset_array_sha256": dataset_manifest["datasets"][dataset][
                "array_sha256"
            ],
            "reference_curve_sha256": reference_hashes,
            "methods": {},
            "thresholds": {},
        }
        for method in METHODS:
            method_payload = payload["methods"][method]
            if int(method_payload["seed"]) != 42:
                raise RuntimeError(f"{dataset}/{method} canonical seed changed")
            embedding = np.asarray(method_payload["embedding"], dtype=np.float32)
            if array_sha256(embedding) != output["datasets"][dataset]["methods"][
                method
            ]["canonical_embedding_float32_sha256"]:
                raise RuntimeError(f"{dataset}/{method} canonical embedding changed")
            embedded_curve = compute_ripser_isolated(
                evaluator_source, embedding[indices]
            )
            dataset_output["methods"][method] = historical.curve_distances(
                reference, embedded_curve, len(indices)
            )

        for metric in TOPOLOGY_METRICS:
            best_method = min(
                COMPARATOR_METHODS,
                key=lambda name: dataset_output["methods"][name][metric],
            )
            dataset_output["thresholds"][metric] = {
                "best_method": best_method,
                "best_value": dataset_output["methods"][best_method][metric],
            }
        ripser_output["datasets"][dataset] = dataset_output

    output["ripser_canonical"] = ripser_output


def main() -> None:
    if len(sys.argv) == 5 and sys.argv[1] == "_ripser-worker":
        ripser_worker(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-logs",
        type=Path,
        required=True,
        help="Path to homological-stability-repro/data/archived/json_logs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frozen-datasets",
        type=Path,
        help="Canonical issue #14 frozen arrays; requires --evaluator-source",
    )
    parser.add_argument(
        "--evaluator-source",
        type=Path,
        help="Pinned 293b622 dire_rapids/betti_curve.py",
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        help="Verified historical reference cache; requires the other Ripser inputs",
    )
    args = parser.parse_args()

    ripser_arguments = (
        args.frozen_datasets,
        args.evaluator_source,
        args.reference_cache,
    )
    if any(value is not None for value in ripser_arguments) and not all(
        value is not None for value in ripser_arguments
    ):
        parser.error(
            "--frozen-datasets, --evaluator-source, and --reference-cache "
            "must be used together"
        )

    result = extract_baselines(args.json_logs.resolve())
    if args.frozen_datasets is not None:
        add_ripser_canonical(
            result,
            args.json_logs.resolve(),
            args.frozen_datasets.resolve(),
            args.evaluator_source.resolve(),
            args.reference_cache.resolve(),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
