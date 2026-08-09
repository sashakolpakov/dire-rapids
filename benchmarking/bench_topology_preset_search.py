#!/usr/bin/env python3
"""Crossed Atlas/Ripser search for issue #14 topology presets.

The search uses four OpenML datasets that are disjoint from the six retained
validation datasets. Each DiRe embedding is scored by both the fixed Atlas and
Ripser evaluators. A deterministic Sobol design keeps the run budget bounded;
no UMAP or t-SNE fits are performed by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np


HISTORICAL_HARNESS = Path(__file__).with_name("bench_topology_historical.py")
HISTORICAL_SPECIFICATION = importlib.util.spec_from_file_location(
    "_dire_issue14_historical_harness", HISTORICAL_HARNESS
)
if (
    HISTORICAL_SPECIFICATION is None
    or HISTORICAL_SPECIFICATION.loader is None
):
    raise RuntimeError(f"cannot load historical harness: {HISTORICAL_HARNESS}")
historical = importlib.util.module_from_spec(HISTORICAL_SPECIFICATION)
HISTORICAL_SPECIFICATION.loader.exec_module(historical)


SCHEMA_VERSION = 1
SEARCH_SEEDS = (42, 43)
SOBOL_SEED = 14
TOPOLOGY_METRICS = ("dtw_beta0", "dtw_beta1")
QUALITY_TOLERANCE = 0.01
STRESS_RELATIVE_TOLERANCE = 0.10
GLOBAL_PAIR_COUNT = 20_000
EXPECTED_AUTO_METHOD = "index_search"
TUNING_DATASETS = {
    "mfeat-factors": {"openml_id": 12},
    "satimage": {"openml_id": 182},
    "pendigits": {"openml_id": 32},
    "isolet": {"openml_id": 300},
}
DEFAULT_PARAMETERS = {
    "init": "pca",
    "n_neighbors": 16,
    "spread": 1.0,
    "min_dist": 1e-2,
    "cutoff": 42.0,
    "neg_ratio": 8,
    "max_iter_layout": 128,
}
WITHDRAWN_PARAMETERS = {
    "init": "spectral",
    "n_neighbors": 15,
    "spread": 3.6,
    "min_dist": 5e-4,
    "cutoff": 4.6,
    "neg_ratio": 4,
    "max_iter_layout": 150,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_sha256(value) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={root.resolve()}",
            "-C",
            str(root),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def prepare_tuning(output: Path) -> dict:
    """Download once, normalize labels, and freeze the disjoint tuning suite."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    output.mkdir(parents=True, exist_ok=True)
    records = {}
    with TemporaryDirectory(prefix="openml-cache-", dir=output) as data_home:
        for name, source in TUNING_DATASETS.items():
            dataset = fetch_openml(
                data_id=source["openml_id"],
                as_frame=False,
                parser="auto",
                data_home=data_home,
            )
            data = np.asarray(dataset.data, dtype=np.float32)
            labels = LabelEncoder().fit_transform(np.asarray(dataset.target)).astype(
                np.float32
            )
            finite = np.all(np.isfinite(data), axis=1)
            data = np.ascontiguousarray(data[finite])
            labels = np.ascontiguousarray(labels[finite])
            data_path = output / f"{name}.npy"
            labels_path = output / f"{name}.labels.npy"
            historical.save_array(data_path, data)
            historical.save_array(labels_path, labels)
            records[name] = {
                **source,
                "data_path": data_path.name,
                "labels_path": labels_path.name,
                "shape": list(data.shape),
                "classes": int(len(np.unique(labels))),
                "data_array_sha256": historical.array_sha256(data),
                "data_file_sha256": historical.sha256_file(data_path),
                "labels_array_sha256": historical.array_sha256(labels),
                "labels_file_sha256": historical.sha256_file(labels_path),
            }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "suite": "issue-14-disjoint-openml-tuning",
        "datasets": records,
        "versions": {
            "numpy": package_version("numpy"),
            "scikit-learn": package_version("scikit-learn"),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def load_tuning(root: Path) -> tuple[dict, dict[str, tuple[np.ndarray, np.ndarray]]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing tuning manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("unsupported tuning manifest schema")
    if set(manifest.get("datasets", {})) != set(TUNING_DATASETS):
        raise RuntimeError("tuning dataset suite changed")
    datasets = {}
    for name in TUNING_DATASETS:
        record = manifest["datasets"][name]
        data_path = root / record["data_path"]
        labels_path = root / record["labels_path"]
        if historical.sha256_file(data_path) != record["data_file_sha256"]:
            raise RuntimeError(f"{name} tuning data file hash changed")
        if historical.sha256_file(labels_path) != record["labels_file_sha256"]:
            raise RuntimeError(f"{name} tuning label file hash changed")
        data = np.load(data_path, allow_pickle=False)
        labels = np.load(labels_path, allow_pickle=False)
        if historical.array_sha256(data) != record["data_array_sha256"]:
            raise RuntimeError(f"{name} tuning data array hash changed")
        if historical.array_sha256(labels) != record["labels_array_sha256"]:
            raise RuntimeError(f"{name} tuning label array hash changed")
        datasets[name] = (
            np.asarray(data, dtype=np.float32),
            np.asarray(labels, dtype=np.int64),
        )
    return manifest, datasets


def candidate_parameters(sobol_count: int) -> dict[str, dict]:
    """Return controls plus a deterministic bounded Sobol design."""
    if sobol_count < 1 or sobol_count & (sobol_count - 1):
        raise ValueError("sobol_count must be a positive power of two")
    from scipy.stats import qmc

    dimensions = 7
    points = qmc.Sobol(d=dimensions, scramble=True, seed=SOBOL_SEED).random_base2(
        int(math.log2(sobol_count))
    )
    candidates = {
        "default": {"role": "control", "parameters": DEFAULT_PARAMETERS},
        "withdrawn_9117": {
            "role": "withdrawn_control",
            "parameters": WITHDRAWN_PARAMETERS,
        },
    }
    init_values = ("pca", "spectral", "random")

    def integer(value, low, high):
        return min(high, low + int(value * (high - low + 1)))

    for index, point in enumerate(points):
        init = init_values[min(2, int(point[0] * len(init_values)))]
        parameters = {
            "init": init,
            "n_neighbors": integer(point[1], 8, 48),
            "cutoff": float(2.0 + point[2] * 40.0),
            "spread": float(0.5 + point[3] * 3.5),
            "min_dist": float(10 ** (-4.0 + point[4] * 3.0)),
            "neg_ratio": integer(point[5], 2, 32),
            "max_iter_layout": integer(point[6], 64, 256),
        }
        candidates[f"sobol_{index:03d}"] = {
            "role": "candidate",
            "parameters": parameters,
        }
    return candidates


def local_candidate_parameters() -> dict[str, dict]:
    """Return an interpretable one-parameter refinement around default DiRe."""
    candidates = {
        "default": {"role": "control", "parameters": DEFAULT_PARAMETERS},
    }
    refinements = {
        "n_neighbors": (12, 20, 24, 32),
        "cutoff": (24.0, 30.0, 36.0),
        "spread": (0.8, 1.2, 1.5),
        "min_dist": (3e-3, 3e-2),
        "neg_ratio": (6, 10, 12),
        "max_iter_layout": (96, 160, 192),
    }
    for parameter, values in refinements.items():
        for value in values:
            configured = {**DEFAULT_PARAMETERS, parameter: value}
            candidates[f"local_{parameter}_{str(value).replace('.', 'p')}"] = {
                "role": "candidate",
                "parameters": configured,
            }
    return candidates


def existing_keys(path: Path) -> set[tuple[str, str, int]]:
    keys = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(f"invalid record schema at line {line_number}")
            key = (record["candidate"], record["dataset"], record["layout_seed"])
            if key in keys:
                raise RuntimeError(f"duplicate search record: {key}")
            keys.add(key)
    return keys


def effective_policy(reducer) -> dict:
    method = getattr(reducer, "effective_cuvs_knn_method_", None)
    if method != EXPECTED_AUTO_METHOD:
        raise RuntimeError(
            f"expected guarded auto {EXPECTED_AUTO_METHOD}, got {method!r}"
        )
    return {
        "method": method,
        "index_type": getattr(reducer, "effective_cuvs_index_type_", None),
        "algorithm": getattr(reducer, "effective_all_neighbors_algo_", None),
    }


def knn_accuracy(embedding: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    indices = np.arange(len(labels))
    train, test = train_test_split(
        indices,
        test_size=0.3,
        random_state=214,
        stratify=labels,
    )
    model = KNeighborsClassifier(n_neighbors=15)
    model.fit(embedding[train], labels[train])
    return float(model.score(embedding[test], labels[test]))


def local_quality(dire_rapids, data: np.ndarray, embedding: np.ndarray) -> dict:
    metrics = dire_rapids.metrics.compute_local_metrics(
        data,
        embedding,
        n_neighbors=16,
        subsample_threshold=1.0,
        use_gpu=True,
    )
    return {
        "stress": float(metrics["stress"]),
        "neighbor_mean": float(metrics["neighbor"][0]),
        "neighbor_std": float(metrics["neighbor"][1]),
        "n_samples": int(metrics["n_samples"]),
    }


def global_distance_correlation(
    data: np.ndarray, embedding: np.ndarray, seed: int
) -> float:
    """Spearman correlation on a fixed sample of global pairwise distances."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed + 3_000)
    left = rng.integers(0, len(data), size=GLOBAL_PAIR_COUNT)
    right = rng.integers(0, len(data) - 1, size=GLOBAL_PAIR_COUNT)
    right = right + (right >= left)
    original_distances = np.linalg.norm(data[left] - data[right], axis=1)
    embedded_distances = np.linalg.norm(
        embedding[left] - embedding[right], axis=1
    )
    correlation = spearmanr(original_distances, embedded_distances).statistic
    if not np.isfinite(correlation):
        raise RuntimeError("global distance correlation is not finite")
    return float(correlation)


def run_search(
    source_root: Path,
    evaluator_source: Path,
    tuning_root: Path,
    output: Path,
    reference_cache: Path,
    sobol_count: int,
    design: str,
) -> None:
    """Run or resume every candidate/dataset/seed evaluation on an H100."""
    import torch

    source_root = source_root.resolve()
    sys.path.insert(0, str(source_root))
    try:
        import dire_rapids
    finally:
        sys.path.pop(0)
    loaded = Path(dire_rapids.__file__).resolve()
    if source_root not in loaded.parents:
        raise RuntimeError(f"loaded DiRe from {loaded}, not {source_root}")
    evaluator = historical.load_fixed_evaluator(evaluator_source)
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("preset search requires an H100 CUDA GPU")

    tuning_manifest, datasets = load_tuning(tuning_root)
    candidates = (
        candidate_parameters(sobol_count)
        if design == "sobol"
        else local_candidate_parameters()
    )
    revision = git_commit(source_root)
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": package_version("torch"),
        "cupy": package_version("cupy-cuda13x") or package_version("cupy-cuda12x"),
        "cuvs": package_version("cuvs-cu13") or package_version("cuvs-cu12"),
        "cuml": package_version("cuml-cu13") or package_version("cuml-cu12"),
        "scikit_learn": package_version("scikit-learn"),
        "ripser": package_version("ripser"),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_revision": revision,
        "evaluator_sha256": historical.EVALUATOR_SHA256,
        "tuning_manifest_sha256": historical.sha256_file(
            tuning_root / "manifest.json"
        ),
        "tuning_data_sha256": {
            name: record["data_array_sha256"]
            for name, record in tuning_manifest["datasets"].items()
        },
        "search_seeds": list(SEARCH_SEEDS),
        "sobol_seed": SOBOL_SEED,
        "sobol_count": sobol_count if design == "sobol" else None,
        "design": design,
        "quality_tolerance": QUALITY_TOLERANCE,
        "stress_relative_tolerance": STRESS_RELATIVE_TOLERANCE,
        "global_pair_count": GLOBAL_PAIR_COUNT,
        "candidates": candidates,
        "candidate_sha256": json_sha256(candidates),
        "environment": environment,
        "environment_sha256": json_sha256(environment),
        "policy": {
            "backend": "cuvs",
            "knn_backend": "cuvs",
            "cuvs_knn_method": "auto",
            "expected_effective_method": EXPECTED_AUTO_METHOD,
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError("search resume manifest changed")
    else:
        write_json(manifest_path, manifest)

    completed = existing_keys(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reference_cache.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for candidate, candidate_record in candidates.items():
            for dataset, (data, labels) in datasets.items():
                data_hash = tuning_manifest["datasets"][dataset]["data_array_sha256"]
                for seed in SEARCH_SEEDS:
                    key = (candidate, dataset, seed)
                    if key in completed:
                        continue
                    indices = historical.topology_subset_indices(len(data), seed)
                    atlas_reference, ripser_reference, reference_hashes = (
                        historical.load_or_compute_reference_curves(
                            evaluator,
                            data,
                            indices,
                            dataset,
                            data_hash,
                            seed,
                            reference_cache,
                        )
                    )
                    kwargs = {
                        "backend": "cuvs",
                        "knn_backend": "cuvs",
                        "cuvs_knn_method": "auto",
                        "cuvs_index_type": "auto",
                        "n_components": 2,
                        "random_state": seed,
                        "verbose": False,
                        **candidate_record["parameters"],
                    }
                    reducer = dire_rapids.create_dire(**kwargs)
                    started = time.perf_counter()
                    embedding = reducer.fit_transform(data)
                    torch.cuda.synchronize()
                    fit_seconds = time.perf_counter() - started
                    if hasattr(embedding, "get"):
                        embedding = embedding.get()
                    if torch.is_tensor(embedding):
                        embedding = embedding.detach().cpu().numpy()
                    embedding = np.asarray(embedding, dtype=np.float32)
                    if embedding.shape != (len(data), 2) or not np.all(
                        np.isfinite(embedding)
                    ):
                        raise RuntimeError(f"invalid embedding for {key}")
                    embedded_subset = embedding[indices]
                    atlas_curve = evaluator.compute_betti_curve_gpu(
                        embedded_subset,
                        k_neighbors=20,
                        density_threshold=0.8,
                        overlap_factor=1.5,
                        n_steps=100,
                    )
                    ripser_curve = evaluator.compute_betti_curve_ripser(
                        embedded_subset, n_steps=100, maxdim=1
                    )
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "source_revision": revision,
                        "environment_sha256": manifest["environment_sha256"],
                        "candidate": candidate,
                        "candidate_role": candidate_record["role"],
                        "parameters": kwargs,
                        "dataset": dataset,
                        "dataset_array_sha256": data_hash,
                        "layout_seed": seed,
                        "subset_seed": seed + historical.TOPOLOGY_SUBSET_SEED_OFFSET,
                        "subset_indices_sha256": historical.subset_sha256(indices),
                        "reference_curve_sha256": reference_hashes,
                        "effective_policy": effective_policy(reducer),
                        "fit_seconds": fit_seconds,
                        "knn_accuracy": knn_accuracy(embedding, labels),
                        "local": local_quality(
                            dire_rapids, data[indices], embedded_subset
                        ),
                        "global_distance_spearman": global_distance_correlation(
                            data[indices], embedded_subset, seed
                        ),
                        "metrics": {
                            "atlas": historical.curve_distances(
                                atlas_reference, atlas_curve, len(indices)
                            ),
                            "ripser": historical.curve_distances(
                                ripser_reference, ripser_curve, len(indices)
                            ),
                        },
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    stream.flush()
                    print(f"completed {candidate} {dataset} seed={seed}", flush=True)
                    completed.add(key)
                    del reducer, embedding, atlas_curve, ripser_curve
                    gc.collect()
                    torch.cuda.empty_cache()


def geometric_mean(values) -> float:
    retained = np.asarray(values, dtype=np.float64)
    if np.any(retained < 0):
        raise ValueError("geometric mean requires non-negative values")
    return float(np.exp(np.mean(np.log(retained + 1e-12))))


def summarize_records(records: list[dict], candidates: dict) -> dict:
    """Select separate Atlas/Ripser candidates under a per-dataset guard."""
    by_key = {
        (record["candidate"], record["dataset"], record["layout_seed"]): record
        for record in records
    }
    expected = {
        (candidate, dataset, seed)
        for candidate in candidates
        for dataset in TUNING_DATASETS
        for seed in SEARCH_SEEDS
    }
    if set(by_key) != expected:
        missing = sorted(expected - set(by_key))
        extra = sorted(set(by_key) - expected)
        raise RuntimeError(f"search matrix incomplete; missing={missing[:3]} extra={extra[:3]}")

    scores = {}
    for candidate, candidate_record in candidates.items():
        ratios = {"atlas": [], "ripser": []}
        dataset_quality = {}
        for dataset in TUNING_DATASETS:
            candidate_accuracies = []
            default_accuracies = []
            candidate_neighbors = []
            default_neighbors = []
            candidate_stress = []
            default_stress = []
            candidate_global = []
            default_global = []
            for seed in SEARCH_SEEDS:
                current = by_key[(candidate, dataset, seed)]
                default = by_key[("default", dataset, seed)]
                candidate_accuracies.append(current["knn_accuracy"])
                default_accuracies.append(default["knn_accuracy"])
                candidate_neighbors.append(current["local"]["neighbor_mean"])
                default_neighbors.append(default["local"]["neighbor_mean"])
                candidate_stress.append(current["local"]["stress"])
                default_stress.append(default["local"]["stress"])
                candidate_global.append(current["global_distance_spearman"])
                default_global.append(default["global_distance_spearman"])
                for backend in ratios:
                    for metric in TOPOLOGY_METRICS:
                        numerator = current["metrics"][backend][metric]
                        denominator = default["metrics"][backend][metric]
                        ratios[backend].append(
                            (float(numerator) + 1e-12) / (float(denominator) + 1e-12)
                        )
            candidate_accuracy = float(np.mean(candidate_accuracies))
            default_accuracy = float(np.mean(default_accuracies))
            candidate_neighbor = float(np.mean(candidate_neighbors))
            default_neighbor = float(np.mean(default_neighbors))
            candidate_stress_mean = float(np.mean(candidate_stress))
            default_stress_mean = float(np.mean(default_stress))
            candidate_global_mean = float(np.mean(candidate_global))
            default_global_mean = float(np.mean(default_global))
            dataset_quality[dataset] = {
                "knn_accuracy": {
                    "candidate_mean": candidate_accuracy,
                    "default_mean": default_accuracy,
                    "gap": candidate_accuracy - default_accuracy,
                },
                "neighbor": {
                    "candidate_mean": candidate_neighbor,
                    "default_mean": default_neighbor,
                    "gap": candidate_neighbor - default_neighbor,
                },
                "stress": {
                    "candidate_mean": candidate_stress_mean,
                    "default_mean": default_stress_mean,
                    "ratio": (candidate_stress_mean + 1e-12)
                    / (default_stress_mean + 1e-12),
                },
                "global_distance_spearman": {
                    "candidate_mean": candidate_global_mean,
                    "default_mean": default_global_mean,
                    "gap": candidate_global_mean - default_global_mean,
                },
            }
        feasible = all(
            value["knn_accuracy"]["gap"] >= -QUALITY_TOLERANCE
            and value["neighbor"]["gap"] >= -QUALITY_TOLERANCE
            and value["global_distance_spearman"]["gap"] >= -QUALITY_TOLERANCE
            and value["stress"]["ratio"] <= 1.0 + STRESS_RELATIVE_TOLERANCE
            for value in dataset_quality.values()
        )
        atlas_score = geometric_mean(ratios["atlas"])
        ripser_score = geometric_mean(ratios["ripser"])
        scores[candidate] = {
            "role": candidate_record["role"],
            "parameters": candidate_record["parameters"],
            "quality_feasible": feasible,
            "quality_by_dataset": dataset_quality,
            "atlas_ratio_to_default": atlas_score,
            "ripser_ratio_to_default": ripser_score,
            "compromise_ratio_to_default": math.sqrt(atlas_score * ripser_score),
        }

    search_candidates = [
        name
        for name, value in scores.items()
        if value["role"] == "candidate"
    ]
    eligible = [name for name in search_candidates if scores[name]["quality_feasible"]]

    def select(pool):
        if not pool:
            return {}
        return {
            "atlas": min(
                pool, key=lambda name: scores[name]["atlas_ratio_to_default"]
            ),
            "ripser": min(
                pool, key=lambda name: scores[name]["ripser_ratio_to_default"]
            ),
            "compromise": min(
                pool, key=lambda name: scores[name]["compromise_ratio_to_default"]
            ),
        }

    selections = select(eligible)
    unconstrained = select(search_candidates)
    selection_status = (
        "eligible candidates selected for held-out validation"
        if eligible
        else "no candidate passed all local/context/global quality guards; "
        "unconstrained winners are diagnostic only and are not presets"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "selection_status": selection_status,
        "eligible_candidate_count": len(eligible),
        "search_candidate_count": len(search_candidates),
        "selection_rule": {
            "quality_guard": (
                "on every tuning dataset, candidate mean 15-NN accuracy, local "
                "neighbor retention, and global distance Spearman are no more than "
                "0.01 below default, while local stress is no more than 10% above default"
            ),
            "topology_score": (
                "geometric mean of paired candidate/default DTW ratios over "
                "four datasets, two seeds, and H0/H1"
            ),
        },
        "selections": selections,
        "unconstrained_diagnostic_winners": unconstrained,
        "scores": scores,
    }


def summarize(input_path: Path, manifest_path: Path, output: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = manifest["candidates"]
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize_records(records, candidates)
    summary["source_revision"] = manifest["source_revision"]
    summary["environment_sha256"] = manifest["environment_sha256"]
    summary["candidate_sha256"] = manifest["candidate_sha256"]
    write_json(output, summary)
    return summary


def parse_seed_spec(value: str) -> tuple[int, ...]:
    if ":" in value:
        start_text, stop_text = value.split(":", 1)
        seeds = tuple(range(int(start_text), int(stop_text)))
    else:
        seeds = tuple(int(item) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("layout seeds must be a non-empty unique list or start:stop")
    return seeds


def import_source_dire(source_root: Path):
    import torch  # Import before dire_rapids; see historical harness rationale.

    source_root = source_root.resolve()
    sys.path.insert(0, str(source_root))
    try:
        import dire_rapids
    finally:
        sys.path.pop(0)
    loaded = Path(dire_rapids.__file__).resolve()
    if source_root not in loaded.parents:
        raise RuntimeError(f"loaded DiRe from {loaded}, not {source_root}")
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("preset validation requires an H100 CUDA GPU")
    return dire_rapids, torch


def run_validation(
    source_root: Path,
    evaluator_source: Path,
    frozen_root: Path,
    search_summary_path: Path,
    output: Path,
    reference_cache: Path,
    layout_seeds: tuple[int, ...],
) -> None:
    """Run or resume shortlisted candidates on the six untouched datasets."""
    dire_rapids, torch = import_source_dire(source_root)
    evaluator = historical.load_fixed_evaluator(evaluator_source)
    dataset_manifest, datasets = historical.load_dataset_manifest(frozen_root)
    search_summary = json.loads(search_summary_path.read_text(encoding="utf-8"))
    candidate_names = tuple(dict.fromkeys(search_summary["selections"].values()))
    candidates = {
        name: search_summary["scores"][name]["parameters"]
        for name in candidate_names
    }
    revision = git_commit(source_root)
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "torch": package_version("torch"),
        "cupy": package_version("cupy-cuda13x") or package_version("cupy-cuda12x"),
        "cuvs": package_version("cuvs-cu13") or package_version("cuvs-cu12"),
        "cuml": package_version("cuml-cu13") or package_version("cuml-cu12"),
        "ripser": package_version("ripser"),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_revision": revision,
        "evaluator_sha256": historical.EVALUATOR_SHA256,
        "frozen_dataset_manifest_sha256": historical.sha256_file(
            frozen_root / "manifest.json"
        ),
        "dataset_array_sha256": {
            name: record["array_sha256"]
            for name, record in dataset_manifest["datasets"].items()
        },
        "search_summary_sha256": historical.sha256_file(search_summary_path),
        "layout_seeds": list(layout_seeds),
        "candidates": candidates,
        "candidate_sha256": json_sha256(candidates),
        "environment": environment,
        "environment_sha256": json_sha256(environment),
        "policy": {
            "backend": "cuvs",
            "knn_backend": "cuvs",
            "cuvs_knn_method": "auto",
            "expected_effective_method": EXPECTED_AUTO_METHOD,
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError("validation resume manifest changed")
    else:
        write_json(manifest_path, manifest)

    completed = existing_keys(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reference_cache.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for candidate, parameters in candidates.items():
            for dataset in historical.DATASETS:
                data = datasets[dataset]
                data_hash = dataset_manifest["datasets"][dataset]["array_sha256"]
                for seed in layout_seeds:
                    key = (candidate, dataset, seed)
                    if key in completed:
                        continue
                    indices = historical.topology_subset_indices(len(data), seed)
                    atlas_reference, ripser_reference, reference_hashes = (
                        historical.load_or_compute_reference_curves(
                            evaluator,
                            data,
                            indices,
                            dataset,
                            data_hash,
                            seed,
                            reference_cache,
                        )
                    )
                    kwargs = {
                        "backend": "cuvs",
                        "knn_backend": "cuvs",
                        "cuvs_knn_method": "auto",
                        "cuvs_index_type": "auto",
                        "n_components": 2,
                        "random_state": seed,
                        "verbose": False,
                        **parameters,
                    }
                    reducer = dire_rapids.create_dire(**kwargs)
                    started = time.perf_counter()
                    embedding = reducer.fit_transform(data)
                    torch.cuda.synchronize()
                    fit_seconds = time.perf_counter() - started
                    if hasattr(embedding, "get"):
                        embedding = embedding.get()
                    if torch.is_tensor(embedding):
                        embedding = embedding.detach().cpu().numpy()
                    embedding = np.asarray(embedding, dtype=np.float32)
                    if embedding.shape != (len(data), 2) or not np.all(
                        np.isfinite(embedding)
                    ):
                        raise RuntimeError(f"invalid validation embedding for {key}")
                    embedded_subset = embedding[indices]
                    atlas_curve = evaluator.compute_betti_curve_gpu(
                        embedded_subset,
                        k_neighbors=20,
                        density_threshold=0.8,
                        overlap_factor=1.5,
                        n_steps=100,
                    )
                    ripser_curve = evaluator.compute_betti_curve_ripser(
                        embedded_subset, n_steps=100, maxdim=1
                    )
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "source_revision": revision,
                        "environment_sha256": manifest["environment_sha256"],
                        "candidate": candidate,
                        "candidate_role": "shortlist",
                        "parameters": kwargs,
                        "dataset": dataset,
                        "dataset_array_sha256": data_hash,
                        "layout_seed": seed,
                        "subset_seed": seed + historical.TOPOLOGY_SUBSET_SEED_OFFSET,
                        "subset_indices_sha256": historical.subset_sha256(indices),
                        "reference_curve_sha256": reference_hashes,
                        "effective_policy": effective_policy(reducer),
                        "fit_seconds": fit_seconds,
                        "local": local_quality(
                            dire_rapids, data[indices], embedded_subset
                        ),
                        "global_distance_spearman": global_distance_correlation(
                            data[indices], embedded_subset, seed
                        ),
                        "metrics": {
                            "atlas": historical.curve_distances(
                                atlas_reference, atlas_curve, len(indices)
                            ),
                            "ripser": historical.curve_distances(
                                ripser_reference, ripser_curve, len(indices)
                            ),
                        },
                    }
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
                    stream.flush()
                    print(f"validated {candidate} {dataset} seed={seed}", flush=True)
                    completed.add(key)
                    del reducer, embedding, atlas_curve, ripser_curve
                    gc.collect()
                    torch.cuda.empty_cache()


def paired_interval(gaps: list[float]) -> tuple[float, float]:
    if len(gaps) < 2:
        return float("nan"), float("nan")
    from scipy.stats import t

    values = np.asarray(gaps, dtype=np.float64)
    half_width = float(
        t.ppf(0.975, len(values) - 1)
        * values.std(ddof=1)
        / math.sqrt(len(values))
    )
    mean = float(values.mean())
    return mean - half_width, mean + half_width


def summarize_validation(
    input_path: Path,
    manifest_path: Path,
    baseline_path: Path,
    output: Path,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_key = {
        (record["candidate"], record["dataset"], record["layout_seed"]): record
        for record in records
    }
    candidates = tuple(manifest["candidates"])
    seeds = tuple(manifest["layout_seeds"])
    expected = {
        (candidate, dataset, seed)
        for candidate in candidates
        for dataset in historical.DATASETS
        for seed in seeds
    }
    if set(by_key) != expected:
        raise RuntimeError("validation matrix is incomplete")

    scores = {}
    for candidate in candidates:
        candidate_output = {"cells": {"atlas": {}, "ripser": {}}}
        atlas_ratios = []
        ripser_ratios = []
        for dataset in historical.DATASETS:
            for metric in TOPOLOGY_METRICS:
                atlas_threshold = baselines["datasets"][dataset]["thresholds"][metric]
                baseline_method = atlas_threshold["best_method"]
                baseline_record = baselines["datasets"][dataset]["methods"][
                    baseline_method
                ]
                baseline_by_seed = dict(
                    zip(
                        baseline_record["seeds"],
                        baseline_record["atlas"][metric]["values"],
                    )
                )
                overlap = [seed for seed in seeds if seed in baseline_by_seed]
                candidate_values = [
                    by_key[(candidate, dataset, seed)]["metrics"]["atlas"][metric]
                    for seed in overlap
                ]
                comparator_values = [baseline_by_seed[seed] for seed in overlap]
                gaps = [
                    float(current) - float(comparator)
                    for current, comparator in zip(candidate_values, comparator_values)
                ]
                low, high = paired_interval(gaps)
                gap_mean = float(np.mean(gaps))
                ratio = (float(np.mean(candidate_values)) + 1e-12) / (
                    float(np.mean(comparator_values)) + 1e-12
                )
                atlas_ratios.append(ratio)
                candidate_output["cells"]["atlas"][f"{dataset}/{metric}"] = {
                    "comparator": baseline_method,
                    "paired_count": len(overlap),
                    "candidate_mean": float(np.mean(candidate_values)),
                    "comparator_mean": float(np.mean(comparator_values)),
                    "paired_gap_candidate_minus_comparator": gap_mean,
                    "paired_95pct_low": low,
                    "paired_95pct_high": high,
                    "mean_win": gap_mean < 0,
                    "interval_win": high < 0,
                }

                if 42 in seeds:
                    ripser_threshold = baselines["ripser_canonical"]["datasets"][
                        dataset
                    ]["thresholds"][metric]
                    candidate_value = by_key[(candidate, dataset, 42)]["metrics"][
                        "ripser"
                    ][metric]
                    ratio = (float(candidate_value) + 1e-12) / (
                        float(ripser_threshold["best_value"]) + 1e-12
                    )
                    ripser_ratios.append(ratio)
                    candidate_output["cells"]["ripser"][f"{dataset}/{metric}"] = {
                        "scope": "seed-42 screen",
                        "comparator": ripser_threshold["best_method"],
                        "candidate_value": candidate_value,
                        "comparator_value": ripser_threshold["best_value"],
                        "win": candidate_value < ripser_threshold["best_value"],
                    }
        atlas_cells = candidate_output["cells"]["atlas"].values()
        ripser_cells = candidate_output["cells"]["ripser"].values()
        candidate_output.update(
            {
                "parameters": manifest["candidates"][candidate],
                "atlas_geometric_ratio_to_best_baseline": geometric_mean(atlas_ratios),
                "atlas_mean_wins": sum(cell["mean_win"] for cell in atlas_cells),
                "atlas_interval_wins": sum(
                    cell["interval_win"] for cell in candidate_output["cells"]["atlas"].values()
                ),
                "ripser_seed42_geometric_ratio_to_best_baseline": (
                    geometric_mean(ripser_ratios) if ripser_ratios else None
                ),
                "ripser_seed42_wins": sum(cell["win"] for cell in ripser_cells),
            }
        )
        scores[candidate] = candidate_output

    selections = {
        "atlas": min(
            candidates,
            key=lambda name: scores[name]["atlas_geometric_ratio_to_best_baseline"],
        )
    }
    if 42 in seeds:
        selections["ripser_seed42_screen"] = min(
            candidates,
            key=lambda name: scores[name][
                "ripser_seed42_geometric_ratio_to_best_baseline"
            ],
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_revision": manifest["source_revision"],
        "validation_manifest_sha256": historical.sha256_file(manifest_path),
        "baseline_fixture_sha256": historical.sha256_file(baseline_path),
        "layout_seeds": list(seeds),
        "selections": selections,
        "scores": scores,
    }
    write_json(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="freeze the OpenML tuning suite")
    prepare.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run", help="run or resume the crossed search")
    run.add_argument("--source-root", type=Path, required=True)
    run.add_argument("--evaluator-source", type=Path, required=True)
    run.add_argument("--tuning-root", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--reference-cache", type=Path, required=True)
    run.add_argument("--sobol-count", type=int, default=32)
    run.add_argument("--design", choices=("sobol", "local"), default="sobol")

    summary_parser = subparsers.add_parser("summarize", help="select candidates")
    summary_parser.add_argument("--input", type=Path, required=True)
    summary_parser.add_argument("--manifest", type=Path, required=True)
    summary_parser.add_argument("--output", type=Path, required=True)

    validation = subparsers.add_parser(
        "validate", help="run shortlisted candidates on untouched validation data"
    )
    validation.add_argument("--source-root", type=Path, required=True)
    validation.add_argument("--evaluator-source", type=Path, required=True)
    validation.add_argument("--frozen-root", type=Path, required=True)
    validation.add_argument("--search-summary", type=Path, required=True)
    validation.add_argument("--output", type=Path, required=True)
    validation.add_argument("--reference-cache", type=Path, required=True)
    validation.add_argument("--layout-seeds", default="42")

    validation_summary = subparsers.add_parser(
        "summarize-validation", help="compare validation candidates to retained baselines"
    )
    validation_summary.add_argument("--input", type=Path, required=True)
    validation_summary.add_argument("--manifest", type=Path, required=True)
    validation_summary.add_argument("--baselines", type=Path, required=True)
    validation_summary.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_tuning(args.output), indent=2, sort_keys=True))
    elif args.command == "run":
        run_search(
            args.source_root,
            args.evaluator_source,
            args.tuning_root,
            args.output,
            args.reference_cache,
            args.sobol_count,
            args.design,
        )
    elif args.command == "summarize":
        print(json.dumps(summarize(args.input, args.manifest, args.output), indent=2))
    elif args.command == "validate":
        run_validation(
            args.source_root,
            args.evaluator_source,
            args.frozen_root,
            args.search_summary,
            args.output,
            args.reference_cache,
            parse_seed_spec(args.layout_seeds),
        )
    else:
        print(
            json.dumps(
                summarize_validation(
                    args.input, args.manifest, args.baselines, args.output
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
