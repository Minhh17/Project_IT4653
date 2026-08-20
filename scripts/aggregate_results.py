#!/usr/bin/env python3
"""Build master and mean±std CSV files directly from immutable run summaries."""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlstudy.config import (  # noqa: E402
    apply_overrides,
    load_yaml,
    semantic_fingerprint,
    validate_config,
)

DEFAULT_SEEDS = {42, 2026}
OFFICIAL_GROUPS = {"optimizer", "schedule", "normalization", "regularization"}
ANCHOR_ID = "anchor_sgdm_wd_bn_b128"
METRICS = [
    "best_val_accuracy",
    "final_val_accuracy",
    "final_val_loss",
    "final_generalization_gap",
    "mean_val_accuracy_over_epochs",
    "training_seconds",
    "clean_train_eval_seconds",
    "total_seconds",
]


def _read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _csv_value(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _audit_official_matrix(matrix_directory, rows):
    """Detect even a completely missing two-seed configuration."""
    issues = []
    expected = {}
    ids_by_fingerprint = defaultdict(list)
    for path in sorted(matrix_directory.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as file:
            matrix = yaml.safe_load(file)
        if not matrix.get("approved", False):
            issues.append("official matrix is not approved: {}".format(path.name))
        base_path = Path(matrix["base_config"])
        if not base_path.is_absolute():
            base_path = PROJECT_ROOT / base_path
        base = load_yaml(base_path)
        for experiment in matrix["experiments"]:
            overrides = dict(experiment.get("overrides", {}))
            overrides.update(
                {
                    "experiment.id": experiment["id"],
                    "experiment.label": experiment["label"],
                    "experiment.comparison_groups": experiment["comparison_groups"],
                    "train.seed": matrix["seeds"][0],
                }
            )
            resolved = apply_overrides(base, overrides)
            validate_config(resolved)
            expected_fingerprint = semantic_fingerprint(resolved)
            ids_by_fingerprint[expected_fingerprint].append(experiment["id"])
            expected[experiment["id"]] = {
                "seeds": {int(seed) for seed in matrix["seeds"]},
                "groups": set(experiment["comparison_groups"]),
                "fingerprint": expected_fingerprint,
            }

    for fingerprint, experiment_ids in ids_by_fingerprint.items():
        if len(experiment_ids) > 1:
            issues.append(
                "matrix duplicates semantic config {} across ids {}".format(
                    fingerprint, sorted(experiment_ids)
                )
            )

    completed = defaultdict(list)
    for row in rows:
        if row["dataset"] != "fake":
            completed[row["experiment_id"]].append(row)
    for experiment_id, requirement in sorted(expected.items()):
        members = completed.get(experiment_id, [])
        seeds = {int(row["seed"]) for row in members}
        if seeds != requirement["seeds"]:
            issues.append(
                "official config {} has seeds {}, expected {}".format(
                    experiment_id, sorted(seeds), sorted(requirement["seeds"])
                )
            )
        for row in members:
            if set(row["comparison_groups"]) != requirement["groups"]:
                issues.append("comparison_groups mismatch for {}".format(experiment_id))
            if row.get("semantic_fingerprint") != requirement["fingerprint"]:
                issues.append(
                    "{} seed {} does not match the current frozen matrix".format(
                        experiment_id, row["seed"]
                    )
                )
    return issues


def _official_experiment_ids(matrix_directory):
    ids = set()
    for path in matrix_directory.glob("*.yaml"):
        with path.open("r", encoding="utf-8") as file:
            matrix = yaml.safe_load(file)
        ids.update(experiment["id"] for experiment in matrix["experiments"])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--strict", action="store_true", help="Fail if the two-seed protocol is incomplete"
    )
    parser.add_argument(
        "--skip-matrix-check",
        action="store_true",
        help="Use only for pilots/tests that are intentionally not the official 54-run matrix.",
    )
    arguments = parser.parse_args()

    rows = []
    for summary_path in sorted(arguments.runs_dir.glob("**/summary.json")):
        summary = _read_json(summary_path)
        if summary.get("state") != "completed":
            continue
        summary["run_directory"] = str(summary_path.parent.resolve())
        rows.append(summary)
    if not rows:
        raise SystemExit("No completed summary.json files found in {}".format(arguments.runs_dir))
    for row in rows:
        for metric in METRICS:
            if metric not in row or not math.isfinite(float(row[metric])):
                raise SystemExit(
                    "Missing/non-finite {} in {} seed {}".format(
                        metric, row.get("experiment_id"), row.get("seed")
                    )
                )

    # One physical run can belong to several comparisons (the shared anchor).
    grouped = defaultdict(list)
    group_gpu_names = defaultdict(set)
    duplicate_guard = defaultdict(int)
    for row in rows:
        duplicate_guard[(row["experiment_id"], int(row["seed"]))] += 1
        for group in row["comparison_groups"]:
            grouped[(group, row["experiment_id"])].append(row)
            hardware = row.get("gpu_name") or row.get("device") or "unknown"
            group_gpu_names[group].add(hardware)

    issues = []
    warnings = []
    matrix_directory = PROJECT_ROOT / "configs/matrices"
    official_ids = _official_experiment_ids(matrix_directory)
    if arguments.strict and not arguments.skip_matrix_check:
        issues.extend(_audit_official_matrix(matrix_directory, rows))
    for group in sorted(OFFICIAL_GROUPS):
        gpu_names = sorted(group_gpu_names[group])
        if len(gpu_names) > 1:
            warnings.append(
                "{} uses multiple GPU types {}; do not compare total_seconds across configs".format(
                    group, gpu_names
                )
            )
    for key, count in duplicate_guard.items():
        if count > 1:
            issues.append("duplicate run for {} seed {}".format(key[0], key[1]))
    duplicate_pairs = [key for key, count in duplicate_guard.items() if count > 1]
    if duplicate_pairs:
        raise SystemExit(
            "Ambiguous duplicate completed runs: {}. Move superseded runs outside --runs-dir.".format(
                sorted(duplicate_pairs)
            )
        )

    official_rows = [
        row for row in rows if row["experiment_id"] in official_ids and row["dataset"] != "fake"
    ]
    split_hashes = sorted({row["split_hash"] for row in official_rows})
    if len(split_hashes) > 1:
        issues.append("official runs use multiple split hashes: {}".format(split_hashes))
    if official_rows:
        cpu_runs = [
            "{}:seed{}".format(row["experiment_id"], row["seed"])
            for row in official_rows
            if row.get("device") != "cuda"
        ]
        if cpu_runs:
            issues.append("official runs must use CUDA: {}".format(cpu_runs))
        debug_runs = [
            "{}:seed{}".format(row["experiment_id"], row["seed"])
            for row in official_rows
            if row.get("debug_active") is not False
        ]
        if debug_runs:
            issues.append("official runs must not use debug truncation: {}".format(debug_runs))
        protocol_versions = sorted({row["protocol_version"] for row in official_rows})
        if len(protocol_versions) > 1:
            issues.append("official runs mix protocol versions: {}".format(protocol_versions))
        commits = sorted({row["git_commit"] for row in official_rows})
        if "unknown" in commits or len(commits) > 1:
            issues.append("official runs must use one known git commit: {}".format(commits))
        dirty_runs = [
            "{}:seed{}".format(row["experiment_id"], row["seed"])
            for row in official_rows
            if row.get("git_status") != ""
        ]
        if dirty_runs:
            issues.append("official runs have dirty/unknown git status: {}".format(dirty_runs))
        software = sorted(
            {
                (
                    row.get("python_version"),
                    row.get("torch_version"),
                    row.get("torchvision_version"),
                )
                for row in official_rows
            }
        )
        if len(software) > 1:
            issues.append("official runs mix Python/PyTorch/torchvision versions")

    aggregate_rows = []
    for (group, experiment_id), members in sorted(grouped.items()):
        seeds = {int(row["seed"]) for row in members}
        if experiment_id in official_ids and group in OFFICIAL_GROUPS and seeds != DEFAULT_SEEDS:
            issues.append(
                "{} / {} has seeds {}, expected [42, 2026]".format(
                    group, experiment_id, sorted(seeds)
                )
            )
        fingerprints = {row["semantic_fingerprint"] for row in members}
        if len(fingerprints) != 1:
            issues.append("{} mixes scientific configs across seeds".format(experiment_id))

        first = members[0]
        output = {
            "comparison_group": group,
            "experiment_id": experiment_id,
            "experiment_label": first["experiment_label"],
            "n_seeds": len(seeds),
            "seeds": sorted(seeds),
            "optimizer": first["optimizer"],
            "scheduler": first["scheduler"],
            "warmup_epochs": first["warmup_epochs"],
            "normalization": first["normalization"],
            "batch_size": first["batch_size"],
            "weight_decay": first["weight_decay"],
            "dropout": first["dropout"],
            "augmentation": first["augmentation"],
            "early_stopping": first["early_stopping"],
            "timing_comparable_in_group": len(group_gpu_names[group]) == 1,
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in members]
            timing_is_invalid = metric.endswith("seconds") and len(group_gpu_names[group]) != 1
            output[metric + "_mean"] = None if timing_is_invalid else statistics.mean(values)
            output[metric + "_std"] = (
                None if timing_is_invalid or len(values) < 2 else statistics.stdev(values)
            )
        aggregate_rows.append(output)

    # Same seeds across configs create a paired design. Preserve per-seed
    # deltas to the anchor so a sign flip cannot be hidden by mean ± std.
    paired_rows = []
    for group in sorted(OFFICIAL_GROUPS):
        anchor_members = grouped.get((group, ANCHOR_ID), [])
        anchor_by_seed = {int(row["seed"]): row for row in anchor_members}
        for (member_group, experiment_id), members in sorted(grouped.items()):
            if member_group != group or experiment_id == ANCHOR_ID:
                continue
            for row in members:
                seed = int(row["seed"])
                if seed not in anchor_by_seed:
                    continue
                paired_rows.append(
                    {
                        "comparison_group": group,
                        "experiment_id": experiment_id,
                        "experiment_label": row["experiment_label"],
                        "seed": seed,
                        "best_val_accuracy": row["best_val_accuracy"],
                        "anchor_best_val_accuracy": anchor_by_seed[seed]["best_val_accuracy"],
                        "delta_to_anchor": row["best_val_accuracy"]
                        - anchor_by_seed[seed]["best_val_accuracy"],
                    }
                )

    master_columns = sorted({key for row in rows for key in row.keys()})
    aggregate_columns = list(aggregate_rows[0].keys())
    _write_csv(arguments.output_dir / "master.csv", rows, master_columns)
    _write_csv(arguments.output_dir / "summary_mean_std.csv", aggregate_rows, aggregate_columns)
    if paired_rows:
        _write_csv(
            arguments.output_dir / "paired_deltas.csv",
            paired_rows,
            list(paired_rows[0].keys()),
        )
    audit = {
        "completed_physical_runs": len(rows),
        "comparison_rows": len(aggregate_rows),
        "split_hashes": split_hashes,
        "run_manifest": sorted(
            "{}|{}|{}|{}".format(
                row["experiment_id"],
                row["seed"],
                row.get("attempt"),
                row["semantic_fingerprint"],
            )
            for row in rows
        ),
        "issues": issues,
        "warnings": warnings,
    }
    with (arguments.output_dir / "audit.json").open("w", encoding="utf-8") as file:
        json.dump(audit, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("Wrote {} physical runs and {} comparison rows".format(len(rows), len(aggregate_rows)))
    if issues:
        print("AUDIT ISSUES:")
        for issue in issues:
            print("- " + issue)
        if arguments.strict:
            raise SystemExit(1)
    if warnings:
        print("AUDIT WARNINGS:")
        for warning in warnings:
            print("- " + warning)


if __name__ == "__main__":
    main()
