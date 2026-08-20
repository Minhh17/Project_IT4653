#!/usr/bin/env python3
"""Aggregate the two pre-selected final test evaluations as mean ± std."""

import argparse
import json
import math
import statistics
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument(
        "--selection", type=Path, default=PROJECT_ROOT / "configs/final_selection.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results/final_test_mean_std.json"
    )
    arguments = parser.parse_args()

    with arguments.selection.open("r", encoding="utf-8") as file:
        selection = yaml.safe_load(file)
    if not selection.get("frozen"):
        raise SystemExit("Final selection is not frozen")

    expected_seeds = {int(seed) for seed in selection["seeds"]}
    by_seed = {}
    for path in arguments.runs_dir.glob("**/test_metrics.json"):
        value = _load(path)
        if (
            value.get("experiment_id") == selection["experiment_id"]
            and value.get("semantic_fingerprint") == selection["semantic_fingerprint"]
            and value.get("protocol_version") == selection["protocol_version"]
        ):
            seed = int(value["seed"])
            if seed in by_seed:
                raise SystemExit("Duplicate final test result for seed {}".format(seed))
            by_seed[seed] = value
    if set(by_seed) != expected_seeds:
        raise SystemExit(
            "Found final test seeds {}, expected {}".format(sorted(by_seed), sorted(expected_seeds))
        )

    # Test metrics are paired only if the evaluation code and software were
    # identical. GPU is included because even deterministic kernels can differ
    # across hardware families.
    environment_keys = (
        "git_commit",
        "python",
        "torch",
        "torchvision",
        "cuda_runtime",
        "cudnn",
        "gpu_name",
    )
    environments = {
        tuple(by_seed[seed]["evaluation_environment"].get(key) for key in environment_keys)
        for seed in by_seed
    }
    if len(environments) != 1:
        raise SystemExit("Final test seeds were evaluated with different code/software/hardware")
    if any(by_seed[seed]["evaluation_environment"].get("git_status") != "" for seed in by_seed):
        raise SystemExit("A final test was evaluated from a dirty Git worktree")

    losses = [float(by_seed[seed]["test_loss"]) for seed in sorted(by_seed)]
    accuracies = [float(by_seed[seed]["test_accuracy"]) for seed in sorted(by_seed)]
    if not all(math.isfinite(value) for value in losses + accuracies):
        raise SystemExit("Final test contains a non-finite metric")
    result = {
        "experiment_id": selection["experiment_id"],
        "semantic_fingerprint": selection["semantic_fingerprint"],
        "protocol_version": selection["protocol_version"],
        "seeds": sorted(by_seed),
        "evaluation_git_commit": next(iter(environments))[0],
        "checkpoint_sha256_by_seed": {
            str(seed): by_seed[seed]["checkpoint_sha256"] for seed in sorted(by_seed)
        },
        "test_loss_mean": statistics.mean(losses),
        "test_loss_std": statistics.stdev(losses),
        "test_accuracy_mean": statistics.mean(accuracies),
        "test_accuracy_std": statistics.stdev(accuracies),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
