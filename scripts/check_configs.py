#!/usr/bin/env python3
"""Validate the base config and all matrices without importing PyTorch."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlstudy.config import (  # noqa: E402
    apply_overrides,
    load_matrix,
    load_yaml,
    semantic_fingerprint,
    validate_config,
)


def main() -> None:
    base = load_yaml(PROJECT_ROOT / "configs/base.yaml")
    validate_config(base)
    matrix_paths = sorted((PROJECT_ROOT / "configs/matrices").glob("*.yaml"))
    all_ids = set()
    total_configs = 0
    total_runs = 0
    approval_states = []
    ids_by_fingerprint = {}

    for path in matrix_paths:
        matrix = load_matrix(path)
        approval_states.append(bool(matrix["approved"]))
        if len(matrix["experiments"]) != int(matrix["expected_unique_configs"]):
            raise ValueError("expected_unique_configs disagrees in {}".format(path))
        if sorted(matrix["seeds"]) != [42, 2026]:
            raise ValueError("Official matrices must use seeds 42 and 2026: {}".format(path))
        for experiment in matrix["experiments"]:
            if experiment["id"] in all_ids:
                raise ValueError("Duplicate global experiment id: {}".format(experiment["id"]))
            all_ids.add(experiment["id"])
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
            fingerprint = semantic_fingerprint(resolved)
            previous_id = ids_by_fingerprint.get(fingerprint)
            if previous_id is not None:
                raise ValueError(
                    "Duplicate scientific config: {} and {}".format(previous_id, experiment["id"])
                )
            ids_by_fingerprint[fingerprint] = experiment["id"]
        config_count = len(matrix["experiments"])
        run_count = config_count * len(matrix["seeds"])
        total_configs += config_count
        total_runs += run_count
        print("OK {:32s} {:2d} configs / {:2d} runs".format(path.stem, config_count, run_count))

    if total_configs != 27 or total_runs != 54:
        raise ValueError("Protocol currently expects 27 unique configs / 54 runs")
    if any(approval_states) and not all(approval_states):
        raise ValueError("Approve all three official matrices together after the protocol review")
    if all(approval_states) and str(base["experiment"]["protocol_version"]).startswith("draft"):
        raise ValueError("Approved matrices cannot use a draft protocol_version")
    print("OK total: {} unique configs / {} seeded runs".format(total_configs, total_runs))


if __name__ == "__main__":
    main()
