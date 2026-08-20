#!/usr/bin/env python3
"""Run one member's YAML work queue sequentially."""

import argparse
import json
import re
import shlex
import subprocess
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


def _run_directory(config):
    output = Path(config["experiment"]["output_dir"])
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    seed_directory = "seed_{}".format(config["train"]["seed"])
    attempt = config["experiment"].get("attempt")
    if attempt:
        safe_attempt = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(attempt)).strip("._")
        seed_directory += "_" + (safe_attempt or "retry")
    return output / config["experiment"]["id"] / seed_directory


def _set_argument(key, value):
    # JSON values are also valid YAML values and survive spaces/booleans/lists.
    return "{}={}".format(key, json.dumps(value, ensure_ascii=False))


def _completed_run_matches(summary_path, expected_config):
    """Never silently reuse a completed run from an older draft protocol."""
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    matches = (
        summary.get("state") == "completed"
        and summary.get("experiment_id") == expected_config["experiment"]["id"]
        and int(summary.get("seed", -1)) == int(expected_config["train"]["seed"])
        and summary.get("semantic_fingerprint") == semantic_fingerprint(expected_config)
    )
    if not matches:
        raise SystemExit(
            "Stale/mismatched completed run at {}. Move old pilots outside runs/ "
            "or use a clean output directory; do not mix protocols.".format(summary_path.parent)
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--only", action="append", default=[], help="Run only this experiment id")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Run only canonical folders that exist but have no completed summary.",
    )
    parser.add_argument("--attempt", help="Retry folder suffix, for example retry1")
    arguments = parser.parse_args()
    if arguments.retry_failed != bool(arguments.attempt):
        raise SystemExit("Use --retry-failed and --attempt together")

    matrix_path = arguments.matrix
    if not matrix_path.is_absolute():
        matrix_path = PROJECT_ROOT / matrix_path
    matrix = load_matrix(matrix_path)
    if not matrix["approved"] and not (arguments.allow_draft or arguments.dry_run):
        raise SystemExit(
            "Matrix is still approved:false. Freeze the protocol or pass --allow-draft for pilots."
        )

    base_path = Path(matrix["base_config"])
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path
    base = load_yaml(base_path)
    selected = set(arguments.only)

    for experiment in matrix["experiments"]:
        if selected and experiment["id"] not in selected:
            continue
        for seed in matrix["seeds"]:
            metadata = {
                "experiment.id": experiment["id"],
                "experiment.label": experiment["label"],
                "experiment.comparison_groups": experiment["comparison_groups"],
                "train.seed": seed,
            }
            overrides = dict(experiment.get("overrides", {}))
            overrides.update(metadata)
            canonical = apply_overrides(base, overrides)
            validate_config(canonical)
            canonical_directory = _run_directory(canonical)
            canonical_summary = canonical_directory / "summary.json"
            if canonical_summary.exists() and _completed_run_matches(canonical_summary, canonical):
                print("SKIP completed {} seed {}".format(experiment["id"], seed))
                continue
            canonical_failed = canonical_directory.exists() and any(canonical_directory.iterdir())
            if arguments.retry_failed and not canonical_failed:
                continue
            if canonical_failed and not arguments.retry_failed:
                print(
                    "SKIP incomplete {} (retry with --retry-failed --attempt retry1)".format(
                        canonical_directory
                    )
                )
                continue

            if arguments.retry_failed:
                overrides["experiment.attempt"] = arguments.attempt
            resolved = apply_overrides(base, overrides)
            validate_config(resolved)
            run_directory = _run_directory(resolved)
            retry_summary = run_directory / "summary.json"
            if retry_summary.exists() and _completed_run_matches(retry_summary, resolved):
                print("SKIP completed retry {}".format(run_directory))
                continue
            if run_directory.exists() and any(run_directory.iterdir()):
                print("SKIP incomplete retry {}; use a new --attempt".format(run_directory))
                continue

            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts/train.py"),
                "--config",
                str(base_path),
            ]
            for key, value in overrides.items():
                command.extend(["--set", _set_argument(key, value)])
            if arguments.dry_run:
                print(shlex.join(command))
            else:
                print("RUN {} seed {}".format(experiment["id"], seed), flush=True)
                subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
