#!/usr/bin/env python3
"""Evaluate one frozen best checkpoint on the official test set exactly once."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlstudy.config import load_yaml, semantic_fingerprint, validate_config  # noqa: E402
from dlstudy.data import build_dataloaders  # noqa: E402
from dlstudy.model import build_model  # noqa: E402
from dlstudy.optimization import build_criterion  # noqa: E402
from dlstudy.training import evaluate_model  # noqa: E402
from dlstudy.utils import (  # noqa: E402
    choose_device,
    collect_environment,
    seed_everything,
    write_json,
)


def _sha256(path: Path) -> str:
    """Identify the exact checkpoint bytes used for this one test result."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument(
        "--selection",
        type=Path,
        default=PROJECT_ROOT / "configs/final_selection.yaml",
        help="Version-controlled selection frozen before test access.",
    )
    parser.add_argument(
        "--confirm-final",
        action="store_true",
        help="Required acknowledgement that model/config selection is finished.",
    )
    arguments = parser.parse_args()
    if not arguments.confirm_final:
        raise SystemExit(
            "Refusing test access: finish selection on validation, then pass --confirm-final"
        )

    selection_path = arguments.selection
    if not selection_path.is_absolute():
        selection_path = PROJECT_ROOT / selection_path
    selection = load_yaml(selection_path)
    if not selection.get("frozen"):
        raise SystemExit("Refusing test access: final_selection.yaml is not frozen")

    run_directory = arguments.run_directory.resolve()
    output_path = run_directory / "test_metrics.json"
    if output_path.exists():
        raise SystemExit("test_metrics.json already exists; do not repeatedly tune on the test set")

    config = load_yaml(run_directory / "config.resolved.yaml")
    validate_config(config)
    with (run_directory / "summary.json").open("r", encoding="utf-8") as file:
        run_summary = json.load(file)
    fingerprint = semantic_fingerprint(config)
    expected_seeds = {int(seed) for seed in selection["seeds"]}
    if (
        selection.get("experiment_id") != config["experiment"]["id"]
        or selection.get("semantic_fingerprint") != fingerprint
        or selection.get("protocol_version") != config["experiment"]["protocol_version"]
        or selection.get("training_git_commit") != run_summary.get("git_commit")
        or int(config["train"]["seed"]) not in expected_seeds
        or run_summary.get("semantic_fingerprint") != fingerprint
    ):
        raise SystemExit("Run does not match the frozen final selection")
    seed_everything(int(config["train"]["seed"]), bool(config["train"]["deterministic"]))
    device = choose_device(config["train"]["device"])
    environment = collect_environment()
    # The selection commit can be newer than the training commit because it
    # records the validation decision. It must still be known, clean, and the
    # same on both test runs; aggregate_test_results.py checks that pairing.
    if environment["git_commit"] == "unknown" or environment["git_status"] != "":
        raise SystemExit("Final test must run from a known, clean Git commit")
    if device.type != "cuda" or not environment["cuda_available"]:
        raise SystemExit("Final official test must run on CUDA, as declared in the protocol")
    data = build_dataloaders(config, include_test=True)
    if data.test is None:
        raise SystemExit("A real CIFAR test set is required")
    checkpoint_path = run_directory / "checkpoints/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint["split_hash"] != data.split_hash:
        raise RuntimeError("Split checksum differs from the training checkpoint")
    checkpoint_config = checkpoint["config"]
    if semantic_fingerprint(checkpoint_config) != fingerprint:
        raise RuntimeError("Checkpoint config differs from the frozen run config")
    if (
        int(checkpoint_config["train"]["seed"]) != int(config["train"]["seed"])
        or checkpoint_config["experiment"]["id"] != config["experiment"]["id"]
        or int(checkpoint["epoch"]) != int(run_summary["best_epoch"])
        or float(checkpoint["val_accuracy"]) != float(run_summary["best_val_accuracy"])
    ):
        raise RuntimeError("Checkpoint identity does not match this seed/run summary")
    model = build_model(config["model"], data.num_classes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    criterion = build_criterion(config).to(device)
    loss, accuracy = evaluate_model(
        model,
        data.test,
        criterion,
        device,
        amp_enabled=bool(config["train"]["amp"]) and device.type == "cuda",
    )
    result = {
        "experiment_id": config["experiment"]["id"],
        "semantic_fingerprint": fingerprint,
        "protocol_version": config["experiment"]["protocol_version"],
        "seed": int(config["train"]["seed"]),
        "test_loss": loss,
        "test_accuracy": accuracy,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_sha256": _sha256(checkpoint_path),
        "evaluation_device": str(device),
        "evaluation_environment": environment,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_path, result)
    print(result)


if __name__ == "__main__":
    main()
