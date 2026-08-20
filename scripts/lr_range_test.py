#!/usr/bin/env python3
"""Increase LR exponentially and record where training loss falls/diverges."""

import argparse
import csv
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "it4653_matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlstudy.config import load_config  # noqa: E402
from dlstudy.data import build_dataloaders  # noqa: E402
from dlstudy.model import build_model  # noqa: E402
from dlstudy.optimization import build_criterion, build_optimizer  # noqa: E402
from dlstudy.utils import (  # noqa: E402
    choose_device,
    collect_environment,
    safe_name,
    seed_everything,
    write_json,
    write_yaml,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/base.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--start-lr", type=float, default=1e-6)
    parser.add_argument("--end-lr", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tag", default="pilot1", help="Unique label; prevents silent overwrite")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/lr_range")
    arguments = parser.parse_args()
    if arguments.start_lr <= 0 or arguments.end_lr <= arguments.start_lr or arguments.steps < 2:
        raise SystemExit("Require 0 < start-lr < end-lr and steps >= 2")

    config_path = arguments.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_config(config_path, arguments.set)
    # The range test owns the LR value; all other optimizer settings remain fixed.
    config["optimizer"]["lr"] = arguments.start_lr
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(
        "{}_{}_seed{}".format(arguments.tag, config["optimizer"]["name"], config["train"]["seed"])
    )
    csv_path = arguments.output_dir / (stem + ".csv")
    figure_path = arguments.output_dir / (stem + ".png")
    metadata_path = arguments.output_dir / (stem + ".metadata.json")
    config_output_path = arguments.output_dir / (stem + ".config.yaml")
    if any(path.exists() for path in (csv_path, figure_path, metadata_path, config_output_path)):
        raise SystemExit("LR finder artifact exists; choose a new --tag instead of overwriting")

    seed_everything(int(config["train"]["seed"]), bool(config["train"]["deterministic"]))
    device = choose_device(config["train"]["device"])
    data = build_dataloaders(config, include_test=False)
    model = build_model(config["model"], data.num_classes).to(device)
    criterion = build_criterion(config).to(device)
    optimizer = build_optimizer(model, config)
    amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    multiplier = (arguments.end_lr / arguments.start_lr) ** (1.0 / (arguments.steps - 1))
    beta = 0.98
    moving_average = 0.0
    best_smoothed_loss = float("inf")
    records = []
    stop_reason = "requested_steps_completed"
    iterator = iter(data.train)

    model.train()
    for step in range(arguments.steps):
        try:
            images, targets = next(iterator)
        except StopIteration:
            iterator = iter(data.train)
            images, targets = next(iterator)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        learning_rate = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            loss = criterion(model(images), targets)
        raw_loss = float(loss.item())
        if not math.isfinite(raw_loss):
            stop_reason = "non_finite_raw_loss"
            print("Stopping LR finder: non-finite loss at step {}".format(step))
            break
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        moving_average = beta * moving_average + (1.0 - beta) * raw_loss
        # Bias correction matters during the first few steps when the EMA starts at zero.
        smoothed_loss = moving_average / (1.0 - beta ** (step + 1))
        if not math.isfinite(smoothed_loss):
            stop_reason = "non_finite_smoothed_loss"
            print("Stopping LR finder: non-finite smoothed loss at step {}".format(step))
            break
        best_smoothed_loss = min(best_smoothed_loss, smoothed_loss)
        records.append(
            {
                "step": step,
                "learning_rate": learning_rate,
                "loss": raw_loss,
                "smoothed_loss": smoothed_loss,
            }
        )

        # Stop after clear divergence to avoid wasting data/GPU time.
        if step > 10 and smoothed_loss > 4 * best_smoothed_loss:
            stop_reason = "loss_diverged"
            break
        new_learning_rate = learning_rate * multiplier
        for group in optimizer.param_groups:
            group["lr"] = new_learning_rate

    if not records:
        raise FloatingPointError("LR finder produced no finite loss; inspect model and optimizer")

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    write_yaml(config_output_path, config)
    best_record = min(records, key=lambda item: item["smoothed_loss"])
    write_json(
        metadata_path,
        {
            "tag": arguments.tag,
            "start_lr": arguments.start_lr,
            "end_lr": arguments.end_lr,
            "requested_steps": arguments.steps,
            "completed_steps": len(records),
            "stop_reason": stop_reason,
            "last_finite_lr": records[-1]["learning_rate"],
            "ema_beta": beta,
            "lowest_smoothed_loss_lr": best_record["learning_rate"],
            "environment": collect_environment(),
        },
    )

    plt.figure(figsize=(7, 4))
    plt.plot(
        [item["learning_rate"] for item in records], [item["smoothed_loss"] for item in records]
    )
    plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("Exponentially smoothed training loss")
    plt.title("LR range test: {}".format(config["optimizer"]["name"]))
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=180)
    plt.close()
    print("Wrote {} and {}".format(csv_path, figure_path))
    print("Choose LR from the stable descending region; do not tune it on the test set.")


if __name__ == "__main__":
    main()
