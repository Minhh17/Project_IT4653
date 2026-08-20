#!/usr/bin/env python3
"""Try to memorize 128 CIFAR images as a model/data/gradient sanity check."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlstudy.config import load_config  # noqa: E402
from dlstudy.training import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    output = arguments.output_dir or Path(tempfile.mkdtemp(prefix="it4653_overfit_"))
    config = load_config(
        PROJECT_ROOT / "configs/base.yaml",
        [
            "experiment.id=overfit_128",
            "experiment.label=Overfit 128 samples",
            "experiment.comparison_groups=[development]",
            "experiment.output_dir={}".format(output),
            "data.augmentation=false",
            "data.batch_size=64",
            "data.num_workers=0",
            "optimizer.name=adam",
            "optimizer.lr=0.001",
            "optimizer.weight_decay=0.0",
            "scheduler.name=constant",
            "scheduler.warmup_epochs=0",
            "train.epochs={}".format(arguments.epochs),
            "debug.train_samples=128",
            "debug.use_train_as_val=true",
        ],
    )
    summary = run_training(config, PROJECT_ROOT)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["clean_train_accuracy"] < 0.90:
        raise SystemExit("Pipeline ran, but did not reach 90% train accuracy; inspect the curve.")
    print("Overfit check passed: final train accuracy >= 90%")


if __name__ == "__main__":
    main()
