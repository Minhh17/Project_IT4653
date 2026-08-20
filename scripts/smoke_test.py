#!/usr/bin/env python3
"""Exercise data -> model -> backward -> validation -> checkpoint on tiny fake data."""

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlstudy.config import load_config  # noqa: E402
from dlstudy.training import run_training  # noqa: E402


def main() -> None:
    temporary_output = Path(tempfile.mkdtemp(prefix="it4653_smoke_"))
    overrides = [
        "experiment.id=smoke",
        "experiment.label=Smoke test",
        "experiment.comparison_groups=[development]",
        "experiment.output_dir={}".format(temporary_output),
        "data.dataset=fake",
        "data.batch_size=16",
        "data.num_workers=0",
        "data.fake_train_size=32",
        "data.fake_val_size=16",
        "model.base_channels=8",
        "train.epochs=1",
        "train.device=cpu",
        "train.amp=false",
        "scheduler.warmup_epochs=0",
        "debug.max_train_batches=2",
        "debug.max_val_batches=1",
    ]
    config = load_config(PROJECT_ROOT / "configs/base.yaml", overrides)
    summary = run_training(config, PROJECT_ROOT)
    print("Smoke test passed. Artifacts: {}".format(temporary_output))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
