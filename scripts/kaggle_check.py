#!/usr/bin/env python3
"""One quick preflight: GPU, Git commit, and attached CIFAR layout."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dlstudy.config import load_config  # noqa: E402
from dlstudy.data import resolve_cifar_root  # noqa: E402
from dlstudy.utils import collect_environment  # noqa: E402


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs/base.yaml")
    environment = collect_environment()
    if not environment["cuda_available"]:
        raise SystemExit("CUDA is unavailable: choose Accelerator > GPU in Kaggle Settings")
    if environment["git_commit"] == "unknown" or environment["git_status"] != "":
        raise SystemExit("Clone/checkout one known commit and keep the worktree clean")

    root = resolve_cifar_root(config["data"], config["data"]["dataset"])
    print("Kaggle preflight OK")
    print("GPU: {}".format(environment["gpu_name"]))
    print("Git commit: {}".format(environment["git_commit"]))
    print("CIFAR root for torchvision: {}".format(root))


if __name__ == "__main__":
    main()
