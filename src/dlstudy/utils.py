"""Reproducibility, environment metadata, and small file helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every random-number generator used by this project.

    A seed controls a pseudo-random sequence; it does not guarantee identical
    floating-point results across different GPU types or library versions.
    Those versions are therefore recorded for every run as well.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # warn_only keeps a long experiment alive if one rare CUDA operation
        # has no deterministic implementation, while still making it visible.
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_data_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic NumPy/Python seed."""
    del worker_id  # worker identity is already encoded in torch.initial_seed().
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def safe_name(value: str) -> str:
    """Make a human label safe to use as one directory name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "run"


def _git_value(arguments: list) -> str:
    try:
        return subprocess.check_output(
            ["git"] + arguments,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_environment() -> Dict[str, Any]:
    """Capture enough context to interpret reproducibility and timing."""
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = "unavailable"

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_name": gpu_name,
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_status": _git_value(["status", "--short"]),
    }


def write_json(path: Path, value: Any) -> None:
    """Write atomically so a stopped notebook does not leave half a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(str(temporary), str(path))


def write_yaml(path: Path, value: Any) -> None:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        yaml.safe_dump(value, file, sort_keys=False, allow_unicode=True)
    os.replace(str(temporary), str(path))
