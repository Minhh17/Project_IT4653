"""Small YAML configuration helpers.

The project intentionally does not use Hydra or another configuration framework.
Nested dictionaries and explicit validation are less powerful, but every group
member can read the whole mechanism before the defence.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or inconsistent."""


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load one YAML file and require a dictionary at its root."""
    with Path(path).open("r", encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ConfigError("YAML root must be a mapping: {}".format(path))
    return value


def set_by_dotted_key(config: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``optimizer.lr`` as ``config['optimizer']['lr']``.

    Dotted keys keep matrix files compact without hiding how values are merged.
    """
    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        child = current.get(key)
        if not isinstance(child, dict):
            raise ConfigError("Unknown or non-mapping override path: {}".format(dotted_key))
        current = child
    if keys[-1] not in current:
        raise ConfigError("Unknown override key: {}".format(dotted_key))
    current[keys[-1]] = value


def parse_cli_overrides(items: Iterable[str]) -> Dict[str, Any]:
    """Parse ``--set key=value`` arguments using YAML scalar syntax."""
    result = {}
    for item in items:
        if "=" not in item:
            raise ConfigError("Override must have key=value form: {}".format(item))
        key, raw_value = item.split("=", 1)
        if not key:
            raise ConfigError("Override key cannot be empty")
        result[key] = yaml.safe_load(raw_value)
    return result


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep copy with dotted-key overrides applied."""
    resolved = copy.deepcopy(dict(config))
    for key, value in overrides.items():
        set_by_dotted_key(resolved, key, value)
    return resolved


def _require(config: Mapping[str, Any], section: str, key: str) -> Any:
    if section not in config or not isinstance(config[section], dict):
        raise ConfigError("Missing section: {}".format(section))
    if key not in config[section]:
        raise ConfigError("Missing key: {}.{}".format(section, key))
    return config[section][key]


def _integer(value: Any, name: str, minimum: int = None) -> int:
    """Require a real YAML integer; ``20.9`` and ``true`` are not integers here."""
    if type(value) is not int:  # exact type matters because bool is a subclass of int
        raise ConfigError("{} must be an integer".format(name))
    if minimum is not None and value < minimum:
        raise ConfigError("{} must be at least {}".format(name, minimum))
    return value


def _number(value: Any, name: str) -> float:
    """Require an unquoted YAML number instead of silently casting a string."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("{} must be a number".format(name))
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ConfigError("{} must be true or false, without quotes".format(name))
    return value


def _optional_positive_integer(value: Any, name: str) -> None:
    if value is not None:
        _integer(value, name, minimum=1)


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail early, before an expensive GPU run starts."""
    dataset = _require(config, "data", "dataset")
    if dataset not in {"cifar10", "cifar100", "fake"}:
        raise ConfigError("data.dataset must be cifar10, cifar100, or fake")
    data_root = _require(config, "data", "root")
    if not isinstance(data_root, str) or not data_root.strip():
        raise ConfigError("data.root must be a non-empty path or auto")

    _integer(_require(config, "data", "batch_size"), "data.batch_size", minimum=1)
    _integer(_require(config, "data", "image_size"), "data.image_size", minimum=1)
    _integer(_require(config, "data", "num_workers"), "data.num_workers", minimum=0)
    _integer(_require(config, "data", "split_seed"), "data.split_seed")
    _integer(_require(config, "data", "crop_padding"), "data.crop_padding", minimum=0)
    _integer(_require(config, "data", "fake_train_size"), "data.fake_train_size", minimum=1)
    _integer(_require(config, "data", "fake_val_size"), "data.fake_val_size", minimum=1)
    _boolean(_require(config, "data", "download"), "data.download")
    _boolean(_require(config, "data", "augmentation"), "data.augmentation")
    flip_probability = _number(
        _require(config, "data", "horizontal_flip_probability"),
        "data.horizontal_flip_probability",
    )
    if not 0.0 <= flip_probability <= 1.0:
        raise ConfigError("data.horizontal_flip_probability must be in [0, 1]")
    if _number(_require(config, "data", "color_jitter"), "data.color_jitter") < 0:
        raise ConfigError("data.color_jitter cannot be negative")

    if dataset.startswith("cifar"):
        train_size = _require(config, "data", "train_size")
        val_size = _require(config, "data", "val_size")
        _integer(train_size, "data.train_size", minimum=1)
        _integer(val_size, "data.val_size", minimum=1)
        if train_size <= 0 or val_size <= 0 or train_size + val_size > 50000:
            raise ConfigError("CIFAR train_size + val_size must be at most 50000")

    if _require(config, "model", "name") != "resnet18_cifar":
        raise ConfigError("model.name must be resnet18_cifar")
    normalization = _require(config, "model", "normalization")
    if normalization not in {"batch", "layer", "group"}:
        raise ConfigError("model.normalization must be batch, layer, or group")
    _integer(_require(config, "model", "base_channels"), "model.base_channels", minimum=1)
    _integer(
        _require(config, "model", "group_norm_groups"),
        "model.group_norm_groups",
        minimum=1,
    )
    dropout = _number(_require(config, "model", "dropout"), "model.dropout")
    if not 0.0 <= dropout < 1.0:
        raise ConfigError("model.dropout must be in [0, 1)")

    optimizer = _require(config, "optimizer", "name")
    valid_optimizers = {"sgd", "sgd_momentum", "nesterov", "rmsprop", "adam", "adamw"}
    if optimizer not in valid_optimizers:
        raise ConfigError("Unknown optimizer: {}".format(optimizer))
    if _number(_require(config, "optimizer", "lr"), "optimizer.lr") <= 0:
        raise ConfigError("optimizer.lr must be positive")
    if _number(_require(config, "optimizer", "weight_decay"), "optimizer.weight_decay") < 0:
        raise ConfigError("optimizer.weight_decay cannot be negative")
    momentum = _number(_require(config, "optimizer", "momentum"), "optimizer.momentum")
    alpha = _number(_require(config, "optimizer", "rmsprop_alpha"), "optimizer.rmsprop_alpha")
    if not 0.0 <= momentum < 1.0 or not 0.0 <= alpha < 1.0:
        raise ConfigError("optimizer momentum and RMSProp alpha must be in [0, 1)")
    if _number(_require(config, "optimizer", "eps"), "optimizer.eps") <= 0:
        raise ConfigError("optimizer.eps must be positive")
    betas = _require(config, "optimizer", "adam_betas")
    if not isinstance(betas, list) or len(betas) != 2:
        raise ConfigError("optimizer.adam_betas must contain two numbers")
    beta_values = [_number(value, "optimizer.adam_betas") for value in betas]
    if not all(0.0 <= value < 1.0 for value in beta_values):
        raise ConfigError("optimizer.adam_betas values must be in [0, 1)")

    scheduler = _require(config, "scheduler", "name")
    if scheduler not in {"constant", "step", "cosine"}:
        raise ConfigError("scheduler.name must be constant, step, or cosine")
    epochs = _integer(_require(config, "train", "epochs"), "train.epochs", minimum=1)
    warmup_epochs = _integer(
        _require(config, "scheduler", "warmup_epochs"),
        "scheduler.warmup_epochs",
        minimum=0,
    )
    if epochs <= 0 or warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ConfigError("Require train.epochs > scheduler.warmup_epochs >= 0")
    _integer(
        _require(config, "scheduler", "step_size_epochs"),
        "scheduler.step_size_epochs",
        minimum=1,
    )
    start_factor = _number(
        _require(config, "scheduler", "warmup_start_factor"),
        "scheduler.warmup_start_factor",
    )
    gamma = _number(_require(config, "scheduler", "gamma"), "scheduler.gamma")
    if not 0.0 < start_factor <= 1.0 or not 0.0 < gamma <= 1.0:
        raise ConfigError("warmup_start_factor and scheduler.gamma must be in (0, 1]")
    if _number(_require(config, "scheduler", "min_lr"), "scheduler.min_lr") < 0:
        raise ConfigError("scheduler.min_lr cannot be negative")
    _integer(_require(config, "train", "seed"), "train.seed")
    _integer(_require(config, "train", "log_every_steps"), "train.log_every_steps", minimum=1)
    _boolean(_require(config, "train", "deterministic"), "train.deterministic")
    _boolean(_require(config, "train", "amp"), "train.amp")
    if _require(config, "train", "device") not in {"auto", "cpu", "cuda"}:
        raise ConfigError("train.device must be auto, cpu, or cuda")
    convergence = _number(
        _require(config, "train", "convergence_accuracy"), "train.convergence_accuracy"
    )
    if not 0.0 <= convergence <= 1.0:
        raise ConfigError("train.convergence_accuracy must be in [0, 1]")

    early = _require(config, "regularization", "early_stopping")
    if not isinstance(early, dict):
        raise ConfigError("regularization.early_stopping must be a mapping")
    _boolean(early.get("enabled"), "early_stopping.enabled")
    monitor = early.get("monitor")
    if monitor not in {"val_loss", "val_accuracy"}:
        raise ConfigError("early_stopping.monitor must be val_loss or val_accuracy")
    _integer(early.get("patience"), "early_stopping.patience", minimum=1)
    if _number(early.get("min_delta"), "early_stopping.min_delta") < 0:
        raise ConfigError("early_stopping.min_delta cannot be negative")
    label_smoothing = _number(
        _require(config, "regularization", "label_smoothing"),
        "regularization.label_smoothing",
    )
    if not 0.0 <= label_smoothing < 1.0:
        raise ConfigError("regularization.label_smoothing must be in [0, 1)")

    _optional_positive_integer(config["debug"].get("max_train_batches"), "debug.max_train_batches")
    _optional_positive_integer(config["debug"].get("max_val_batches"), "debug.max_val_batches")
    _optional_positive_integer(config["debug"].get("train_samples"), "debug.train_samples")
    _boolean(config["debug"].get("use_train_as_val"), "debug.use_train_as_val")

    groups = _require(config, "experiment", "comparison_groups")
    if (
        not isinstance(groups, list)
        or not groups
        or not all(isinstance(item, str) for item in groups)
    ):
        raise ConfigError("experiment.comparison_groups must be a non-empty list")


def load_config(path: Path, cli_overrides: Iterable[str] = ()) -> Dict[str, Any]:
    """Load, override, and validate one training configuration."""
    config = load_yaml(path)
    config = apply_overrides(config, parse_cli_overrides(cli_overrides))
    validate_config(config)
    return config


def semantic_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash scientific settings while ignoring run identity and seed.

    Equal hashes mean two runs may be reused across comparison branches. The
    aggregator still checks both required seeds before reporting mean and std.
    """
    value = copy.deepcopy(dict(config))
    value.get("train", {}).pop("seed", None)
    experiment = value.get("experiment", {})
    for key in ("id", "label", "attempt", "comparison_groups", "output_dir"):
        experiment.pop(key, None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_matrix(path: Path) -> Dict[str, Any]:
    """Load and perform lightweight validation of a work-queue matrix."""
    matrix = load_yaml(path)
    required = {"name", "owner", "base_config", "approved", "seeds", "experiments"}
    missing = sorted(required.difference(matrix))
    if missing:
        raise ConfigError("Matrix {} misses: {}".format(path, ", ".join(missing)))
    if type(matrix["approved"]) is not bool:
        raise ConfigError("Matrix approved must be true or false, without quotes")
    if not isinstance(matrix["experiments"], list) or not matrix["experiments"]:
        raise ConfigError("Matrix experiments must be a non-empty list")
    if not matrix["seeds"] or not all(type(seed) is int for seed in matrix["seeds"]):
        raise ConfigError("Matrix seeds must be a non-empty integer list")
    ids = [item.get("id") for item in matrix["experiments"]]
    if None in ids or len(ids) != len(set(ids)):
        raise ConfigError("Experiment ids must be present and unique within a matrix")
    return matrix
