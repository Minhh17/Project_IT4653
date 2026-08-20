"""Optimizer and learning-rate schedule definitions kept in one visible file."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torch import nn


def _parameter_groups(model: nn.Module, weight_decay: float):
    """Apply decay to learned Conv/Linear weights, not biases or norm scales.

    This common convention avoids shrinking BatchNorm/LayerNorm/GroupNorm
    affine parameters. The rule is identical for every optimizer and run.
    """
    decay_parameters = []
    no_decay_parameters = []
    for module in model.modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            is_learned_weight = parameter_name == "weight" and isinstance(
                module, (nn.Conv2d, nn.Linear)
            )
            if is_learned_weight:
                decay_parameters.append(parameter)
            else:
                no_decay_parameters.append(parameter)
    return [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Map each experiment name to its exact PyTorch optimizer arguments."""
    values = config["optimizer"]
    name = values["name"]
    parameters = _parameter_groups(model, float(values["weight_decay"]))
    common = {"lr": float(values["lr"])}

    if name == "sgd":
        return torch.optim.SGD(parameters, momentum=0.0, nesterov=False, **common)
    if name == "sgd_momentum":
        return torch.optim.SGD(
            parameters, momentum=float(values["momentum"]), nesterov=False, **common
        )
    if name == "nesterov":
        return torch.optim.SGD(
            parameters, momentum=float(values["momentum"]), nesterov=True, **common
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            alpha=float(values["rmsprop_alpha"]),
            momentum=float(values["momentum"]),
            eps=float(values["eps"]),
            **common,
        )
    if name in {"adam", "adamw"}:
        optimizer_class = torch.optim.Adam if name == "adam" else torch.optim.AdamW
        return optimizer_class(
            parameters,
            betas=tuple(float(item) for item in values["adam_betas"]),
            eps=float(values["eps"]),
            **common,
        )
    raise ValueError("Unknown optimizer: {}".format(name))


def build_criterion(config: Dict[str, Any]) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=float(config["regularization"]["label_smoothing"]))


class LearningRateSchedule:
    """Set learning rate once per optimizer update.

    Writing this small controller explicitly makes warm-up order and schedule
    formulas easy to defend. ``global_step`` starts at zero.
    """

    def __init__(self, optimizer, config: Dict[str, Any], steps_per_epoch: int):
        self.optimizer = optimizer
        self.values = config["scheduler"]
        self.total_steps = int(config["train"]["epochs"]) * steps_per_epoch
        self.warmup_steps = int(self.values["warmup_epochs"]) * steps_per_epoch
        self.steps_per_epoch = steps_per_epoch
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def _warmup_factor(self, step: int) -> float:
        start = float(self.values["warmup_start_factor"])
        if self.warmup_steps <= 1:
            return 1.0
        progress = step / float(self.warmup_steps - 1)
        return start + (1.0 - start) * progress

    def factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return self._warmup_factor(step)

        name = self.values["name"]
        post_warmup_step = step - self.warmup_steps
        if name == "constant":
            return 1.0
        if name == "step":
            epoch_after_warmup = post_warmup_step // self.steps_per_epoch
            decays = epoch_after_warmup // int(self.values["step_size_epochs"])
            return float(self.values["gamma"]) ** decays
        if name == "cosine":
            remaining = max(1, self.total_steps - self.warmup_steps - 1)
            progress = min(1.0, post_warmup_step / float(remaining))
            minimum_ratio = float(self.values["min_lr"]) / self.base_lrs[0]
            return (
                minimum_ratio + (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * progress)) / 2.0
            )
        raise ValueError("Unknown scheduler: {}".format(name))

    def apply(self, step: int) -> float:
        factor = self.factor(step)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor
        return self.optimizer.param_groups[0]["lr"]
