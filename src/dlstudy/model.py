"""A readable ResNet-18 adapted to 32x32 CIFAR images."""

from __future__ import annotations

from typing import Callable, Dict, List, Type

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    """Layer normalization for NCHW feature maps.

    BatchNorm estimates statistics across a mini-batch. Here, each image is
    normalized using all CxHxW values of its own feature map, so the result does
    not depend on batch size. Learnable scale/bias remain per channel.
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=(1, 2, 3), keepdim=True)
        variance = inputs.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        normalized = (inputs - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.weight + self.bias


def make_norm_factory(name: str, requested_groups: int) -> Callable[[int], nn.Module]:
    """Return a function so every ResNet normalization layer is replaced alike."""
    if name == "batch":
        return lambda channels: nn.BatchNorm2d(channels)
    if name == "layer":
        return lambda channels: LayerNorm2d(channels)
    if name == "group":

        def group_norm(channels: int) -> nn.Module:
            # Pick the largest valid group count no greater than the requested one.
            groups = min(requested_groups, channels)
            while channels % groups != 0:
                groups -= 1
            return nn.GroupNorm(groups, channels)

        return group_norm
    raise ValueError("Unknown normalization: {}".format(name))


class BasicBlock(nn.Module):
    """The two-convolution residual block used by ResNet-18."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm: Callable[[int], nn.Module],
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = norm(out_channels)

        # If shape changes, a 1x1 convolution makes the shortcut compatible.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        outputs = self.relu(self.norm1(self.conv1(inputs)))
        outputs = self.norm2(self.conv2(outputs))
        # The identity path lets gradients bypass the two learned convolutions.
        outputs = self.relu(outputs + residual)
        return outputs


class CifarResNet(nn.Module):
    """ResNet with a CIFAR stem: 3x3 stride 1 and no initial max-pooling."""

    def __init__(
        self,
        block: Type[BasicBlock],
        blocks_per_stage: List[int],
        num_classes: int,
        base_channels: int,
        normalization: str,
        group_norm_groups: int,
        dropout: float,
    ):
        super().__init__()
        norm = make_norm_factory(normalization, group_norm_groups)
        self.current_channels = base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            norm(base_channels),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(block, base_channels, blocks_per_stage[0], 1, norm)
        self.stage2 = self._make_stage(block, base_channels * 2, blocks_per_stage[1], 2, norm)
        self.stage3 = self._make_stage(block, base_channels * 4, blocks_per_stage[2], 2, norm)
        self.stage4 = self._make_stage(block, base_channels * 8, blocks_per_stage[3], 2, norm)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Dropout is kept at one fixed location for the regularization ablation.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(base_channels * 8 * block.expansion, num_classes)
        self._initialize_weights()

    def _make_stage(self, block, out_channels, count, stride, norm):
        layers = [block(self.current_channels, out_channels, stride, norm)]
        self.current_channels = out_channels * block.expansion
        for _ in range(1, count):
            layers.append(block(self.current_channels, out_channels, 1, norm))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, LayerNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)  # 32x32
        outputs = self.stage1(outputs)  # 32x32
        outputs = self.stage2(outputs)  # 16x16
        outputs = self.stage3(outputs)  # 8x8
        outputs = self.stage4(outputs)  # 4x4
        outputs = self.pool(outputs)
        outputs = torch.flatten(outputs, 1)
        return self.classifier(self.dropout(outputs))


def build_model(model_config: Dict, num_classes: int) -> nn.Module:
    if model_config["name"] != "resnet18_cifar":
        raise ValueError("Only resnet18_cifar is supported in this controlled study")
    return CifarResNet(
        block=BasicBlock,
        blocks_per_stage=[2, 2, 2, 2],
        num_classes=num_classes,
        base_channels=int(model_config["base_channels"]),
        normalization=model_config["normalization"],
        group_norm_groups=int(model_config["group_norm_groups"]),
        dropout=float(model_config["dropout"]),
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
