"""Small CNN written from scratch (no torchvision/timm, no pretrained weights)."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ConvBlock(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2 -> MaxPool 2x2."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class SmallCNN(nn.Module):
    """Stack of ConvBlocks -> global average pooling -> dropout -> linear classifier."""

    def __init__(self, num_classes: int, channels: list[int], dropout: float = 0.3,
                 in_channels: int = 3) -> None:
        super().__init__()
        blocks = []
        prev = in_channels
        for ch in channels:
            blocks.append(ConvBlock(prev, ch))
            prev = ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(prev, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(self.dropout(x))


def build_model(cfg: dict[str, Any]) -> SmallCNN:
    """Create the model described by config['model'] with len(config['classes']) outputs."""
    m = cfg["model"]
    return SmallCNN(num_classes=len(cfg["classes"]), channels=list(m["channels"]),
                    dropout=float(m["dropout"]))


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
