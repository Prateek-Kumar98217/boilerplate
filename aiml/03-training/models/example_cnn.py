"""
Example ResNet-style CNN for image classification.
Demonstrates: residual blocks, batch norm, adaptive pooling, dropout.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel


class ResidualBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout2d(dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ConvNet(BaseModel):
    """
    Configurable residual CNN classifier.

    Args:
        in_channels:  Input channels (3 for RGB, 1 for grayscale).
        num_classes:  Number of output classes.
        channel_list: Output channels per stage.
        dropout:      Dropout probability.

    Example:
        model = ConvNet(in_channels=3, num_classes=10)
        out = model(torch.randn(4, 3, 32, 32))   # (4, 10)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        channel_list: List[int] = (64, 128, 256, 512),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channel_list[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channel_list[0]),
            nn.ReLU(inplace=True),
        )

        layers = []
        in_ch = channel_list[0]
        for out_ch in channel_list[1:]:
            layers.append(ResidualBlock(in_ch, out_ch, stride=2, dropout=dropout))
            layers.append(ResidualBlock(out_ch, out_ch, dropout=dropout))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channel_list[-1], num_classes),
        )
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.backbone(x)
        x = self.pool(x)
        return self.classifier(x)

    def enable_gradient_checkpointing(self) -> None:
        """Use torch.utils.checkpoint on backbone blocks."""
        import torch.utils.checkpoint as ckpt

        for block in self.backbone:
            if isinstance(block, ResidualBlock):
                original_fwd = block.forward
                block.forward = lambda x, fn=original_fwd: ckpt.checkpoint(
                    fn, x, use_reentrant=False
                )
