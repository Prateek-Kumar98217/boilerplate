"""
Abstract base model for all PyTorch models in this boilerplate.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseModel(nn.Module, abc.ABC):
    """
    All custom models inherit from this.
    Provides:
    - forward() contract
    - parameter count helpers
    - gradient checkpointing hook
    - weight initialization scaffold
    """

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any: ...

    # ── Parameter utilities ───────────────────────────────────────────

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameter_summary(self) -> Dict[str, Any]:
        total = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
            "trainable_pct": round(100 * trainable / max(total, 1), 2),
        }

    # ── Initialization ────────────────────────────────────────────────

    def init_weights(self) -> None:
        """Override in subclasses for custom weight initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ── Gradient checkpointing ────────────────────────────────────────

    def enable_gradient_checkpointing(self) -> None:
        """Override for models that support gradient checkpointing."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement gradient checkpointing."
        )

    # ── Device helpers ────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
