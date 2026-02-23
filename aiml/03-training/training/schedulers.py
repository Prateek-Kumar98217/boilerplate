"""
LR scheduler factory.

Supported: cosine | cosine_warmup | step | plateau | onecycle | linear_warmup | none
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Cosine decay with linear warmup (HuggingFace style)."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int
) -> LambdaLR:
    """Linear decay with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def create_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    num_training_steps: Optional[int] = None,
    num_warmup_steps: int = 0,
    warmup_ratio: float = 0.0,
    step_size: int = 10,
    gamma: float = 0.1,
    lr_min: float = 1e-6,
    max_lr: float = 1e-3,
):
    """
    Factory for LR schedulers.

    Args:
        scheduler_type: One of cosine | cosine_warmup | linear_warmup |
                        step | plateau | onecycle | none.
        num_training_steps: Total training steps (required for warmup variants).
        num_warmup_steps:   Warmup steps (overridden by warmup_ratio if > 0).
    """
    if warmup_ratio > 0 and num_training_steps:
        num_warmup_steps = int(warmup_ratio * num_training_steps)

    if scheduler_type == "none" or scheduler_type is None:
        return None

    elif scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=num_training_steps or 100, eta_min=lr_min
        )

    elif scheduler_type == "cosine_warmup":
        assert num_training_steps, "num_training_steps required for cosine_warmup"
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, lr_min / max_lr
        )

    elif scheduler_type == "linear_warmup":
        assert num_training_steps, "num_training_steps required for linear_warmup"
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps
        )

    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == "plateau":
        sch = ReduceLROnPlateau(
            optimizer, mode="min", factor=gamma, patience=step_size, min_lr=lr_min
        )
        sch.step_on_batch = False  # type: ignore[attr-defined]
        return sch

    elif scheduler_type == "onecycle":
        assert num_training_steps, "num_training_steps required for onecycle"
        sch = OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=num_training_steps, pct_start=0.1
        )
        sch.step_on_batch = True  # type: ignore[attr-defined]
        return sch

    elif scheduler_type == "cosine_restarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=step_size, eta_min=lr_min)

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
