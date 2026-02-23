"""Checkpoint save / load manager."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints.

    Example:
        mgr = CheckpointManager(Path("./checkpoints"))
        path = mgr.save(model, optimizer, epoch=5, tag="best")
        state = mgr.load(path)
        model.load_state_dict(state["model"])
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        global_step: int = 0,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        tag: str = "latest",
        extra: Optional[Dict] = None,
    ) -> Path:
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state["scheduler"] = scheduler.state_dict()
        if scaler is not None and hasattr(scaler, "state_dict"):
            state["scaler"] = scaler.state_dict()
        if extra:
            state.update(extra)

        path = self.checkpoint_dir / f"checkpoint_{tag}.pt"
        torch.save(state, path)
        logger.info("Checkpoint saved → %s (epoch %d)", path, epoch)
        return path

    def load(self, path: str, map_location: str = "cpu") -> Dict[str, Any]:
        state = torch.load(str(path), map_location=map_location, weights_only=True)
        logger.info("Checkpoint loaded from %s", path)
        return state

    def latest(self) -> Optional[Path]:
        p = self.checkpoint_dir / "checkpoint_latest.pt"
        return p if p.exists() else None

    def best(self) -> Optional[Path]:
        p = self.checkpoint_dir / "checkpoint_best.pt"
        return p if p.exists() else None
