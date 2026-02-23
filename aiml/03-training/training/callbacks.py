"""
Training callbacks — EarlyStopping, ModelCheckpoint, LRLogger, WandB, TensorBoard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Base callback ─────────────────────────────────────────────────────


class Callback:
    def on_train_begin(self, trainer) -> None:
        pass

    def on_train_end(self, trainer) -> None:
        pass

    def on_epoch_begin(self, epoch: int, trainer) -> None:
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer) -> None:
        pass

    def on_batch_end(self, step: int, loss: float, trainer) -> None:
        pass


class CallbackList:
    def __init__(self, callbacks: List[Callback]) -> None:
        self.callbacks = callbacks

    def on_train_begin(self, trainer) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, epoch, trainer) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(epoch, trainer)

    def on_epoch_end(self, epoch, metrics, trainer) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics, trainer)

    def on_batch_end(self, step, loss, trainer) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(step, loss, trainer)


# ── Early stopping ────────────────────────────────────────────────────


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric stops improving.

    Args:
        monitor:   Metric key to watch (default: "val_loss").
        patience:  Number of epochs to wait.
        mode:      "min" or "max".
        min_delta: Minimum improvement to count as improvement.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 1e-4,
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self._best: Optional[float] = None
        self._counter = 0

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer) -> None:
        value = metrics.get(self.monitor)
        if value is None:
            return

        improved = (
            self._best is None
            or (self.mode == "min" and value < self._best - self.min_delta)
            or (self.mode == "max" and value > self._best + self.min_delta)
        )
        if improved:
            self._best = value
            self._counter = 0
        else:
            self._counter += 1
            logger.info(
                "EarlyStopping: %s did not improve (%d/%d)",
                self.monitor,
                self._counter,
                self.patience,
            )
            if self._counter >= self.patience:
                trainer.stop_training()


# ── Model checkpoint ──────────────────────────────────────────────────


class ModelCheckpoint(Callback):
    """
    Save the best model checkpoint based on a monitored metric.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        save_last: bool = True,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.save_last = save_last
        self._best: Optional[float] = None

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer) -> None:
        value = metrics.get(self.monitor)
        is_best = value is not None and (
            self._best is None
            or (self.mode == "min" and value < self._best)
            or (self.mode == "max" and value > self._best)
        )
        if is_best:
            self._best = value
            path = trainer.save_checkpoint(tag="best")
            logger.info(
                "ModelCheckpoint: new best %s=%.4f saved to %s",
                self.monitor,
                value,
                path,
            )
        if self.save_last:
            trainer.save_checkpoint(tag="last")


# ── LR Logger ────────────────────────────────────────────────────────


class LRLogger(Callback):
    """Log current learning rate at the end of each epoch."""

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer) -> None:
        lrs = [pg["lr"] for pg in trainer.optimizer.param_groups]
        logger.info("Epoch %d | LR: %s", epoch + 1, lrs)


# ── WandB callback ────────────────────────────────────────────────────


class WandBCallback(Callback):
    """
    Log metrics to Weights & Biases.
    Requires: pip install wandb and WANDB_API_KEY in environment.
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
    ) -> None:
        self.project = project
        self.name = name
        self.config = config or {}

    def on_train_begin(self, trainer) -> None:
        try:
            import wandb

            wandb.init(
                project=self.project, name=self.name, config=self.config, reinit=True
            )
        except ImportError:
            logger.warning("wandb not installed — WandBCallback disabled.")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer) -> None:
        try:
            import wandb

            wandb.log({"epoch": epoch + 1, **metrics})
        except Exception:
            pass

    def on_train_end(self, trainer) -> None:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass


# ── TensorBoard callback ──────────────────────────────────────────────


class TensorBoardCallback(Callback):
    """Log metrics to TensorBoard."""

    def __init__(self, log_dir: str = "./tb_logs") -> None:
        self.log_dir = log_dir
        self._writer = None

    def on_train_begin(self, trainer) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=self.log_dir)
        except ImportError:
            logger.warning("tensorboard not installed — TensorBoardCallback disabled.")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer) -> None:
        if self._writer:
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, epoch)

    def on_train_end(self, trainer) -> None:
        if self._writer:
            self._writer.close()
