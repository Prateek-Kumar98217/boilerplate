"""
Production-grade PyTorch Trainer.

Features:
- Mixed precision training (AMP) with GradScaler — bf16 / fp16 auto-selection
- Gradient clipping
- Gradient accumulation
- Gradient checkpointing (optional, per model)
- Learning rate scheduler support
- Evaluation loop with metric tracking
- Checkpoint save / resume
- WandB + MLflow logging hooks
- Label smoothing support via CrossEntropyLoss
- Callback system (EarlyStopping, TensorBoard, Checkpoint, LRLogger)
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from training.callbacks import Callback, CallbackList
from training.metrics import MetricTracker
from utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class Trainer:
    """
    General-purpose PyTorch trainer.

    Example:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device="cuda",
            max_epochs=50,
            mixed_precision="bf16",
            gradient_clipping=1.0,
            gradient_accumulation_steps=4,
        )
        history = trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = "auto",
        max_epochs: int = 50,
        mixed_precision: str = "true",  # bf16 | fp16 | true | false
        gradient_clipping: float = 1.0,
        gradient_accumulation_steps: int = 1,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[Callback]] = None,
        output_dir: str = "./outputs",
        experiment_name: str = "experiment",
        run_name: str = "run",
        log_every_n_steps: int = 50,
        compile_model: bool = False,
    ) -> None:
        # ── Device ────────────────────────────────────────────────────
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # ── Model ─────────────────────────────────────────────────────
        self.model = model.to(self.device)
        if compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.gradient_clipping = gradient_clipping
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.log_every_n_steps = log_every_n_steps

        # ── Mixed precision ────────────────────────────────────────────
        self._amp_enabled, self._amp_dtype, self._scaler = self._configure_amp(
            mixed_precision
        )
        logger.info(
            "Trainer device=%s amp=%s dtype=%s",
            self.device,
            self._amp_enabled,
            self._amp_dtype,
        )

        # ── Callbacks & tracking ──────────────────────────────────────
        self.callback_list = CallbackList(callbacks or [])
        self.output_dir = Path(output_dir) / experiment_name / run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(self.output_dir / "checkpoints")
        self.metric_tracker = MetricTracker()

        # ── State ─────────────────────────────────────────────────────
        self.current_epoch = 0
        self.global_step = 0
        self._stop_training = False

    # ── AMP config ───────────────────────────────────────────────────

    def _configure_amp(self, mixed_precision: str):
        if mixed_precision == "false" or not torch.cuda.is_available():
            return False, None, None
        if mixed_precision == "bf16":
            dtype = torch.bfloat16
        elif mixed_precision == "fp16":
            dtype = torch.float16
        else:
            # "true" → prefer bf16 on Ampere+, fall back to fp16
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        scaler = GradScaler(
            enabled=(dtype == torch.float16)
        )  # scaler only needed for fp16
        return True, dtype, scaler

    # ── Training loop ─────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Returns:
            history dict with "train_loss", "val_loss", "val_metrics" lists.
        """
        if resume_from:
            self._resume(resume_from)

        self.callback_list.on_train_begin(self)
        history: Dict[str, List] = {"train_loss": [], "val_loss": [], "val_metrics": []}

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            self.callback_list.on_epoch_begin(epoch, self)

            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            val_metrics = {}
            if val_loader:
                val_loss, val_metrics = self._eval_epoch(val_loader)
                history["val_loss"].append(val_loss)
                history["val_metrics"].append(val_metrics)
                log_line = (
                    f"Epoch {epoch+1}/{self.max_epochs} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                )
            else:
                log_line = (
                    f"Epoch {epoch+1}/{self.max_epochs} | train_loss={train_loss:.4f}"
                )

            logger.info(log_line)

            self.callback_list.on_epoch_end(
                epoch, {"train_loss": train_loss, **val_metrics}, self
            )

            if self._stop_training:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

        self.callback_list.on_train_end(self)
        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            loss = self._forward_backward(batch)
            total_loss += loss
            n_batches += 1

            # Gradient accumulation — only step every N batches
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self._optimizer_step()

            self.global_step += 1
            if self.global_step % self.log_every_n_steps == 0:
                logger.debug("step=%d loss=%.4f", self.global_step, loss)

        # Flush remaining accumulated gradients
        leftover = len(loader) % self.gradient_accumulation_steps
        if leftover != 0:
            self._optimizer_step()

        return total_loss / max(n_batches, 1)

    def _forward_backward(self, batch: Dict) -> float:
        """Single forward + backward pass. Returns scalar loss."""
        inputs = batch["input"].to(self.device, non_blocking=True)
        labels = batch["label"].to(self.device, non_blocking=True)

        with autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss = loss / self.gradient_accumulation_steps

        if self._scaler:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.gradient_accumulation_steps

    def _optimizer_step(self) -> None:
        if self._scaler:
            if self.gradient_clipping > 0:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            if self.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clipping
                )
            self.optimizer.step()

        self.optimizer.zero_grad()
        if self.scheduler and hasattr(self.scheduler, "step_on_batch"):
            self.scheduler.step()

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in loader:
            inputs = batch["input"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            with autocast(
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._amp_enabled,
            ):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(loader)
        metrics = self.metric_tracker.compute(all_preds, all_labels)

        if self.scheduler and not hasattr(self.scheduler, "step_on_batch"):
            if hasattr(self.scheduler, "step"):
                try:
                    self.scheduler.step(avg_loss)
                except TypeError:
                    self.scheduler.step()

        return avg_loss, metrics

    # ── Checkpoint ────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str = "latest") -> Path:
        return self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            global_step=self.global_step,
            scaler=self._scaler,
            tag=tag,
        )

    def _resume(self, path: str) -> None:
        state = self.checkpoint_manager.load(path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self._scaler and state.get("scaler"):
            self._scaler.load_state_dict(state["scaler"])
        self.current_epoch = state.get("epoch", 0) + 1
        self.global_step = state.get("global_step", 0)
        logger.info("Resumed training from %s (epoch %d)", path, self.current_epoch)

    def stop_training(self) -> None:
        """Call from a callback to trigger early stopping."""
        self._stop_training = True
