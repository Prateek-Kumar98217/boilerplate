"""
End-to-end image classification training example.

Run:
    python examples/train_classification.py
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.optim import AdamW

from config import get_settings
from data.dataloader import create_dataloaders
from data.dataset import ArrayDataset
from models.example_cnn import ConvNet
from training.callbacks import (
    EarlyStopping,
    LRLogger,
    ModelCheckpoint,
    TensorBoardCallback,
)
from training.schedulers import create_scheduler
from training.trainer import Trainer
from utils.cuda_utils import log_gpu_memory, set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    settings = get_settings()
    set_seed(settings.tracking.seed, settings.tracking.deterministic)

    # ── Synthetic dataset (replace with real data) ────────────────────
    import numpy as np
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=2000, n_features=3 * 32 * 32, n_informative=100, random_state=42
    )
    X = X.reshape(-1, 3, 32, 32).astype("float32")
    split = int(0.8 * len(X))
    train_ds = ArrayDataset(X[:split], y[:split])
    val_ds = ArrayDataset(X[split:], y[split:])

    # ── DataLoaders ───────────────────────────────────────────────────
    loaders = create_dataloaders(
        train_ds,
        val_ds,
        batch_size=settings.loop.batch_size,
        num_workers=settings.hardware.num_workers,
        pin_memory=settings.hardware.pin_memory,
        seed=settings.tracking.seed,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = ConvNet(
        in_channels=3,
        num_classes=2,
        dropout=settings.regularization.dropout,
    )
    logger.info("Model parameters: %s", model.parameter_summary())

    if settings.gradient.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr=settings.optimizer.learning_rate,
        weight_decay=settings.optimizer.weight_decay,
        betas=settings.optimizer.betas,
    )

    # ── Criterion ─────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        label_smoothing=settings.regularization.label_smoothing
    )

    # ── Scheduler ─────────────────────────────────────────────────────
    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * settings.loop.epochs
    scheduler = create_scheduler(
        settings.scheduler.scheduler,
        optimizer,
        num_training_steps=total_steps,
        num_warmup_steps=settings.scheduler.warmup_steps,
        warmup_ratio=settings.scheduler.warmup_ratio,
        lr_min=settings.scheduler.lr_min,
        max_lr=settings.optimizer.learning_rate,
    )

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=settings.loop.early_stopping_patience
        ),
        ModelCheckpoint(monitor="val_loss", mode="min"),
        LRLogger(),
        TensorBoardCallback(log_dir=str(settings.experiment.output_dir / "tb")),
    ]

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=settings.hardware.device,
        max_epochs=settings.loop.epochs,
        mixed_precision=settings.hardware.mixed_precision,
        gradient_clipping=settings.gradient.gradient_clipping,
        gradient_accumulation_steps=settings.gradient.gradient_accumulation_steps,
        scheduler=scheduler,
        callbacks=callbacks,
        output_dir=str(settings.experiment.output_dir),
        experiment_name=settings.experiment.experiment_name,
        run_name=settings.experiment.run_name,
        log_every_n_steps=settings.tracking.log_every_n_steps,
    )

    log_gpu_memory()
    history = trainer.train(loaders["train"], loaders.get("val"))

    best_val_loss = min(history["val_loss"]) if history["val_loss"] else None
    logger.info("Training complete. Best val_loss: %s", best_val_loss)


if __name__ == "__main__":
    main()
