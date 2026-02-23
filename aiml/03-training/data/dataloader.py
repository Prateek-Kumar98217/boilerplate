"""
DataLoader factory — handles weighted sampling, worker init, and seed workers.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducibility."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle class imbalance.
    Pass dataset labels as a flat list/array.
    """
    import collections

    label_arr = np.array(labels)
    counts = collections.Counter(label_arr.tolist())
    class_weights = {cls: 1.0 / cnt for cls, cnt in counts.items()}
    sample_weights = np.array([class_weights[y] for y in label_arr])
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    drop_last: bool = False,
    use_weighted_sampler: bool = False,
    collate_fn=None,
) -> DataLoader:
    """
    Create a configured DataLoader.

    Args:
        dataset:               PyTorch Dataset.
        batch_size:            Batch size.
        shuffle:               Shuffle (set False when using sampler).
        num_workers:           Worker processes. Set 0 for debugging.
        pin_memory:            Pin tensors to page-locked memory (speeds GPU transfer).
        seed:                  Random seed for workers.
        drop_last:             Drop incomplete last batch.
        use_weighted_sampler:  Use WeightedRandomSampler for class imbalance.
                               Requires dataset to have `.labels` or `.class_counts()`.
        collate_fn:            Custom collate function.
    """
    sampler = None
    if use_weighted_sampler:
        # Try to get labels from dataset
        if hasattr(dataset, "labels"):
            labels = dataset.labels.tolist()
        elif hasattr(dataset, "samples"):
            labels = [s[1] for s in dataset.samples]
        else:
            raise ValueError(
                "Dataset must have .labels or .samples for weighted sampling."
            )
        sampler = make_weighted_sampler(labels)
        shuffle = False  # Sampler and shuffle are mutually exclusive

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
        persistent_workers=num_workers > 0,
    )


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    use_weighted_sampler: bool = False,
):
    """
    Convenience factory: creates train / val / test loaders at once.

    Returns:
        dict with keys "train", "val" (optional), "test" (optional)
    """
    loaders = {}
    loaders["train"] = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        drop_last=True,
        use_weighted_sampler=use_weighted_sampler,
    )
    if val_dataset:
        loaders["val"] = create_dataloader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )
    if test_dataset:
        loaders["test"] = create_dataloader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )
    return loaders
