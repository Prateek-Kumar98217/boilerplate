"""
Flexible PyTorch Dataset base classes.

Provides:
- TensorDataset wrapper (numpy → tensor)
- CSVDataset (tabular data from CSV)
- ImageFolderDataset (images in class sub-directories)
- Abstract AbstractDataset with length / item logging
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────


class AbstractDataset(Dataset, abc.ABC):
    """Base dataset with length / override hook."""

    @abc.abstractmethod
    def __len__(self) -> int: ...

    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Any: ...

    def class_counts(self) -> Optional[Dict[int, int]]:
        return None


# ── Numpy / Tensor dataset ────────────────────────────────────────────


class ArrayDataset(AbstractDataset):
    """
    Wrap numpy arrays or torch tensors as a Dataset.

    Example:
        ds = ArrayDataset(X_train, y_train)
        sample = ds[0]   # {"input": tensor, "label": tensor}
    """

    def __init__(
        self,
        features: Union[np.ndarray, torch.Tensor],
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = (
            torch.as_tensor(labels, dtype=torch.long) if labels is not None else None
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.features[idx]
        if self.transform:
            x = self.transform(x)
        if self.labels is not None:
            return {"input": x, "label": self.labels[idx]}
        return {"input": x}


# ── CSV dataset ───────────────────────────────────────────────────────


class CSVDataset(AbstractDataset):
    """
    Load a CSV where feature columns are floats and label_col is the target.

    Example:
        ds = CSVDataset("data/train.csv", label_col="target")
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        label_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        df = pd.read_csv(csv_path)
        self.label_col = label_col
        feature_cols = feature_cols or [c for c in df.columns if c != label_col]
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.labels = (
            torch.tensor(df[label_col].values, dtype=torch.long) if label_col else None
        )
        self.transform = transform
        logger.info(
            "CSVDataset loaded %d samples from %s", len(self.features), csv_path
        )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.features[idx]
        if self.transform:
            x = self.transform(x)
        if self.labels is not None:
            return {"input": x, "label": self.labels[idx]}
        return {"input": x}


# ── Image folder dataset ──────────────────────────────────────────────


class ImageFolderDataset(AbstractDataset):
    """
    Dataset for image classification with folder structure:
        root/
          class_a/img1.jpg, img2.jpg, ...
          class_b/img1.jpg, ...

    Example:
        ds = ImageFolderDataset("data/train", transform=train_transform)
        img, label = ds[0]["input"], ds[0]["label"]
    """

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        extensions: Optional[set] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.extensions = extensions or self.IMG_EXTENSIONS

        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int]] = []
        for cls_name, idx in self.class_to_idx.items():
            cls_dir = self.root / cls_name
            for p in cls_dir.iterdir():
                if p.suffix.lower() in self.extensions:
                    self.samples.append((p, idx))

        logger.info(
            "ImageFolderDataset: %d images, %d classes from %s",
            len(self.samples),
            len(self.classes),
            root,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"input": img, "label": label, "path": str(path)}

    def class_counts(self) -> Dict[int, int]:
        from collections import Counter

        return dict(Counter(label for _, label in self.samples))
