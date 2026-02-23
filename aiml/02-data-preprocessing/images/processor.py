"""
Image preprocessing utilities.

Covers:
- Resize, crop, pad
- Normalization (ImageNet, custom)
- Augmentation pipeline (train vs val/test)
- PIL ↔ NumPy ↔ Tensor conversions
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF

    _TV_AVAILABLE = True
except ImportError:
    _TV_AVAILABLE = False


def _require_torchvision():
    if not _TV_AVAILABLE:
        raise ImportError("torchvision required: pip install torchvision")


# ── Conversion helpers ────────────────────────────────────────────────


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def pil_to_tensor(img: Image.Image) -> "torch.Tensor":
    _require_torchvision()
    return TF.to_tensor(img)


def tensor_to_pil(t: "torch.Tensor") -> Image.Image:
    _require_torchvision()
    return TF.to_pil_image(t)


# ── ImageProcessor ────────────────────────────────────────────────────


class ImageProcessor:
    """
    Configurable image preprocessing pipeline.

    Example:
        proc = ImageProcessor(size=224)
        train_transform = proc.get_train_transforms()
        val_transform   = proc.get_val_transforms()

        img = proc.load("photo.jpg")
        tensor = train_transform(img)   # → torch.Tensor (C, H, W)
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        size: int = 224,
        mean: Tuple[float, ...] = IMAGENET_MEAN,
        std: Tuple[float, ...] = IMAGENET_STD,
        color_mode: str = "RGB",  # RGB | L | RGBA
    ) -> None:
        _require_torchvision()
        self.size = size
        self.mean = mean
        self.std = std
        self.color_mode = color_mode

    def load(self, path: Union[str, Path]) -> Image.Image:
        return Image.open(str(path)).convert(self.color_mode)

    # ── Transform factories ───────────────────────────────────────────

    def get_train_transforms(self) -> T.Compose:
        """Standard augmented training transform."""
        return T.Compose(
            [
                T.RandomResizedCrop(self.size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.RandomGrayscale(p=0.05),
                T.RandomRotation(degrees=15),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def get_val_transforms(self) -> T.Compose:
        """Deterministic val / test transform."""
        return T.Compose(
            [
                T.Resize(int(self.size * 1.14)),  # resize shorter side
                T.CenterCrop(self.size),
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def get_inference_transforms(self) -> T.Compose:
        """TTA-ready inference transform (no augmentation)."""
        return self.get_val_transforms()

    # ── Denormalization ───────────────────────────────────────────────

    def denormalize(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """Reverse normalization for visualization."""
        import torch

        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        return tensor * std + mean

    # ── Batch processing ──────────────────────────────────────────────

    def process_paths(
        self, paths: List[Union[str, Path]], augment: bool = False
    ) -> "torch.Tensor":
        """Load and process a list of image paths → (N, C, H, W) tensor."""
        import torch

        transform = (
            self.get_train_transforms() if augment else self.get_val_transforms()
        )
        tensors = [transform(self.load(p)) for p in paths]
        return torch.stack(tensors)
