"""
CUDA / device utilities.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """
    Resolve the best available device.
    preference: 'auto' | 'cuda' | 'mps' | 'cpu'
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed everything for reproducibility.

    Args:
        seed:          Integer seed value.
        deterministic: If True, forces CUDA deterministic ops (slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True  # speeds up fixed-size workloads


def gpu_memory_stats(device: Optional[int] = None) -> Dict[str, float]:
    """
    Get GPU memory usage in MB.

    Returns dict with: allocated_mb, reserved_mb, free_mb, total_mb
    """
    if not torch.cuda.is_available():
        return {}
    dev = device or torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    alloc = torch.cuda.memory_allocated(dev)
    reserved = torch.cuda.memory_reserved(dev)
    total = props.total_memory
    return {
        "device_name": props.name,
        "allocated_mb": alloc / (1024**2),
        "reserved_mb": reserved / (1024**2),
        "free_mb": (total - reserved) / (1024**2),
        "total_mb": total / (1024**2),
        "utilization_pct": round(100 * alloc / total, 1),
    }


def log_gpu_memory() -> None:
    stats = gpu_memory_stats()
    if stats:
        logger.info(
            "GPU [%s] alloc=%.0fMB reserved=%.0fMB free=%.0fMB (%.1f%% used)",
            stats["device_name"],
            stats["allocated_mb"],
            stats["reserved_mb"],
            stats["free_mb"],
            stats["utilization_pct"],
        )


def empty_cache() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def optimal_dtype() -> torch.dtype:
    """Return best available float precision for the current GPU."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def to_device(batch: dict, device: torch.device) -> dict:
    """Move a batch dict of tensors to the target device."""
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
