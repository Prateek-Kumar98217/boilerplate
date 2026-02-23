"""
Base model wrapper — all deployed models inherit from this.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class BaseModel(abc.ABC):
    """
    Minimal contract every deployable model must satisfy.
    Subclass this and implement `load`, `predict`, and optionally `preprocess`/`postprocess`.
    """

    def __init__(self, name: str, device: Optional[str] = None) -> None:
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Any = None
        self._is_loaded: bool = False

    # ── Abstract interface ────────────────────────────────────────────
    @abc.abstractmethod
    async def load(self, path: Path) -> None:
        """Load model weights/artifacts from *path*."""

    @abc.abstractmethod
    async def predict(self, inputs: Any) -> Any:
        """Run inference and return raw outputs."""

    async def preprocess(self, raw: Any) -> Any:
        """Optional input pre-processing hook."""
        return raw

    async def postprocess(self, raw: Any) -> Any:
        """Optional output post-processing hook."""
        return raw

    async def unload(self) -> None:
        """Release GPU/CPU memory."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._is_loaded = False

    # ── Convenience pipeline ──────────────────────────────────────────
    async def __call__(self, inputs: Any) -> Any:
        processed = await self.preprocess(inputs)
        raw_output = await self.predict(processed)
        return await self.postprocess(raw_output)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
            "is_loaded": self._is_loaded,
        }
