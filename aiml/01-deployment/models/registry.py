"""
Model registry — discovers, loads, and serves multiple model variants.
Supports PyTorch (.pt/.pth) and HuggingFace checkpoints.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type

import torch
from transformers import AutoModel, AutoTokenizer, pipeline as hf_pipeline

from models.base import BaseModel

logger = logging.getLogger(__name__)


# ── Concrete PyTorch model example ──────────────────────────────────


class TorchScriptModel(BaseModel):
    """Wraps a TorchScript (.pt) exported model."""

    async def load(self, path: Path) -> None:
        self._model = torch.jit.load(str(path), map_location=self.device)
        self._model.eval()
        self._is_loaded = True
        logger.info("TorchScript model '%s' loaded on %s", self.name, self.device)

    async def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._model(inputs.to(self.device))


# ── Concrete HuggingFace model example ──────────────────────────────


class HFPipelineModel(BaseModel):
    """
    Wraps a HuggingFace pipeline (text-classification, etc.).
    Pass `task` and `model_name_or_path` at construction time.
    """

    def __init__(self, name: str, task: str, hf_model_id: str, **kwargs) -> None:
        super().__init__(name)
        self._task = task
        self._hf_model_id = hf_model_id
        self._pipe_kwargs = kwargs

    async def load(self, path: Optional[Path] = None) -> None:
        model_id = str(path) if path and path.exists() else self._hf_model_id
        self._model = hf_pipeline(
            task=self._task,
            model=model_id,
            device=0 if self.device == "cuda" else -1,
            **self._pipe_kwargs,
        )
        self._is_loaded = True
        logger.info("HF pipeline '%s' (%s) ready", self.name, self._task)

    async def predict(self, inputs) -> list:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._model, inputs)


# ── Registry ─────────────────────────────────────────────────────────


class ModelRegistry:
    """
    Scans `model_dir` for sub-directories, each treated as a versioned model.
    Auto-instantiates TorchScript or HF pipeline models based on file extensions.

    Directory layout expected:
        model_dir/
          my_model_v1/
            model.pt          ← TorchScript
          my_classifier/
            model.safetensors ← HuggingFace
    """

    def __init__(self, model_dir: Path) -> None:
        self._dir = Path(model_dir)
        self._models: Dict[str, BaseModel] = {}

    async def load_all(self) -> None:
        if not self._dir.exists():
            logger.warning(
                "Model dir '%s' does not exist — skipping auto-load", self._dir
            )
            return
        tasks = []
        for model_path in self._dir.iterdir():
            if model_path.is_dir():
                tasks.append(self._load_one(model_path))
        await asyncio.gather(*tasks)

    async def _load_one(self, path: Path) -> None:
        name = path.name
        pt_file = next(path.glob("*.pt"), None) or next(path.glob("*.pth"), None)
        if pt_file:
            model = TorchScriptModel(name=name)
            await model.load(pt_file)
        else:
            model = HFPipelineModel(
                name=name, task="feature-extraction", hf_model_id=str(path)
            )
            await model.load(path)
        self._models[name] = model

    def get(self, name: str) -> Optional[BaseModel]:
        return self._models.get(name)

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def register(self, model: BaseModel) -> None:
        """Register a model instance manually (useful in tests / custom loaders)."""
        self._models[model.name] = model

    async def unload_all(self) -> None:
        await asyncio.gather(*(m.unload() for m in self._models.values()))
        self._models.clear()
