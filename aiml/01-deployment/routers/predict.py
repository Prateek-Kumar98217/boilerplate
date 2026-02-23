"""Prediction router — handles single and batch inference requests."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from config import get_settings

router = APIRouter()
settings = get_settings()


# ── Request / Response schemas ────────────────────────────────────────


class PredictRequest(BaseModel):
    model_name: Optional[str] = Field(
        None, description="Model to use; defaults to DEFAULT_MODEL"
    )
    inputs: Union[str, List[Any], Dict[str, Any]] = Field(
        ..., description="Model inputs"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Optional inference params"
    )


class PredictResponse(BaseModel):
    model_name: str
    outputs: Any
    latency_ms: float
    timestamp: float


class BatchPredictRequest(BaseModel):
    model_name: Optional[str] = None
    inputs: List[Union[str, List[Any], Dict[str, Any]]] = Field(..., max_length=32)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class BatchPredictResponse(BaseModel):
    model_name: str
    outputs: List[Any]
    latency_ms: float
    timestamp: float


# ── Helpers ───────────────────────────────────────────────────────────


def get_registry(request: Request):
    return request.app.state.model_registry


def resolve_model(registry, model_name: Optional[str]):
    name = model_name or settings.model.default_model
    model = registry.get(name)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{name}' not found. Available: {registry.list_models()}",
        )
    return model, name


# ── Endpoints ─────────────────────────────────────────────────────────


@router.post("/predict", response_model=PredictResponse, summary="Single inference")
async def predict(body: PredictRequest, registry=Depends(get_registry)):
    model, name = resolve_model(registry, body.model_name)
    t0 = time.perf_counter()
    outputs = await model(body.inputs)
    latency = (time.perf_counter() - t0) * 1000
    return PredictResponse(
        model_name=name,
        outputs=outputs,
        latency_ms=round(latency, 2),
        timestamp=time.time(),
    )


@router.post(
    "/predict/batch", response_model=BatchPredictResponse, summary="Batch inference"
)
async def predict_batch(body: BatchPredictRequest, registry=Depends(get_registry)):
    model, name = resolve_model(registry, body.model_name)
    t0 = time.perf_counter()
    outputs = await model(body.inputs)
    latency = (time.perf_counter() - t0) * 1000
    return BatchPredictResponse(
        model_name=name,
        outputs=outputs,
        latency_ms=round(latency, 2),
        timestamp=time.time(),
    )


@router.get("/models", summary="List available models")
async def list_models(registry=Depends(get_registry)):
    models_info = []
    for name in registry.list_models():
        m = registry.get(name)
        models_info.append(m.info())
    return {"models": models_info, "count": len(models_info)}
