"""Health-check routers."""

from __future__ import annotations

import platform
import time

import torch
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()
_start_time = time.time()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    python_version: str
    cuda_available: bool
    cuda_device_count: int
    models_loaded: int


@router.get("", response_model=HealthResponse, summary="Liveness probe")
async def health(request: Request):
    registry = request.app.state.model_registry
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 1),
        python_version=platform.python_version(),
        cuda_available=torch.cuda.is_available(),
        cuda_device_count=torch.cuda.device_count(),
        models_loaded=len(registry.list_models()),
    )


@router.get("/ready", summary="Readiness probe")
async def ready(request: Request):
    registry = request.app.state.model_registry
    if not registry.list_models():
        return {"status": "not_ready", "reason": "No models loaded"}
    return {"status": "ready"}
