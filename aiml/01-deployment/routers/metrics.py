"""Simple in-process metrics router (use Prometheus + prometheus_fastapi_instrumentator for prod)."""

from __future__ import annotations

import gc
import os
from collections import defaultdict, deque
from typing import Deque, Dict

import torch
from fastapi import APIRouter, Request

router = APIRouter()

# Shared counter store — for a real deployment wire this into Prometheus / StatsD
_call_counts: Dict[str, int] = defaultdict(int)
_latencies: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=1000))


def record_call(model_name: str, latency_ms: float) -> None:
    _call_counts[model_name] += 1
    _latencies[model_name].append(latency_ms)


@router.get("", summary="Runtime metrics snapshot")
async def metrics(request: Request):
    registry = request.app.state.model_registry

    # GPU memory
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            alloc = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            gpu_info.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_mb": props.total_memory // (1024**2),
                    "allocated_mb": alloc // (1024**2),
                    "reserved_mb": reserved // (1024**2),
                }
            )

    # Per-model latency stats
    latency_stats = {}
    for model_name, lats in _latencies.items():
        if lats:
            s = sorted(lats)
            n = len(s)
            latency_stats[model_name] = {
                "calls": _call_counts[model_name],
                "p50_ms": s[int(n * 0.50)],
                "p95_ms": s[int(n * 0.95)],
                "p99_ms": s[int(n * 0.99)],
                "mean_ms": sum(s) / n,
            }

    return {
        "models": registry.list_models(),
        "gpu": gpu_info,
        "inference": latency_stats,
        "process_pid": os.getpid(),
    }
