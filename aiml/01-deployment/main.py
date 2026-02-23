"""
FastAPI application entrypoint.

Usage:
    uvicorn main:app --reload
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from middleware.auth import APIKeyMiddleware
from middleware.logging import RequestLoggingMiddleware
from models.registry import ModelRegistry
from routers import health, metrics, predict

settings = get_settings()
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=settings.app.log_level.upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Load models on startup, clean up on shutdown."""
    logger.info("Starting up — loading model registry...")
    registry = ModelRegistry(model_dir=settings.model.model_dir)
    await registry.load_all()
    app.state.model_registry = registry
    logger.info("Model registry ready. Models loaded: %s", registry.list_models())
    yield
    logger.info("Shutting down — releasing resources...")
    await registry.unload_all()


app = FastAPI(
    title=settings.app.name,
    version="1.0.0",
    description="Production ML model serving API",
    docs_url="/docs" if settings.app.env != "production" else None,
    redoc_url="/redoc" if settings.app.env != "production" else None,
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.app.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(APIKeyMiddleware, api_key=settings.app.api_key)

# ── Routers ───────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])
app.include_router(predict.router, prefix="/v1", tags=["Inference"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.app.host,
        port=settings.app.port,
        workers=settings.app.workers,
        log_level=settings.app.log_level,
        reload=settings.app.env == "development",
    )
