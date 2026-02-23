"""Structured request/response logging middleware."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "rid=%s method=%s path=%s error=%s",
                request_id,
                request.method,
                request.url.path,
                exc,
            )
            raise

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "rid=%s method=%s path=%s status=%s latency=%.1fms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response
