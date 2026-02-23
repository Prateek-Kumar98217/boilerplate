"""API Key authentication middleware."""

from __future__ import annotations

import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

_OPEN_PATHS = {"/health", "/health/ready", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip auth when no key configured (dev mode) or for open paths
        if not self._api_key or request.url.path in _OPEN_PATHS:
            return await call_next(request)

        provided = request.headers.get("X-API-Key") or request.headers.get(
            "Authorization", ""
        ).removeprefix("Bearer ")
        if provided != self._api_key:
            logger.warning("Unauthorized request from %s", request.client)
            return Response(
                content='{"detail":"Unauthorized"}',
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)
