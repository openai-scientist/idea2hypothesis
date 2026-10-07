"""Inbound service-key authentication (``X-Service-Key``)."""

from __future__ import annotations

import hmac
import os
import re
from typing import Any

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from idea2hypothesis.config import ApiConfig

HEADER = "x-service-key"
_PROTECTED = re.compile(
    r"^/(runs(/.*)?|api/phase1(/.*)?|api/stage[1-8](/.*)?|api/health/bedrock(/.*)?)$"
)


def is_protected(path: str) -> bool:
    """Routes that act on runs, artifacts or cloud credentials. ``/docs``, ``/openapi.json``
    and ``/api/health`` stay open."""
    return bool(_PROTECTED.match(path))


class ServiceKeyMiddleware:
    """Require the server's service key on protected routes when one is configured.

    The key is read from the environment variable named by ``api.service_key_env`` on every
    request; when the variable is unset or empty the check is disabled.
    """

    def __init__(self, app: ASGIApp, api: ApiConfig) -> None:
        self.app = app
        self.env_name = api.service_key_env

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return
        expected = os.environ.get(self.env_name, "")
        if expected and is_protected(scope["path"]):
            supplied = _header(scope, HEADER)
            if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
                response = JSONResponse({"detail": "Invalid or missing service key"}, 401)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _header(scope: dict[str, Any], name: str) -> str:
    for key, value in scope.get("headers", []):
        if key.decode("latin-1").lower() == name:
            return str(value.decode("latin-1"))
    return ""
