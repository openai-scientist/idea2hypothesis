"""ASGI application factory.

``create_app(config)`` builds the FastAPI application from a loaded
:class:`~idea2hypothesis.config.Config`. ``app`` is a lazily built instance for
``uvicorn idea2hypothesis.api.app:app``; it reads the configuration file named by the
``I2H_CONFIG`` environment variable (default ``configs/example.yaml``) on first use.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from idea2hypothesis import __version__
from idea2hypothesis.api.routes import bedrock, engine_runs, stages
from idea2hypothesis.api.security import ServiceKeyMiddleware
from idea2hypothesis.api.service import RunService, ServiceError
from idea2hypothesis.config import Config, load_config
from idea2hypothesis.pipeline.models import Services

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "configs/example.yaml"
CONFIG_ENV = "I2H_CONFIG"


def create_app(
    config: Config,
    *,
    services: Services | None = None,
    callback_transport: httpx.AsyncBaseTransport | None = None,
    sleep: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
) -> FastAPI:
    """Build the application.

    ``services``, ``callback_transport`` and ``sleep`` let tests inject the engine ports, the
    webhook HTTP transport and the retry sleep; by default everything comes from ``config``.
    """
    service = RunService(
        config, services=services, callback_transport=callback_transport, sleep=sleep
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        report = await service.startup()
        if report["recovered"] or report["delivery_resumed"]:
            logger.info("startup recovery: %s", report)
        try:
            yield
        finally:
            await service.shutdown()

    app = FastAPI(
        title="idea2hypothesis",
        description="Idea to hypothesis engine: Platform runs, stage review and diagnostics.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.run_service = service
    app.state.config = config

    app.add_middleware(ServiceKeyMiddleware, api=config.api)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.api.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)

    @app.get("/api/health", tags=["Health"], summary="Liveness")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(engine_runs.router)
    app.include_router(stages.router)
    app.include_router(bedrock.router)
    return app


class _LazyApp:
    """ASGI callable that builds the real application on first use."""

    def __init__(self) -> None:
        self._app: FastAPI | None = None

    def build(self) -> FastAPI:
        if self._app is None:
            path = os.environ.get(CONFIG_ENV, DEFAULT_CONFIG_PATH)
            self._app = create_app(load_config(path))
        return self._app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.build()(scope, receive, send)


app = _LazyApp()
