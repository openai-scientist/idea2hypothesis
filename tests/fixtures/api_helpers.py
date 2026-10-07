"""Test harness for the HTTP adapter: ASGI app, fake Platform BE and request helpers.

Everything runs in-process (``httpx.ASGITransport`` and ``httpx.MockTransport``); there is no
network. The fake Platform BE enforces the same contiguity rule as the real consumer.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from idea2hypothesis.api.app import create_app
from idea2hypothesis.api.service import RunService
from idea2hypothesis.config import Config
from tests.conftest import TOPIC, make_config, make_services
from tests.fixtures import FixtureLiterature, FixtureLLM

CALLBACK = "http://platform.test/internal/runs/00000000-0000-0000-0000-000000000001"
SERVER_KEY = "server-secret-key"


async def no_sleep(_: float) -> None:
    return None


@dataclass
class FakePlatform:
    """Stand-in for the BE ingest endpoint: ordered, deduplicating, scriptable failures."""

    expected: int = 1
    received: list[dict[str, Any]] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)
    script: list[int] = field(default_factory=list)  # status codes served before normal handling
    closed: bool = False

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.script:
            return httpx.Response(self.script.pop(0), json={"detail": "scripted"})
        if self.closed:
            return httpx.Response(409, json={"detail": "RUN_FINISHED"})
        events = json.loads(request.content)["events"]
        for event in events:
            if event["source_seq"] < self.expected:
                continue
            if event["source_seq"] != self.expected:
                return httpx.Response(422, json={"detail": "EVENT_GAP"})
            self.received.append(event)
            self.expected += 1
        return httpx.Response(200, json={"data": {"last_source_seq": self.expected - 1}})

    @property
    def seqs(self) -> list[int]:
        return [e["source_seq"] for e in self.received]

    @property
    def types(self) -> list[str]:
        return [e["type"] for e in self.received]


@dataclass
class Harness:
    app: Any
    client: httpx.AsyncClient
    service: RunService
    config: Config
    platform: FakePlatform

    async def start(self, platform_run_id: str = "p-1", **overrides: Any) -> httpx.Response:
        body = run_body(platform_run_id, **overrides)
        return await self.client.post("/runs", json=body)

    async def settle(self) -> None:
        await self.service.wait_idle()

    async def state(self, run_id: str) -> dict[str, Any]:
        response = await self.client.get(f"/runs/{run_id}")
        assert response.status_code == 200, response.text
        return response.json()

    async def events(self, run_id: str, after: int = 0) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"/runs/{run_id}/events", params={"after_source_seq": after, "limit": 1000}
        )
        assert response.status_code == 200, response.text
        return response.json()["events"]


def run_body(platform_run_id: str = "p-1", **overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "platform_run_id": platform_run_id,
        "topic": TOPIC,
        "domains": ["education", "psychology"],
        "review_mode": "copilot",
        "budget_usd": "5.00",
        "callback_url": CALLBACK,
    }
    body.update(overrides)
    return body


def api_config(tmp_path: Path, **sections: dict[str, Any]) -> Config:
    api = {
        "callback_allowed_hosts": ["platform.test"],
        "delivery_max_attempts": 3,
        "delivery_backoff_max_sec": 0,
    }
    api.update(sections.pop("api", {}))
    return make_config(tmp_path, api=api, **sections)


@asynccontextmanager
async def harness(
    tmp_path: Path,
    *,
    llm: FixtureLLM | None = None,
    literature: FixtureLiterature | None = None,
    platform: FakePlatform | None = None,
    send_key: bool = True,
    **sections: dict[str, Any],
) -> AsyncIterator[Harness]:
    """App + client with the lifespan running; ``tmp_path`` identifies the run storage."""
    config = api_config(tmp_path, **sections)
    services = make_services(tmp_path, llm=llm, literature=literature, config=config)
    fake = platform or FakePlatform()
    app = create_app(config, services=services, callback_transport=fake.transport(), sleep=no_sleep)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        key = os.environ.get(config.api.service_key_env, "") if send_key else ""
        headers = {"X-Service-Key": key} if key else {}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", headers=headers
        ) as client:
            yield Harness(app, client, app.state.run_service, config, fake)
            await app.state.run_service.wait_idle()
