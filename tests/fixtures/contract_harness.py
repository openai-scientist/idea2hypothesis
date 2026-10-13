"""Harness for the engine contract tests: the app, a fake Platform and fixture models.

Everything runs in-process. Each model of a run is a :class:`FixtureModelLLM` that records the
phase and the model name of every call it answers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from idea2hypothesis.api.app import create_app
from idea2hypothesis.api.service import RunService
from idea2hypothesis.config import Config, LLMConfig
from idea2hypothesis.llm.models import ChatMessage, LLMResponse
from tests.conftest import TOPIC, make_services
from tests.contract.fake_platform import FakePlatform
from tests.fixtures import FixtureLiterature, FixtureLLM
from tests.fixtures.api_helpers import SERVER_KEY, api_config, no_sleep

SECRET_MARKER = "sk-contract-secret-9f3a1c"
INTEGRATION_MARKER = "integration-secret-77d2"
PLATFORM = "http://platform.test/internal/engines/i2h/runs"


@dataclass
class FixtureModelLLM:
    """A scripted client for one configured model; records ``(phase, model)`` per call."""

    model: str
    calls: list[tuple[str, str]]
    phase_of: Callable[[], str]
    inner: FixtureLLM = field(default_factory=FixtureLLM)

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append((self.phase_of(), self.model))
        return await self.inner.chat(
            messages, system=system, json_mode=json_mode, max_tokens=max_tokens,
            temperature=temperature,
        )  # fmt: skip


def run_body(
    stage_run_id: str = "stage-run-1",
    *,
    mode: str = "light",
    stage_model: str = "stage-model",
    review_model: str = "review-model",
    **overrides: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "contract_version": "1.0.0",
        "platform_run_id": "platform-run-1",
        "stage_run_id": stage_run_id,
        "stage_key": "hypothesis",
        "interaction_mode": mode,
        "models": {
            "stage": {"provider": "openai_compatible", "base_url": "http://stage.test/v1",
                      "model": stage_model, "api_key": SECRET_MARKER},
            "review": {"provider": "openai_compatible", "base_url": "http://review.test/v1",
                       "model": review_model, "api_key": SECRET_MARKER},
        },  # fmt: skip
        "budget_usd": "5.00",
        "callback_url": f"{PLATFORM}/{stage_run_id}",
        "inputs": {"params": {"topic": TOPIC, "domains": []}},
        "integrations": {"semantic_scholar": INTEGRATION_MARKER},
    }
    body.update(overrides)
    return body


@dataclass
class ContractHarness:
    app: Any
    client: httpx.AsyncClient
    service: RunService
    config: Config
    platform: FakePlatform
    calls: list[tuple[str, str]]

    async def create(self, stage_run_id: str = "stage-run-1", **overrides: Any) -> httpx.Response:
        return await self.client.post("/v1/runs", json=run_body(stage_run_id, **overrides))

    async def settle(self) -> None:
        await self.service.wait_idle()

    async def state(self, run_id: str) -> dict[str, Any]:
        response = await self.client.get(f"/v1/runs/{run_id}")
        assert response.status_code == 200, response.text
        return response.json()

    async def reach(self, stage_run_id: str = "stage-run-1", **overrides: Any) -> dict[str, Any]:
        """Create a run and wait until the engine stops; returns its state."""
        response = await self.create(stage_run_id, **overrides)
        assert response.status_code == 201, response.text
        await self.settle()
        return await self.state(response.json()["engine_run_id"])

    def run_dir(self, run_id: str) -> Path:
        return self.service.store.run_dir(run_id)

    def llm_calls(self) -> list[tuple[str, str]]:
        return list(self.calls)

    def current_phase(self) -> str:
        ids = self.service.store.list_run_ids()
        return self.service.contract.phase(ids[-1]) if ids else "drafting"


@asynccontextmanager
async def contract_harness(
    tmp_path: Path,
    *,
    platform: FakePlatform | None = None,
    literature: FixtureLiterature | None = None,
    overrides: dict[str, Any] | None = None,
    **sections: dict[str, Any],
) -> AsyncIterator[ContractHarness]:
    config = api_config(tmp_path, **sections)
    services = make_services(tmp_path, literature=literature, config=config)
    fake = platform or FakePlatform(service_key=SERVER_KEY)
    calls: list[tuple[str, str]] = []
    holder: dict[str, ContractHarness] = {}

    def factory(cfg: LLMConfig, *, api_key: str | None = None) -> FixtureModelLLM:
        assert api_key == SECRET_MARKER
        return FixtureModelLLM(
            cfg.model, calls, lambda: holder["h"].current_phase(),
            FixtureLLM(overrides=dict(overrides or {})),
        )  # fmt: skip

    app = create_app(
        config, services=services, callback_transport=fake.transport(), sleep=no_sleep,
        llm_factory=factory,
    )  # fmt: skip
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://i2h.test",
        headers={"X-Service-Key": SERVER_KEY},
    ) as client:  # fmt: skip
        harness = ContractHarness(app, client, app.state.run_service, config, fake, calls)
        holder["h"] = harness
        yield harness
