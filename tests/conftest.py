"""Shared pytest fixtures: configuration, services and run helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from idea2hypothesis.config import Config, config_from_dict
from idea2hypothesis.pipeline.models import RunRequest, Services
from idea2hypothesis.prompts.loader import PromptLoader
from idea2hypothesis.resources.hardware import HardwareProfile
from idea2hypothesis.storage.runs import RunStore
from tests.fixtures import FixtureLiterature, FixtureLLM

TOPIC = "Effect of sleep duration on exam performance in university students"


def make_config(tmp_path: Path, **sections: dict[str, Any]) -> Config:
    data: dict[str, Any] = {
        "research": {
            "topic": TOPIC,
            "domains": ["education", "psychology"],
            "hardware_advisory": True,
            "novelty_check": True,
        },
        "llm": {"model": "fixture-model"},
        "literature": {"inter_query_delay_sec": 0},
        "runtime": {"max_retries": 1, "retry_delay_sec": 0, "concurrency": 3},
        "review": {"mode": "copilot"},
        "storage": {
            "runs_root": str(tmp_path / "runs"),
            "cache_root": str(tmp_path / "cache"),
        },
    }
    for name, values in sections.items():
        data.setdefault(name, {}).update(values)
    return config_from_dict(data)


def fixture_hardware() -> HardwareProfile:
    return HardwareProfile(
        has_gpu=False, gpu_type="cpu", gpu_name="CPU only", vram_mb=None,
        tier="cpu_only", warning="fixture hardware",
    )  # fmt: skip


def make_services(
    tmp_path: Path,
    *,
    llm: FixtureLLM | None = None,
    literature: FixtureLiterature | None = None,
    config: Config | None = None,
    **sections: dict[str, Any],
) -> Services:
    cfg = config or make_config(tmp_path, **sections)
    return Services(
        config=cfg,
        llm=llm or FixtureLLM(),
        literature=literature or FixtureLiterature(),
        prompts=PromptLoader(cfg.prompts.domain),
        store=RunStore(cfg.runs_root),
        hardware=fixture_hardware,
    )


async def no_sleep(_: float) -> None:
    return None


@pytest.fixture
def tmp_services(tmp_path: Path) -> Services:
    return make_services(tmp_path)


def request(**overrides: Any) -> RunRequest:
    values: dict[str, Any] = {"topic": TOPIC, "domains": ("education", "psychology")}
    values.update(overrides)
    return RunRequest(**values)
