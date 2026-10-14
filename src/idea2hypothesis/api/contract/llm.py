"""Build independent stage and review clients from in-memory run credentials."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from idea2hypothesis.api.contract.models import ModelConfig, Models
from idea2hypothesis.config import LLMConfig
from idea2hypothesis.llm.factory import build_llm
from idea2hypothesis.llm.models import ChatMessage, LLMConfigError, LLMPort, LLMResponse
from idea2hypothesis.pipeline.models import Services


class LlmFactory(Protocol):
    def __call__(self, cfg: LLMConfig, *, api_key: str | None = None) -> LLMPort: ...


class UnconfiguredLLM:
    """Stands in for the ambient client of a service whose runs bring their own models.

    It never answers: a call fails with a clear configuration error.
    """

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        raise LLMConfigError("contract runs use the models of their request, not the service's")


def llm_config_for(base: LLMConfig, model: ModelConfig) -> LLMConfig:
    if model.provider != "openai_compatible":
        raise LLMConfigError("contract runs currently support only openai_compatible")
    return replace(
        base, provider=model.provider, model=model.model, base_url=model.base_url or "",
        api_key_env="", reviewer=None, fallback_models=(), pricing=None,
    )


def services_for_request(
    base: Services, models: Models, factory: LlmFactory = build_llm
) -> Services:
    configs = [llm_config_for(base.config.llm, m) for m in (models.stage, models.review)]
    keys = [m.api_key.get_secret_value() if m.api_key else ""
            for m in (models.stage, models.review)]
    if not all(keys):
        raise LLMConfigError("stage and review model API keys are required")
    if not all(cfg.base_url for cfg in configs):
        raise LLMConfigError("stage and review model base URLs are required")
    stage = factory(configs[0], api_key=keys[0])
    reviewer = factory(configs[1], api_key=keys[1])
    cfg = replace(base.config, llm=replace(configs[0], reviewer=configs[1]))
    return replace(base, config=cfg, llm=stage, reviewer=reviewer)
