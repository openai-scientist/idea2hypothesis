"""Build LLM clients from configuration, failing early on missing credentials."""

from __future__ import annotations

import os

from idea2hypothesis.config import Config, LLMConfig
from idea2hypothesis.llm.bedrock import BedrockLLM
from idea2hypothesis.llm.models import LLMConfigError, LLMPort
from idea2hypothesis.llm.openai_compatible import OpenAICompatibleLLM


def build_llm(cfg: LLMConfig) -> LLMPort:
    """Create the client described by ``cfg`` (raises :class:`LLMConfigError` when unusable)."""
    common = {
        "fallback_models": cfg.fallback_models,
        "max_retries": cfg.max_retries,
        "retry_base_delay": cfg.retry_base_delay,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "pricing": cfg.pricing,
    }
    if cfg.provider == "bedrock":
        region = cfg.region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        return BedrockLLM(
            model=cfg.model,
            region=region or "",
            profile=cfg.aws_profile or os.environ.get("AWS_PROFILE", ""),
            timeout_sec=cfg.timeout_sec,
            **common,
        )
    api_key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
    if not api_key:
        raise LLMConfigError(
            f"environment variable {cfg.api_key_env!r} (llm.api_key_env) is not set"
        )
    return OpenAICompatibleLLM(
        base_url=cfg.base_url,
        api_key=api_key,
        model=cfg.model,
        timeout_sec=cfg.timeout_sec,
        **common,
    )


def build_reviewer(cfg: Config) -> LLMPort | None:
    """Create the independent reviewer/judge client, or ``None`` when none is configured."""
    reviewer = cfg.llm.reviewer
    if reviewer is None or not reviewer.model:
        return None
    return build_llm(reviewer)
