"""Explicit per-run keys take precedence without reading the environment."""

import pytest

from idea2hypothesis.config import LLMConfig
from idea2hypothesis.llm.factory import build_llm
from idea2hypothesis.llm.models import LLMConfigError


async def test_build_llm_uses_explicit_key_without_env(monkeypatch):
    monkeypatch.delenv("I2H_LLM_API_KEY", raising=False)
    client = build_llm(LLMConfig(model="stage", base_url="http://model.test"), api_key="sk-x")
    assert client._api_key == "sk-x"
    await client.aclose()


def test_empty_explicit_key_does_not_fall_back_to_env(monkeypatch):
    monkeypatch.setenv("I2H_LLM_API_KEY", "fixture-env-key")
    with pytest.raises(LLMConfigError):
        build_llm(LLMConfig(model="stage", base_url="http://model.test"), api_key="")
