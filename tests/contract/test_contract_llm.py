"""Run model configuration cannot use ambient credentials or share reviewers."""

import json

import pytest
from pydantic import ValidationError

from idea2hypothesis.api.contract.llm import services_for_request
from idea2hypothesis.api.contract.models import ModelConfig, Models
from idea2hypothesis.llm.models import LLMConfigError
from tests.conftest import make_services
from tests.fixtures import FixtureLLM


def model(name, key="fixture-run-key"):
    return ModelConfig(provider="openai_compatible", base_url="http://model.test", model=name,
                       api_key=key)


def test_services_for_request_sets_distinct_reviewer(tmp_path):
    calls = []

    def factory(cfg, *, api_key):
        calls.append((cfg, api_key))
        return FixtureLLM()

    base = make_services(tmp_path)
    services = services_for_request(base, Models(stage=model("stage"), review=model("review")),
                                    factory)
    assert services.llm is not services.reviewer
    assert [c.model for c, _ in calls] == ["stage", "review"]
    assert all(c.api_key_env == "" and key == "fixture-run-key" for c, key in calls)
    assert "fixture-run-key" not in json.dumps(services.config.snapshot())
    assert base.config.llm.model == "fixture-model"


def test_services_for_request_missing_key_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("I2H_LLM_API_KEY", "fixture-ambient-key")
    with pytest.raises(LLMConfigError):
        services_for_request(make_services(tmp_path),
                             Models(stage=model("stage", None), review=model("review")))


def test_review_model_must_differ():
    with pytest.raises(ValidationError):
        Models(stage=model("same"), review=model("same"))


def test_model_secrets_are_masked_and_unknown_fields_rejected():
    value = model("stage")
    assert "fixture-run-key" not in repr(value)
    assert "fixture-run-key" not in value.model_dump_json()
    with pytest.raises(ValidationError):
        ModelConfig(provider="openai_compatible", model="stage", unknown="unsafe")
