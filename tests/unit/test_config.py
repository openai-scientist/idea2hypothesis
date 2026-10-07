from __future__ import annotations

from pathlib import Path

import pytest

from idea2hypothesis.config import ConfigError, load_config

EXAMPLE = Path(__file__).resolve().parents[2] / "configs" / "example.yaml"


def test_defaults_load_from_none() -> None:
    cfg = load_config(None)
    assert cfg.review.mode == "copilot"
    assert cfg.llm.provider == "openai_compatible"
    assert cfg.literature.sources == ("openalex", "semantic_scholar", "arxiv")
    assert cfg.api.callback_allowed_hosts == () and cfg.api.callback_allow_any is False


def test_example_config_is_valid() -> None:
    cfg = load_config(EXAMPLE)
    assert cfg.llm.model
    assert cfg.llm.api_key_env == "I2H_LLM_API_KEY"


def test_only_the_eight_sections_exist() -> None:
    with pytest.raises(ConfigError, match=r"^experiment: unknown key"):
        load_config({"experiment": {"mode": "docker"}})


@pytest.mark.parametrize(
    ("data", "path"),
    [
        ({"llm": {"modle": "x"}}, "llm.modle"),
        ({"research": {"extra": 1}}, "research.extra"),
        ({"llm": {"timeout_sec": "fast"}}, "llm.timeout_sec"),
        ({"llm": {"max_retries": 0}}, "llm.max_retries"),
        ({"llm": {"provider": "anthropic"}}, "llm.provider"),
        ({"llm": {"reviewer": {"model": 3}}}, "llm.reviewer.model"),
        ({"llm": {"pricing": {"input_per_1k": "x"}}}, "llm.pricing.input_per_1k"),
        ({"literature": {"sources": ["google"]}}, "literature.sources"),
        ({"literature": {"sources": []}}, "literature.sources"),
        ({"literature": {"cache": "yes"}}, "literature.cache"),
        ({"prompts": {"domain": "chemistry"}}, "prompts.domain"),
        ({"review": {"mode": "yolo"}}, "review.mode"),
        ({"runtime": {"concurrency": 0}}, "runtime.concurrency"),
        ({"api": {"port": True}}, "api.port"),
        ({"api": {"cors_origins": "*"}}, "api.cors_origins"),
        ({"storage": {"runs_root": 3}}, "storage.runs_root"),
    ],
)
def test_errors_name_the_dotted_path(data: dict, path: str) -> None:
    with pytest.raises(ConfigError) as info:
        load_config(data)
    assert path in str(info.value)


def test_reviewer_inherits_unset_values_from_the_main_model() -> None:
    cfg = load_config(
        {
            "llm": {
                "model": "main",
                "base_url": "https://example.test/v1",
                "api_key_env": "MY_KEY",
                "fallback_models": ["fallback"],
                "debate_rounds": 2,
                "reviewer": {"model": "judge"},
            }
        }
    )
    reviewer = cfg.llm.reviewer
    assert reviewer is not None
    assert (reviewer.model, reviewer.base_url, reviewer.api_key_env) == (
        "judge",
        "https://example.test/v1",
        "MY_KEY",
    )
    assert reviewer.fallback_models == ()  # the judge never falls back to author models
    assert reviewer.reviewer is None
    assert cfg.llm.debate_rounds == 2


def test_pricing_cost() -> None:
    cfg = load_config({"llm": {"pricing": {"input_per_1k": 0.5, "output_per_1k": 1.5}}})
    assert cfg.llm.pricing is not None
    assert cfg.llm.pricing.cost(2000, 1000) == pytest.approx(2.5)


def test_snapshot_contains_names_but_no_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("I2H_LLM_API_KEY", "super-secret-value")
    cfg = load_config({"llm": {"model": "m", "aws_profile": "work"}})
    snapshot = cfg.snapshot()
    assert snapshot["llm"]["api_key_env"] == "I2H_LLM_API_KEY"
    assert snapshot["llm"]["aws_profile"] == "work"
    assert "super-secret-value" not in str(snapshot)


def test_load_from_yaml_file_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("review:\n  mode: full\nllm:\n  model: m\n", encoding="utf-8")
    assert load_config(path).review.mode == "full"
    path.write_text("review: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(path)
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")


def test_platform_local_config_matches_the_platform_be_setup() -> None:
    cfg = load_config(EXAMPLE.with_name("platform-local.yaml"))
    assert cfg.llm.provider == "bedrock"
    assert (cfg.api.host, cfg.api.port) == ("0.0.0.0", 8001)
    assert {"localhost", "host.docker.internal"} <= set(cfg.api.callback_allowed_hosts)
    assert (cfg.api.service_key_env, cfg.api.callback_key_env) == (
        "I2H_SERVICE_KEY",
        "I2H_CALLBACK_KEY",
    )
