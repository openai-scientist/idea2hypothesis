"""Typed configuration.

Exactly eight sections are supported: ``research``, ``llm``, ``literature``, ``prompts``,
``runtime``, ``review``, ``storage`` and ``api``. Unknown keys and wrong types raise
:class:`ConfigError` naming the dotted path of the offending key.

Credentials are never stored in configuration: only the *names* of environment variables
(or AWS profile names) are kept, so ``Config.snapshot()`` is safe to persist.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REVIEW_MODES = ("auto", "light", "copilot", "full")
LLM_PROVIDERS = ("openai_compatible", "bedrock")
PROMPT_DOMAINS = ("ml", "hep", "biology")
LITERATURE_SOURCES = ("openalex", "semantic_scholar", "arxiv")


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types or invalid values."""


@dataclass(frozen=True)
class Pricing:
    """USD price per 1000 tokens."""

    input_per_1k: float
    output_per_1k: float

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens / 1000.0 * self.input_per_1k
            + completion_tokens / 1000.0 * self.output_per_1k
        )


@dataclass(frozen=True)
class ResearchConfig:
    topic: str = ""
    domains: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    hardware_advisory: bool = False
    min_relevance: float = 0.7
    min_quality: float = 0.5
    min_topic_score: float = 5.0
    novelty_check: bool = True


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "openai_compatible"
    model: str = ""
    fallback_models: tuple[str, ...] = ()
    base_url: str = ""
    api_key_env: str = "I2H_LLM_API_KEY"
    region: str = ""
    aws_profile: str = ""
    timeout_sec: float = 120.0
    max_retries: int = 3
    retry_base_delay: float = 2.0
    temperature: float = 0.4
    max_tokens: int = 4096
    reviewer: LLMConfig | None = None
    debate_rounds: int = 0
    pricing: Pricing | None = None


@dataclass(frozen=True)
class LiteratureConfig:
    sources: tuple[str, ...] = LITERATURE_SOURCES
    max_results_per_query: int = 20
    inter_query_delay_sec: float = 1.0
    timeout_sec: float = 30.0
    max_retries: int = 3
    openalex_email: str = ""
    openalex_api_key_env: str = ""
    s2_api_key_env: str = "S2_API_KEY"
    cache: bool = False
    default_year_min: int = 0


@dataclass(frozen=True)
class PromptsConfig:
    override_file: str = ""
    domain: str = "ml"


@dataclass(frozen=True)
class RuntimeConfig:
    max_retries: int = 1
    retry_delay_sec: float = 2.0
    concurrency: int = 4
    ideation_memory: bool = False


@dataclass(frozen=True)
class ReviewConfig:
    mode: str = "copilot"


@dataclass(frozen=True)
class StorageConfig:
    runs_root: str = "runs"
    cache_root: str = ".cache"


@dataclass(frozen=True)
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8001
    service_key_env: str = "I2H_SERVICE_KEY"
    callback_key_env: str = "I2H_CALLBACK_KEY"
    callback_allowed_hosts: tuple[str, ...] = ()
    callback_allow_any: bool = False
    delivery_max_attempts: int = 5
    delivery_backoff_max_sec: float = 60.0
    cors_origins: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    research: ResearchConfig = field(default_factory=ResearchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    literature: LiteratureConfig = field(default_factory=LiteratureConfig)
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    @property
    def runs_root(self) -> Path:
        return Path(self.storage.runs_root)

    @property
    def cache_root(self) -> Path:
        return Path(self.storage.cache_root)

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe dict of the whole configuration (no credentials by construction)."""
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class _Reader:
    """Typed accessor over one mapping that tracks consumed keys."""

    def __init__(self, data: Any, path: str) -> None:
        if data is None:
            data = {}
        if not isinstance(data, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
        self._data = data
        self._path = path
        self._used: set[str] = set()

    def _key(self, key: str) -> str:
        return f"{self._path}.{key}" if self._path else key

    def _get(self, key: str, default: Any) -> Any:
        self._used.add(key)
        return self._data.get(key, default)

    def str(self, key: str, default: str) -> str:
        value = self._get(key, default)
        if value is None:
            return default
        if not isinstance(value, str):
            raise ConfigError(f"{self._key(key)}: expected a string, got {type(value).__name__}")
        return value

    def bool(self, key: str, default: bool) -> bool:
        value = self._get(key, default)
        if not isinstance(value, bool):
            raise ConfigError(f"{self._key(key)}: expected a boolean, got {type(value).__name__}")
        return value

    def int(self, key: str, default: int, *, minimum: int | None = None) -> int:
        value = self._get(key, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{self._key(key)}: expected an integer, got {type(value).__name__}")
        if minimum is not None and value < minimum:
            raise ConfigError(f"{self._key(key)}: must be >= {minimum}, got {value}")
        return value

    def float(self, key: str, default: float, *, minimum: float | None = None) -> float:
        value = self._get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{self._key(key)}: expected a number, got {type(value).__name__}")
        if minimum is not None and value < minimum:
            raise ConfigError(f"{self._key(key)}: must be >= {minimum}, got {value}")
        return float(value)

    def str_list(self, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
        value = self._get(key, list(default))
        if value is None:
            return default
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{self._key(key)}: expected a list of strings")
        return tuple(value)

    def choice(self, key: str, default: str, options: tuple[str, ...]) -> str:
        value = self.str(key, default)
        if value not in options:
            raise ConfigError(f"{self._key(key)}: must be one of {list(options)}, got {value!r}")
        return value

    def section(self, key: str) -> Any:
        return self._get(key, None)

    def finish(self) -> None:
        unknown = sorted(set(self._data) - self._used)
        if unknown:
            raise ConfigError(f"{self._key(unknown[0])}: unknown key")


def _parse_research(data: Any) -> ResearchConfig:
    r = _Reader(data, "research")
    d = ResearchConfig()
    cfg = ResearchConfig(
        topic=r.str("topic", d.topic),
        domains=r.str_list("domains", d.domains),
        constraints=r.str_list("constraints", d.constraints),
        hardware_advisory=r.bool("hardware_advisory", d.hardware_advisory),
        min_relevance=r.float("min_relevance", d.min_relevance, minimum=0.0),
        min_quality=r.float("min_quality", d.min_quality, minimum=0.0),
        min_topic_score=r.float("min_topic_score", d.min_topic_score, minimum=0.0),
        novelty_check=r.bool("novelty_check", d.novelty_check),
    )
    r.finish()
    return cfg


def _parse_pricing(data: Any, path: str) -> Pricing | None:
    if data is None:
        return None
    r = _Reader(data, path)
    pricing = Pricing(
        input_per_1k=r.float("input_per_1k", 0.0, minimum=0.0),
        output_per_1k=r.float("output_per_1k", 0.0, minimum=0.0),
    )
    r.finish()
    return pricing


def _parse_llm(data: Any, path: str = "llm", parent: LLMConfig | None = None) -> LLMConfig:
    """Parse an ``llm`` block; a reviewer block inherits unset keys from ``parent``."""
    r = _Reader(data, path)
    d = parent or LLMConfig()
    is_reviewer = parent is not None
    cfg = LLMConfig(
        provider=r.choice("provider", d.provider, LLM_PROVIDERS),
        model=r.str("model", d.model),
        fallback_models=r.str_list("fallback_models", () if is_reviewer else d.fallback_models),
        base_url=r.str("base_url", d.base_url),
        api_key_env=r.str("api_key_env", d.api_key_env),
        region=r.str("region", d.region),
        aws_profile=r.str("aws_profile", d.aws_profile),
        timeout_sec=r.float("timeout_sec", d.timeout_sec, minimum=1.0),
        max_retries=r.int("max_retries", d.max_retries, minimum=1),
        retry_base_delay=r.float("retry_base_delay", d.retry_base_delay, minimum=0.0),
        temperature=r.float("temperature", d.temperature, minimum=0.0),
        max_tokens=r.int("max_tokens", d.max_tokens, minimum=1),
        pricing=_parse_pricing(r.section("pricing"), f"{path}.pricing") or d.pricing,
    )
    if not is_reviewer:
        reviewer_data = r.section("reviewer")
        reviewer = (
            _parse_llm(reviewer_data, f"{path}.reviewer", parent=cfg)
            if reviewer_data is not None
            else None
        )
        cfg = dataclasses.replace(
            cfg,
            reviewer=reviewer,
            debate_rounds=r.int("debate_rounds", d.debate_rounds, minimum=0),
        )
    r.finish()
    return cfg


def _parse_literature(data: Any) -> LiteratureConfig:
    r = _Reader(data, "literature")
    d = LiteratureConfig()
    sources = r.str_list("sources", d.sources)
    for source in sources:
        if source not in LITERATURE_SOURCES:
            raise ConfigError(
                f"literature.sources: unknown source {source!r}, "
                f"expected {list(LITERATURE_SOURCES)}"
            )
    if not sources:
        raise ConfigError("literature.sources: at least one source is required")
    cfg = LiteratureConfig(
        sources=sources,
        max_results_per_query=r.int("max_results_per_query", d.max_results_per_query, minimum=1),
        inter_query_delay_sec=r.float(
            "inter_query_delay_sec", d.inter_query_delay_sec, minimum=0.0
        ),
        timeout_sec=r.float("timeout_sec", d.timeout_sec, minimum=1.0),
        max_retries=r.int("max_retries", d.max_retries, minimum=1),
        openalex_email=r.str("openalex_email", d.openalex_email),
        openalex_api_key_env=r.str("openalex_api_key_env", d.openalex_api_key_env),
        s2_api_key_env=r.str("s2_api_key_env", d.s2_api_key_env),
        cache=r.bool("cache", d.cache),
        default_year_min=r.int("default_year_min", d.default_year_min, minimum=0),
    )
    r.finish()
    return cfg


def _parse_prompts(data: Any) -> PromptsConfig:
    r = _Reader(data, "prompts")
    d = PromptsConfig()
    cfg = PromptsConfig(
        override_file=r.str("override_file", d.override_file),
        domain=r.choice("domain", d.domain, PROMPT_DOMAINS),
    )
    r.finish()
    return cfg


def _parse_runtime(data: Any) -> RuntimeConfig:
    r = _Reader(data, "runtime")
    d = RuntimeConfig()
    cfg = RuntimeConfig(
        max_retries=r.int("max_retries", d.max_retries, minimum=0),
        retry_delay_sec=r.float("retry_delay_sec", d.retry_delay_sec, minimum=0.0),
        concurrency=r.int("concurrency", d.concurrency, minimum=1),
        ideation_memory=r.bool("ideation_memory", d.ideation_memory),
    )
    r.finish()
    return cfg


def _parse_review(data: Any) -> ReviewConfig:
    r = _Reader(data, "review")
    cfg = ReviewConfig(mode=r.choice("mode", ReviewConfig().mode, REVIEW_MODES))
    r.finish()
    return cfg


def _parse_storage(data: Any) -> StorageConfig:
    r = _Reader(data, "storage")
    d = StorageConfig()
    cfg = StorageConfig(
        runs_root=r.str("runs_root", d.runs_root),
        cache_root=r.str("cache_root", d.cache_root),
    )
    r.finish()
    return cfg


def _parse_api(data: Any) -> ApiConfig:
    r = _Reader(data, "api")
    d = ApiConfig()
    cfg = ApiConfig(
        host=r.str("host", d.host),
        port=r.int("port", d.port, minimum=1),
        service_key_env=r.str("service_key_env", d.service_key_env),
        callback_key_env=r.str("callback_key_env", d.callback_key_env),
        callback_allowed_hosts=r.str_list("callback_allowed_hosts", d.callback_allowed_hosts),
        callback_allow_any=r.bool("callback_allow_any", d.callback_allow_any),
        delivery_max_attempts=r.int("delivery_max_attempts", d.delivery_max_attempts, minimum=1),
        delivery_backoff_max_sec=r.float(
            "delivery_backoff_max_sec", d.delivery_backoff_max_sec, minimum=0.0
        ),
        cors_origins=r.str_list("cors_origins", d.cors_origins),
    )
    r.finish()
    return cfg


def config_from_dict(data: Mapping[str, Any] | None) -> Config:
    """Build a :class:`Config` from a plain mapping, validating every key."""
    r = _Reader(data, "")
    cfg = Config(
        research=_parse_research(r.section("research")),
        llm=_parse_llm(r.section("llm")),
        literature=_parse_literature(r.section("literature")),
        prompts=_parse_prompts(r.section("prompts")),
        runtime=_parse_runtime(r.section("runtime")),
        review=_parse_review(r.section("review")),
        storage=_parse_storage(r.section("storage")),
        api=_parse_api(r.section("api")),
    )
    r.finish()
    return cfg


def load_config(source: str | Path | Mapping[str, Any] | None = None) -> Config:
    """Load configuration from a YAML file path or a mapping (``None`` gives defaults)."""
    if source is None or isinstance(source, Mapping):
        return config_from_dict(source)
    path = Path(source)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    return config_from_dict(data)
