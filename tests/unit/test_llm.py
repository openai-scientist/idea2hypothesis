from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from idea2hypothesis.config import Pricing, load_config
from idea2hypothesis.llm.bedrock import BedrockLLM
from idea2hypothesis.llm.factory import build_llm, build_reviewer
from idea2hypothesis.llm.models import (
    ChatMessage,
    LLMConfigError,
    LLMRateLimited,
    LLMResponseError,
    LLMTimeout,
)
from idea2hypothesis.llm.openai_compatible import OpenAICompatibleLLM

USER = [ChatMessage.user("hello")]


def ok(text: str = "hi", **extra: Any) -> httpx.Response:
    body = {
        "model": "m",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        **extra,
    }
    return httpx.Response(200, json=body)


def make_llm(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> tuple[OpenAICompatibleLLM, list[float]]:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = OpenAICompatibleLLM(
        base_url="https://llm.test/v1",
        api_key="key",
        model=kwargs.pop("model", "m"),
        client=client,
        sleep=fake_sleep,
        **kwargs,
    )
    return llm, sleeps


async def test_successful_call_and_usage() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return ok("hello back")

    llm, _ = make_llm(handler, pricing=Pricing(1.0, 2.0))
    response = await llm.chat(USER, system="be brief")
    assert response.text == "hello back"
    assert (response.prompt_tokens, response.completion_tokens) == (100, 50)
    assert response.cost_usd == pytest.approx(0.1 + 0.1)
    body = json.loads(seen[0].content)
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert seen[0].headers["authorization"] == "Bearer key"
    assert str(seen[0].url) == "https://llm.test/v1/chat/completions"


async def test_cost_is_none_without_pricing() -> None:
    llm, _ = make_llm(lambda r: ok())
    assert (await llm.chat(USER)).cost_usd is None


async def test_thinking_tags_are_stripped() -> None:
    llm, _ = make_llm(lambda r: ok("<think>plan</think>answer"))
    assert (await llm.chat(USER)).text == "answer"


async def test_rate_limit_is_retried_with_retry_after() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return ok()

    llm, sleeps = make_llm(handler, max_retries=3, retry_base_delay=1.0)
    assert (await llm.chat(USER)).text == "hi"
    assert attempts["n"] == 3
    assert len(sleeps) == 2 and all(s >= 7 for s in sleeps)


async def test_rate_limit_exhaustion_raises_typed_error() -> None:
    llm, _ = make_llm(lambda r: httpx.Response(429), max_retries=2)
    with pytest.raises(LLMRateLimited):
        await llm.chat(USER)


async def test_server_errors_are_retried_then_raised() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="down")

    llm, _ = make_llm(handler, max_retries=3)
    with pytest.raises(LLMResponseError) as info:
        await llm.chat(USER)
    assert info.value.status == 503 and info.value.retryable
    assert calls["n"] == 3


async def test_non_retryable_error_is_raised_immediately() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    llm, _ = make_llm(handler, max_retries=3)
    with pytest.raises(LLMResponseError) as info:
        await llm.chat(USER)
    assert info.value.status == 401 and not info.value.retryable
    assert calls["n"] == 1


async def test_overload_reported_as_400_is_treated_as_transient() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(400, text="Server is overloaded, retry later")
        return ok()

    llm, _ = make_llm(handler)
    assert (await llm.chat(USER)).text == "hi"


async def test_timeout_is_typed_and_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("too slow", request=request)

    llm, _ = make_llm(handler, max_retries=2)
    with pytest.raises(LLMTimeout):
        await llm.chat(USER)
    assert calls["n"] == 2


async def test_connection_errors_are_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    llm, _ = make_llm(handler, max_retries=2)
    with pytest.raises(LLMResponseError) as info:
        await llm.chat(USER)
    assert info.value.retryable


async def test_fallback_model_is_used_after_the_primary_fails() -> None:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        models.append(model)
        return (
            httpx.Response(404, text="no such model") if model == "primary" else ok("from fallback")
        )

    llm, _ = make_llm(handler, model="primary", fallback_models=["backup"])
    assert (await llm.chat(USER)).text == "from fallback"
    assert models == ["primary", "backup"]


async def test_malformed_payloads_are_response_errors() -> None:
    llm, _ = make_llm(lambda r: httpx.Response(200, json={"choices": []}), max_retries=1)
    with pytest.raises(LLMResponseError, match="missing choices"):
        await llm.chat(USER)
    llm, _ = make_llm(lambda r: httpx.Response(200, text="<html>"), max_retries=1)
    with pytest.raises(LLMResponseError, match="non-JSON"):
        await llm.chat(USER)


async def test_json_mode_uses_response_format_or_a_system_hint() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return ok("{}")

    llm, _ = make_llm(handler, model="gpt-4o-mini")
    await llm.chat(USER, json_mode=True)
    assert bodies[-1]["response_format"] == {"type": "json_object"}

    llm, _ = make_llm(handler, model="claude-3-haiku")
    await llm.chat(USER, json_mode=True)
    assert "response_format" not in bodies[-1]
    assert bodies[-1]["messages"][0]["role"] == "system"
    assert "valid JSON only" in bodies[-1]["messages"][0]["content"]


async def test_reasoning_models_use_max_completion_tokens() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return ok()

    llm, _ = make_llm(handler, model="gpt-5.2")
    await llm.chat(USER, max_tokens=100)
    assert bodies[0]["max_completion_tokens"] >= 32768 and "max_tokens" not in bodies[0]


def sse(*chunks: Any, done: bool = True) -> httpx.Response:
    lines = [": PROCESSING"] + [f"data: {json.dumps(c)}" for c in chunks]
    text = "\n\n".join(lines + (["data: [DONE]"] if done else [])) + "\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=text)


def delta(text: str, finish: str | None = None) -> dict[str, Any]:
    return {"model": "m-1", "choices": [{"delta": {"content": text}, "finish_reason": finish}]}


async def test_streamed_chunks_are_joined_with_their_usage() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        usage = {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
        return sse(delta('{"ok": '), delta("true}", "stop"), usage)

    llm, _ = make_llm(handler, pricing=Pricing(1.0, 2.0))
    response = await llm.chat(USER, json_mode=True)
    assert response.text == '{"ok": true}' and response.model == "m-1"
    assert (response.prompt_tokens, response.completion_tokens) == (100, 50)
    assert response.cost_usd == pytest.approx(0.2)
    assert bodies[0]["stream"] is True
    assert bodies[0]["stream_options"] == {"include_usage": True}


async def test_a_stream_cut_by_the_token_limit_is_truncated() -> None:
    llm, _ = make_llm(lambda r: sse(delta("partial", "length")))
    response = await llm.chat(USER)
    assert response.text == "partial" and response.truncated


async def test_stream_errors_and_empty_streams_are_response_errors() -> None:
    failing = {"error": {"message": "provider overloaded"}, "choices": []}
    llm, _ = make_llm(lambda r: sse(delta("x"), failing), max_retries=1)
    with pytest.raises(LLMResponseError, match="provider overloaded") as info:
        await llm.chat(USER)
    assert info.value.retryable
    llm, _ = make_llm(lambda r: sse(done=True), max_retries=1)
    with pytest.raises(LLMResponseError, match="missing choices"):
        await llm.chat(USER)


def test_missing_credentials_fail_at_build_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("I2H_LLM_API_KEY", raising=False)
    cfg = load_config({"llm": {"model": "m", "base_url": "https://x.test/v1"}})
    with pytest.raises(LLMConfigError, match="I2H_LLM_API_KEY"):
        build_llm(cfg.llm)


def test_missing_model_and_base_url_are_config_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("I2H_LLM_API_KEY", "k")
    with pytest.raises(LLMConfigError, match="model"):
        build_llm(load_config({"llm": {"base_url": "https://x.test/v1"}}).llm)
    with pytest.raises(LLMConfigError, match="base_url"):
        build_llm(load_config({"llm": {"model": "m"}}).llm)


def test_factory_builds_client_and_reviewer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("I2H_LLM_API_KEY", "k")
    cfg = load_config(
        {"llm": {"model": "m", "base_url": "https://x.test/v1", "reviewer": {"model": "judge"}}}
    )
    assert isinstance(build_llm(cfg.llm), OpenAICompatibleLLM)
    reviewer = build_reviewer(cfg)
    assert isinstance(reviewer, OpenAICompatibleLLM) and reviewer.models == ["judge"]
    assert build_reviewer(load_config({"llm": {"model": "m"}})) is None


# -- Bedrock (fake boto client, no network) -------------------------------------


class FakeBotoError(Exception):
    def __init__(self, code: str, status: int = 400) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}


class FakeBedrockClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.requests: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def converse_ok(text: str = "answer") -> dict[str, Any]:
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "usage": {"inputTokens": 10, "outputTokens": 4},
        "stopReason": "end_turn",
    }


async def _no_sleep(_: float) -> None:
    return None


def make_bedrock(outcomes: list[Any], **kwargs: Any) -> tuple[BedrockLLM, FakeBedrockClient]:
    client = FakeBedrockClient(outcomes)
    return BedrockLLM(model="anthropic.claude", client=client, sleep=_no_sleep, **kwargs), client


async def test_bedrock_request_shape_and_usage() -> None:
    llm, client = make_bedrock([converse_ok("done")], pricing=Pricing(1.0, 1.0))
    response = await llm.chat(
        [ChatMessage.user("a"), ChatMessage.user("b")],
        system="sys",
        json_mode=True,
        max_tokens=99999,
    )
    request = client.requests[0]
    assert response.text == "done" and response.prompt_tokens == 10
    assert request["modelId"] == "anthropic.claude"
    assert request["inferenceConfig"]["maxTokens"] == 8192
    assert request["messages"] == [{"role": "user", "content": [{"text": "a"}, {"text": "b"}]}]
    assert "sys" in request["system"][0]["text"] and "valid JSON" in request["system"][0]["text"]
    assert response.cost_usd == pytest.approx(0.014)


async def test_bedrock_throttling_is_retried_and_other_errors_are_typed() -> None:
    llm, client = make_bedrock(
        [FakeBotoError("ThrottlingException", 429), converse_ok()], max_retries=2
    )
    assert (await llm.chat(USER)).text == "answer"
    assert len(client.requests) == 2

    llm, _ = make_bedrock([FakeBotoError("ThrottlingException")] * 2, max_retries=2)
    with pytest.raises(LLMRateLimited):
        await llm.chat(USER)

    llm, client = make_bedrock([FakeBotoError("ValidationException")], max_retries=3)
    with pytest.raises(LLMResponseError) as info:
        await llm.chat(USER)
    assert not info.value.retryable and len(client.requests) == 1

    llm, _ = make_bedrock([FakeBotoError("ModelTimeoutException")] * 2, max_retries=2)
    with pytest.raises(LLMTimeout):
        await llm.chat(USER)


async def test_bedrock_credential_errors_are_config_errors() -> None:
    class NoCredentialsError(Exception):
        pass

    llm, _ = make_bedrock([NoCredentialsError("no creds")])
    with pytest.raises(LLMConfigError):
        await llm.chat(USER)


def test_bedrock_without_credentials_fails_at_build_time(monkeypatch: pytest.MonkeyPatch) -> None:
    import boto3

    class NoCredsSession:
        def __init__(self, **_: Any) -> None:
            pass

        def get_credentials(self) -> None:
            return None

    monkeypatch.setattr(boto3, "Session", NoCredsSession)
    cfg = load_config({"llm": {"provider": "bedrock", "model": "anthropic.claude"}})
    with pytest.raises(LLMConfigError, match="AWS credentials"):
        build_llm(cfg.llm)


async def test_bedrock_sends_a_zero_temperature_instead_of_its_default() -> None:
    llm, client = make_bedrock([converse_ok("a"), converse_ok("b")])
    await llm.chat([ChatMessage.user("x")], temperature=0)
    await llm.chat([ChatMessage.user("x")], temperature=0.4)
    assert client.requests[0]["inferenceConfig"]["temperature"] == 0
    assert client.requests[1]["inferenceConfig"]["temperature"] == 0.4
