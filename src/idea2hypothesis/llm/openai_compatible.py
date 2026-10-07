"""OpenAI-compatible chat-completions client built on :mod:`httpx`."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from idea2hypothesis.llm.chain import ChainedLLM, RawCompletion
from idea2hypothesis.llm.models import (
    ChatMessage,
    LLMConfigError,
    LLMRateLimited,
    LLMResponseError,
    LLMTimeout,
)

logger = logging.getLogger(__name__)

# Models that need ``max_completion_tokens`` and a large reasoning headroom.
_NEW_PARAM_MODELS = ("o3", "o4-mini", "gpt-5")
_NO_TEMPERATURE_MODELS = ("o3", "o4-mini")
# Providers that reject ``response_format``; JSON is requested through a system hint instead.
_NO_RESPONSE_FORMAT_PREFIXES = (
    "claude",
    "deepseek",
    "qwen",
    "yi-",
    "glm",
    "moonshot",
    "minimax",
    "doubao",
    "abab",
    "hunyuan",
    "ernie",
    "spark",
    "gemma",
)
_JSON_HINT = (
    "You MUST respond with valid JSON only. Do not include any text outside the JSON object."
)
_REASONING_MIN_TOKENS = 32768
_TRANSIENT_400_HINTS = (
    "rate limit",
    "ratelimit",
    "overloaded",
    "temporarily",
    "capacity",
    "throttl",
    "too many",
    "retry",
)
_USER_AGENT = "idea2hypothesis/0.1"


class OpenAICompatibleLLM(ChainedLLM):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_sec: float = 120.0,
        client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        if not base_url:
            raise LLMConfigError("llm.base_url is required for the openai_compatible provider")
        if not api_key:
            raise LLMConfigError("missing API key for the openai_compatible provider")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout_sec
        self._client = client or httpx.AsyncClient(timeout=timeout_sec)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_body(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_mode: bool,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        body: dict[str, Any] = {"model": model, "messages": msgs}
        if not model.startswith(_NO_TEMPERATURE_MODELS):
            body["temperature"] = temperature
        if model.startswith(_NEW_PARAM_MODELS):
            body["max_completion_tokens"] = max(max_tokens, _REASONING_MIN_TOKENS)
        else:
            body["max_tokens"] = max_tokens
        if json_mode:
            if model.lower().startswith(_NO_RESPONSE_FORMAT_PREFIXES):
                if msgs and msgs[0]["role"] == "system":
                    msgs[0]["content"] = _JSON_HINT + "\n\n" + msgs[0]["content"]
                else:
                    msgs.insert(0, {"role": "system", "content": _JSON_HINT})
            else:
                body["response_format"] = {"type": "json_object"}
        return body

    async def _complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_mode: bool,
        max_tokens: int,
        temperature: float,
    ) -> RawCompletion:
        body = self._build_body(
            model, messages, json_mode=json_mode, max_tokens=max_tokens, temperature=temperature
        )
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=headers,
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeout(f"timeout calling {model}: {exc!r}") from exc
        except httpx.TransportError as exc:
            raise LLMResponseError(
                f"transport error calling {model}: {exc!r}", retryable=True
            ) from exc
        return self._parse(response, model)

    def _parse(self, response: httpx.Response, model: str) -> RawCompletion:
        status = response.status_code
        if status >= 400:
            self._raise_for_status(response, model)
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError(f"non-JSON response from {model}", status=status) from exc
        if not isinstance(data, dict):
            raise LLMResponseError(f"malformed response from {model}: expected an object")
        error = data.get("error")
        if error:
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise LLMResponseError(f"{model}: {message}", status=status, retryable=True)
        choices = data.get("choices")
        if not choices:
            raise LLMResponseError(f"malformed response from {model}: missing choices")
        choice = choices[0]
        content = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        finish = choice.get("finish_reason") or ""
        return RawCompletion(
            text=content,
            model=data.get("model", model),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            finish_reason=finish,
            truncated=finish == "length",
        )

    @staticmethod
    def _raise_for_status(response: httpx.Response, model: str) -> None:
        status = response.status_code
        body = response.text[:500]
        if status == 429:
            retry_after: float | None
            try:
                retry_after = float(response.headers.get("Retry-After", ""))
            except ValueError:
                retry_after = None
            raise LLMRateLimited(f"{model}: rate limited (HTTP 429)", retry_after)
        if status in (500, 502, 503, 504, 529):
            raise LLMResponseError(f"{model}: HTTP {status}", status=status, retryable=True)
        if status == 400 and any(hint in body.lower() for hint in _TRANSIENT_400_HINTS):
            raise LLMResponseError(f"{model}: HTTP 400 ({body})", status=status, retryable=True)
        raise LLMResponseError(f"{model}: HTTP {status} {body}", status=status)
