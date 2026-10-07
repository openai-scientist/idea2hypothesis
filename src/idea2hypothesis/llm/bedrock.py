"""AWS Bedrock client using the Converse API.

``boto3`` is imported lazily so the core package works without the ``bedrock`` extra.
Calls run in a worker thread; botocore's own retries are disabled so the outer retry loop is
the only one.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from idea2hypothesis.llm.chain import ChainedLLM, RawCompletion
from idea2hypothesis.llm.models import (
    ChatMessage,
    LLMConfigError,
    LLMRateLimited,
    LLMResponseError,
    LLMTimeout,
)

logger = logging.getLogger(__name__)

_MAX_OUTPUT_TOKENS = 8192
_JSON_HINT = (
    "You MUST respond with valid JSON only. Do not include any introductory or concluding "
    "text outside the JSON object."
)
_RATE_LIMIT_CODES = {"ThrottlingException", "TooManyRequestsException"}
_TIMEOUT_CODES = {"ModelTimeoutException"}
_TRANSIENT_CODES = {
    "InternalServerException",
    "ServiceUnavailableException",
    "ModelNotReadyException",
    "ModelErrorException",
}
_CONFIG_CODES = {
    "UnrecognizedClientException",
    "ExpiredTokenException",
    "InvalidSignatureException",
}


def make_bedrock_client(region: str, profile: str, timeout_sec: float) -> Any:
    """Create a ``bedrock-runtime`` boto3 client (raises :class:`LLMConfigError` if unusable)."""
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise LLMConfigError(
            "the bedrock provider requires the 'bedrock' extra "
            "(pip install idea2hypothesis[bedrock])"
        ) from exc
    boto_cfg = BotoConfig(
        region_name=region,
        retries={"max_attempts": 1, "mode": "standard"},
        connect_timeout=15,
        read_timeout=timeout_sec,
    )
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    if session.get_credentials() is None:
        raise LLMConfigError(
            "no AWS credentials found: set AWS_PROFILE or the standard AWS credential variables"
        )
    return session.client("bedrock-runtime", config=boto_cfg)


class BedrockLLM(ChainedLLM):
    def __init__(
        self,
        *,
        model: str,
        region: str = "",
        profile: str = "",
        timeout_sec: float = 120.0,
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, **kwargs)
        self.region = region or "us-east-1"
        self._client = client or make_bedrock_client(self.region, profile, timeout_sec)

    async def _complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_mode: bool,
        max_tokens: int,
        temperature: float,
    ) -> RawCompletion:
        kwargs = self._build_request(
            model, messages, json_mode=json_mode, max_tokens=max_tokens, temperature=temperature
        )
        try:
            response = await asyncio.to_thread(self._client.converse, **kwargs)
        except Exception as exc:  # noqa: BLE001 - mapped to typed errors below
            raise self._map_error(exc, model) from exc
        return self._parse(response, model)

    @staticmethod
    def _build_request(
        model: str,
        messages: list[ChatMessage],
        *,
        json_mode: bool,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        system_parts = [m.content for m in messages if m.role == "system"]
        if json_mode:
            system_parts.append(_JSON_HINT)
        conversation: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            role = "assistant" if message.role == "assistant" else "user"
            text = message.content if message.content.strip() else "(empty)"
            if conversation and conversation[-1]["role"] == role:
                conversation[-1]["content"].append({"text": text})
            else:
                conversation.append({"role": role, "content": [{"text": text}]})
        if not conversation or conversation[0]["role"] != "user":
            conversation.insert(0, {"role": "user", "content": [{"text": "Continue"}]})

        inference: dict[str, Any] = {
            "maxTokens": max(1, min(max_tokens or 4096, _MAX_OUTPUT_TOKENS))
        }
        if temperature > 0:
            inference["temperature"] = min(max(temperature, 0.0), 1.0)
        request: dict[str, Any] = {
            "modelId": model,
            "messages": conversation,
            "inferenceConfig": inference,
        }
        if system_parts:
            request["system"] = [{"text": "\n\n".join(system_parts)}]
        return request

    @staticmethod
    def _parse(response: dict[str, Any], model: str) -> RawCompletion:
        content = response.get("output", {}).get("message", {}).get("content", [])
        text = "".join(block["text"] for block in content if "text" in block)
        usage = response.get("usage", {})
        stop = response.get("stopReason", "end_turn")
        truncated = stop in ("max_tokens", "length")
        return RawCompletion(
            text=text,
            model=model,
            prompt_tokens=int(usage.get("inputTokens", 0) or 0),
            completion_tokens=int(usage.get("outputTokens", 0) or 0),
            finish_reason="length" if truncated else "stop",
            truncated=truncated,
        )

    @staticmethod
    def _map_error(exc: Exception, model: str) -> Exception:
        name = type(exc).__name__
        code = ""
        status: int | None = None
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            code = str(response.get("Error", {}).get("Code", ""))
            status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        message = f"bedrock {model}: {code or name}: {exc}"
        if name in ("NoCredentialsError", "PartialCredentialsError") or code in _CONFIG_CODES:
            return LLMConfigError(message)
        if code in _RATE_LIMIT_CODES:
            return LLMRateLimited(message)
        if code in _TIMEOUT_CODES or "Timeout" in name:
            return LLMTimeout(message)
        if code in _TRANSIENT_CODES or name in ("EndpointConnectionError", "ConnectionClosedError"):
            return LLMResponseError(message, status=status, retryable=True)
        return LLMResponseError(message, status=status)
