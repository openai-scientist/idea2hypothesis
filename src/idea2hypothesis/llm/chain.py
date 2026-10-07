"""Base class shared by providers: model fallback chain, retry, pricing, tag stripping."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from idea2hypothesis.config import Pricing
from idea2hypothesis.llm.models import (
    ChatMessage,
    LLMConfigError,
    LLMError,
    LLMResponse,
)
from idea2hypothesis.llm.parsing import strip_thinking_tags
from idea2hypothesis.llm.retry import call_with_retry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawCompletion:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    truncated: bool = False


class ChainedLLM(ABC):
    """Tries the primary model, then each fallback; every model gets bounded retries."""

    def __init__(
        self,
        model: str,
        *,
        fallback_models: Sequence[str] = (),
        max_retries: int = 3,
        retry_base_delay: float = 2.0,
        temperature: float = 0.4,
        max_tokens: int = 4096,
        pricing: Pricing | None = None,
        strip_thinking: bool = True,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not model:
            raise LLMConfigError("llm.model is required")
        self.models = [model, *fallback_models]
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pricing = pricing
        self.strip_thinking = strip_thinking
        self._sleep = sleep

    @abstractmethod
    async def _complete(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        json_mode: bool,
        max_tokens: int,
        temperature: float,
    ) -> RawCompletion: ...

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        full: list[ChatMessage] = list(messages)
        if system:
            full.insert(0, ChatMessage("system", system))
        tokens = max_tokens or self.max_tokens
        temp = self.temperature if temperature is None else temperature

        last: LLMError | None = None
        for model in self.models:
            try:
                raw = await call_with_retry(
                    lambda model=model: self._complete(
                        model, full, json_mode=json_mode, max_tokens=tokens, temperature=temp
                    ),
                    max_retries=self.max_retries,
                    base_delay=self.retry_base_delay,
                    label=f"model {model}",
                    sleep=self._sleep,
                )
            except LLMConfigError:
                raise
            except LLMError as exc:
                logger.warning("Model %s failed: %s", model, exc)
                last = exc
                continue
            return self._to_response(raw)
        assert last is not None
        raise last

    def _to_response(self, raw: RawCompletion) -> LLMResponse:
        text = strip_thinking_tags(raw.text) if self.strip_thinking else raw.text
        cost = self.pricing.cost(raw.prompt_tokens, raw.completion_tokens) if self.pricing else None
        return LLMResponse(
            text=text,
            model=raw.model,
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
            cost_usd=cost,
            finish_reason=raw.finish_reason,
            truncated=raw.truncated,
        )
