"""Token and cost metering around any :class:`LLMPort`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from idea2hypothesis.llm.models import ChatMessage, LLMPort, LLMResponse


class UsageMeter:
    """Accumulates usage. ``cost_usd`` stays ``None`` until a priced response is seen."""

    def __init__(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float | None = None,
        calls: int = 0,
    ) -> None:  # noqa: PLR0913
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost_usd = cost_usd
        self.calls = calls

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> UsageMeter:
        data = data or {}
        return cls(
            int(data.get("prompt_tokens", 0)),
            int(data.get("completion_tokens", 0)),
            data.get("cost_usd"),
            int(data.get("calls", 0)),
        )

    def record(self, response: LLMResponse) -> None:
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.calls += 1
        if response.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + response.cost_usd

    def snapshot(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "calls": self.calls,
        }

    def delta_since(self, before: dict[str, Any]) -> dict[str, Any]:
        cost = self.cost_usd
        prior = before.get("cost_usd")
        delta_cost = None if cost is None else cost - (prior or 0.0)
        return {
            "prompt_tokens": self.prompt_tokens - int(before.get("prompt_tokens", 0)),
            "completion_tokens": self.completion_tokens - int(before.get("completion_tokens", 0)),
            "cost_usd": delta_cost,
            "calls": self.calls - int(before.get("calls", 0)),
        }


class MeteredLLM:
    """Wraps an :class:`LLMPort` and records every response in a :class:`UsageMeter`."""

    def __init__(self, inner: LLMPort, meter: UsageMeter) -> None:
        self._inner = inner
        self.meter = meter

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        response = await self._inner.chat(
            messages,
            system=system,
            json_mode=json_mode,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.meter.record(response)
        return response
