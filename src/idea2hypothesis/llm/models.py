"""LLM data models, errors and the :class:`LLMPort` protocol."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        return cls("assistant", content)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    finish_reason: str = ""
    truncated: bool = False


class LLMError(Exception):
    """Base class for provider failures."""

    retryable: bool = False


class LLMConfigError(LLMError):
    """Missing or invalid LLM configuration (for example absent credentials)."""


class LLMTimeout(LLMError):
    retryable = True


class LLMRateLimited(LLMError):
    retryable = True

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMResponseError(LLMError):
    """The provider answered with an error or an unusable payload."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@runtime_checkable
class LLMPort(Protocol):
    """Asynchronous chat interface used by every stage."""

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse: ...
