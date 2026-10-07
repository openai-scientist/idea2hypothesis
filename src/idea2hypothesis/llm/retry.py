"""Shared retry/backoff loop used by every provider."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from idea2hypothesis.llm.models import LLMError, LLMRateLimited

logger = logging.getLogger(__name__)

T = TypeVar("T")

MAX_BACKOFF_SEC = 300.0


def backoff_delay(attempt: int, base: float, retry_after: float | None = None) -> float:
    """Exponential delay with jitter, honouring a provider ``Retry-After`` hint."""
    delay = min(base * (2**attempt), MAX_BACKOFF_SEC)
    if retry_after is not None:
        delay = max(delay, min(retry_after, MAX_BACKOFF_SEC))
    return delay + random.uniform(0, delay * 0.3)


async def call_with_retry(
    func: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay: float,
    label: str,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run ``func`` up to ``max_retries`` times, retrying only retryable :class:`LLMError`."""
    attempts = max(1, max_retries)
    last: LLMError | None = None
    for attempt in range(attempts):
        try:
            return await func()
        except LLMError as exc:
            if not exc.retryable:
                raise
            last = exc
            if attempt >= attempts - 1:
                break
            retry_after = exc.retry_after if isinstance(exc, LLMRateLimited) else None
            delay = backoff_delay(attempt, base_delay, retry_after)
            logger.info(
                "%s failed (%s); retry %d/%d in %.1fs", label, exc, attempt + 1, attempts, delay
            )
            await sleep(delay)
    assert last is not None
    raise last
