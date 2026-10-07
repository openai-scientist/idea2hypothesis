"""Shared HTTP plumbing for literature providers: retry, rate spacing, circuit breaker."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]
MAX_WAIT_SEC = 60.0
LONG_RETRY_AFTER_SEC = 300.0
_RETRYABLE_STATUS = frozenset({500, 502, 503, 504})


class ProviderError(Exception):
    """A provider request failed after bounded retries."""


class CircuitBreaker:
    """Three-state breaker (closed, open, half-open) that stops hammering a limited API."""

    def __init__(
        self,
        name: str,
        *,
        threshold: int = 3,
        cooldown_sec: float = 120.0,
        max_cooldown: float = 600.0,
    ) -> None:
        self.name = name
        self._threshold = threshold
        self._initial_cooldown = cooldown_sec
        self._cooldown = cooldown_sec
        self._max_cooldown = max_cooldown
        self._state = "closed"
        self._failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> str:
        return self._state

    def allow(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open":
            if time.monotonic() - self._opened_at >= self._cooldown:
                self._state = "half_open"
                return True
            return False
        return True

    def on_success(self) -> None:
        self._failures = 0
        if self._state != "closed":
            logger.info("%s circuit breaker closed", self.name)
            self._state = "closed"
            self._cooldown = self._initial_cooldown

    def on_failure(self) -> bool:
        """Record a throttling failure; returns True if the breaker is now open."""
        self._failures += 1
        if self._state == "half_open":
            self._cooldown = min(self._cooldown * 2, self._max_cooldown)
            self._trip()
            return True
        if self._failures >= self._threshold:
            self._trip()
            return True
        return False

    def _trip(self) -> None:
        self._state = "open"
        self._opened_at = time.monotonic()
        logger.warning("%s circuit breaker open for %.0fs", self.name, self._cooldown)


class RateSpacer:
    """Enforces a minimum interval between consecutive requests of one provider."""

    def __init__(self, min_interval: float, sleep: Sleep = asyncio.sleep) -> None:
        self._interval = min_interval
        self._sleep = sleep
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            elapsed = time.monotonic() - self._last
            if elapsed < self._interval:
                await self._sleep(self._interval - elapsed)
            self._last = time.monotonic()


def _retry_after(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    return float(2 ** (attempt + 1))


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    provider: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    json_body: Any = None,
    max_retries: int = 3,
    timeout: float = 30.0,  # noqa: ASYNC109 - forwarded to httpx, not an asyncio timeout
    breaker: CircuitBreaker | None = None,
    spacer: RateSpacer | None = None,
    sleep: Sleep = asyncio.sleep,
) -> httpx.Response:
    """Send a request with bounded retries on 429/5xx/transport errors.

    Raises :class:`ProviderError` when retries are exhausted or the error is not retryable.
    API keys must never be placed in ``url`` so error messages stay free of secrets.
    """
    last_error = "no attempt made"
    for attempt in range(max(1, max_retries)):
        if breaker is not None and not breaker.allow():
            raise ProviderError(f"{provider}: circuit breaker open")
        if spacer is not None:
            await spacer.wait()
        try:
            response = await client.request(
                method, url, params=params, headers=headers, json=json_body, timeout=timeout
            )
        except httpx.TimeoutException:
            last_error = f"timeout after {timeout:.0f}s"
        except httpx.TransportError as exc:
            last_error = f"transport error: {type(exc).__name__}"
        else:
            status = response.status_code
            if status < 400:
                if breaker is not None:
                    breaker.on_success()
                return response
            if status == 429:
                last_error = "HTTP 429 (rate limited)"
                if breaker is not None and breaker.on_failure():
                    raise ProviderError(f"{provider}: {last_error}; circuit breaker opened")
                wait = _retry_after(response, attempt)
                if wait > LONG_RETRY_AFTER_SEC:
                    raise ProviderError(
                        f"{provider}: {last_error}; Retry-After {wait:.0f}s too long"
                    )
                if attempt < max_retries - 1:
                    await sleep(min(wait, MAX_WAIT_SEC) + random.uniform(0, 0.2 * wait))
                continue
            if status in _RETRYABLE_STATUS:
                last_error = f"HTTP {status}"
            else:
                raise ProviderError(f"{provider}: HTTP {status}")
        if attempt < max_retries - 1:
            wait = min(2.0**attempt, MAX_WAIT_SEC)
            await sleep(wait + random.uniform(0, wait * 0.2))
    raise ProviderError(f"{provider}: {last_error} after {max_retries} attempts")
