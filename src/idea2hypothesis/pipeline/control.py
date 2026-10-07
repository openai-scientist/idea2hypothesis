"""Cooperative pause/cancel requests and the one-worker-per-run lock."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

PAUSE = "pause"
CANCEL = "cancel"


class RunInterrupted(Exception):
    """Raised at a safe point when a pause or cancel was requested."""

    def __init__(self, kind: str, reason: str = "") -> None:
        super().__init__(f"{kind}: {reason}" if reason else kind)
        self.kind = kind
        self.reason = reason


class RunBusyError(Exception):
    """Another worker already executes this run in this process."""


class RunControl:
    """In-process control flags and per-run worker locks."""

    def __init__(self) -> None:
        self._requests: dict[str, tuple[str, str]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def request(self, run_id: str, kind: str, reason: str = "") -> None:
        """Ask the worker to stop at its next safe point; cancel overrides pause."""
        if kind not in (PAUSE, CANCEL):
            raise ValueError(f"unknown control request {kind!r}")
        current = self._requests.get(run_id)
        if current and current[0] == CANCEL:
            return
        self._requests[run_id] = (kind, reason)

    def pending(self, run_id: str) -> tuple[str, str] | None:
        return self._requests.get(run_id)

    def clear(self, run_id: str) -> None:
        self._requests.pop(run_id, None)

    def check(self, run_id: str) -> None:
        pending = self._requests.get(run_id)
        if pending:
            raise RunInterrupted(*pending)

    def is_active(self, run_id: str) -> bool:
        lock = self._locks.get(run_id)
        return bool(lock and lock.locked())

    @asynccontextmanager
    async def worker(self, run_id: str) -> AsyncIterator[None]:
        """Hold the run's worker slot; raises :class:`RunBusyError` if already taken."""
        lock = self._locks.setdefault(run_id, asyncio.Lock())
        if lock.locked():
            raise RunBusyError(run_id)
        async with lock:
            yield
