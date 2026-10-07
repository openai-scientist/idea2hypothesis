"""Webhook delivery of persisted Platform events.

One delivery task per run reads ``platform_events.jsonl`` in order, posts batches to
``<callback_url>/events`` and records the last delivered ``source_seq`` in
``runs/<id>/delivery.json``. Delivery is at-least-once (consumers dedupe on ``source_seq``),
retries are bounded, a 409 from the consumer stops delivery and cancels the run, and delivery
resumes from the cursor after a restart. A delivery failure never affects the research run:
the events stay replayable through ``GET /runs/{id}/events``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from idea2hypothesis.api.platform_events import PlatformEventLog
from idea2hypothesis.config import ApiConfig
from idea2hypothesis.storage.artifacts import write_json_atomic
from idea2hypothesis.storage.runs import RunStore, utc_now

logger = logging.getLogger(__name__)

PLATFORM_META = "platform.json"
DELIVERY_FILE = "delivery.json"
BATCH_SIZE = 100
REQUEST_TIMEOUT_SEC = 15.0

ACTIVE = "active"
IDLE = "idle"
FAILED = "delivery_failed"
STOPPED = "stopped_by_platform"


class CallbackRejected(ValueError):
    """The callback URL is malformed or its host is not allowed."""


def validate_callback_url(url: str, api: ApiConfig) -> str:
    """Return the events endpoint for ``url`` or raise :class:`CallbackRejected`."""
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise CallbackRejected("callback_url must be an http(s) URL with a host")
    if not api.callback_allow_any:
        allowed = {h.strip().lower() for h in api.callback_allowed_hosts if h.strip()}
        host = parts.hostname.lower()
        netloc = parts.netloc.lower()
        if host not in allowed and netloc not in allowed:
            raise CallbackRejected(f"callback host {host!r} is not in api.callback_allowed_hosts")
    base = url.strip().rstrip("/")
    return base if base.endswith("/events") else f"{base}/events"


def callback_key(api: ApiConfig) -> str | None:
    """Key sent as ``X-Service-Key`` on webhook deliveries, never a caller-supplied one.

    Read from ``api.callback_key_env`` (Platform BE ``POPPER_CALLBACK_KEY``); when that variable
    is unset the inbound service key (``api.service_key_env``) is used, which matches deployments
    where both keys are the same.
    """
    value = os.environ.get(api.callback_key_env, "") or os.environ.get(api.service_key_env, "")
    return value or None


class DeliveryManager:
    """Owns the per-run delivery tasks."""

    def __init__(
        self,
        store: RunStore,
        log: PlatformEventLog,
        api: ApiConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_platform_closed: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.store = store
        self.log = log
        self.api = api
        self._transport = transport
        self._sleep = sleep
        self._on_closed = on_platform_closed
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._closing = False

    # -- sidecar files ------------------------------------------------------

    def _meta_path(self, run_id: str) -> Path:
        return self.store.run_dir(run_id) / PLATFORM_META

    def _delivery_path(self, run_id: str) -> Path:
        return self.store.run_dir(run_id) / DELIVERY_FILE

    def read_meta(self, run_id: str) -> dict[str, Any] | None:
        try:
            return json.loads(self._meta_path(run_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def write_meta(self, run_id: str, meta: dict[str, Any]) -> None:
        write_json_atomic(self._meta_path(run_id), meta)

    def read_delivery(self, run_id: str) -> dict[str, Any]:
        try:
            data = json.loads(self._delivery_path(run_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data.setdefault("delivered_source_seq", 0)
        data.setdefault("status", IDLE)
        return data

    def _write_delivery(self, run_id: str, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now()
        write_json_atomic(self._delivery_path(run_id), state)

    # -- scheduling ---------------------------------------------------------

    def notify(self, run_id: str) -> None:
        """New events exist for ``run_id``; make sure a delivery task is running."""
        if self._closing or self.read_meta(run_id) is None:
            return
        if self.read_delivery(run_id)["status"] == STOPPED:
            return
        task = self._tasks.get(run_id)
        if task is None or task.done():
            self._tasks[run_id] = asyncio.get_running_loop().create_task(self._deliver(run_id))

    def resume_all(self) -> list[str]:
        """Restart delivery for every run whose cursor is behind its event log."""
        resumed = []
        for run_id in self.store.list_run_ids():
            if self.read_meta(run_id) is None:
                continue
            state = self.read_delivery(run_id)
            if (
                state["status"] != STOPPED
                and self.log.last_seq(run_id) > state["delivered_source_seq"]
            ):
                self.notify(run_id)
                resumed.append(run_id)
        return resumed

    async def wait_idle(self) -> None:
        """Wait until no delivery task is running (used by tests and shutdown)."""
        while True:
            pending = [t for t in self._tasks.values() if not t.done()]
            if not pending:
                return
            await asyncio.gather(*pending, return_exceptions=True)

    async def close(self) -> None:
        self._closing = True
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()

    # -- delivery loop ------------------------------------------------------

    async def _deliver(self, run_id: str) -> None:
        meta = self.read_meta(run_id)
        if meta is None:
            return
        url = meta["events_url"]
        state = self.read_delivery(run_id)
        failures = 0
        try:
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT_SEC, transport=self._transport
            ) as client:
                while True:
                    batch = self.log.read(run_id, state["delivered_source_seq"], BATCH_SIZE)
                    if not batch:
                        state.update(status=IDLE, attempts=0)
                        self._write_delivery(run_id, state)
                        self._tasks.pop(run_id, None)
                        return
                    outcome = await self._post(client, url, batch)
                    if outcome == "ok":
                        failures = 0
                        state.update(
                            delivered_source_seq=int(batch[-1]["source_seq"]),
                            status=ACTIVE,
                            attempts=0,
                            last_error=None,
                        )
                        self._write_delivery(run_id, state)
                    elif outcome == "closed":
                        state.update(status=STOPPED, last_error="platform answered 409")
                        self._write_delivery(run_id, state)
                        await self._platform_closed(run_id)
                        return
                    else:
                        failures += 1
                        state.update(status=ACTIVE, attempts=failures, last_error=outcome)
                        if failures >= self.api.delivery_max_attempts:
                            state["status"] = FAILED
                            self._write_delivery(run_id, state)
                            logger.error("delivery for run %s failed: %s", run_id, outcome)
                            return
                        self._write_delivery(run_id, state)
                        await self._sleep(self._backoff(failures))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - delivery must never crash the process
            logger.exception("delivery task for run %s crashed", run_id)

    def _backoff(self, failures: int) -> float:
        return float(min(2.0 ** (failures - 1), self.api.delivery_backoff_max_sec))

    async def _post(self, client: httpx.AsyncClient, url: str, batch: list[dict[str, Any]]) -> str:
        headers = {"Content-Type": "application/json"}
        key = callback_key(self.api)
        if key:
            headers["X-Service-Key"] = key
        try:
            response = await client.post(url, json={"events": batch}, headers=headers)
        except httpx.HTTPError as exc:
            return f"{type(exc).__name__}: {exc}"[:300]
        if response.status_code == 409:
            return "closed"
        if 200 <= response.status_code < 300:
            return "ok"
        return f"HTTP {response.status_code}: {response.text[:200]}"

    async def _platform_closed(self, run_id: str) -> None:
        logger.warning("run %s was closed by the platform (409); cancelling", run_id)
        self._tasks.pop(run_id, None)
        if self._on_closed is not None:
            try:
                await self._on_closed(run_id)
            except Exception:  # noqa: BLE001
                logger.exception("could not cancel run %s after platform 409", run_id)
