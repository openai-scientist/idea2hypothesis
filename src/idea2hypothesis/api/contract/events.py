"""Contract event log and the projector that turns core events into envelopes."""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from idea2hypothesis.api.contract.artifacts import ArtifactPublisher, PushFailed
from idea2hypothesis.api.contract.mapping import MapContext, Timeline, map_event
from idea2hypothesis.api.platform_events import PlatformEventLog
from idea2hypothesis.pipeline import events as ev
from idea2hypothesis.storage.runs import RunStore

logger = logging.getLogger(__name__)

CONTRACT_FILE = "contract_events.jsonl"


class ContractEventLog(PlatformEventLog):
    """Append-only ``contract_events.jsonl`` with contiguous ``source_seq`` (same mechanism)."""

    def path(self, run_id: str) -> Path:
        return self.store.run_dir(run_id) / CONTRACT_FILE


class ContractProjector:
    """Persists the contract envelopes of every core event not yet projected.

    Artifacts an event refers to are uploaded first; when an upload fails the event (and the
    ones after it) stay pending and are projected on the next call.
    """

    def __init__(
        self,
        store: RunStore,
        log: ContractEventLog,
        publisher: ArtifactPublisher,
        on_closed: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.store = store
        self.log = log
        self.publisher = publisher
        self.listeners: list[Callable[[str], None]] = []
        self._on_closed = on_closed
        self._timelines: dict[str, Timeline] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def timeline(self, run_id: str) -> Timeline:
        if run_id not in self._timelines:
            self._timelines[run_id] = Timeline.from_rows(self.log.read(run_id))
        return self._timelines[run_id]

    def forget(self, run_id: str) -> None:
        self._timelines.pop(run_id, None)

    async def project(self, run_id: str) -> int:
        """Persist envelopes for every pending core event; returns how many were added."""
        lock = self._locks.setdefault(run_id, asyncio.Lock())
        async with lock:
            added = await self._project(run_id)
        if added:
            for listener in self.listeners:
                listener(run_id)
        return added

    async def _project(self, run_id: str) -> int:
        state = self.log.state(run_id)
        pending = self.store.read_events(run_id, after_seq=state.memory_core)
        timeline = self.timeline(run_id)
        added = 0
        for event in pending:
            try:
                ctx = await self._context(run_id, event)
            except PushFailed as exc:
                logger.warning("run %s: %s is waiting for an artifact upload: %s",
                               run_id, event.type, exc)  # fmt: skip
                if exc.closed and self._on_closed is not None:
                    await self._on_closed(run_id)
                break
            before = copy.copy(timeline)
            try:
                envelopes = map_event(event, timeline, ctx)
            except Exception:  # noqa: BLE001 - a mapping problem must not stop the run
                logger.exception("could not map %s of run %s", event.type, run_id)
                timeline.phase, timeline.step = before.phase, before.step
                timeline.before_pause = before.before_pause
                envelopes = []
            self.log.append(run_id, event.seq, envelopes)
            added += len(envelopes)
        return added

    async def _context(self, run_id: str, event: ev.Event) -> MapContext:
        ctx = MapContext(ids=await self.publisher.before(run_id, event))
        record = self.store.read_run(run_id)
        ctx.actors = dict(record.get("contract_gate_actors") or {})
        if event.type == ev.GATE_OPENED:
            ctx.gate_state = self.store.read_checkpoint(run_id).get("gates", {}).get(
                event.data.get("gate_id")
            )
        if event.type == ev.REVIEW_COMPLETED:
            ctx.review = self.store.artifacts(run_id).read_json(8, "review.json")
        return ctx
