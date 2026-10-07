"""Application service shared by the Platform engine routes and the stage review routes.

Both route groups drive the same engine (:mod:`idea2hypothesis.pipeline.runner`) through
:class:`RunService`; the service owns background tasks, the run store, the Platform event
projection and webhook delivery, and maps HTTP-level requests onto engine operations.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import re
import shutil
from collections.abc import Callable, Coroutine
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import httpx

from idea2hypothesis.api.platform_events import (
    PlatformEventLog,
    PlatformProjector,
)
from idea2hypothesis.api.schemas import GateAnswerRequest, RunCreateRequest
from idea2hypothesis.api.webhooks import (
    CallbackRejected,
    DeliveryManager,
    validate_callback_url,
)
from idea2hypothesis.config import Config, LLMConfig
from idea2hypothesis.llm.factory import build_llm
from idea2hypothesis.llm.models import LLMConfigError
from idea2hypothesis.pipeline import events as ev
from idea2hypothesis.pipeline import gates
from idea2hypothesis.pipeline.contracts import missing_inputs, validate_stage
from idea2hypothesis.pipeline.control import RunBusyError, RunControl, RunInterrupted
from idea2hypothesis.pipeline.events import Event
from idea2hypothesis.pipeline.models import (
    GateAnswer,
    RunRequest,
    RunStatus,
    Services,
    Stage,
    StageResult,
    StageStatus,
)
from idea2hypothesis.pipeline.runner import (
    RunStateError,
    apply_gate_answer,
    build_context,
    cancel_run,
    create_run,
    execute_stage,
    new_run_id,
    pause_run,
    resume_pipeline,
    run_pipeline,
)
from idea2hypothesis.pipeline.services import build_services
from idea2hypothesis.pipeline.usage import MeteredLLM, UsageMeter
from idea2hypothesis.storage.artifacts import STAGE_COUNT
from idea2hypothesis.storage.runs import RunNotFoundError, RunStore, utc_now

logger = logging.getLogger(__name__)

STAGE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,80}$")
STAGE_NAMES = {s.value: s.name for s in Stage}
STAGE_RUN_PAUSE = "stage_run"
_PROVIDERS = {
    "bedrock": "bedrock",
    "openai": "openai_compatible",
    "openai_compatible": "openai_compatible",
}


class ServiceError(Exception):
    """An error with an HTTP status; ``detail`` is returned as ``{"detail": ...}``."""

    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


def _conflict(code: str, message: str = "") -> ServiceError:
    return ServiceError(409, {"code": code, "message": message} if message else code)


class RunService:
    """Owns the run store, engine services, background tasks and Platform delivery."""

    def __init__(
        self,
        config: Config,
        *,
        services: Services | None = None,
        callback_transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Coroutine[Any, Any, None]] = asyncio.sleep,
    ) -> None:
        self.config = config
        self._injected = services
        self._services: Services | None = None
        self.store = services.store if services else RunStore(config.runs_root)
        self.control = services.control if services else RunControl()
        self.log = PlatformEventLog(self.store)
        self.projector = PlatformProjector(self.store, self.log)
        self.deliveries = DeliveryManager(
            self.store,
            self.log,
            config.api,
            transport=callback_transport,
            sleep=sleep,
            on_platform_closed=self._platform_closed,
        )
        self.projector.listeners.append(self.deliveries.notify)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._started: dict[str, asyncio.Event] = {}
        self._create_lock = asyncio.Lock()
        self._edit_locks: dict[str, asyncio.Lock] = {}
        # What pause/cancel need from the engine services; no LLM credentials involved.
        self._control_view = SimpleNamespace(
            store=self.store, control=self.control, events=self, config=config
        )

    # ------------------------------------------------------------------
    # Engine services and event sink
    # ------------------------------------------------------------------

    async def emit(self, event: Event) -> None:
        """Core event sink: project to Platform events, then signal run creation."""
        await self.projector.emit(event)
        if event.type == ev.RUN_STARTED and event.run_id in self._started:
            self._started[event.run_id].set()

    def services(self) -> Services:
        """The engine services; built from configuration on first use."""
        if self._services is not None:
            return self._services
        if self._injected is not None:
            built = self._injected
            previous = built.events
            built.events = _Fanout(self, previous) if previous is not None else self
        else:
            try:
                built = build_services(self.config, events=self, store=self.store)
            except LLMConfigError as exc:
                raise ServiceError(
                    503, {"code": "LLM_NOT_CONFIGURED", "message": str(exc)}
                ) from exc
            built.control = self.control
        self._services = built
        return built

    def _services_for(self, provider: str | None, model: str | None) -> Services:
        base = self.services()
        if not provider and not model:
            return base
        if self._injected is not None:
            raise ServiceError(422, "llm overrides are not available with injected services")
        llm_cfg: LLMConfig = base.config.llm
        if provider:
            if provider not in _PROVIDERS:
                raise ServiceError(
                    422, f"llm_provider {provider!r} is not supported; use 'bedrock' or 'openai'"
                )
            llm_cfg = dataclasses.replace(llm_cfg, provider=_PROVIDERS[provider])
        if model:
            llm_cfg = dataclasses.replace(llm_cfg, model=model)
        try:
            llm = build_llm(llm_cfg)
        except LLMConfigError as exc:
            raise ServiceError(503, {"code": "LLM_NOT_CONFIGURED", "message": str(exc)}) from exc
        cfg = dataclasses.replace(base.config, llm=llm_cfg)
        return dataclasses.replace(base, config=cfg, llm=llm)

    def spawn(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("background task failed", exc_info=task.exception())

    async def wait_idle(self) -> None:
        """Wait for background run tasks and webhook deliveries (tests and shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
        await self.deliveries.wait_idle()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> dict[str, list[str]]:
        """Recover runs left ``running`` by a crash, finish projections, resume deliveries."""
        recovered = await self._recover_interrupted()
        for run_id in self.store.list_run_ids():
            self.projector.project(run_id)
        resumed = self.deliveries.resume_all()
        return {"recovered": recovered, "delivery_resumed": resumed}

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*list(self._tasks), return_exceptions=True)
        await self.deliveries.close()

    async def _recover_interrupted(self) -> list[str]:
        # Mirrors ``runner.recover_interrupted`` without needing LLM credentials.
        recovered: list[str] = []
        for run_id in self.store.list_run_ids():
            record = self.store.read_run(run_id)
            if record["status"] != RunStatus.RUNNING.value or self.control.is_active(run_id):
                continue
            self.store.update_run(
                run_id, status=RunStatus.PAUSED.value, pause_reason="interrupted", worker=None
            )
            event = self.store.append_event(
                run_id,
                ev.RUN_PAUSED,
                attempt=int(record["attempt"]),
                data={"reason": "interrupted"},
            )
            await self.emit(event)
            recovered.append(run_id)
        return recovered

    async def _platform_closed(self, run_id: str) -> None:
        with contextlib.suppress(RunStateError):
            await cancel_run(run_id, self._control_view, reason="platform_closed")  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Lookup and state
    # ------------------------------------------------------------------

    def resolve(self, identifier: str) -> str:
        """Run id for a run id or a Platform run id."""
        if self.store.exists(identifier):
            return identifier
        found = self.store.find_by_platform(identifier)
        if found:
            return found
        raise ServiceError(404, "Run not found")

    def record(self, run_id: str) -> dict[str, Any]:
        try:
            return self.store.read_run(run_id)
        except RunNotFoundError as exc:
            raise ServiceError(404, f"Run {run_id} not found") from exc

    def stage_run_id(self, run_id: str) -> str:
        if not STAGE_RUN_ID.match(run_id):
            raise ServiceError(400, f"Invalid run id: {run_id}")
        if not self.store.exists(run_id):
            raise ServiceError(404, f"Run {run_id} not found")
        return run_id

    def state(self, run_id: str) -> dict[str, Any]:
        """Wire state of a run for the Platform routes."""
        self.projector.project(run_id)
        record = self.record(run_id)
        status = RunStatus(record["status"])
        error = record.get("error")
        message: str | None = None
        api_status = status.value
        if status is RunStatus.CANCELLED:
            api_status, message = "failed", "Cancelled by user"
        elif status is RunStatus.FAILED and error:
            message = f"{error['code']}: {error['message']}"
        elif status is RunStatus.PAUSED:
            message = record.get("pause_reason")
        cost = (record.get("usage") or {}).get("cost_usd")
        return {
            "popper_run_id": run_id,
            "status": api_status,
            "cost_usd": None if cost is None else f"{float(cost):.4f}",
            "message": message,
            "last_source_seq": self.log.last_seq(run_id),
        }

    def events(self, run_id: str, after: int, limit: int) -> list[dict[str, Any]]:
        self.projector.project(run_id)
        return self.log.read(run_id, after, limit)

    # ------------------------------------------------------------------
    # Platform runs
    # ------------------------------------------------------------------

    async def create_platform_run(self, req: RunCreateRequest) -> tuple[dict[str, Any], bool]:
        """Start (or find) the run for ``req.platform_run_id``; ``True`` when newly created."""
        try:
            events_url = validate_callback_url(req.callback_url, self.config.api)
        except CallbackRejected as exc:
            raise ServiceError(422, str(exc)) from exc
        async with self._create_lock:
            existing = self.store.find_by_platform(req.platform_run_id)
            if existing:
                return self.state(existing), False
            services = self.services()
            run_id = self.store.claim_platform_id(req.platform_run_id, new_run_id())
            if self.store.exists(run_id):
                return self.state(run_id), False
            self.store.run_dir(run_id).mkdir(parents=True, exist_ok=True)
            self.deliveries.write_meta(
                run_id,
                {
                    "platform_run_id": req.platform_run_id,
                    "callback_url": req.callback_url,
                    "events_url": events_url,
                    "created_at": utc_now(),
                },
            )
            request = RunRequest(
                topic=req.topic,
                domains=tuple(req.domains),
                review_mode=req.review_mode,
                run_id=run_id,
                platform_run_id=req.platform_run_id,
                budget_usd=float(req.budget_usd),
            )
            await self._start(run_pipeline(request, services), run_id)
        return self.state(run_id), True

    async def _start(self, coro: Coroutine[Any, Any, Any], run_id: str) -> None:
        """Run ``coro`` in the background and return once the run record exists."""
        started = self._started.setdefault(run_id, asyncio.Event())
        task = self.spawn(coro)
        waiter = asyncio.ensure_future(started.wait())
        try:
            await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
            self._started.pop(run_id, None)
        if task.done() and not started.is_set():
            exc = task.exception()
            if isinstance(exc, ValueError):
                raise ServiceError(422, str(exc)) from exc
            if exc is not None:
                raise exc

    async def answer_gate(self, run_id: str, gate_id: str, body: GateAnswerRequest) -> str:
        record = self.record(run_id)
        gate = record.get("gate")
        answers: dict[str, str] = record.get("platform_gate_answers") or {}
        if not gate or gate.get("status") != "open":
            if answers.get(gate_id) == body.option_id:
                return "Gate already resolved"
            raise _conflict("GATE_NOT_OPEN")
        if gate["gate_id"] != gate_id:
            raise ServiceError(404, "Gate ID mismatch")
        answer = _core_answer(body)
        services = self.services()
        try:
            result = await apply_gate_answer(run_id, gate_id, answer, services)
        except gates.GateError as exc:
            raise ServiceError(422, str(exc)) from exc
        self.store.update_run(run_id, platform_gate_answers={**answers, gate_id: body.option_id})
        if result.status is not RunStatus.FAILED:
            self.spawn(resume_pipeline(run_id, services))
        return "Gate answer accepted"

    async def pause(self, run_id: str) -> None:
        try:
            await pause_run(run_id, self._control_view, "user")  # type: ignore[arg-type]
        except RunStateError as exc:
            raise _conflict("RUN_FINISHED") from exc

    async def resume(self, run_id: str) -> str:
        record = self.record(run_id)
        status = RunStatus(record["status"])
        if status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            raise _conflict("RUN_FINISHED")
        if self.control.is_active(run_id):
            self.control.clear(run_id)  # a pause that was requested but not yet taken
            return "Run resumed"
        if (
            status is RunStatus.AWAITING_REVIEW
            and (record.get("gate") or {}).get("status") == "open"
        ):
            return "Run awaits a gate decision"
        self.spawn(resume_pipeline(run_id, self.services()))
        return "Run resumed"

    async def cancel(self, run_id: str) -> str:
        record = self.record(run_id)
        if RunStatus(record["status"]) is RunStatus.CANCELLED:
            return "Run already cancelled"
        try:
            await cancel_run(run_id, self._control_view, "user")  # type: ignore[arg-type]
        except RunStateError as exc:
            raise _conflict("RUN_FINISHED") from exc
        return "Run cancelled"

    # ------------------------------------------------------------------
    # Stage API runs
    # ------------------------------------------------------------------

    async def start_phase1(
        self,
        topic: str,
        domains: list[str],
        *,
        auto_approve: bool,
        provider: str | None,
        model: str | None,
    ) -> str:
        services = self._services_for(provider, model)
        run_id = new_run_id()
        request = RunRequest(
            topic=topic,
            domains=tuple(domains),
            run_id=run_id,
            review_mode="auto" if auto_approve else "copilot",
        )
        await self._start(run_pipeline(request, services), run_id)
        return run_id

    def latest_run(self) -> str | None:
        runs = [
            (self.store.read_run(r).get("created_at", ""), r) for r in self.store.list_run_ids()
        ]
        return max(runs)[1] if runs else None

    def list_runs(self) -> list[dict[str, Any]]:
        rows = []
        for run_id in reversed(self.store.list_run_ids()):
            record = self.store.read_run(run_id)
            rows.append(
                {
                    "run_id": run_id,
                    "topic": record.get("topic", ""),
                    "status": record["status"],
                    "has_stage8_hypotheses": self.store.artifacts(run_id).exists(
                        8, "hypotheses.md"
                    ),
                }
            )
        return rows

    async def delete_run(self, run_id: str) -> None:
        record = self.record(run_id)
        if self.control.is_active(run_id):
            raise _conflict("RUN_ACTIVE", "stop the run before deleting it")
        if record.get("platform_run_id"):
            raise _conflict("PLATFORM_RUN", "runs created by the Platform cannot be deleted here")
        shutil.rmtree(self.store.run_dir(run_id))
        self.log._state.pop(run_id, None)  # noqa: SLF001 - forget cached sequence numbers
        self.store._next_seq.pop(run_id, None)  # noqa: SLF001

    async def create_stage_run(
        self, topic: str, run_id: str | None, provider: str | None, model: str | None
    ) -> tuple[str, Services]:
        """Create the run for ``POST /api/stage1/run`` (or reuse an existing one)."""
        services = self._services_for(provider, model)
        if run_id and not STAGE_RUN_ID.match(run_id):
            raise ServiceError(400, f"Invalid run id: {run_id}")
        if run_id and self.store.exists(run_id):
            self.store.update_run(run_id, topic=topic.strip())
            return run_id, services
        request = RunRequest(topic=topic, review_mode="auto", run_id=run_id or new_run_id())
        try:
            created = await create_run(request, services)
        except ValueError as exc:
            raise ServiceError(422, str(exc)) from exc
        return created, services

    async def run_stage(
        self,
        run_id: str,
        stage: Stage,
        *,
        services: Services | None = None,
        auto_approve: bool | None = None,
    ) -> dict[str, Any]:
        """Execute one stage of an existing run (the same engine code the runner uses)."""
        services = services or self.services()
        record = self.record(run_id)
        status = RunStatus(record["status"])
        if status is RunStatus.CANCELLED:
            raise _conflict("RUN_CANCELLED")
        if (
            status is RunStatus.AWAITING_REVIEW
            and (record.get("gate") or {}).get("status") == "open"
        ):
            raise _conflict("GATE_OPEN", "answer the open gate before running stages by hand")
        try:
            async with self.control.worker(run_id):
                return await self._run_stage_locked(run_id, stage, services, auto_approve)
        except RunBusyError as exc:
            raise _conflict("RUN_BUSY", "the run already has a worker") from exc

    async def _run_stage_locked(
        self, run_id: str, stage: Stage, services: Services, auto_approve: bool | None
    ) -> dict[str, Any]:
        store = self.store
        art = store.artifacts(run_id)
        missing = missing_inputs(stage, art)
        if missing:
            raise _conflict("MISSING_INPUT", f"missing upstream artifacts: {', '.join(missing)}")
        self._invalidate(run_id, int(stage))
        if auto_approve is False and store.read_run(run_id)["review_mode"] in ("auto", "light"):
            store.update_run(run_id, review_mode="copilot")
        record = store.update_run(
            run_id,
            status=RunStatus.RUNNING.value,
            pause_reason=None,
            error=None,
            current_stage=int(stage),
        )
        attempt = int(record["attempt"])
        await self._emit(run_id, ev.STAGE_STARTED, stage=int(stage), attempt=attempt)
        meter = UsageMeter.from_dict(record.get("usage"))
        llm = MeteredLLM(services.llm, meter)
        reviewer = MeteredLLM(services.reviewer, meter) if services.reviewer else None
        before = meter.snapshot()
        context = build_context(run_id, stage, services, record=record, llm=llm, reviewer=reviewer)
        try:
            result = await execute_stage(stage, context)
        except RunInterrupted as exc:
            await self._interrupted(run_id, attempt, exc)
            raise _conflict("RUN_INTERRUPTED", exc.kind) from exc
        except BaseException:
            store.update_run(run_id, status=RunStatus.PAUSED.value, pause_reason="interrupted")
            raise
        usage = meter.delta_since(before)
        store.update_run(run_id, usage=meter.snapshot())
        await self._finish_stage(run_id, stage, attempt, result, usage)
        return {
            "run_id": run_id,
            "stage": int(stage),
            "stage_name": STAGE_NAMES[int(stage)],
            "status": result.status.value,
            "artifacts": list(result.artifacts),
            "warnings": list(result.warnings),
            "error": result.error.to_dict() if result.error else None,
        }

    async def _interrupted(self, run_id: str, attempt: int, exc: RunInterrupted) -> None:
        self.control.clear(run_id)
        if exc.kind == "cancel":
            self.store.update_run(run_id, status=RunStatus.CANCELLED.value, pause_reason=None)
            await self._emit(run_id, ev.RUN_CANCELLED, attempt=attempt, data={"reason": exc.reason})
            return
        reason = exc.reason or "user"
        self.store.update_run(run_id, status=RunStatus.PAUSED.value, pause_reason=reason)
        await self._emit(run_id, ev.RUN_PAUSED, attempt=attempt, data={"reason": reason})

    async def _finish_stage(
        self, run_id: str, stage: Stage, attempt: int, result: StageResult, usage: dict[str, Any]
    ) -> None:
        store = self.store
        checkpoint = store.read_checkpoint(run_id)
        if result.status is not StageStatus.COMPLETED or result.error is not None:
            error = result.error.to_dict() if result.error else {"code": "FAILED", "message": ""}
            checkpoint["stages"][str(int(stage))] = {
                "status": "failed",
                "attempt": attempt,
                "error": error,
                "usage": usage,
            }
            store.write_checkpoint(run_id, checkpoint)
            await self._emit(
                run_id,
                ev.STAGE_FAILED,
                stage=int(stage),
                attempt=attempt,
                data={**error, "warnings": list(result.warnings), "usage": usage},
            )
            store.update_run(
                run_id, status=RunStatus.FAILED.value, error={**error, "stage": int(stage)}
            )
            await self._emit(run_id, ev.RUN_FAILED, stage=int(stage), attempt=attempt, data=error)
            return
        checkpoint["stages"][str(int(stage))] = {
            "status": "completed",
            "attempt": attempt,
            "completed_at": utc_now(),
            "artifacts": list(result.artifacts),
            "usage": usage,
        }
        store.write_checkpoint(run_id, checkpoint)
        await self._emit(
            run_id,
            ev.STAGE_COMPLETED,
            stage=int(stage),
            attempt=attempt,
            data={
                "artifacts": list(result.artifacts),
                "evidence_refs": list(result.evidence_refs),
                "warnings": list(result.warnings),
                "usage": usage,
            },
        )
        store.update_run(
            run_id, status=RunStatus.PAUSED.value, pause_reason=STAGE_RUN_PAUSE, current_stage=None
        )
        await self._emit(run_id, ev.RUN_PAUSED, attempt=attempt, data={"reason": STAGE_RUN_PAUSE})

    async def _emit(
        self,
        run_id: str,
        type_: str,
        *,
        stage: int | None = None,
        attempt: int = 1,
        data: dict[str, Any] | None = None,
    ) -> None:
        event = self.store.append_event(run_id, type_, stage=stage, attempt=attempt, data=data)
        await self.emit(event)

    # ------------------------------------------------------------------
    # Invalidation and edits
    # ------------------------------------------------------------------

    def _invalidate(self, run_id: str, from_stage: int) -> bool:
        """Archive the outputs of ``from_stage`` and later stages and drop their checkpoints.

        Returns ``True`` when something was archived. The attempt number is then incremented,
        gates of still valid stages are re-keyed so they do not reopen, and gates of
        invalidated stages are forgotten.
        """
        store = self.store
        art = store.artifacts(run_id)
        record = store.read_run(run_id)
        attempt = int(record["attempt"])
        moved = art.invalidate_from(from_stage, attempt)
        checkpoint = store.read_checkpoint(run_id)
        for n in range(from_stage, STAGE_COUNT + 1):
            checkpoint["stages"].pop(str(n), None)
        if not moved:
            store.write_checkpoint(run_id, checkpoint)
            return False
        new_attempt = attempt + 1
        kept: dict[str, Any] = {}
        for gate_id, state in (checkpoint.get("gates") or {}).items():
            if int(state["stage"]) >= from_stage:
                continue  # its stage output no longer exists
            spec = gates.gate_by_kind(state["kind"])
            moved_state = {
                **state,
                "attempt": new_attempt,
                "gate_id": gates.make_gate_id(spec, new_attempt),
            }
            kept[gate_id] = state
            kept[moved_state["gate_id"]] = moved_state
        checkpoint["gates"] = kept
        checkpoint["attempt"] = new_attempt
        store.write_checkpoint(run_id, checkpoint)
        updates: dict[str, Any] = {"attempt": new_attempt}
        gate = record.get("gate")
        if gate:
            updates["gate"] = kept.get(
                gates.make_gate_id(gates.gate_by_kind(gate["kind"]), new_attempt)
            )
        store.update_run(run_id, **updates)
        return True

    def edit_lock(self, run_id: str) -> asyncio.Lock:
        return self._edit_locks.setdefault(run_id, asyncio.Lock())

    async def edit_artifacts(
        self, run_id: str, stage: Stage, writes: dict[str, str], *, note: str = ""
    ) -> dict[str, Any]:
        """Replace artifact files of a completed stage; validate; invalidate later stages."""
        record = self.record(run_id)
        status = RunStatus(record["status"])
        if status is RunStatus.CANCELLED:
            raise _conflict("RUN_CANCELLED")
        if status.terminal and record.get("platform_run_id"):
            raise _conflict("RUN_FINISHED", "a finished Platform run is immutable")
        if self.control.is_active(run_id):
            raise _conflict("RUN_BUSY", "the run is executing; pause it before editing")
        async with self.edit_lock(run_id):
            store = self.store
            art = store.artifacts(run_id)
            entry = store.read_checkpoint(run_id)["stages"].get(str(int(stage)))
            if not entry or entry.get("status") != "completed":
                raise _conflict("STAGE_NOT_COMPLETED", f"stage {int(stage)} has not completed")
            backup: dict[str, str | None] = {}
            for name in writes:
                backup[name] = (
                    art.read_text(int(stage), name) if art.exists(int(stage), name) else None
                )
            try:
                for name, text in writes.items():
                    art.write_text(int(stage), name, text)
                findings = validate_stage(stage, art)
            except (OSError, ValueError) as exc:
                self._restore(art, int(stage), backup)
                raise ServiceError(422, {"message": str(exc), "errors": [str(exc)]}) from exc
            if not findings.ok:
                self._restore(art, int(stage), backup)
                raise ServiceError(
                    422,
                    {
                        "message": "the edited artifact violates the stage contract",
                        "errors": findings.errors[:20],
                    },
                )
            attempt = int(record["attempt"])
            files = art.list_files(int(stage))
            art.write_manifest(int(stage), run_id=run_id, attempt=attempt, files=files)
            checkpoint = store.read_checkpoint(run_id)
            checkpoint["stages"][str(int(stage))]["artifacts"] = files
            store.write_checkpoint(run_id, checkpoint)
            invalidated = self._invalidate(run_id, int(stage) + 1)
            if invalidated and status.terminal:
                store.update_run(
                    run_id, status=RunStatus.PAUSED.value, pause_reason="edited", error=None
                )
                await self._emit(
                    run_id,
                    ev.RUN_PAUSED,
                    attempt=int(store.read_run(run_id)["attempt"]),
                    data={"reason": "edited"},
                )
            edits = [
                *(store.read_run(run_id).get("human_edits") or []),
                {"stage": int(stage), "files": sorted(writes), "at": utc_now(), "note": note},
            ]
            store.update_run(run_id, human_edits=edits)
            return {
                "status": "updated",
                "run_id": run_id,
                "files": sorted(writes),
                "downstream_invalidated": invalidated,
                "warnings": findings.warnings,
            }

    @staticmethod
    def _restore(art: Any, stage: int, backup: dict[str, str | None]) -> None:
        for name, text in backup.items():
            if text is None:
                art.path(stage, name).unlink(missing_ok=True)
            else:
                art.write_text(stage, name, text)

    async def approve_open_gate(self, run_id: str, kind: str, reason: str) -> None:
        record = self.record(run_id)
        gate = record.get("gate")
        if not gate or gate.get("status") != "open" or gate.get("kind") != kind:
            raise _conflict("GATE_NOT_OPEN", f"run {run_id} has no open {kind} gate")
        services = self.services()
        result = await apply_gate_answer(
            run_id, gate["gate_id"], GateAnswer(gates.APPROVE, note=reason), services
        )
        if result.status is not RunStatus.FAILED:
            self.spawn(resume_pipeline(run_id, services))

    # ------------------------------------------------------------------
    # Status, health, decisions
    # ------------------------------------------------------------------

    def stage_status(self, run_id: str, n: int) -> str:
        record = self.record(run_id)
        entry = self.store.read_checkpoint(run_id)["stages"].get(str(n))
        if entry:
            return str(entry["status"])
        if self.control.is_active(run_id) and record.get("current_stage") == n:
            return "running"
        return "pending"

    def overview(self, run_id: str | None) -> dict[str, Any]:
        if run_id is None:
            return {
                "is_running": False,
                "run_id": None,
                "topic": None,
                "progress_percentage": 0,
                "stages": [],
                "last_log": None,
                "error": None,
            }
        record = self.record(run_id)
        art = self.store.artifacts(run_id)
        stages = [
            {
                "stage": n,
                "name": STAGE_NAMES[n],
                "status": self.stage_status(run_id, n),
                "artifacts": art.list_files(n),
            }
            for n in range(1, STAGE_COUNT + 1)
        ]
        done = sum(1 for s in stages if s["status"] == "completed")
        events = self.store.read_events(run_id)
        last = events[-1] if events else None
        error = record.get("error")
        return {
            "is_running": self.control.is_active(run_id),
            "run_id": run_id,
            "topic": record.get("topic"),
            "status": record["status"],
            "progress_percentage": int(done / STAGE_COUNT * 100),
            "stages": stages,
            "last_log": f"{last.type} (stage {last.stage})"
            if last and last.stage
            else (last.type if last else None),
            "error": f"{error['code']}: {error['message']}" if error else None,
        }

    def health(self, run_id: str, n: int) -> dict[str, Any]:
        record = self.record(run_id)
        entry = self.store.read_checkpoint(run_id)["stages"].get(str(n))
        if not entry:
            raise ServiceError(404, f"Stage {n} has not run yet")
        events = [e for e in self.store.read_events(run_id) if e.stage == n]
        started = next((e.timestamp for e in reversed(events) if e.type == ev.STAGE_STARTED), None)
        ended = entry.get("completed_at") or next(
            (e.timestamp for e in reversed(events) if e.type == ev.STAGE_FAILED), None
        )
        return {
            "stage_id": f"{n:02d}",
            "stage": n,
            "name": STAGE_NAMES[n],
            "status": entry["status"],
            "attempt": entry.get("attempt", record["attempt"]),
            "started_at": started,
            "completed_at": ended,
            "duration_sec": _duration(started, ended),
            "artifacts": entry.get("artifacts", []),
            "usage": entry.get("usage"),
            "error": entry.get("error"),
        }

    def health_overview(self, run_id: str) -> list[dict[str, Any]]:
        out = []
        for n in range(1, STAGE_COUNT + 1):
            try:
                out.append(self.health(run_id, n))
            except ServiceError:
                out.append(
                    {
                        "stage_id": f"{n:02d}",
                        "stage": n,
                        "name": STAGE_NAMES[n],
                        "status": self.stage_status(run_id, n),
                    }
                )
        return out

    def decision(self, run_id: str, n: int) -> dict[str, Any]:
        checkpoint = self.store.read_checkpoint(run_id)
        entry = checkpoint["stages"].get(str(n))
        if not entry:
            raise ServiceError(404, f"Stage {n} has not run yet")
        gate = next(
            (
                g
                for g in (checkpoint.get("gates") or {}).values()
                if int(g["stage"]) == n and int(g["attempt"]) == int(entry.get("attempt", 1))
            ),
            None,
        )
        status = {"completed": "PASSED", "failed": "FAILED"}.get(entry["status"], "PENDING")
        reason = (entry.get("error") or {}).get("message")
        timestamp = entry.get("completed_at")
        if gate:
            status = {
                "open": "AWAITING_REVIEW",
                "approved": "APPROVED",
                "rejected": "REJECTED",
            }.get(gate["status"], status)
            answer = gate.get("answer") or {}
            reason = answer.get("note") or reason
            timestamp = gate.get("resolved_at") or timestamp
        return {
            "stage": n,
            "status": status,
            "reason": reason,
            "timestamp": timestamp,
            "attempt": entry.get("attempt"),
            "gate": gate,
        }


class _Fanout:
    """Event sink forwarding to several sinks."""

    def __init__(self, *sinks: Any) -> None:
        self._sinks = sinks

    async def emit(self, event: Event) -> None:
        for sink in self._sinks:
            await sink.emit(event)


def _core_answer(body: GateAnswerRequest) -> GateAnswer:
    option = body.option_id
    dropped = tuple(body.dropped)
    note = body.note or ""
    if option in ("approve", "drop"):
        if option == "drop" and not dropped:
            raise ServiceError(422, "option 'drop' needs at least one id in 'dropped'")
        return GateAnswer(gates.APPROVE, dropped, note)
    if option == "reject":
        return GateAnswer(gates.REJECT, (), note)
    raise ServiceError(422, f"unknown option_id {option!r}; use approve, drop or reject")


def _duration(started: str | None, ended: str | None) -> float | None:
    if not started or not ended:
        return None
    try:
        return round(
            (datetime.fromisoformat(ended) - datetime.fromisoformat(started)).total_seconds(), 3
        )
    except ValueError:
        return None
