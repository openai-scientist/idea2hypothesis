"""Run orchestration: stage execution, checkpointing, gates, pause/cancel and resume.

Public interface::

    async def run_pipeline(request, services) -> RunResult
    async def resume_pipeline(run_id, services) -> RunResult
    async def execute_stage(stage, context) -> StageResult
    async def answer_gate(run_id, gate_id, answer, services) -> RunResult

One worker executes a run at a time (:class:`RunControl.worker`). A gate does not block a
worker: the run is persisted as ``awaiting_review`` and the worker returns; answering the gate
and resuming continues from the checkpoint.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime
from typing import Any

from idea2hypothesis.config import REVIEW_MODES
from idea2hypothesis.llm.models import (
    LLMConfigError,
    LLMError,
    LLMRateLimited,
    LLMTimeout,
)
from idea2hypothesis.pipeline import events as ev
from idea2hypothesis.pipeline import gates
from idea2hypothesis.pipeline.contracts import CONTRACTS, missing_inputs, validate_stage
from idea2hypothesis.pipeline.control import RunInterrupted
from idea2hypothesis.pipeline.models import (
    STAGE_SEQUENCE,
    GateAnswer,
    RunRequest,
    RunResult,
    RunStatus,
    Services,
    Stage,
    StageContext,
    StageError,
    StageResult,
    StageStatus,
)
from idea2hypothesis.pipeline.usage import MeteredLLM, UsageMeter
from idea2hypothesis.stages import STAGE_RUNNERS
from idea2hypothesis.stages.base import StageFailure
from idea2hypothesis.storage.runs import RunNotFoundError, utc_now

logger = logging.getLogger(__name__)

MEMORY_FAILURE_CODES = frozenset(
    {"TOPIC_NOT_RESEARCHABLE", "TOPIC_BELOW_THRESHOLD", "NO_LITERATURE", "EMPTY_SHORTLIST"}
)


class RunStateError(Exception):
    """The requested operation is not valid for the run's current state."""


def new_run_id() -> str:
    return f"i2h-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Single stage
# ---------------------------------------------------------------------------


def _failed(stage: Stage, code: str, message: str, warnings: tuple[str, ...] = ()) -> StageResult:
    return StageResult(
        stage, StageStatus.FAILED, warnings=warnings, error=StageError(code, message)
    )


def _llm_code(exc: LLMError) -> str:
    if isinstance(exc, LLMConfigError):
        return "LLM_CONFIG"
    if isinstance(exc, LLMTimeout):
        return "LLM_TIMEOUT"
    if isinstance(exc, LLMRateLimited):
        return "LLM_RATE_LIMITED"
    return "LLM_ERROR"


async def execute_stage(stage: Stage, context: StageContext) -> StageResult:
    """Run one stage: check inputs, execute with bounded retries, validate, write the manifest.

    :class:`RunInterrupted` propagates so the caller can pause or cancel; every other failure
    becomes a ``FAILED`` result with a code. No substitute output is ever produced.
    """
    art = context.artifacts
    missing = missing_inputs(stage, art)
    if missing:
        return _failed(stage, "MISSING_INPUT", f"missing upstream artifacts: {', '.join(missing)}")

    runtime = context.config.runtime
    attempts = 1 + max(0, runtime.max_retries)
    warnings: list[str] = []
    for attempt in range(attempts):
        art.reset_stage(int(stage))
        context.try_index = attempt
        if attempt:
            # Results announced by the failed try are void; the next try announces its own.
            await context.progress("restart")
        try:
            warnings = await STAGE_RUNNERS[stage](context)
            break
        except StageFailure as exc:
            return _failed(stage, exc.code, exc.message, exc.warnings)
        except LLMError as exc:
            if exc.retryable and attempt + 1 < attempts:
                logger.warning("stage %s: transient LLM error, retrying: %s", int(stage), exc)
                await context.sleep(runtime.retry_delay_sec * (attempt + 1))
                continue
            return _failed(stage, _llm_code(exc), str(exc))
        except (RunInterrupted, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001 - reported as a failed stage, never swallowed silently
            logger.exception("stage %s crashed", int(stage))
            return _failed(stage, "STAGE_ERROR", f"{type(exc).__name__}: {exc}")

    findings = validate_stage(stage, art)
    if not findings.ok:
        return _failed(
            stage,
            CONTRACTS[stage].error_code,
            "; ".join(findings.errors[:8]),
            tuple(warnings),
        )
    files = art.list_files(int(stage))
    art.write_manifest(int(stage), run_id=context.run_id, attempt=context.attempt, files=files)
    return StageResult(
        stage,
        StageStatus.COMPLETED,
        artifacts=tuple(files),
        evidence_refs=tuple(f"stage-{int(stage):02d}/{f}" for f in files),
        warnings=(*warnings, *findings.warnings),
    )


def build_context(
    run_id: str,
    stage: Stage,
    services: Services,
    *,
    record: dict[str, Any] | None = None,
    llm: Any = None,
    reviewer: Any = None,
) -> StageContext:
    """Create a :class:`StageContext` for ``stage`` of an existing run."""
    record = record or services.store.read_run(run_id)
    feedback = record.get("feedback") or {}
    text = feedback.get("text", "") if int(stage) in feedback.get("stages", []) else ""
    attempt = int(record["attempt"])
    stage_run = _stage_run(services, run_id, int(stage))

    async def report(kind: str, data: dict[str, Any]) -> None:
        await _emit(
            services, run_id, ev.STAGE_PROGRESS, stage=int(stage), attempt=attempt,
            data={"kind": kind, "stage_run": stage_run, **data},
        )  # fmt: skip

    return StageContext(
        run_id=run_id,
        attempt=attempt,
        stage=stage,
        topic=record["topic"],
        domains=tuple(record.get("domains", [])),
        constraints=tuple(record.get("constraints", [])),
        config=services.config,
        llm=llm or services.llm,
        literature=services.literature,
        prompts=services.prompts,
        artifacts=services.store.artifacts(run_id),
        reviewer=reviewer or services.reviewer,
        memory=services.memory,
        feedback=text,
        checkpoint=lambda: _check(services, run_id),
        hardware=services.hardware,
        sleep=services.sleep,
        report=report,
    )


def _stage_run(services: Services, run_id: str, stage: int) -> int:
    """Sequence number of the ``stage.started`` event of the stage's current execution."""
    started = [
        e.seq
        for e in services.store.read_events(run_id)
        if e.type == ev.STAGE_STARTED and e.stage == stage
    ]
    return started[-1] if started else 0


async def _check(services: Services, run_id: str) -> None:
    services.control.check(run_id)


# ---------------------------------------------------------------------------
# Events and state helpers
# ---------------------------------------------------------------------------


async def _emit(
    services: Services,
    run_id: str,
    type_: str,
    *,
    stage: int | None = None,
    attempt: int = 1,
    data: dict[str, Any] | None = None,
) -> None:
    event = services.store.append_event(run_id, type_, stage=stage, attempt=attempt, data=data)
    if services.events is not None:
        try:
            await services.events.emit(event)
        except Exception:  # noqa: BLE001 - observers must not break the run
            logger.exception("event sink failed for %s", type_)


def _result(services: Services, run_id: str) -> RunResult:
    record = services.store.read_run(run_id)
    checkpoint = services.store.read_checkpoint(run_id)
    done = tuple(
        sorted(int(n) for n, e in checkpoint["stages"].items() if e.get("status") == "completed")
    )
    error = record.get("error")
    return RunResult(
        run_id=run_id,
        status=RunStatus(record["status"]),
        attempt=int(record["attempt"]),
        completed_stages=done,
        error=StageError(error["code"], error["message"]) if error else None,
        gate=record.get("gate"),
        pause_reason=record.get("pause_reason"),
        usage=record.get("usage", {}),
    )


def _stage_valid(services: Services, run_id: str, stage: Stage, checkpoint: dict[str, Any]) -> bool:
    entry = checkpoint["stages"].get(str(int(stage)))
    if not entry or entry.get("status") != "completed":
        return False
    art = services.store.artifacts(run_id)
    return art.verify_manifest(int(stage), run_id=run_id) and validate_stage(stage, art).ok


def _record_memory(services: Services, run_id: str, outcome: str, reason: str = "") -> None:
    memory = services.memory
    if memory is None or not services.config.runtime.ideation_memory:
        return
    try:
        record = services.store.read_run(run_id)
        art = services.store.artifacts(run_id)
        score = 1.0
        if art.exists(2, "topic_evaluation.json"):
            score = float(art.read_json(2, "topic_evaluation.json").get("overall", 1.0))
        memory.record_topic_outcome(record["topic"], outcome, score, run_id=run_id, reason=reason)
        if outcome == "success" and art.exists(8, "hypotheses.json"):
            for h in art.read_json(8, "hypotheses.json")["hypotheses"]:
                memory.record_hypothesis(
                    h["statement"], True, f"proposed; falsified if: {h['falsification_criteria']}",
                    run_id=run_id,
                )  # fmt: skip
        memory.save()
    except Exception:  # noqa: BLE001 - memory is an aid, never a reason to fail a run
        logger.exception("could not record ideation memory")


# ---------------------------------------------------------------------------
# Driving a run
# ---------------------------------------------------------------------------


async def create_run(request: RunRequest, services: Services) -> str:
    """Create the run record and emit ``run.started``; returns the run id (no stage executes)."""
    cfg = services.config
    topic = request.topic.strip()
    if not topic:
        raise ValueError("topic must not be empty")
    mode = request.review_mode or cfg.review.mode
    if mode not in REVIEW_MODES:
        raise ValueError(f"unknown review mode {mode!r}; expected one of {list(REVIEW_MODES)}")
    run_id = request.run_id or new_run_id()
    domains = list(request.domains or cfg.research.domains)
    constraints = list(request.constraints or cfg.research.constraints)

    record = {
        "status": RunStatus.RUNNING.value,
        "review_mode": mode,
        "attempt": 1,
        "topic": topic,
        "domains": domains,
        "constraints": constraints,
        "platform_run_id": request.platform_run_id,
        "budget_usd": request.budget_usd,
        "current_stage": None,
        "pause_reason": None,
        "error": None,
        "gate": None,
        "feedback": None,
        "worker": None,
        "usage": UsageMeter().snapshot(),
    }
    snapshot = {
        **cfg.snapshot(),
        "run": {"review_mode": mode, "domains": domains, "constraints": constraints},
    }
    services.store.create(
        run_id, record, config_snapshot=snapshot, prompts_snapshot=services.prompts.snapshot()
    )
    await _emit(
        services, run_id, ev.RUN_STARTED,
        data={"topic": topic, "domains": domains, "review_mode": mode, "constraints": constraints},
    )  # fmt: skip
    return run_id


async def run_pipeline(request: RunRequest, services: Services) -> RunResult:
    """Create a run and drive it until it completes, fails, pauses or awaits review."""
    run_id = await create_run(request, services)
    return await _drive(run_id, services)


async def resume_pipeline(run_id: str, services: Services) -> RunResult:
    """Continue a paused, interrupted or failed run from its checkpoint.

    A completed run returns its result unchanged; a run waiting on an unanswered gate stays
    ``awaiting_review``; a cancelled run cannot be resumed.
    """
    try:
        record = services.store.read_run(run_id)
    except RunNotFoundError:
        raise
    status = RunStatus(record["status"])
    if status is RunStatus.COMPLETED:
        return _result(services, run_id)
    if status is RunStatus.CANCELLED:
        raise RunStateError(f"run {run_id} was cancelled and cannot be resumed")
    gate = record.get("gate")
    if status is RunStatus.AWAITING_REVIEW and gate and gate.get("status") == "open":
        return _result(services, run_id)
    services.control.clear(run_id)
    await _emit(
        services, run_id, ev.RUN_RESUMED, attempt=int(record["attempt"]),
        data={"previous_status": status.value},
    )  # fmt: skip
    return await _drive(run_id, services)


async def _drive(run_id: str, services: Services) -> RunResult:
    store = services.store
    async with services.control.worker(run_id):
        store.update_run(
            run_id,
            status=RunStatus.RUNNING.value,
            pause_reason=None,
            worker={"pid": os.getpid(), "started_at": utc_now()},
        )
        try:
            await _loop(run_id, services)
        except RunInterrupted as exc:
            await _handle_interrupt(run_id, services, exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("run %s crashed", run_id)
            await _fail_run(
                run_id, services, None, StageError("INTERNAL_ERROR", f"{type(exc).__name__}: {exc}")
            )
        finally:
            if store.exists(run_id):
                store.update_run(run_id, worker=None)
    return _result(services, run_id)


async def _handle_interrupt(run_id: str, services: Services, exc: RunInterrupted) -> None:
    store = services.store
    record = store.read_run(run_id)
    attempt = int(record["attempt"])
    services.control.clear(run_id)
    if exc.kind == "cancel":
        store.update_run(run_id, status=RunStatus.CANCELLED.value, pause_reason=None)
        await _emit(
            services, run_id, ev.RUN_CANCELLED, attempt=attempt, data={"reason": exc.reason}
        )
    else:
        reason = exc.reason or "user"
        store.update_run(run_id, status=RunStatus.PAUSED.value, pause_reason=reason)
        await _emit(services, run_id, ev.RUN_PAUSED, attempt=attempt, data={"reason": reason})


async def _fail_run(
    run_id: str, services: Services, stage: Stage | None, error: StageError
) -> None:
    store = services.store
    attempt = int(store.read_run(run_id)["attempt"])
    store.update_run(
        run_id,
        status=RunStatus.FAILED.value,
        error={**error.to_dict(), "stage": int(stage) if stage else None},
    )
    await _emit(
        services, run_id, ev.RUN_FAILED, stage=int(stage) if stage else None, attempt=attempt,
        data=error.to_dict(),
    )  # fmt: skip
    if error.code in MEMORY_FAILURE_CODES:
        _record_memory(services, run_id, "failure", error.code)


def _budget_check(record: dict[str, Any], meter: UsageMeter) -> None:
    budget = record.get("budget_usd")
    if budget is not None and meter.cost_usd is not None and meter.cost_usd > float(budget):
        raise RunInterrupted("pause", "budget_exceeded")


async def _loop(run_id: str, services: Services) -> None:
    store = services.store
    meter = UsageMeter.from_dict(store.read_run(run_id).get("usage"))
    llm = MeteredLLM(services.llm, meter)
    reviewer = MeteredLLM(services.reviewer, meter) if services.reviewer else None

    for stage in STAGE_SEQUENCE:
        record = store.read_run(run_id)
        attempt = int(record["attempt"])
        mode = record["review_mode"]
        services.control.check(run_id)
        checkpoint = store.read_checkpoint(run_id)

        if not _stage_valid(services, run_id, stage, checkpoint):
            _budget_check(record, meter)
            if not await _run_stage(run_id, services, stage, record, meter, llm, reviewer):
                return
            record = store.read_run(run_id)

        spec = gates.gate_after(mode, stage)
        if spec is not None:
            checkpoint = store.read_checkpoint(run_id)
            gate_id = gates.make_gate_id(spec, attempt)
            state = checkpoint.get("gates", {}).get(gate_id)
            if state is None:
                await _open_gate(run_id, services, spec, gate_id, attempt)
                return
            if state["status"] == "open":
                store.update_run(run_id, status=RunStatus.AWAITING_REVIEW.value)
                return

    store.update_run(run_id, status=RunStatus.COMPLETED.value, current_stage=None, error=None)
    record = store.read_run(run_id)
    await _emit(
        services, run_id, ev.RUN_COMPLETED, attempt=int(record["attempt"]),
        data={"usage": record["usage"]},
    )  # fmt: skip
    _record_memory(services, run_id, "success")


async def _run_stage(
    run_id: str,
    services: Services,
    stage: Stage,
    record: dict[str, Any],
    meter: UsageMeter,
    llm: MeteredLLM,
    reviewer: MeteredLLM | None,
) -> bool:
    """Execute one stage; returns False when the run stopped (failure)."""
    store = services.store
    attempt = int(record["attempt"])
    store.update_run(run_id, current_stage=int(stage))
    await _emit(services, run_id, ev.STAGE_STARTED, stage=int(stage), attempt=attempt)
    before = meter.snapshot()
    context = build_context(run_id, stage, services, record=record, llm=llm, reviewer=reviewer)
    result = await execute_stage(stage, context)
    usage = meter.delta_since(before)
    store.update_run(run_id, usage=meter.snapshot())
    checkpoint = store.read_checkpoint(run_id)

    if result.status is not StageStatus.COMPLETED:
        assert result.error is not None
        checkpoint["stages"][str(int(stage))] = {
            "status": "failed", "attempt": attempt, "error": result.error.to_dict(), "usage": usage,
        }  # fmt: skip
        store.write_checkpoint(run_id, checkpoint)
        await _emit(
            services, run_id, ev.STAGE_FAILED, stage=int(stage), attempt=attempt,
            data={**result.error.to_dict(), "warnings": list(result.warnings), "usage": usage},
        )  # fmt: skip
        await _fail_run(run_id, services, stage, result.error)
        return False

    checkpoint["stages"][str(int(stage))] = {
        "status": "completed", "attempt": attempt, "completed_at": utc_now(),
        "artifacts": list(result.artifacts), "usage": usage,
    }  # fmt: skip
    store.write_checkpoint(run_id, checkpoint)
    data: dict[str, Any] = {
        "artifacts": list(result.artifacts),
        "evidence_refs": list(result.evidence_refs),
        "warnings": list(result.warnings),
        "usage": usage,
    }
    if record["review_mode"] == "light":
        data["advisories"] = gates.quality_advisories(stage, services.store.artifacts(run_id))
    await _emit(services, run_id, ev.STAGE_COMPLETED, stage=int(stage), attempt=attempt, data=data)
    return True


async def _open_gate(
    run_id: str, services: Services, spec: gates.GateSpec, gate_id: str, attempt: int
) -> None:
    store = services.store
    payload = gates.gate_payload(spec, store.artifacts(run_id))
    state = {
        "gate_id": gate_id, "kind": spec.kind, "stage": int(spec.after_stage), "attempt": attempt,
        "status": "open", "opened_at": utc_now(), "answer": None,
    }  # fmt: skip
    checkpoint = store.read_checkpoint(run_id)
    checkpoint.setdefault("gates", {})[gate_id] = state
    store.write_checkpoint(run_id, checkpoint)
    store.update_run(run_id, status=RunStatus.AWAITING_REVIEW.value, gate=state)
    await _emit(
        services, run_id, ev.GATE_OPENED, stage=int(spec.after_stage), attempt=attempt,
        data={"gate_id": gate_id, "kind": spec.kind, **payload},
    )  # fmt: skip


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


async def apply_gate_answer(
    run_id: str, gate_id: str, answer: GateAnswer, services: Services
) -> RunResult:
    """Record a reviewer decision. Approving keeps the run waiting for ``resume_pipeline``;
    rejecting rolls the run back (new attempt) and also leaves it ready to resume."""
    store = services.store
    record = store.read_run(run_id)
    gate = record.get("gate")
    if not gate or gate.get("gate_id") != gate_id:
        raise gates.GateError(f"gate {gate_id} is not the current gate of run {run_id}")
    if gate.get("status") != "open":
        raise gates.GateError(f"gate {gate_id} was already answered ({gate.get('status')})")
    if answer.decision not in (gates.APPROVE, gates.REJECT):
        raise gates.GateError("decision must be 'approve' or 'reject'")
    spec = gates.gate_by_kind(gate["kind"])
    attempt = int(record["attempt"])
    art = store.artifacts(run_id)
    checkpoint = store.read_checkpoint(run_id)
    answer_doc = {"decision": answer.decision, "dropped": list(answer.dropped), "note": answer.note}
    resolved = {**gate, "status": "approved" if answer.decision == gates.APPROVE else "rejected",
                "resolved_at": utc_now(), "answer": answer_doc}  # fmt: skip
    empty_after_drop = False

    if answer.decision == gates.APPROVE:
        if answer.dropped and spec.kind != gates.SCREEN:
            raise gates.GateError("papers can only be dropped at the screening gate")
        if spec.kind == gates.SCREEN and answer.dropped:
            remaining = gates.apply_screen_drops(
                art, run_id=run_id, attempt=attempt, dropped=answer.dropped, note=answer.note
            )
            empty_after_drop = remaining == 0
        update: dict[str, Any] = {"gate": resolved}
    else:
        feedback_text = _rejection_feedback(spec, art, answer.note)
        art.invalidate_from(int(spec.rollback_to), attempt)
        for n in range(int(spec.rollback_to), len(STAGE_SEQUENCE) + 1):
            checkpoint["stages"].pop(str(n), None)
        update = {
            "gate": resolved,
            "attempt": attempt + 1,
            "feedback": {
                "stages": [int(s) for s in spec.feedback_stages],
                "note": answer.note,
                "text": feedback_text,
            },
            "current_stage": None,
        }
    checkpoint.setdefault("gates", {})[gate_id] = resolved
    checkpoint["attempt"] = update.get("attempt", attempt)
    store.write_checkpoint(run_id, checkpoint)
    store.update_run(run_id, **update)
    await _emit(
        services, run_id, ev.GATE_RESOLVED, stage=int(spec.after_stage), attempt=attempt,
        data={"gate_id": gate_id, "kind": spec.kind, **answer_doc},
    )  # fmt: skip
    if empty_after_drop:
        error = StageError("EMPTY_SHORTLIST", "the reviewer dropped every shortlisted paper")
        await _fail_run(run_id, services, Stage.LITERATURE_SCREEN, error)
    return _result(services, run_id)


def _rejection_feedback(spec: gates.GateSpec, art: Any, note: str) -> str:
    lines = ["Reviewer feedback on the previous attempt (address it in this attempt):"]
    if note.strip():
        lines.append(f"- {note.strip()}")
    if spec.kind == gates.SCREEN:
        try:
            summary = art.read_json(5, "review.json")["summary"]
            lines.append(
                f"- Previous screening: {summary.get('candidates')} candidates, "
                f"{summary.get('kept')} kept, {summary.get('rejected')} rejected."
            )
            queries = art.read_json(3, "queries.json")["queries"]
            lines.append("- Previous queries: " + "; ".join(q["text"] for q in queries[:12]))
        except (OSError, ValueError, KeyError):
            pass
    return "\n".join(lines)


async def answer_gate(
    run_id: str, gate_id: str, answer: GateAnswer, services: Services
) -> RunResult:
    """Apply a gate answer and continue the run to its next stopping point."""
    result = await apply_gate_answer(run_id, gate_id, answer, services)
    if result.status is RunStatus.FAILED:
        return result
    return await resume_pipeline(run_id, services)


# ---------------------------------------------------------------------------
# Control and recovery
# ---------------------------------------------------------------------------


async def pause_run(run_id: str, services: Services, reason: str = "user") -> RunResult:
    """Pause at the next safe point (active worker) or immediately (no worker)."""
    record = services.store.read_run(run_id)
    status = RunStatus(record["status"])
    if status.terminal:
        raise RunStateError(f"run {run_id} is {status.value}")
    if services.control.is_active(run_id):
        services.control.request(run_id, "pause", reason)
        return _result(services, run_id)
    if status is not RunStatus.PAUSED:
        services.store.update_run(run_id, status=RunStatus.PAUSED.value, pause_reason=reason)
        await _emit(
            services, run_id, ev.RUN_PAUSED, attempt=int(record["attempt"]), data={"reason": reason}
        )
    return _result(services, run_id)


async def cancel_run(run_id: str, services: Services, reason: str = "user") -> RunResult:
    """Cancel at the next safe point (active worker) or immediately (no worker)."""
    record = services.store.read_run(run_id)
    status = RunStatus(record["status"])
    if status.terminal:
        raise RunStateError(f"run {run_id} is {status.value}")
    if services.control.is_active(run_id):
        services.control.request(run_id, "cancel", reason)
        return _result(services, run_id)
    services.store.update_run(run_id, status=RunStatus.CANCELLED.value, pause_reason=None)
    await _emit(
        services, run_id, ev.RUN_CANCELLED, attempt=int(record["attempt"]), data={"reason": reason}
    )
    return _result(services, run_id)


async def recover_interrupted(services: Services) -> list[str]:
    """Mark runs left ``running`` by a crash as ``paused`` (``interrupted``); no auto-restart."""
    recovered: list[str] = []
    for run_id in services.store.list_run_ids():
        record = services.store.read_run(run_id)
        if record["status"] != RunStatus.RUNNING.value or services.control.is_active(run_id):
            continue
        services.store.update_run(
            run_id, status=RunStatus.PAUSED.value, pause_reason="interrupted", worker=None
        )
        await _emit(
            services, run_id, ev.RUN_PAUSED, attempt=int(record["attempt"]),
            data={"reason": "interrupted"},
        )  # fmt: skip
        recovered.append(run_id)
    return recovered
