"""Core run events and records to contract envelopes, gates and run state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from idea2hypothesis.api.contract.models import (
    FailureInfo,
    Gate,
    GateOption,
    Phase,
    RunState,
)
from idea2hypothesis.api.contract.review import review_findings
from idea2hypothesis.api.platform_events import RunView, map_core_event
from idea2hypothesis.pipeline import events as ev
from idea2hypothesis.pipeline import gates
from idea2hypothesis.pipeline.events import Event
from idea2hypothesis.pipeline.models import RunStatus, Stage

Envelope = dict[str, Any]
STEP_KEYS = tuple(stage.name.lower() for stage in Stage)

# Only content is activity; run/stage/step/gate envelopes stay lifecycle.
_ACTIVITY_LABELS = {
    "scope.profile": "Assessed the available research resources",
    "scope.goal": "Defined the research goal",
    "scope.approved": "Approved the research scope",
    "problem.subquestion": "Defined a research sub-question",
    "problem.risk": "Identified a research risk",
    "topic.evaluated": "Evaluated the research topic",
    "search.strategy": "Defined a literature search strategy",
    "search.query": "Prepared a literature query",
    "search.sources": "Selected literature sources",
    "literature.request": "Queried a literature source",
    "literature.batch": "Received literature search results",
    "literature.merged": "Merged duplicate literature records",
    "literature.collected": "Completed literature collection",
    "screen.criteria": "Defined paper screening criteria",
    "screen.scored": "Scored candidate papers",
    "screen.kept": "Kept a paper in the shortlist",
    "screen.rejected": "Excluded a candidate paper",
    "card.extracted": "Extracted an abstract-level evidence card",
    "synthesis.cluster": "Grouped evidence cards",
    "synthesis.set_aside": "Set aside evidence cards",
    "synthesis.overview": "Synthesised the retrieved evidence",
    "synthesis.tension": "Identified conflicting evidence",
    "synthesis.gap": "Identified a research gap",
    "synthesis.ranked": "Ranked research gaps",
    "debate.turn": "Contributed to the hypothesis debate",
    "hypothesis.drafted": "Drafted a falsifiable hypothesis",
    "hypothesis.selected": "Selected a hypothesis",
    "hypothesis.checked": "Assessed hypothesis novelty heuristically",
    "idea.set_aside": "Set aside a hypothesis candidate",
    "rule.checked": "Checked a hypothesis requirement",
    "map.node": "Mapped a research entity",
    "map.edge": "Mapped a research relationship",
    "canvas.piece": "Updated the research canvas",
    "agent.message": "Reported research progress",
    "review.finding": "Reported an independent review finding",
    "question.defaulted": "Applied the default answer to an optional question",
}
ACTIVITY_TYPES = tuple(f"i2h.{kind}" for kind in _ACTIVITY_LABELS)

HYPOTHESES_PROMPT = (
    "Review the hypotheses and the independent review findings. Approve to publish them, "
    "or comment with findings to have them revised."
)
QUESTION_PROMPTS = {
    gates.SCREEN: "Review the shortlisted papers. Approve to continue or reject to search again.",
    gates.SCOPE: "Review the research goal and sub-questions. Approve to continue or reject.",
}


@dataclass
class Timeline:
    """Where a run stands on the wire: phase and step, rebuilt from its own envelopes."""

    phase: str | None = None
    step: str | None = None
    before_pause: str | None = None

    def observe(self, envelope: Envelope) -> None:
        if envelope["type"] == "run.step_changed":
            self.step = envelope["step_key"]
        phase = envelope.get("phase")
        if phase:
            if phase == "paused" and self.phase not in (None, "paused"):
                self.before_pause = self.phase
            self.phase = phase

    @classmethod
    def from_rows(cls, rows: list[Envelope]) -> Timeline:
        timeline = cls()
        for row in rows:
            timeline.observe(row)
        return timeline


@dataclass
class MapContext:
    """What the mapping needs besides the event: pushed artifact ids and gate bookkeeping."""

    gate_state: dict[str, Any] | None = None
    actors: dict[str, str] = field(default_factory=dict)
    ids: dict[str, str] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    question_payload: dict[str, Any] = field(default_factory=dict)
    view: RunView | None = None


def _envelope(
    type_: str,
    event: Event,
    *,
    phase: str | None = None,
    step_key: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Envelope:
    envelope: Envelope = {
        "type": type_, "class": "lifecycle", "occurred_at": event.timestamp,
        "payload": payload or {},
    }  # fmt: skip
    if phase:
        envelope["phase"] = phase
    if step_key:
        envelope["step_key"] = step_key
    return envelope


def step_of(stage: int | None) -> str | None:
    return STEP_KEYS[stage - 1] if stage and 1 <= stage <= len(STEP_KEYS) else None


def _refs(payload: dict[str, Any]) -> list[str]:
    """Record ids already present in the projected content, in stable encounter order."""
    keys = {"paper_id", "card_id", "hypothesis_id", "artifact_id", "card_ids",
            "hypothesis_ids", "evidence_refs", "sub_question_ids", "gap_id", "gap",
            "sub_question", "tension_ids"}
    containers = {"paper", "card", "hypothesis", "gap", "sub_question", "node", "points"}
    found: list[str] = []

    def visit(value: Any, parent: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                link = parent in {"gap", "hypothesis", "edge"} and key in {"from", "to"}
                if key in keys or link or (key == "id" and parent in containers):
                    ids = item if isinstance(item, list) else [item]
                    found.extend(v for v in ids if isinstance(v, str) and v)
                if key == "locator" and isinstance(item, str) and item.startswith("hypothesis:"):
                    found.append(item.removeprefix("hypothesis:").split("/", 1)[0])
                visit(item, key)
        elif isinstance(value, list):
            for item in value:
                visit(item, parent)

    visit(payload)
    return list(dict.fromkeys(found))


def _activity(
    kind: str, payload: dict[str, Any], actor: str, event: Event, timeline: Timeline
) -> Envelope:
    refs = _refs(payload)
    detail = payload.get("text") if kind == "agent.message" else None
    summary = " ".join(str(detail or _ACTIVITY_LABELS[kind]).split()).rstrip(".!?")
    if not detail and refs:
        summary += f" ({', '.join(refs[:3])})"
    summary = summary[:239].rstrip(".!? ") + "."
    return {
        "type": f"i2h.{kind}", "class": "activity", "occurred_at": event.timestamp,
        "phase": timeline.phase, "step_key": step_of(event.stage), "actor": actor,
        "summary": summary, "refs": refs, "payload": payload,
    }


def _activities(event: Event, timeline: Timeline, ctx: MapContext) -> list[Envelope]:
    out: list[Envelope] = []
    if ctx.view is not None and event.stage and event.type in {
        ev.STAGE_STARTED, ev.STAGE_PROGRESS, ev.STAGE_COMPLETED, ev.STAGE_FAILED, ev.GATE_RESOLVED,
    }:
        for row in map_core_event(ctx.view, event):
            kind = row["type"]
            if kind not in _ACTIVITY_LABELS:
                continue
            if kind == "card.extracted" and _cached_card_reported(event, ctx.view):
                continue
            payload = row["payload"]
            actor = (payload.get("turn") or {}).get("actor") or row.get("actor") or "i2h"
            # ``pi`` is the Platform's research lead; i2h's own lead role is the chair.
            actor = "chair" if actor == "pi" else actor
            out.append(_activity(kind, payload, actor, event, timeline))
        if (event.type == ev.STAGE_COMPLETED and ctx.view.flags.mode in {"auto", "light"}
                and event.stage in {int(Stage.PROBLEM_DECOMPOSE), int(Stage.LITERATURE_SCREEN)}):
            gate = gates.SCOPE if event.stage == int(Stage.PROBLEM_DECOMPOSE) else gates.SCREEN
            out.append(_activity("question.defaulted", {"kind": gate, "option_id": "approve"},
                                 "system", event, timeline))
    if event.type == ev.REVIEW_COMPLETED:
        findings = review_findings(ctx.review, artifact_id=ctx.ids["hypotheses"])
        out.extend(_activity("review.finding", finding.model_dump(mode="json"),
                             "reviewer", event, timeline) for finding in findings.findings)
    return out


def _cached_card_reported(event: Event, view: RunView) -> bool:
    """A valid cached card reannounced by a retry is already on the append-only timeline."""
    if event.type != ev.STAGE_PROGRESS or event.data.get("kind") != "card" or view.store is None:
        return False
    return any(
        prior.seq < event.seq and prior.stage == event.stage and prior.attempt == event.attempt
        and prior.type == ev.STAGE_PROGRESS and prior.data.get("kind") == "card"
        and prior.data.get("card_id") == event.data["card_id"]
        for prior in view.store.read_events(view.run_id)
    )


def gate_of(
    state: dict[str, Any], artifact_ids: list[str], question_payload: dict[str, Any] | None = None
) -> Gate:
    """The wire gate of a core gate state (``gate_id``, ``kind``, ``stage``, ``opened_at``)."""
    if state["kind"] == gates.HYPOTHESES:
        return Gate(
            gate_id=state["gate_id"], kind="artifact_review", step_key=step_of(state["stage"]),
            prompt=HYPOTHESES_PROMPT, opened_at=state["opened_at"], artifact_ids=artifact_ids,
        )  # fmt: skip
    options = [GateOption(option_id="approve", label="Approve")]
    prompt = QUESTION_PROMPTS[state["kind"]]
    if state["kind"] == gates.SCREEN:
        options.append(GateOption(option_id="drop", label="Drop selected papers"))
        papers = (question_payload or {}).get("shortlist", [])
        prompt += " Drop selected papers to continue without them."
        for paper in papers:
            prompt += (
                f"\n{paper['paper_id']}: {paper.get('title') or 'Untitled'}"
                f" ({paper.get('year')}); relevance={paper.get('relevance_score')},"
                f" quality={paper.get('quality_score')}. {paper.get('reason') or ''}"
            )
    options.append(GateOption(option_id="reject", label="Reject"))
    return Gate(
        gate_id=state["gate_id"], kind="question", step_key=step_of(state["stage"]),
        prompt=prompt, opened_at=state["opened_at"], options=options,
        default_option_id="approve",
    )  # fmt: skip


def map_event(event: Event, timeline: Timeline, ctx: MapContext) -> list[Envelope]:
    """Lifecycle and content envelopes; ``timeline`` advances with each lifecycle envelope."""
    out: list[Envelope] = []

    def add(envelope: Envelope) -> None:
        out.append(envelope)
        timeline.observe(envelope)

    def change_phase(new: str) -> None:
        if timeline.phase != new:
            add(_envelope("run.phase_changed", event, phase=new, payload={"from": timeline.phase}))

    kind = event.type
    data = event.data
    if kind == ev.RUN_STARTED:
        change_phase("drafting")
    elif kind == ev.STAGE_STARTED:
        step = step_of(event.stage)
        if event.stage == int(Stage.HYPOTHESIS_GEN) and timeline.phase == "awaiting_researcher":
            change_phase("revising")
        if step:
            payload = {"from": timeline.step}
            add(_envelope("run.step_changed", event, step_key=step, payload=payload))
    elif kind == ev.PHASE_CHANGED:
        change_phase(str(data["phase"]))
    elif kind == ev.REVIEW_COMPLETED:
        add(_envelope("review.completed", event, payload={
            "round": int(data["round"]), "passed": bool(data["passed"]),
            "artifact_id": ctx.ids.get("findings"), "error": ctx.review.get("error"),
        }))  # fmt: skip
    elif kind == ev.GATE_OPENED:
        if data["kind"] == gates.HYPOTHESES:
            change_phase("awaiting_researcher")
        state = ctx.gate_state or {
            "gate_id": data["gate_id"], "kind": data["kind"], "stage": event.stage,
            "opened_at": event.timestamp,
        }  # fmt: skip
        ids = [ctx.ids["hypotheses"]] if "hypotheses" in ctx.ids else []
        add(_envelope(
            "run.gate_opened", event, phase=timeline.phase,
            payload=gate_of(state, ids, ctx.question_payload).model_dump(mode="json"),
        ))  # fmt: skip
    elif kind == ev.GATE_RESOLVED:
        if data["kind"] == gates.HYPOTHESES:
            action = "approve" if data["decision"] == gates.APPROVE else "comment"
        else:
            action = "answer"
        add(_envelope("run.gate_closed", event, payload={
            "gate_id": data["gate_id"], "action": action,
            "actor": ctx.actors.get(data["gate_id"], "user"),
        }))  # fmt: skip
    elif kind == ev.RUN_PAUSED:
        change_phase("paused")
    elif kind == ev.RUN_RESUMED:
        if timeline.phase == "paused" and timeline.before_pause:
            change_phase(timeline.before_pause)
    elif kind == ev.RUN_FAILED:
        add(_envelope("run.failed", event, phase="failed", payload={
            "code": str(data.get("code", "")), "message": str(data.get("message", "")),
        }))  # fmt: skip
    elif kind == ev.RUN_CANCELLED:
        add(_envelope("run.cancelled", event, phase="cancelled"))
    elif kind == ev.RUN_COMPLETED:
        add(_envelope("run.published", event, phase="published",
                      payload={"artifact_id": ctx.ids["hypotheses"]}))  # fmt: skip
    return [*out, *_activities(event, timeline, ctx)]


def run_state(
    run_id: str,
    record: dict[str, Any],
    meta: dict[str, Any],
    timeline: Timeline,
    last_seq: int,
    hypotheses_id: str | None,
    question_payload: dict[str, Any] | None = None,
) -> RunState:
    gate_state = record.get("gate")
    finished = RunStatus(record["status"]).terminal  # a closed run waits for nobody
    open_gate = (
        gate_of(gate_state, [hypotheses_id] if hypotheses_id else [], question_payload)
        if gate_state and gate_state.get("status") == "open" and not finished
        else None
    )
    error = record.get("error")
    cost = (record.get("usage") or {}).get("cost_usd")
    phase: Phase = timeline.phase or "drafting"  # type: ignore[assignment]
    return RunState(
        engine_run_id=run_id,
        stage_run_id=str(meta["stage_run_id"]),
        phase=phase,
        step_key=timeline.step,
        waiting_input=open_gate is not None,
        open_gate=open_gate,
        failure=FailureInfo(code=error["code"], message=error["message"]) if error else None,
        cost_usd="0" if cost is None else float(cost),
        last_source_seq=last_seq,
    )
