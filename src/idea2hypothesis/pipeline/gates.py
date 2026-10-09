"""Review modes, gates, human screening decisions and light-mode quality advisories."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from idea2hypothesis.pipeline.models import Stage
from idea2hypothesis.storage.artifacts import ArtifactStore

SCREEN = "screen"
SCOPE = "scope"
HYPOTHESES = "hypotheses"
APPROVE = "approve"
REJECT = "reject"


class GateError(Exception):
    """The gate answer is invalid or does not match the open gate."""


@dataclass(frozen=True)
class GateSpec:
    kind: str
    after_stage: Stage
    rollback_to: Stage
    feedback_stages: tuple[Stage, ...]


SCREEN_GATE = GateSpec(
    SCREEN, Stage.LITERATURE_SCREEN, Stage.SEARCH_STRATEGY, (Stage.SEARCH_STRATEGY,)
)
SCOPE_GATE = GateSpec(
    SCOPE, Stage.PROBLEM_DECOMPOSE, Stage.TOPIC_INIT, (Stage.TOPIC_INIT, Stage.PROBLEM_DECOMPOSE)
)
#: The reviewer decides on the final set before it is mapped: drop hypotheses, keep a held-back
#: one despite its objection, or send the set back to be written again.
HYPOTHESES_GATE = GateSpec(
    HYPOTHESES, Stage.HYPOTHESIS_GEN, Stage.HYPOTHESIS_GEN, (Stage.HYPOTHESIS_GEN,)
)

_MODE_GATES: dict[str, tuple[GateSpec, ...]] = {
    "auto": (),
    "light": (),
    "copilot": (SCREEN_GATE, HYPOTHESES_GATE),
    "full": (SCOPE_GATE, SCREEN_GATE, HYPOTHESES_GATE),
}


def gates_for(mode: str) -> tuple[GateSpec, ...]:
    return _MODE_GATES[mode]


def gate_after(mode: str, stage: Stage) -> GateSpec | None:
    return next((g for g in gates_for(mode) if g.after_stage == stage), None)


def gate_by_kind(kind: str) -> GateSpec:
    return {SCREEN: SCREEN_GATE, SCOPE: SCOPE_GATE, HYPOTHESES: HYPOTHESES_GATE}[kind]


def make_gate_id(spec: GateSpec, attempt: int) -> str:
    return f"gate-s{int(spec.after_stage):02d}-a{attempt}"


def gate_payload(spec: GateSpec, artifacts: ArtifactStore) -> dict[str, Any]:
    """Information a reviewer needs to decide (included in the ``gate.opened`` event)."""
    if spec.kind == SCREEN:
        shortlist = artifacts.read_jsonl(5, "shortlist.jsonl")
        review = artifacts.read_json(5, "review.json")
        return {
            "shortlist": [
                {
                    "paper_id": r["paper_id"], "title": r.get("title"), "year": r.get("year"),
                    "relevance_score": r.get("relevance_score"),
                    "quality_score": r.get("quality_score"), "reason": r.get("keep_reason"),
                }
                for r in shortlist
            ],
            "summary": review.get("summary", {}),
        }  # fmt: skip
    if spec.kind == HYPOTHESES:
        doc = artifacts.read_json(8, "hypotheses.json")
        return {
            "hypotheses": [
                {"id": h["id"], "statement": h["statement"], "from": h.get("from") or [],
                 "contested": h.get("contested") or [], "tension_ids": h.get("tension_ids") or []}
                for h in doc["hypotheses"]
            ],
            "held_back": [
                {"candidate": b["candidate"], "role": b["role"], "number": b["number"],
                 "statement": b["hypothesis"].get("statement", ""), "objections": b["objections"]}
                for b in doc.get("held_back") or []
            ],
            "not_used": [
                {"candidate": u["candidate"], "role": u["role"], "number": u["number"],
                 "statement": u["hypothesis"].get("statement", ""), "reason": u["reason"],
                 "of": u.get("of"), "text": u.get("text", "")}
                for u in doc.get("not_used") or []
            ],
            "open_tensions": doc.get("open_tensions") or [],
        }  # fmt: skip
    goal = artifacts.read_json(1, "goal.json")
    tree = artifacts.read_json(2, "problem_tree.json")
    evaluation = artifacts.read_json(2, "topic_evaluation.json")
    return {
        "goal": {k: goal.get(k) for k in ("working_title", "problem", "objective", "scope")},
        "sub_questions": [{"id": q["id"], "text": q["text"]} for q in tree["sub_questions"]],
        "topic_evaluation": evaluation,
    }


def apply_screen_drops(
    artifacts: ArtifactStore, *, run_id: str, attempt: int, dropped: tuple[str, ...], note: str
) -> int:
    """Remove reviewer-dropped papers from the stage 5 outputs; returns remaining shortlist size."""
    shortlist = artifacts.read_jsonl(5, "shortlist.jsonl")
    ids = {str(r["paper_id"]) for r in shortlist}
    unknown = sorted(set(dropped) - ids)
    if unknown:
        raise GateError(f"dropped papers are not in the shortlist: {unknown}")
    remaining = [r for r in shortlist if str(r["paper_id"]) not in set(dropped)]
    review = artifacts.read_json(5, "review.json")
    for decision in review["decisions"]:
        if str(decision["paper_id"]) in set(dropped):
            decision["decision"] = "dropped_by_reviewer"
            decision["reason"] = f"{decision['reason']} | dropped by reviewer"
    review["summary"]["kept"] = len(remaining)
    review["summary"]["dropped_by_reviewer"] = len(dropped)
    review["human_review"] = {"decision": APPROVE, "dropped": list(dropped), "note": note}
    artifacts.write_jsonl(5, "shortlist.jsonl", remaining)
    artifacts.write_json(5, "review.json", review)
    meta = artifacts.read_json(5, "screen_meta.json")
    meta["kept"] = len(remaining)
    meta["dropped_by_reviewer"] = len(dropped)
    artifacts.write_json(5, "screen_meta.json", meta)
    artifacts.write_manifest(5, run_id=run_id, attempt=attempt)
    return len(remaining)


def apply_hypothesis_review(
    artifacts: ArtifactStore,
    *,
    run_id: str,
    attempt: int,
    dropped: tuple[str, ...],
    kept: tuple[str, ...],
    note: str,
) -> int:
    """Apply the reviewer's changes to the final set; returns its new size.

    Dropped hypotheses leave the set. A kept candidate (held back or not used) joins it under a
    new id, a held-back one still marked contested with its objection, and must pass the
    hypothesis contract on its own.
    """
    from idea2hypothesis.pipeline.contracts import (  # noqa: PLC0415 - avoids an import cycle
        check_hypotheses,
        hypothesis_reference_sets,
        tension_ids,
    )
    from idea2hypothesis.stages.hypothesis_gen import (  # noqa: PLC0415
        normalise_hypotheses,
        render_hypotheses_markdown,
    )

    doc = artifacts.read_json(8, "hypotheses.json")
    ids = {str(h["id"]) for h in doc["hypotheses"]}
    unknown = sorted(set(dropped) - ids)
    if unknown:
        raise GateError(f"dropped hypotheses are not in the set: {unknown}")
    held = {str(b["candidate"]): b for b in doc.get("held_back") or []}
    unused = {str(u["candidate"]): u for u in doc.get("not_used") or []}
    unknown = sorted(set(kept) - set(held) - set(unused))
    if unknown:
        raise GateError(f"kept candidates are not held back or set aside: {unknown}")
    remaining = [h for h in doc["hypotheses"] if str(h["id"]) not in set(dropped)]
    synthesis = artifacts.read_json(7, "synthesis.json")
    cards = [
        artifacts.read_json(6, n)
        for n in artifacts.list_files(6)
        if n.startswith("cards/") and n.endswith(".json")
    ]
    gaps, refs = hypothesis_reference_sets(synthesis, cards)
    tensions = tension_ids(synthesis)
    numbers = [int(str(h["id"])[1:]) for h in doc["hypotheses"] if str(h["id"])[1:].isdigit()]
    next_number = max(numbers, default=0) + 1
    restored = []
    for cid in kept:
        entry = held.pop(cid) if cid in held else unused.pop(cid)
        h = {**entry["hypothesis"], "id": f"H{next_number}"}
        next_number += 1
        h["from"] = [cid]
        h["contested"] = entry.get("objections") or []
        h["caveats"] = entry.get("caveats") or []
        limits = h.get("limitations")
        limits = list(limits) if isinstance(limits, list) else [limits] if limits else []
        h["limitations"] = limits + [
            f"Debate caveat from the {c['from']} perspective: {c['text']}" for c in h["caveats"]
        ]
        h["tension_ids"] = [t for t in h.get("tension_ids") or [] if t in tensions]
        normalise_hypotheses({"hypotheses": [h]})
        h["kept_by_reviewer"] = note.strip() or (
            "Kept by the reviewer despite the objection."
            if h["contested"]
            else "Kept by the reviewer although the set had left it out."
        )
        problems = check_hypotheses({"hypotheses": [h, *remaining]}, gaps, refs, tensions).errors
        # Only what is wrong with the kept hypothesis itself (or that it repeats another).
        mine = [p for p in problems if re.search(rf"\b{h['id']}\b", p)]
        if mine:
            raise GateError(f"{cid} cannot join the set as written: {'; '.join(mine[:3])}")
        restored.append(h)
    remaining += restored
    if not remaining:
        raise GateError("keep at least one hypothesis")
    settled = {t for h in remaining for t in h.get("tension_ids") or []}
    doc["hypotheses"] = remaining
    doc["held_back"] = list(held.values())
    doc["not_used"] = list(unused.values())
    doc["open_tensions"] = sorted(tensions - settled)
    doc["human_review"] = {"decision": APPROVE, "dropped": list(dropped), "kept": list(kept),
                           "note": note}  # fmt: skip
    artifacts.write_json(8, "hypotheses.json", doc)
    artifacts.write_text(8, "hypotheses.md", render_hypotheses_markdown(doc))
    artifacts.write_manifest(8, run_id=run_id, attempt=attempt)
    return len(remaining)


# -- light mode ----------------------------------------------------------


def quality_advisories(stage: Stage, artifacts: ArtifactStore) -> list[str]:
    """Advisory quality checks run after a stage in ``light`` review mode."""
    notes: list[str] = []
    n = int(stage)
    if stage == Stage.LITERATURE_COLLECT:
        meta = artifacts.read_json(n, "search_meta.json")
        if meta.get("unique", 0) < 20:
            notes.append(f"only {meta.get('unique', 0)} unique papers were retrieved")
        failed = [s for s, v in meta.get("per_source", {}).items() if v.get("errors")]
        if failed:
            notes.append(f"sources with errors: {', '.join(sorted(failed))}")
    elif stage == Stage.LITERATURE_SCREEN:
        shortlist = artifacts.read_jsonl(n, "shortlist.jsonl")
        if len(shortlist) < 5:
            notes.append(f"the shortlist has only {len(shortlist)} papers")
        if shortlist:
            mean = sum(r["relevance_score"] for r in shortlist) / len(shortlist)
            if mean < 0.75:
                notes.append(f"mean relevance of the shortlist is {mean:.2f}")
    elif stage == Stage.KNOWLEDGE_EXTRACT:
        meta = artifacts.read_json(n, "knowledge_meta.json")
        if meta.get("skipped"):
            notes.append(f"{len(meta['skipped'])} shortlisted papers had no abstract")
    elif stage == Stage.SYNTHESIS:
        synthesis = artifacts.read_json(n, "synthesis.json")
        thin = [g["id"] for g in synthesis["gaps"] if len(g.get("card_ids", [])) < 2]
        if thin:
            notes.append(f"gaps supported by a single card: {', '.join(thin)}")
    elif stage == Stage.HYPOTHESIS_GEN and artifacts.exists(n, "novelty_report.json"):
        report = artifacts.read_json(n, "novelty_report.json")
        if report.get("assessment") in ("low", "critical"):
            notes.append(f"novelty assessment is {report['assessment']} (heuristic)")
        if report.get("search_coverage") == "run_corpus_only":
            notes.append("novelty was compared only with the run's own papers (search found none)")
    return notes
