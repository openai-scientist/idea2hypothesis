"""Reuse of an earlier run's stages 1-7 when the Platform sends the hypotheses back.

The new run starts at stage 8 with the earlier run's evidence, and stage 8 gets the earlier final
hypotheses as the set to revise. When the earlier run cannot be reused, :func:`problem` says why
and the run executes every stage.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from typing import Any

from idea2hypothesis.pipeline import gates
from idea2hypothesis.pipeline.contracts import validate_stage
from idea2hypothesis.pipeline.models import Stage
from idea2hypothesis.pipeline.usage import UsageMeter
from idea2hypothesis.stages.hypothesis_gen import LEAD_DIRECTION_PREFIXES, PREVIOUS_ROUND
from idea2hypothesis.stages.problem_decompose import LEAD_FRAMING_PREFIXES
from idea2hypothesis.storage.runs import RunStore, utc_now

#: Stages whose artifacts a new run copies (everything before the hypotheses).
REUSED_STAGES = tuple(range(1, int(Stage.HYPOTHESIS_GEN)))

def _domains(names: Sequence[str]) -> list[str]:
    return sorted({n.strip().lower() for n in names if n.strip()})


def _framing(constraints: Sequence[str]) -> list[str]:
    """The lead's framing lines; direction for stage 8 (a handoff, send-back feedback) is not."""
    return sorted(
        c.strip()
        for c in constraints
        if c.startswith(LEAD_FRAMING_PREFIXES) and not c.startswith(LEAD_DIRECTION_PREFIXES)
    )


def problem(
    store: RunStore,
    prior_id: str,
    topic: str,
    domains: Sequence[str] = (),
    constraints: Sequence[str] = (),
) -> str | None:
    """Why ``prior_id`` cannot supply stages 1-7 to a run on ``topic``, or ``None``.

    The run's ``domains`` and its framing ``constraints`` (the research lead's lines that
    stage 1 treats as framing) must equal the earlier run's; other feedback lines may differ.
    """
    if not store.exists(prior_id):
        return f"run {prior_id} is not in the local store"
    record = store.read_run(prior_id)
    if record.get("topic", "").strip() != topic:
        return f"run {prior_id} has a different topic"
    if _domains(record.get("domains") or []) != _domains(domains):
        return f"run {prior_id} has different domains"
    if _framing(record.get("constraints") or []) != _framing(constraints):
        return f"run {prior_id} has different framing constraints"
    checkpoint = store.read_checkpoint(prior_id)
    art = store.artifacts(prior_id)
    for n in REUSED_STAGES:
        if (checkpoint["stages"].get(str(n)) or {}).get("status") != "completed":
            return f"stage {n} of run {prior_id} is not complete"
        if not (art.verify_manifest(n, run_id=prior_id) and validate_stage(Stage(n), art).ok):
            return f"stage {n} of run {prior_id} has no valid artifacts"
    return None


def _revision_text(prior: dict[str, Any] | None) -> str:
    lines = [
        "Hypotheses sent back by a later stage (address the research lead's lines in this attempt):"
    ]
    if prior is not None:
        previous = "; ".join(f"{h['id']}: {h.get('statement')}" for h in prior["hypotheses"])
        lines.append(
            "- Keep every hypothesis that the research lead's lines do not concern unchanged "
            "(same id, same statement); change only what they require. "
            f"Previous hypotheses: {previous}"
        )
    return "\n".join(lines)


def _prior_set(source: Any) -> dict[str, Any] | None:
    """The earlier run's final set, or ``None`` when it never wrote a readable one."""
    try:
        prior = source.read_json(int(Stage.HYPOTHESIS_GEN), "hypotheses.json")
    except (OSError, ValueError):
        return None
    items = prior.get("hypotheses") if isinstance(prior, dict) else None
    if not isinstance(items, list) or not all(isinstance(h, dict) and "id" in h for h in items):
        return None
    return prior


def adopt(
    store: RunStore, prior_id: str, run_id: str, *, mode: str, review_phase: bool
) -> None:
    """Copy stages 1-7 of ``prior_id`` into ``run_id`` as completed and set the revision feedback.

    Everything that can fail is built before the run is changed. On any failure the copied
    files and the earlier checkpoint are restored and the exception propagates.
    """
    source, target = store.artifacts(prior_id), store.artifacts(run_id)
    prior = _prior_set(source)
    text = _revision_text(prior)
    review = source.read_json(int(Stage.HYPOTHESIS_GEN), "review.json") if prior else None
    original = store.read_checkpoint(run_id)
    try:
        for n in REUSED_STAGES:
            shutil.copytree(source.stage_dir(n), target.stage_dir(n))
            target.write_manifest(n, run_id=run_id, attempt=1)
        checkpoint = store.read_checkpoint(run_id)
        now = utc_now()
        for n in REUSED_STAGES:
            checkpoint["stages"][str(n)] = {
                "status": "completed", "attempt": 1, "completed_at": now,
                "artifacts": target.list_files(n), "usage": UsageMeter().snapshot(),
                "reused_from": prior_id,
            }  # fmt: skip
        # The earlier run's researcher already answered the gates before stage 8.
        for spec in gates.gates_for(mode, review_phase):
            if int(spec.after_stage) < int(Stage.HYPOTHESIS_GEN):
                gate_id = gates.make_gate_id(spec, 1)
                checkpoint.setdefault("gates", {})[gate_id] = {
                    "gate_id": gate_id, "kind": spec.kind, "stage": int(spec.after_stage),
                    "attempt": 1, "status": "approved", "opened_at": now, "resolved_at": now,
                    "answer": None, "carried_from": prior_id,
                }  # fmt: skip
        if prior is not None and isinstance(review, dict):
            target.write_run_json(
                PREVIOUS_ROUND, {"hypotheses": prior["hypotheses"], "review": review}
            )
        store.write_checkpoint(run_id, checkpoint)
        store.update_run(
            run_id,
            reused_from=prior_id,
            feedback={"stages": [int(Stage.HYPOTHESIS_GEN)], "note": "", "text": text},
        )
    except Exception:
        for n in REUSED_STAGES:
            shutil.rmtree(target.stage_dir(n), ignore_errors=True)
        (target.run_dir / PREVIOUS_ROUND).unlink(missing_ok=True)
        store.write_checkpoint(run_id, original)
        raise
