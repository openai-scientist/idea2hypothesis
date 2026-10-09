"""Stage 8 - HYPOTHESIS_GEN: perspectives, debate, final set and novelty assessment.

The perspectives propose candidates, the debate (``stages/hypothesis_debate.py``) challenges,
answers and reviews them, and the final merge builds the set from the candidates that survived:
each final hypothesis names the candidates it is built from, carries the caveats that still stand
against them, and is marked contested when one of them has a fatal objection standing. Candidates
with a standing fatal objection that the set does not use are kept on record as held back.
"""

from __future__ import annotations

import logging
from typing import Any

from idea2hypothesis.literature.novelty import check_novelty
from idea2hypothesis.pipeline.contracts import (
    MIN_HYPOTHESES,
    Findings,
    check_hypotheses,
    hypothesis_reference_sets,
    normalise_prediction,
    tension_ids,
)
from idea2hypothesis.pipeline.models import StageContext
from idea2hypothesis.stages.base import (
    StageFailure,
    bullet_list,
    compact_json,
    gather_limited,
    request_json,
)
from idea2hypothesis.stages.hypothesis_debate import (
    Candidate,
    Debate,
    check_perspective,
    run_debate,
)
from idea2hypothesis.stages.synthesis import load_cards
from idea2hypothesis.storage.runs import utc_now

logger = logging.getLogger(__name__)

STAGE = 8


def normalise_hypotheses(data: dict[str, Any]) -> None:
    """Canonicalise prediction spellings in place before validation."""
    for hyp in data.get("hypotheses", []) if isinstance(data.get("hypotheses"), list) else []:
        if isinstance(hyp, dict):
            hyp["prediction"] = normalise_prediction(hyp.get("prediction"))


def render_hypotheses_markdown(doc: dict[str, Any]) -> str:
    lines = ["# Hypotheses", "", f"**Topic:** {doc['topic']}", ""]
    for h in doc["hypotheses"]:
        lines += [f"## {h['id']}: {h['statement']}", ""]
        rows = (
            ("Gap", h["gap_id"]),
            ("Evidence", ", ".join(h["evidence_refs"])),
            ("Exposure", h.get("exposure")),
            ("Outcome", h["outcome"]),
            ("Estimand", h.get("estimand")),
            ("Method", h.get("method")),
            ("Conditions", h.get("conditions")),
            ("Prediction", h["prediction"]),
            ("Falsified if", h["falsification_criteria"]),
            ("Rationale", h["rationale"]),
            ("Novelty", h["novelty"]),
            ("Risk", h.get("risk")),
            ("Settles tensions", ", ".join(h.get("tension_ids") or [])),
            ("Built from", ", ".join(h.get("from") or [])),
        )
        lines += [f"- **{k}:** {v}" for k, v in rows if v]
        limits = h["limitations"] if isinstance(h["limitations"], list) else [h["limitations"]]
        lines += ["- **Limitations:**"] + [f"  - {item}" for item in limits]
        if h.get("contested"):
            lines += ["- **Contested:** a fatal objection still stands:"]
            lines += [f"  - {o['from']}: {o['text']}" for o in h["contested"]]
        if h.get("kept_by_reviewer"):
            lines.append(f"- **Kept by the reviewer:** {h['kept_by_reviewer']}")
        lines.append("")
    if doc.get("held_back"):
        lines += ["## Held back", "", "A fatal objection stands against these candidates:", ""]
        for b in doc["held_back"]:
            lines.append(f"- **{b['candidate']}**: {b['hypothesis'].get('statement', '')}")
            lines += [f"  - {o['from']} ({o.get('flaw')}): {o['text']}" for o in b["objections"]]
        lines.append("")
    if doc.get("open_tensions"):
        lines += ["## Open tensions", "", "Not settled by any hypothesis: "
                  + ", ".join(doc["open_tensions"]), ""]  # fmt: skip
    if doc.get("disagreements"):
        lines += (
            ["## Unresolved disagreements", ""] + [f"- {d}" for d in doc["disagreements"]] + [""]
        )
    return "\n".join(lines)


def _perspective_markdown(role: str, hypotheses: list[dict[str, Any]]) -> str:
    lines = [f"# Perspective: {role}", ""]
    for i, h in enumerate(hypotheses, 1):
        lines += [f"{i}. {h.get('statement', '')}", ""]
    return "\n".join(lines)


def _memory_hint(ctx: StageContext) -> str:
    if ctx.memory is None:
        return ""
    anti = ctx.memory.get_anti_patterns()[:5]
    return (
        ("Weak directions seen before (avoid repeating them):\n- " + "\n- ".join(anti))
        if anti
        else ""
    )


async def _generate_perspectives(
    ctx: StageContext, variables: dict[str, Any], warnings: list[str]
) -> dict[str, list[dict[str, Any]]]:
    roles = ctx.prompts.role_names()

    async def one(role: str) -> tuple[str, list[dict[str, Any]] | None]:
        # A perspective written before a pause or a retried try of this attempt is reused.
        hyps = ctx.artifacts.read_partial(STAGE, ctx.attempt, f"perspective-{role}.json")
        if not isinstance(hyps, list):
            prompt = ctx.prompts.render_role(role, **variables)
            try:
                data, _ = await request_json(
                    ctx, prompt, label=f"perspective {role}", validate=check_perspective
                )
            except StageFailure as exc:
                warnings.append(f"perspective {role} failed: {exc.message}")
                return role, None
            hyps = [h for h in data["hypotheses"] if isinstance(h, dict)]
            ctx.artifacts.write_partial(STAGE, ctx.attempt, f"perspective-{role}.json", hyps)
        if hyps:
            ctx.artifacts.write_json(
                STAGE, f"perspectives/{role}.json", {"role": role, "round": 0, "hypotheses": hyps}
            )
            ctx.artifacts.write_text(
                STAGE, f"perspectives/{role}.md", _perspective_markdown(role, hyps)
            )
            await ctx.progress("perspective", file=f"perspectives/{role}.json", role=role, round=0)
        return role, hyps

    await ctx.progress("perspectives_plan", roles=list(roles))
    results = await gather_limited(
        [lambda r=r: one(r) for r in roles], ctx.config.runtime.concurrency
    )
    current = {role: hyps for role, hyps in results if hyps}
    if not current:
        raise StageFailure("NO_PERSPECTIVES", "no perspective produced usable hypotheses")
    return current


def check_merge(
    data: Any, candidates: dict[str, Candidate], minimum: int, maximum: int
) -> Findings:
    """The final set's size, its sources, and the rule that a candidate with a fatal objection
    standing is used only when the set cannot be filled from cleared ones."""
    f = Findings()
    items = data.get("hypotheses") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return f  # check_hypotheses reports the missing list
    low = max(MIN_HYPOTHESES, min(minimum, len(candidates)))
    f.require(
        low <= len(items) <= maximum,
        f"write between {low} and {maximum} hypotheses (every candidate that survived, merged "
        f"where two say the same thing); got {len(items)}",
    )
    clean, contested = 0, []
    for i, h in enumerate(items):
        if not isinstance(h, dict):
            continue
        tag = f"hypothesis {h.get('id') or i}"
        sources = h.get("from")
        if not (isinstance(sources, list) and sources and all(isinstance(x, str) for x in sources)):
            f.error(f"{tag}: from must list the candidate ids it is built from")
            continue
        unknown = sorted(set(sources) - set(candidates))
        if unknown:
            f.error(f"{tag}: from {unknown} are not candidates (allowed: {sorted(candidates)})")
            continue
        if any(candidates[x].fatal for x in sources):
            contested.append(str(h.get("id") or i))
        else:
            clean += 1
    if contested and clean >= minimum:
        f.error(
            f"hypotheses {contested} build on candidates whose fatal objection stands, while "
            f"{clean} are built from cleared ones; leave them out"
        )
    return f


def _annotate(hypotheses: list[dict[str, Any]], candidates: dict[str, Candidate]) -> None:
    """Carry what still stands against each hypothesis's sources onto it: fatal objections make
    it contested, caveats become limitations."""
    for h in hypotheses:
        sources = [candidates[x] for x in h["from"]]
        h["contested"] = [o.summary() for c in sources for o in c.fatal]
        h["caveats"] = [o.summary() for c in sources for o in c.caveats]
        limits = h.get("limitations")
        limits = list(limits) if isinstance(limits, list) else [limits] if limits else []
        limits += [
            f"Debate caveat from the {c['from']} perspective: {c['text']}" for c in h["caveats"]
        ]
        h["limitations"] = limits


def _held_back(hypotheses: list[dict[str, Any]], candidates: dict[str, Candidate]) -> list[dict]:
    """Candidates with a fatal objection standing that the set does not use."""
    used = {x for h in hypotheses for x in h["from"]}
    return [
        {
            "candidate": c.id,
            "role": c.role,
            "number": c.number,
            "hypothesis": c.hypothesis,
            "objections": [o.summary() for o in c.fatal],
        }
        for c in candidates.values()
        if c.fatal and c.id not in used
    ]


async def run(ctx: StageContext) -> list[str]:
    synthesis = ctx.artifacts.read_json(7, "synthesis.json")
    cards = load_cards(ctx)
    valid_gaps, valid_refs = hypothesis_reference_sets(synthesis, cards)
    valid_tensions = tension_ids(synthesis)
    research = ctx.config.research
    warnings: list[str] = []

    synthesis_view = {
        "overview": synthesis.get("overview"),
        "clusters": synthesis["clusters"],
        # Only tensions with an id can be settled by a hypothesis (older syntheses have none).
        "tensions": [
            t for t in synthesis.get("tensions") or [] if isinstance(t, dict) and t.get("id")
        ],
        "gaps": synthesis["gaps"],
        "prioritized_opportunities": synthesis.get("prioritized_opportunities", []),
    }
    variables: dict[str, Any] = {
        "topic": ctx.topic,
        "constraints": bullet_list(ctx.constraints),
        "feedback": ctx.feedback,
        "synthesis_json": compact_json(synthesis_view),
        "valid_refs": ", ".join(sorted(valid_refs)),
        "valid_gaps": ", ".join(sorted(valid_gaps)),
    }

    positions = await _generate_perspectives(ctx, variables, warnings)
    debate: Debate = await run_debate(ctx, positions, variables, valid_refs, warnings)
    candidates = debate.candidates()
    if not candidates:
        raise StageFailure("NO_PERSPECTIVES", "every hypothesis of the debate was withdrawn")

    prompt = ctx.prompts.render(
        "hypothesis_gen",
        topic=ctx.topic,
        constraints=variables["constraints"],
        feedback=ctx.feedback,
        memory_context=_memory_hint(ctx),
        valid_refs=variables["valid_refs"],
        valid_gaps=variables["valid_gaps"],
        synthesis_json=variables["synthesis_json"],
        perspectives="\n\n".join(c.block() for c in candidates.values()),
        judge_assessment=debate.assessment,
        min_hypotheses=str(research.min_hypotheses),
        max_hypotheses=str(research.max_hypotheses),
    )

    def validate(data: dict[str, Any]) -> Findings:
        normalise_hypotheses(data)
        f = check_hypotheses(data, valid_gaps, valid_refs, valid_tensions)
        f.extend(check_merge(data, candidates, research.min_hypotheses, research.max_hypotheses))
        return f

    await ctx.progress("merge", perspectives=sorted(debate.positions))
    data, soft = await request_json(ctx, prompt, label="hypothesis_gen", validate=validate)
    hypotheses = data["hypotheses"]
    for h in hypotheses:
        h["tension_ids"] = [t for t in h.get("tension_ids") or [] if t in valid_tensions]
    _annotate(hypotheses, candidates)
    settled = {t for h in hypotheses for t in h["tension_ids"]}
    doc = {
        "schema_version": 1,
        "topic": ctx.topic,
        "hypotheses": hypotheses,
        # Candidates left out with a fatal objection standing; a reviewer may keep one.
        "held_back": _held_back(hypotheses, candidates),
        # Tensions of the synthesis that no hypothesis of the set settles.
        "open_tensions": sorted(valid_tensions - settled),
        "disagreements": [str(d) for d in data.get("disagreements") or []],
        "perspectives": sorted(debate.positions),
        "debate_rounds": ctx.config.llm.debate_rounds,
        "generated_at": utc_now(),
    }
    contested = [h["id"] for h in hypotheses if h["contested"]]
    if contested:
        warnings.append(
            f"hypotheses {', '.join(contested)} are contested: a fatal objection to what they "
            "are built from still stands"
        )
    ctx.artifacts.write_json(STAGE, "hypotheses.json", doc)
    ctx.artifacts.write_text(STAGE, "hypotheses.md", render_hypotheses_markdown(doc))

    if research.novelty_check:
        await ctx.progress("novelty", hypotheses=len(doc["hypotheses"]))
        queries_doc = ctx.artifacts.read_json(3, "queries.json")
        seen = ctx.artifacts.read_jsonl(4, "candidates.jsonl")
        report = await check_novelty(
            ctx.topic,
            doc["hypotheses"],
            literature=ctx.literature,
            papers_already_seen=seen,
            year_min=int(queries_doc.get("year_min") or 0),
        )
        ctx.artifacts.write_json(STAGE, "novelty_report.json", report)
        if report["assessment"] == "insufficient_data":
            warnings.append("novelty assessment had insufficient search coverage")
    return [*warnings, *soft]
