"""Stage 8 - HYPOTHESIS_GEN: multi-perspective generation, optional debate, novelty assessment."""

from __future__ import annotations

import logging
from typing import Any

from idea2hypothesis.literature.novelty import check_novelty
from idea2hypothesis.pipeline.contracts import (
    Findings,
    check_hypotheses,
    hypothesis_reference_sets,
    normalise_prediction,
)
from idea2hypothesis.pipeline.models import StageContext
from idea2hypothesis.stages.base import (
    StageFailure,
    bullet_list,
    compact_json,
    gather_limited,
    request_json,
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


def check_perspective(data: Any) -> Findings:
    """Light structural check of one role's output; the strict contract applies to the final set."""
    f = Findings()
    items = data.get("hypotheses") if isinstance(data, dict) else None
    if f.require(
        isinstance(items, list) and items, "the answer needs a non-empty 'hypotheses' list"
    ):
        for i, h in enumerate(items):
            f.require(
                isinstance(h, dict)
                and isinstance(h.get("statement"), str)
                and h["statement"].strip(),
                f"hypotheses[{i}] needs a statement",
            )
    return f


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
        )
        lines += [f"- **{k}:** {v}" for k, v in rows if v]
        limits = h["limitations"] if isinstance(h["limitations"], list) else [h["limitations"]]
        lines += ["- **Limitations:**"] + [f"  - {item}" for item in limits] + [""]
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
        prompt = ctx.prompts.render_role(role, **variables)
        try:
            data, _ = await request_json(
                ctx, prompt, label=f"perspective {role}", validate=check_perspective
            )
        except StageFailure as exc:
            warnings.append(f"perspective {role} failed: {exc.message}")
            return role, None
        return role, [h for h in data["hypotheses"] if isinstance(h, dict)]

    results = await gather_limited(
        [lambda r=r: one(r) for r in roles], ctx.config.runtime.concurrency
    )
    current = {role: hyps for role, hyps in results if hyps}
    if not current:
        raise StageFailure("NO_PERSPECTIVES", "no perspective produced usable hypotheses")
    for role, hyps in current.items():
        ctx.artifacts.write_json(
            STAGE, f"perspectives/{role}.json", {"role": role, "round": 0, "hypotheses": hyps}
        )
        ctx.artifacts.write_text(
            STAGE, f"perspectives/{role}.md", _perspective_markdown(role, hyps)
        )
    return current


async def _debate(
    ctx: StageContext,
    current: dict[str, list[dict[str, Any]]],
    variables: dict[str, Any],
    warnings: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], str, dict[str, Any]]:
    """Rebuttal rounds followed by an independent judge; returns the positions and a record."""
    rounds = ctx.config.llm.debate_rounds
    record: dict[str, Any] = {"rounds": rounds, "roles": sorted(current), "concessions": {}}
    for r in range(1, rounds + 1):
        if len(current) < 2:
            break
        previous = dict(current)

        async def rebut(
            role: str, previous: dict[str, Any] = previous, r: int = r
        ) -> tuple[str, Any]:
            others = "\n\n---\n\n".join(
                f"### {other}\n{compact_json(previous[other])}"
                for other in previous
                if other != role
            )
            prompt = ctx.prompts.render(
                "debate_rebuttal",
                role=role,
                own_position=compact_json(previous[role]),
                others=others,
                valid_refs=variables["valid_refs"],
                valid_gaps=variables["valid_gaps"],
                synthesis_json=variables["synthesis_json"],
            )
            try:
                data, _ = await request_json(
                    ctx, prompt, label=f"debate {role} r{r}", validate=check_perspective
                )
            except StageFailure as exc:
                warnings.append(f"debate round {r}: {role} kept its prior position ({exc.message})")
                return role, None
            return role, data

        outcomes = await gather_limited(
            [lambda role=role: rebut(role) for role in previous], ctx.config.runtime.concurrency
        )
        for role, data in outcomes:
            if data is None:
                continue
            hyps = [h for h in data["hypotheses"] if isinstance(h, dict)]
            current[role] = hyps
            record["concessions"].setdefault(role, []).extend(data.get("concessions") or [])
            ctx.artifacts.write_json(
                STAGE,
                f"perspectives/{role}.r{r}.json",
                {"role": role, "round": r, "hypotheses": hyps},
            )

    judge = ctx.reviewer or ctx.llm
    record["independent_judge"] = ctx.reviewer is not None
    if ctx.reviewer is None:
        warnings.append("no reviewer model configured; the debate judge is not independent")
    combined = "\n\n---\n\n".join(
        f"### Perspective: {r}\n{compact_json(h)}" for r, h in current.items()
    )
    prompt = ctx.prompts.render("debate_judge", perspectives=combined)
    assessment = ""
    try:
        data, _ = await request_json(
            ctx, prompt, label="debate judge", llm=judge,
            validate=lambda d: _check_rankings(d),
        )  # fmt: skip
    except StageFailure as exc:
        warnings.append(f"debate judge failed: {exc.message}")
    else:
        record["rankings"] = data["rankings"]
        assessment = "Independent reviewer assessment:\n" + compact_json(data["rankings"])
    ctx.artifacts.write_json(STAGE, "perspectives/debate_record.json", record)
    return current, assessment, record


def _check_rankings(data: Any) -> Findings:
    f = Findings()
    rankings = data.get("rankings") if isinstance(data, dict) else None
    f.require(
        isinstance(rankings, list) and rankings, "the answer needs a non-empty 'rankings' list"
    )
    return f


async def run(ctx: StageContext) -> list[str]:
    synthesis = ctx.artifacts.read_json(7, "synthesis.json")
    cards = load_cards(ctx)
    valid_gaps, valid_refs = hypothesis_reference_sets(synthesis, cards)
    warnings: list[str] = []

    synthesis_view = {
        "overview": synthesis.get("overview"),
        "clusters": synthesis["clusters"],
        "gaps": synthesis["gaps"],
        "prioritized_opportunities": synthesis.get("prioritized_opportunities", []),
    }
    variables: dict[str, Any] = {
        "topic": ctx.topic,
        "constraints": bullet_list(ctx.constraints),
        "synthesis_json": compact_json(synthesis_view),
        "valid_refs": ", ".join(sorted(valid_refs)),
        "valid_gaps": ", ".join(sorted(valid_gaps)),
    }

    current = await _generate_perspectives(ctx, variables, warnings)
    assessment = ""
    debated = ctx.config.llm.debate_rounds > 0
    if debated:
        current, assessment, _ = await _debate(ctx, current, variables, warnings)

    perspectives_text = "\n\n---\n\n".join(
        f"### Perspective: {role}\n{compact_json(hyps)}" for role, hyps in current.items()
    )
    prompt = ctx.prompts.render(
        "hypothesis_gen",
        topic=ctx.topic,
        constraints=variables["constraints"],
        memory_context=_memory_hint(ctx),
        valid_refs=variables["valid_refs"],
        valid_gaps=variables["valid_gaps"],
        synthesis_json=variables["synthesis_json"],
        perspectives=perspectives_text,
        judge_assessment=assessment,
    )

    def validate(data: dict[str, Any]) -> Findings:
        normalise_hypotheses(data)
        return check_hypotheses(data, valid_gaps, valid_refs)

    data, soft = await request_json(ctx, prompt, label="hypothesis_gen", validate=validate)
    doc = {
        "schema_version": 1,
        "topic": ctx.topic,
        "hypotheses": data["hypotheses"],
        "disagreements": [str(d) for d in data.get("disagreements") or []],
        "perspectives": sorted(current),
        "debate_rounds": ctx.config.llm.debate_rounds if debated else 0,
        "generated_at": utc_now(),
    }
    ctx.artifacts.write_json(STAGE, "hypotheses.json", doc)
    ctx.artifacts.write_text(STAGE, "hypotheses.md", render_hypotheses_markdown(doc))

    if ctx.config.research.novelty_check:
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
