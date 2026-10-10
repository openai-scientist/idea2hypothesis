"""Stage 8 - HYPOTHESIS_GEN: perspectives, debate, final set and novelty assessment.

The perspectives propose candidates, the debate (``stages/hypothesis_debate.py``) challenges,
answers and reviews them, and the final merge builds the set from the candidates that survived:
each final hypothesis names the candidates it is built from, carries the caveats that still stand
against them, and is marked contested when one of them has a fatal objection standing. Candidates
with a standing fatal objection that the set does not use are kept on record as held back; any
other candidate the set leaves out is on record as not used, with the reason (a duplicate of a
final hypothesis, or over the limit). A hypothesis merged from several candidates says what it
takes from each, and candidates that predict different effects are never merged.
"""

from __future__ import annotations

import logging
from typing import Any

from idea2hypothesis.literature.novelty import Judge, check_novelty
from idea2hypothesis.pipeline.contracts import (
    EQUIVALENCE,
    MIN_HYPOTHESES,
    PREDICTIONS,
    Findings,
    check_hypotheses,
    check_novelty_judgements,
    check_novelty_queries,
    hypothesis_reference_sets,
    normalise_margin,
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
    """Canonicalise prediction spellings and equivalence margins in place before validation."""
    for hyp in data.get("hypotheses", []) if isinstance(data.get("hypotheses"), list) else []:
        if isinstance(hyp, dict):
            hyp["prediction"] = normalise_prediction(hyp.get("prediction"))
            if "equivalence_margin" in hyp:
                hyp["equivalence_margin"] = normalise_margin(hyp["equivalence_margin"])


def _prediction(h: dict[str, Any]) -> str:
    if h["prediction"] == EQUIVALENCE:
        return f"{h['prediction']} (within ±{h.get('equivalence_margin')})"
    return str(h["prediction"])


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
            ("Prediction", _prediction(h)),
            ("Falsified if", h["falsification_criteria"]),
            ("Rationale", h["rationale"]),
            ("Novelty", h["novelty"]),
            ("Risk", h.get("risk")),
            ("Settles tensions", ", ".join(h.get("tension_ids") or [])),
            ("Built from", ", ".join(h.get("from") or [])),
            ("Merged", h.get("merge_note")),
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
    if doc.get("not_used"):
        lines += ["## Not used", "", "No fatal objection stands against these candidates:", ""]
        for u in doc["not_used"]:
            why = f"same claim as {u['of']}" if u["reason"] == "duplicate" else "over the limit"
            lines.append(f"- **{u['candidate']}** ({why}): {u['hypothesis'].get('statement', '')}")
            lines.append(f"  - {u['text']}")
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


#: Card fields a hypothesis is checked against: how the paper studied it, what it found, its limits.
EVIDENCE_FIELDS = ("method", "findings", "limitations")
MAX_EVIDENCE_CHARS = 600


def _evidence_view(card: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {
        "card_id": card["card_id"], "title": card.get("title"), "year": card.get("year"),
    }  # fmt: skip
    for key in EVIDENCE_FIELDS:
        if card.get(key):
            view[key] = str(card[key])[:MAX_EVIDENCE_CHARS]
    return view


#: Why a candidate with no fatal objection standing is left out of the final set.
NOT_USED_REASONS = ("duplicate", "over_limit")


def _merge_problems(
    tag: str, h: dict[str, Any], sources: list[str], candidates: dict[str, Candidate]
) -> list[str]:
    """A merge joins candidates that predict the same effect and says what each contributes."""
    if len(set(sources)) < 2:
        return []
    problems = []
    signs = {normalise_prediction(candidates[x].hypothesis.get("prediction")) for x in sources}
    # Candidates are checked lightly; compare only predictions written in a canonical form.
    if len(signs) > 1 and signs <= set(PREDICTIONS):
        problems.append(
            f"{tag}: merges candidates that predict different effects ({sorted(map(str, signs))});"
            " keep them as separate hypotheses"
        )
    # Negligible-effect candidates also claim a size: "within ± margin".
    margins = {
        normalise_margin(candidates[x].hypothesis.get("equivalence_margin"))
        for x in sources
        if normalise_prediction(candidates[x].hypothesis.get("prediction")) == EQUIVALENCE
    }
    if len(margins) > 1:
        problems.append(
            f"{tag}: merges negligible-effect candidates with different equivalence margins "
            f"({sorted(map(str, margins))}); keep the one whose margin the cards support and "
            "list the other in not_used as its duplicate, naming both margins in text"
        )
    if not (isinstance(h.get("merge_note"), str) and h["merge_note"].strip()):
        problems.append(f"{tag}: built from {sorted(set(sources))}; say in merge_note what it "
                        "takes from each")  # fmt: skip
    return problems


def _predicts(h: dict[str, Any]) -> Any:
    return normalise_prediction(h.get("prediction"))


def _not_used_problems(
    data: dict[str, Any],
    items: list,
    unused: set[str],
    maximum: int,
    candidates: dict[str, Candidate],
) -> list[str]:
    """Every candidate without a fatal objection that the set leaves out says why."""
    raw = data.get("not_used") or []
    if not isinstance(raw, list):
        return ["not_used must be a list"]
    final = {str(h.get("id")): h for h in items if isinstance(h, dict)}
    problems, seen = [], set()
    for i, entry in enumerate(raw):
        c = entry.get("candidate") if isinstance(entry, dict) else None
        tag = f"not_used[{i}]"
        if c not in unused:
            problems.append(f"{tag}: candidate {c!r} is used by the set or is held back by a "
                            f"fatal objection; list only {sorted(unused)}")  # fmt: skip
            continue
        seen.add(c)
        reason = entry.get("reason")
        if reason not in NOT_USED_REASONS:
            problems.append(f"{tag}: reason must be one of {list(NOT_USED_REASONS)}")
        elif reason == "duplicate" and str(entry.get("of")) not in final:
            problems.append(f"{tag}: a duplicate names in 'of' the final hypothesis that already "
                            "makes its claim")  # fmt: skip
        elif reason == "duplicate" and _predicts(candidates[c].hypothesis) != _predicts(
            final[str(entry["of"])]
        ):
            # Two negligible-effect claims that differ only in margin are one study: a duplicate.
            problems.append(
                f"{tag}: {c} predicts {_predicts(candidates[c].hypothesis)} but {entry['of']} "
                f"predicts {_predicts(final[str(entry['of'])])}; it is not a duplicate: use it, "
                "or give over_limit when the set is full"
            )
        elif reason == "over_limit" and len(items) < maximum:
            problems.append(f"{tag}: the set has {len(items)} of at most {maximum} hypotheses; "
                            f"there is room for {c}")  # fmt: skip
        if not _text(entry.get("text")):
            problems.append(f"{tag}: say why in text")
    missing = sorted(unused - seen)
    if missing:
        problems.append(f"candidates {missing} are not used; build hypotheses from them or list "
                        "them in not_used with the reason")  # fmt: skip
    return problems


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def check_merge(
    data: Any, candidates: dict[str, Candidate], minimum: int, maximum: int
) -> Findings:
    """The final set's size, its sources, the rule that a candidate with a fatal objection
    standing is used only when the set cannot be filled from cleared ones, merges that keep
    each source's claim, and a reason for every cleared candidate left out."""
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
    clean, contested, used = 0, [], set()
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
        used |= set(sources)
        for problem in _merge_problems(tag, h, sources, candidates):
            f.error(problem)
        if any(candidates[x].fatal for x in sources):
            contested.append(str(h.get("id") or i))
        else:
            clean += 1
    if contested and clean >= minimum:
        f.error(
            f"hypotheses {contested} build on candidates whose fatal objection stands, while "
            f"{clean} are built from cleared ones; leave them out"
        )
    if f.ok:
        unused = {c.id for c in candidates.values() if not c.fatal and c.id not in used}
        for problem in _not_used_problems(data, items, unused, maximum, candidates):
            f.error(problem)
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


def _not_used(data: dict[str, Any], candidates: dict[str, Candidate]) -> list[dict[str, Any]]:
    """Candidates with no fatal objection standing that the set leaves out, with the reason."""
    out = []
    for entry in data.get("not_used") or []:
        c = candidates[entry["candidate"]]
        out.append(
            {
                "candidate": c.id, "role": c.role, "number": c.number, "hypothesis": c.hypothesis,
                "reason": entry["reason"],
                "of": str(entry["of"]) if entry["reason"] == "duplicate" else None,
                "text": _text(entry.get("text")),
                "caveats": [o.summary() for o in c.caveats],
            }
        )  # fmt: skip
    return out


async def _novelty_queries(
    ctx: StageContext, hypotheses: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """One keyword query per hypothesis, in the set's order, and the problems met writing them.

    The novelty check is advisory: when no valid queries come back the search is skipped and the
    report says why, rather than the stage failing after the debate.
    """
    ids = {str(h["id"]) for h in hypotheses}
    views = [
        {k: h.get(k) for k in ("id", "statement", "exposure", "outcome", "conditions")}
        for h in hypotheses
    ]
    prompt = ctx.prompts.render(
        "novelty_queries", topic=ctx.topic, hypotheses_json=compact_json(views)
    )
    try:
        data, _ = await request_json(
            ctx, prompt, label="novelty_queries",
            validate=lambda d: check_novelty_queries(d, ids),
        )  # fmt: skip
    except StageFailure as exc:
        return [], [f"no search: the search queries could not be written ({exc.message})"]
    by_id = {str(r["hypothesis_id"]): " ".join(str(r["query"]).split()) for r in data["queries"]}
    return [by_id[str(h["id"])] for h in hypotheses], []


#: Characters of each abstract the novelty judge reads.
MAX_JUDGED_ABSTRACT_CHARS = 1500


def _novelty_judge(ctx: StageContext, problems: list[str]) -> Judge:
    """The model reads the papers closest to each hypothesis and says whether one tests it.

    When no valid judgement comes back the report says so (``problems``) instead of the stage
    failing after the debate: the novelty check is advisory.
    """

    async def judge(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
        papers = {str(i["hypothesis"]["id"]): {p["paper_id"] for p in i["papers"]} for i in items}
        views = [
            {
                **{k: i["hypothesis"].get(k)
                   for k in ("id", "statement", "exposure", "outcome", "conditions", "prediction")},
                "papers": [
                    {**p, "abstract": p["abstract"][:MAX_JUDGED_ABSTRACT_CHARS]}
                    for p in i["papers"]
                ],
            }
            for i in items
        ]  # fmt: skip
        prompt = ctx.prompts.render(
            "novelty_judge", topic=ctx.topic, items_json=compact_json(views)
        )
        try:
            data, _ = await request_json(
                ctx, prompt, label="novelty_judge",
                validate=lambda d: check_novelty_judgements(d, papers),
            )  # fmt: skip
        except StageFailure as exc:
            problems.append(f"no judgement: the papers could not be judged ({exc.message})")
            return None
        return {str(r["hypothesis_id"]): r for r in data["judgements"]}

    return judge


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
        # What each citable card reports: without it a hypothesis cannot be checked against the
        # evidence it cites (already shown, or contradicted).
        "cards": [_evidence_view(c) for c in cards],
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
        if len(set(h["from"])) < 2:
            h.pop("merge_note", None)  # only a hypothesis merged from several candidates has one
        if h["prediction"] != EQUIVALENCE:
            h.pop("equivalence_margin", None)
    _annotate(hypotheses, candidates)
    settled = {t for h in hypotheses for t in h["tension_ids"]}
    doc = {
        "schema_version": 1,
        "topic": ctx.topic,
        "hypotheses": hypotheses,
        # Candidates left out with a fatal objection standing; a reviewer may keep one.
        "held_back": _held_back(hypotheses, candidates),
        # Candidates with no fatal objection that the set leaves out, with the reason; a
        # reviewer may keep one too.
        "not_used": _not_used(data, candidates),
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
        searches, query_problems = await _novelty_queries(ctx, doc["hypotheses"])
        judge_problems: list[str] = []
        report = await check_novelty(
            ctx.topic,
            doc["hypotheses"],
            queries=searches,
            literature=ctx.literature,
            judge=_novelty_judge(ctx, judge_problems),
            papers_already_seen=seen,
            year_min=int(queries_doc.get("year_min") or 0),
        )
        report["search_errors"] = [*query_problems, *report["search_errors"]]
        report["judge_errors"] = judge_problems
        ctx.artifacts.write_json(STAGE, "novelty_report.json", report)
        tested = [r["hypothesis_id"] for r in report["per_hypothesis"] if r["verdict"] == "tested"]
        if tested:
            warnings.append(
                f"novelty: a paper found may already test {', '.join(tested)} "
                "(see novelty_report.json)"
            )
        if judge_problems:
            warnings.append(f"novelty was not judged: {judge_problems[0]}")
        elif report["search_coverage"] == "insufficient":
            warnings.append("novelty assessment had insufficient search coverage")
        if report["search_coverage"] == "run_corpus_only":
            errors = len(report["search_errors"])
            why = (
                f"{errors} search errors, listed in novelty_report.json"
                if errors
                else "the search returned nothing"
            )
            warnings.append(
                "the novelty search found no new papers, so the hypotheses were compared only "
                f"with the run's own papers ({why})"
            )
    return [*warnings, *soft]
