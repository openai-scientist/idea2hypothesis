"""Stage 5 - LITERATURE_SCREEN: keyword pre-filter, batched LLM screening, per-paper decisions."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from idea2hypothesis.literature.novelty import extract_keywords
from idea2hypothesis.pipeline.contracts import Findings
from idea2hypothesis.pipeline.models import StageContext
from idea2hypothesis.stages.base import StageFailure, compact_json, gather_limited, request_json
from idea2hypothesis.storage.runs import utc_now

STAGE = 5
MAX_ABSTRACT_CHARS = 800
MAX_BATCH_CHARS = 30_000
MAX_BATCH_PAPERS = 40
MAX_MISSING_FRACTION = 0.2
SCREEN_RULES = [
    "Domain match",
    "Method relevance",
    "Cross-domain rejection",
    "Recency preference",
    "Quality floor",
]


def topic_keywords(topic: str, domains: tuple[str, ...]) -> list[str]:
    keywords = extract_keywords(topic)
    for domain in domains:
        for part in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", domain.lower()):
            if len(part) >= 2 and part not in keywords:
                keywords.append(part)
    return keywords


def prefilter(
    candidates: list[dict[str, Any]], keywords: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into ``(to_screen, prefiltered)`` by keyword overlap with the topic and domains."""
    to_screen: list[tuple[int, dict[str, Any]]] = []
    dropped: list[dict[str, Any]] = []
    for row in candidates:
        blob = f"{row.get('title', '')} {row.get('abstract', '')}".lower()
        overlap = sum(1 for kw in keywords if kw in blob)
        if overlap >= 1:
            to_screen.append((overlap, row))
        else:
            dropped.append(row)
    to_screen.sort(key=lambda item: item[0], reverse=True)
    return [r for _, r in to_screen], dropped


def make_batches(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group papers into prompt-sized batches so that none is silently truncated."""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for row in rows:
        item = _prompt_view(row)
        length = len(compact_json(item))
        if current and (size + length > MAX_BATCH_CHARS or len(current) >= MAX_BATCH_PAPERS):
            batches.append(current)
            current, size = [], 0
        current.append(row)
        size += length
    if current:
        batches.append(current)
    return batches


def _prompt_view(row: dict[str, Any]) -> dict[str, Any]:
    abstract = str(row.get("abstract") or "")
    if len(abstract) > MAX_ABSTRACT_CHARS:
        abstract = abstract[:MAX_ABSTRACT_CHARS] + "..."
    return {
        "paper_id": row["paper_id"],
        "title": row.get("title", ""),
        "year": row.get("year") or None,
        "venue": row.get("venue") or None,
        "citation_count": row.get("citation_count", 0),
        "abstract": abstract or None,
    }


def check_screen_batch(data: Any, expected: set[str]) -> Findings:
    f = Findings()
    entries = data.get("screened") if isinstance(data, dict) else None
    if not f.require(isinstance(entries, list), "the answer needs a 'screened' list"):
        return f
    seen: set[str] = set()
    for i, entry in enumerate(entries):
        if not f.require(isinstance(entry, dict), f"screened[{i}] must be an object"):
            continue
        pid = entry.get("paper_id")
        if pid not in expected:
            f.warn(f"screened[{i}] has unknown paper_id {pid!r}")
            continue
        seen.add(str(pid))
        f.require(
            entry.get("decision") in ("keep", "reject"), f"{pid}: decision must be keep or reject"
        )
        for key in ("relevance_score", "quality_score"):
            value = entry.get(key)
            f.require(
                isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1,
                f"{pid}: {key} must be a number in [0, 1]",
            )
        f.require(
            isinstance(entry.get("reason"), str) and entry["reason"].strip(),
            f"{pid}: reason is required",
        )
    missing = sorted(expected - seen)
    if len(missing) > MAX_MISSING_FRACTION * len(expected):
        f.error(f"{len(missing)} of {len(expected)} papers were not screened: {missing[:5]}...")
    elif missing:
        f.warn(f"{len(missing)} papers were not scored")
    return f


def _batch_name(batch: list[dict[str, Any]]) -> str:
    """Partial-result file of a batch, keyed by its papers (another batch never reuses it)."""
    ids = "\n".join(str(r["paper_id"]) for r in batch)
    return f"batch-{hashlib.sha256(ids.encode('utf-8')).hexdigest()[:16]}.json"


async def _screen_batch(
    ctx: StageContext, batch: list[dict[str, Any]], index: int
) -> dict[str, dict[str, Any]]:
    """Scores of one batch; a batch already scored in this attempt (before a pause) is reused."""
    name = _batch_name(batch)
    cached = ctx.artifacts.read_partial(STAGE, ctx.attempt, name)
    if isinstance(cached, dict):
        return cached
    scored = await _request_batch(ctx, batch, index)
    ctx.artifacts.write_partial(STAGE, ctx.attempt, name, scored)
    return scored


async def _request_batch(
    ctx: StageContext, batch: list[dict[str, Any]], index: int
) -> dict[str, dict[str, Any]]:
    research = ctx.config.research
    prompt = ctx.prompts.render(
        "literature_screen",
        topic=ctx.topic,
        domains=", ".join(ctx.domains) or "general",
        min_relevance=research.min_relevance,
        min_quality=research.min_quality,
        candidates_json=compact_json([_prompt_view(r) for r in batch]),
    )
    expected = {str(r["paper_id"]) for r in batch}
    data, _ = await request_json(
        ctx,
        prompt,
        label=f"literature_screen batch {index + 1}",
        validate=lambda d: check_screen_batch(d, expected),
    )
    return {str(e["paper_id"]): e for e in data["screened"] if str(e.get("paper_id")) in expected}


def _decision(
    row: dict[str, Any], screened: dict[str, Any] | None, min_rel: float, min_qual: float
) -> dict[str, Any]:
    base = {
        "paper_id": row["paper_id"],
        "title": row.get("title", ""),
        "year": row.get("year") or None,
        "venue": row.get("venue") or None,
        "relevance_score": None,
        "quality_score": None,
        "false_friend": None,
    }
    if screened is None:
        return {
            **base,
            "decision": "unscored",
            "reason": "not scored by the reviewer model; excluded",
        }
    rel = round(float(screened["relevance_score"]), 3)
    qual = round(float(screened["quality_score"]), 3)
    reason = str(screened["reason"]).strip()
    kept = screened["decision"] == "keep" and rel >= min_rel and qual >= min_qual
    if screened["decision"] == "keep" and not kept:
        reason += f" (below thresholds: relevance {rel} / quality {qual})"
    return {
        **base,
        "decision": "kept" if kept else "rejected",
        "relevance_score": rel,
        "quality_score": qual,
        "reason": reason,
        "false_friend": screened.get("false_friend") or None,
    }


def _prefiltered(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": str(row["paper_id"]), "title": row.get("title", ""),
        "year": row.get("year") or None, "venue": row.get("venue") or None,
        "relevance_score": None, "quality_score": None, "false_friend": None,
        "decision": "prefiltered",
        "reason": "no keyword overlap with the topic or domains; not sent to the reviewer",
    }  # fmt: skip


def _cut_reason(reason: str, cap: int) -> str:
    return f"{reason} (cleared both bars but ranked below the {cap} best-scored papers kept)"


async def run(ctx: StageContext) -> list[str]:
    candidates = ctx.artifacts.read_jsonl(4, "candidates.jsonl")
    research = ctx.config.research
    keywords = topic_keywords(ctx.topic, ctx.domains)
    to_screen, dropped = prefilter(candidates, keywords)
    bypassed = False
    if not to_screen:  # nothing overlaps: let the reviewer model judge everything instead
        to_screen, dropped, bypassed = list(candidates), [], True

    batches = make_batches(to_screen)
    await ctx.progress(
        "screen_plan",
        candidates=len(candidates),
        to_screen=len(to_screen),
        batches=len(batches),
        prefiltered=[_prefiltered(r) for r in dropped],
        rules=SCREEN_RULES,
        min_relevance=research.min_relevance,
        min_quality=research.min_quality,
    )
    done = 0

    async def screen(index: int, batch: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        nonlocal done
        scored = await _screen_batch(ctx, batch, index)
        done += 1
        # Decisions against the bars; whether a paper makes the capped shortlist is only known
        # once every batch is in, so kept papers are announced with the shortlist.
        bars = (research.min_relevance, research.min_quality)
        rows = [_decision(r, scored.get(str(r["paper_id"])), *bars) for r in batch]
        await ctx.progress("screen_batch", index=done, total=len(batches), decisions=rows)
        return scored

    results = await gather_limited(
        [lambda i=i, b=b: screen(i, b) for i, b in enumerate(batches)],
        ctx.config.runtime.concurrency,
    )
    scored: dict[str, dict[str, Any]] = {}
    for result in results:
        scored.update(result)

    decisions: list[dict[str, Any]] = []
    shortlist: list[dict[str, Any]] = []
    by_id = {str(r["paper_id"]): r for r in candidates}
    dropped_ids = {str(r["paper_id"]) for r in dropped}
    for row in candidates:
        pid = str(row["paper_id"])
        if pid in dropped_ids:
            decisions.append(_prefiltered(row))
            continue
        decision = _decision(row, scored.get(pid), research.min_relevance, research.min_quality)
        decisions.append(decision)
        if decision["decision"] == "kept":
            shortlist.append(
                {
                    **by_id[pid],
                    "relevance_score": decision["relevance_score"],
                    "quality_score": decision["quality_score"],
                    "keep_reason": decision["reason"],
                }
            )
    shortlist.sort(key=lambda r: (r["relevance_score"], r["quality_score"]), reverse=True)
    cap = research.max_shortlist
    if cap and len(shortlist) > cap:
        cut_ids = {str(r["paper_id"]) for r in shortlist[cap:]}
        shortlist = shortlist[:cap]
        for decision in decisions:
            if decision["paper_id"] in cut_ids:
                decision["decision"] = "below_cutoff"
                decision["reason"] = _cut_reason(decision["reason"], cap)

    counts = {
        "candidates": len(candidates),
        "kept": len(shortlist),
        "rejected": sum(1 for d in decisions if d["decision"] == "rejected"),
        "below_cutoff": sum(1 for d in decisions if d["decision"] == "below_cutoff"),
        "unscored": sum(1 for d in decisions if d["decision"] == "unscored"),
        "prefiltered": len(dropped_ids),
    }
    review = {
        "schema_version": 1,
        "rules": SCREEN_RULES,
        "thresholds": {
            "min_relevance": research.min_relevance,
            "min_quality": research.min_quality,
            "max_shortlist": cap,
        },
        "summary": counts,
        "decisions": decisions,
        "human_review": None,
    }
    ctx.artifacts.write_jsonl(STAGE, "shortlist.jsonl", shortlist)
    ctx.artifacts.write_json(STAGE, "review.json", review)
    ctx.artifacts.write_json(
        STAGE,
        "screen_meta.json",
        {
            "schema_version": 1,
            "outcome": "shortlist" if shortlist else "empty_shortlist",
            **counts,
            "batches": len(batches),
            "keywords": keywords,
            "prefilter_bypassed": bypassed,
            "generated_at": utc_now(),
        },
    )
    if not shortlist:
        raise StageFailure(
            "EMPTY_SHORTLIST",
            f"screening kept 0 of {len(candidates)} papers; refine the search strategy",
        )
    warnings = []
    if counts["unscored"]:
        warnings.append(
            f"{counts['unscored']} papers were not scored by the reviewer and were excluded"
        )
    return warnings
