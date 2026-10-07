"""Stage 3 - SEARCH_STRATEGY: search plan, concrete queries and configured sources."""

from __future__ import annotations

import re
from typing import Any

import yaml

from idea2hypothesis.literature.search import SOURCE_INFO
from idea2hypothesis.pipeline.contracts import check_search_plan, sub_question_ids
from idea2hypothesis.pipeline.models import StageContext
from idea2hypothesis.stages.base import compact_json, request_json
from idea2hypothesis.storage.runs import utc_now

STAGE = 3
MAX_QUERY_CHARS = 60
MIN_PLANNED_QUERIES = 8
SEARCH_SUFFIXES = ("benchmark", "survey", "seminal", "state of the art")
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "for",
        "in",
        "on",
        "and",
        "or",
        "with",
        "to",
        "by",
        "from",
        "its",
        "is",
        "are",
        "was",
        "be",
        "as",
        "at",
        "via",
        "using",
        "based",
        "study",
        "analysis",
        "empirical",
        "towards",
        "toward",
        "into",
        "exploring",
        "comparison",
        "tasks",
        "effectiveness",
        "investigation",
        "comprehensive",
        "novel",
        "challenge",
        "challenges",
        "gaps",
        "gap",
        "critical",
        "survey",
        "review",
    ]
)


def _search_terms(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-zA-Z0-9]+", text) if w.lower() not in _STOP and len(w) > 1]


def shorten_query(query: str, max_keywords: int = 6) -> str:
    """Shorten an over-long query to its keywords, keeping a trailing search suffix."""
    stripped = query.strip()
    suffix, core = "", stripped
    for candidate in SEARCH_SUFFIXES:
        if stripped.lower().endswith(candidate):
            suffix, core = candidate, stripped[: -len(candidate)].strip()
            break
    shortened = " ".join(_search_terms(core)[:max_keywords])
    return f"{shortened} {suffix}".strip() if suffix and shortened else shortened


def sanitize_queries(queries: list[str]) -> list[str]:
    """Shorten long queries and drop empty or case-insensitive duplicates (order kept)."""
    out: list[str] = []
    seen: set[str] = set()
    for query in queries:
        text = shorten_query(query) if len(query) > MAX_QUERY_CHARS else query.strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def render_sources(cfg_sources: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {"id": s, "name": SOURCE_INFO[s][0], "type": "api", "url": SOURCE_INFO[s][1],
         "status": "configured"}
        for s in cfg_sources
    ]  # fmt: skip


async def run(ctx: StageContext) -> list[str]:
    tree = ctx.artifacts.read_json(2, "problem_tree.json")
    known = sub_question_ids(tree)
    cfg = ctx.config.literature
    year_hint = (
        f"If the plan has no better reason, use {cfg.default_year_min} as the earliest year."
        if cfg.default_year_min
        else ""
    )
    prompt = ctx.prompts.render(
        "search_strategy",
        topic=ctx.topic,
        problem_tree_json=compact_json(
            {"sub_questions": [
                {"id": q["id"], "text": q["text"]} for q in tree["sub_questions"]
            ]}
        ),
        feedback=ctx.feedback,
        year_hint=year_hint,
    )  # fmt: skip
    plan, warnings = await request_json(
        ctx, prompt, label="search_strategy", validate=lambda d: check_search_plan(d, known)
    )

    strategies: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for strategy in plan["search_strategies"]:
        queries = sanitize_queries([str(q) for q in strategy["queries"]])
        queries = [q for q in queries if q.lower() not in seen]
        seen.update(q.lower() for q in queries)
        if not queries:
            continue
        strategies.append({**strategy, "queries": queries})
        for q in queries:
            rows.append(
                {
                    "id": f"q{len(rows) + 1}",
                    "text": q,
                    "strategy": strategy["name"],
                    "sub_question_ids": list(strategy["sub_question_ids"]),
                }
            )
    filters = plan.get("filters") if isinstance(plan.get("filters"), dict) else {}
    min_year = filters.get("min_year")
    year_min = min_year if isinstance(min_year, int) and min_year > 0 else cfg.default_year_min
    if len(rows) < MIN_PLANNED_QUERIES:
        warnings = (
            *warnings,
            f"only {len(rows)} distinct queries were planned (target {MIN_PLANNED_QUERIES})",
        )

    final_plan = {
        "topic": ctx.topic,
        "generated": utc_now(),
        "search_strategies": strategies,
        "filters": {"min_year": year_min or None},
    }
    ctx.artifacts.write_text(
        STAGE, "search_plan.yaml", yaml.safe_dump(final_plan, sort_keys=False, allow_unicode=True)
    )
    ctx.artifacts.write_json(
        STAGE, "queries.json", {"schema_version": 1, "year_min": year_min, "queries": rows}
    )
    sources = render_sources(cfg.sources)
    ctx.artifacts.write_json(
        STAGE, "sources.json",
        {"schema_version": 1, "sources": sources, "count": len(sources), "generated": utc_now()},
    )  # fmt: skip
    return list(warnings)
