"""Stage 4 - LITERATURE_COLLECT: real retrieval, deduplication, provenance and BibTeX."""

from __future__ import annotations

import re
from typing import Any

from idea2hypothesis.literature.citations import papers_to_bibtex
from idea2hypothesis.literature.dedup import deduplicate
from idea2hypothesis.literature.models import Paper, SearchReport, SourceStats
from idea2hypothesis.pipeline.models import StageContext
from idea2hypothesis.stages.base import StageFailure
from idea2hypothesis.storage.runs import utc_now

STAGE = 4
# Words that carry no subject. A topic written as a question ("Does X reduce Y, and when ...")
# would otherwise give queries such as "Does X reduce survey".
_FILLER = frozenset(
    [
        "a", "an", "the", "and", "or", "but", "nor", "of", "in", "on", "at", "to", "for",
        "with", "by", "from", "as", "into", "onto", "about", "than", "that", "this", "these",
        "those", "it", "its", "their", "there", "do", "does", "did", "is", "are", "was", "were",
        "be", "been", "being", "can", "could", "will", "would", "should", "may", "might", "must",
        "shall", "how", "what", "when", "where", "which", "who", "whom", "whose", "why",
        "whether",
    ]
)  # fmt: skip


def expand_queries(queries: list[str], topic: str) -> list[str]:
    """Broader variants of the planned queries (shorter topic windows, survey/benchmark forms)."""
    expanded: list[str] = []
    seen = {q.lower().strip() for q in queries}
    # Keep words only: punctuation such as "?" makes some sources reject the query.
    words = [
        w
        for w in (re.sub(r"[^\w-]", "", word) for word in topic.split())
        if w and w.lower() not in _FILLER
    ]
    if not words:
        return []

    def add(candidate: str) -> None:
        key = candidate.lower().strip()
        if key and key not in seen:
            seen.add(key)
            expanded.append(candidate.strip())

    if len(words) > 5:
        add(" ".join(words[:5]))
        add(" ".join(words[-5:]))
    for suffix in ("survey", "benchmark", "comparison"):
        add(f"{' '.join(words[:4])} {suffix}")
    return expanded


def _merge_stats(total: dict[str, SourceStats], report: SearchReport) -> None:
    for name, stats in report.per_source.items():
        current = total.setdefault(name, SourceStats())
        current.requests += stats.requests
        current.papers += stats.papers
        current.errors.extend(stats.errors)
        current.served_from_cache += stats.served_from_cache


async def run(ctx: StageContext) -> list[str]:
    queries_doc = ctx.artifacts.read_json(3, "queries.json")
    year_min = int(queries_doc.get("year_min") or 0)
    planned = [q["text"] for q in queries_doc["queries"]]
    extra = expand_queries(planned, ctx.topic)
    all_queries = [*planned, *extra]
    limit = ctx.config.literature.max_results_per_query

    raw = 0
    dropped = 0
    papers: list[Paper] = []
    per_source: dict[str, SourceStats] = {}
    per_query: dict[str, dict[str, int]] = {}
    await ctx.progress("queries", planned=len(planned), expanded=len(extra))
    for index, query in enumerate(all_queries):
        await ctx.safe_point()
        if index > 0:
            await ctx.sleep(ctx.config.literature.inter_query_delay_sec)
        report = await ctx.literature.search([query], limit=limit, year_min=year_min)
        raw += report.raw_count
        dropped += report.dropped_without_title
        papers.extend(report.papers)
        _merge_stats(per_source, report)
        per_query.update(report.per_query)
        await ctx.progress(
            "query",
            text=query,
            index=index + 1,
            total=len(all_queries),
            hits=dict(report.per_query.get(query, {})),
            raw=raw,
        )

    unique, _ = deduplicate(papers)
    unique.sort(key=lambda p: (p.citation_count, p.year), reverse=True)
    duplicates = raw - dropped - len(unique)
    errors = [f"{n}: {e}" for n, s in per_source.items() for e in s.errors]

    meta: dict[str, Any] = {
        "schema_version": 1,
        "queries_used": [
            *({"text": q, "origin": "plan"} for q in planned),
            *({"text": q, "origin": "expansion"} for q in extra),
        ],
        "year_min": year_min,
        "raw": raw,
        "unique": len(unique),
        "duplicates": duplicates,
        "dropped_without_title": dropped,
        "per_source": {name: stats.to_dict() for name, stats in per_source.items()},
        "per_query": per_query,
        "errors": errors,
        "ts": utc_now(),
    }

    if not unique:
        ctx.artifacts.write_json(STAGE, "search_meta.json", meta)
        raise StageFailure(
            "NO_LITERATURE",
            "no papers were retrieved from any configured source"
            + (f" (errors: {'; '.join(errors[:5])})" if errors else ""),
        )

    ctx.artifacts.write_jsonl(STAGE, "candidates.jsonl", [p.to_dict() for p in unique])
    ctx.artifacts.write_text(STAGE, "references.bib", papers_to_bibtex(unique))
    ctx.artifacts.write_json(STAGE, "search_meta.json", meta)

    # One line per source: a rate limit fails every query the same way (all are in search_meta).
    warnings = [
        f"source error: {name}: {len(s.errors)} searches failed"
        + (f", {s.served_from_cache} papers served from the cache instead" if s.served_from_cache
           else "")
        + f" (first: {s.errors[0]})"
        for name, s in sorted(per_source.items())
        if s.errors
    ]  # fmt: skip
    failed_sources = [n for n, s in per_source.items() if s.requests and s.papers == 0 and s.errors]
    if failed_sources:
        warnings.append(f"no results from: {', '.join(sorted(failed_sources))}")
    return warnings
