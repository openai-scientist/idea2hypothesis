"""Heuristic novelty assessment of hypotheses against retrieved literature.

The result is an *assessment*, not a proof of novelty: it measures keyword and title overlap
between hypotheses and papers that were actually retrieved.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

from idea2hypothesis.literature.models import LiteraturePort

logger = logging.getLogger(__name__)

DISCLAIMER = "heuristic assessment, not proof of novelty"

_STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "of",
        "for",
        "to",
        "with",
        "by",
        "at",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "between",
        "into",
        "through",
        "during",
        "before",
        "under",
        "over",
        "using",
        "based",
        "via",
        "toward",
        "towards",
        "new",
        "novel",
        "approach",
        "method",
        "study",
        "research",
        "paper",
        "work",
        "propose",
        "proposed",
        "show",
        "results",
        "performance",
        "evaluation",
    ]
)


def extract_keywords(text: str) -> list[str]:
    """Lower-cased distinct keywords of 3+ characters without stop words."""
    seen: set[str] = set()
    out: list[str] = []
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()):
        if token not in _STOP_WORDS and len(token) >= 3 and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def similarity(
    keywords: Sequence[str], title: str, abstract: str, reference_title: str = ""
) -> float:
    kw = _jaccard(keywords, extract_keywords(f"{title} {abstract}"))
    if reference_title and title:
        ratio = SequenceMatcher(None, reference_title.lower(), title.lower()).ratio()
        return round(0.7 * kw + 0.3 * ratio, 4)
    return round(kw, 4)


def build_queries(topic: str, statements: Sequence[str]) -> list[str]:
    queries = [topic]
    for statement in statements:
        statement = statement.strip()
        if len(statement) > 10:
            queries.append(statement[:200])
    keywords = extract_keywords(" ".join(statements))[:5]
    if keywords:
        joined = " ".join(keywords)
        if joined not in queries:
            queries.append(joined)
    return queries[:5]


def _assess(similar: list[dict[str, Any]]) -> tuple[float, str]:
    if not similar:
        return 1.0, "high"
    top = similar[:5]
    score = 1.0 - max(p["similarity"] for p in top)
    if sum(1 for p in top if p["similarity"] >= 0.4 and p.get("citation_count", 0) >= 50) >= 2:
        score *= 0.7
    score = round(max(0.0, min(1.0, score)), 3)
    if score >= 0.7:
        return score, "high"
    if score >= 0.45:
        return score, "moderate"
    if score >= 0.25:
        return score, "low"
    return score, "critical"


def _paper_view(
    paper_id: str,
    title: str,
    year: Any,
    venue: str,
    citations: Any,
    url: str,
    cite_key: str,
    sim: float,
) -> dict[str, Any]:  # noqa: PLR0913
    return {
        "paper_id": paper_id,
        "title": title,
        "year": year,
        "venue": venue,
        "citation_count": citations,
        "similarity": sim,
        "url": url,
        "cite_key": cite_key,
    }


async def check_novelty(
    topic: str,
    hypotheses: Sequence[dict[str, Any]],
    *,
    literature: LiteraturePort | None,
    papers_already_seen: Sequence[dict[str, Any]] = (),
    max_search_results: int = 30,
    similarity_threshold: float = 0.25,
    year_min: int = 0,
) -> dict[str, Any]:
    """Compare hypotheses with retrieved papers; searches the literature port when provided."""
    statements = [str(h.get("statement", "")) for h in hypotheses]
    keywords = extract_keywords(f"{topic}\n" + "\n".join(statements))
    queries = build_queries(topic, statements)

    candidates: list[dict[str, Any]] = []
    search_errors: list[str] = []
    retrieved = 0
    if literature is not None:
        try:
            report = await literature.search(
                queries, limit=min(15, max_search_results), year_min=year_min
            )
        except Exception as exc:  # noqa: BLE001 - novelty is advisory and must not stop the run
            search_errors.append(f"{type(exc).__name__}: {exc}")
        else:
            retrieved = len(report.papers)
            search_errors.extend(report.errors)
            for paper in report.papers[:max_search_results]:
                candidates.append(
                    {
                        "paper_id": paper.paper_id, "title": paper.title, "year": paper.year,
                        "venue": paper.venue, "citation_count": paper.citation_count,
                        "url": paper.url, "cite_key": paper.cite_key, "abstract": paper.abstract,
                    }
                )  # fmt: skip
    for row in papers_already_seen:
        if isinstance(row, dict):
            candidates.append(row)

    similar: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for row in candidates:
        title = str(row.get("title", ""))
        sim = similarity(keywords, title, str(row.get("abstract", "")))
        if sim >= similarity_threshold and title.lower() not in seen_titles:
            seen_titles.add(title.lower())
            similar.append(
                _paper_view(
                    str(row.get("paper_id", "")), title, row.get("year", 0),
                    str(row.get("venue", "")), row.get("citation_count", 0),
                    str(row.get("url", "")), str(row.get("cite_key", "")), sim,
                )
            )  # fmt: skip
    similar.sort(key=lambda p: p["similarity"], reverse=True)

    score, assessment = _assess(similar)
    if retrieved == 0 and not papers_already_seen:
        coverage = "insufficient"
    elif retrieved < 5:
        coverage = "partial"
    else:
        coverage = "full"
    if coverage == "insufficient" and not similar:
        assessment, recommendation = "insufficient_data", "proceed_with_caution"
    elif assessment == "critical":
        recommendation = "differentiate_or_reconsider"
    elif assessment == "low":
        recommendation = "differentiate"
    else:
        recommendation = "proceed"

    per_hypothesis = []
    for hyp in hypotheses:
        statement = str(hyp.get("statement", ""))
        hyp_keywords = extract_keywords(statement)
        best: dict[str, Any] | None = None
        for row in candidates:
            sim = similarity(
                hyp_keywords, str(row.get("title", "")), str(row.get("abstract", "")), statement
            )
            if best is None or sim > best["similarity"]:
                best = {
                    "title": str(row.get("title", "")),
                    "paper_id": str(row.get("paper_id", "")),
                    "similarity": sim,
                }
        per_hypothesis.append({"hypothesis_id": hyp.get("id", ""), "closest_paper": best})

    return {
        "kind": "novelty_assessment",
        "disclaimer": DISCLAIMER,
        "topic": topic,
        "hypotheses_checked": len(hypotheses),
        "search_queries": queries,
        "similar_papers_found": len(similar),
        "novelty_score": score,
        "assessment": assessment,
        "similar_papers": similar[:20],
        "per_hypothesis": per_hypothesis,
        "recommendation": recommendation,
        "similarity_threshold": similarity_threshold,
        "search_coverage": coverage,
        "total_papers_retrieved": retrieved,
        "search_errors": search_errors,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
    }
