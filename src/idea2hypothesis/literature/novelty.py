"""Heuristic novelty assessment of hypotheses against retrieved literature.

The result is an *assessment*, not a proof of novelty: for each hypothesis a judge (the stage's
model) reads the retrieved papers closest to it by keyword overlap and says whether one already
tests its prediction. Papers that were not retrieved, or not among the closest, are not read.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
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


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word


def overlap(keywords: Sequence[str], title: str, abstract: str) -> float:
    """Share of a hypothesis's keywords that a paper's title and abstract contain.

    It only ranks papers for the judge to read: a paper can test the same prediction in other
    words, and a high share does not mean it does.
    """
    wanted = {_stem(k) for k in keywords}
    if not wanted:
        return 0.0
    found = {_stem(k) for k in extract_keywords(f"{title} {abstract}")}
    return round(len(wanted & found) / len(wanted), 4)


#: Papers per hypothesis that the judge reads: the closest by keyword overlap.
PAPERS_PER_HYPOTHESIS = 5
#: Hypotheses to judge, each with the papers to read -> verdict per hypothesis id, each
#: ``{"verdict", "paper_ids", "reason"}``; None when no judgement could be made. A verdict is
#: "tested" (a paper already tests the prediction), "related" (one studies the same exposure or
#: outcome without testing it) or "new" (none of the papers read does).
Judge = Callable[[list[dict[str, Any]]], Awaitable[dict[str, dict[str, Any]] | None]]


def _paper_view(row: dict[str, Any], share: float) -> dict[str, Any]:
    return {
        "paper_id": str(row.get("paper_id", "")),
        "title": str(row.get("title", "")),
        "year": row.get("year", 0),
        "venue": str(row.get("venue", "")),
        "citation_count": row.get("citation_count", 0),
        "url": str(row.get("url", "")),
        "cite_key": str(row.get("cite_key", "")),
        "similarity": share,
    }


def _assess(verdicts: list[str]) -> tuple[float, str]:
    """Share of hypotheses no paper tests (related work counts half)."""
    score = sum(1.0 if v == "new" else 0.5 if v == "related" else 0.0 for v in verdicts)
    score = round(score / len(verdicts), 3)
    if score >= 0.7:
        return score, "high"
    if score >= 0.45:
        return score, "moderate"
    if score >= 0.25:
        return score, "low"
    return score, "critical"


def _recommendation(verdicts: list[str], coverage: str) -> str:
    tested = verdicts.count("tested")
    if tested and tested * 2 >= len(verdicts):
        return "differentiate_or_reconsider"
    if tested:
        return "differentiate"
    if coverage == "run_corpus_only":
        return "proceed_with_caution"
    return "proceed"


async def check_novelty(
    topic: str,
    hypotheses: Sequence[dict[str, Any]],
    *,
    queries: Sequence[str],
    literature: LiteraturePort | None,
    judge: Judge,
    papers_already_seen: Sequence[dict[str, Any]] = (),
    per_query: int = 15,
    year_min: int = 0,
) -> dict[str, Any]:
    """Search for prior work on each hypothesis and have ``judge`` read the closest papers.

    ``queries`` are short keyword queries, one per hypothesis, written by the stage: a whole
    hypothesis sentence finds nothing on sources that match every word (arXiv). Keyword overlap
    only picks which papers the judge reads; the verdicts decide what the report says.
    """
    queries = [q.strip() for q in queries if q.strip()]

    rows: list[dict[str, Any]] = []
    search_errors: list[str] = []
    retrieved = 0
    if literature is not None and queries:
        try:
            report = await literature.search(queries, limit=per_query, year_min=year_min)
        except Exception as exc:  # noqa: BLE001 - novelty is advisory and must not stop the run
            search_errors.append(f"{type(exc).__name__}: {exc}")
        else:
            retrieved = len(report.papers)
            search_errors.extend(report.errors)
            for paper in report.papers:
                rows.append(
                    {
                        "paper_id": paper.paper_id, "title": paper.title, "year": paper.year,
                        "venue": paper.venue, "citation_count": paper.citation_count,
                        "url": paper.url, "cite_key": paper.cite_key, "abstract": paper.abstract,
                    }
                )  # fmt: skip
    rows.extend(row for row in papers_already_seen if isinstance(row, dict))
    # One row per paper: the search may return a paper the run already has.
    papers: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("title", "")).strip().lower()
        if key and str(row.get("abstract") or "").strip():
            papers.setdefault(key, row)

    # Each hypothesis is ranked against every paper on its own; keywords pooled over the set
    # would dilute every overlap.
    shortlists: list[list[tuple[float, dict[str, Any]]]] = []
    for hyp in hypotheses:
        keywords = extract_keywords(str(hyp.get("statement", "")))
        ranked = sorted(
            ((overlap(keywords, str(r.get("title", "")), str(r.get("abstract", ""))), r)
             for r in papers.values()),
            key=lambda pair: pair[0], reverse=True,
        )  # fmt: skip
        shortlists.append(ranked[:PAPERS_PER_HYPOTHESIS])

    if retrieved == 0 and not papers_already_seen:
        coverage = "insufficient"
    elif retrieved == 0:
        # Only the run's own papers were compared: no match there says nothing about the field.
        coverage = "run_corpus_only"
    elif retrieved < 5:
        coverage = "partial"
    else:
        coverage = "full"

    judged: dict[str, dict[str, Any]] | None = None
    if papers:
        judged = await judge(
            [
                {
                    "hypothesis": hyp,
                    "papers": [
                        {"paper_id": str(r.get("paper_id", "")), "title": r.get("title", ""),
                         "year": r.get("year") or None, "abstract": str(r.get("abstract", ""))}
                        for _, r in shortlist
                    ],
                }
                for hyp, shortlist in zip(hypotheses, shortlists, strict=True)
            ]
        )  # fmt: skip

    per_hypothesis: list[dict[str, Any]] = []
    similar: list[dict[str, Any]] = []
    for hyp, shortlist in zip(hypotheses, shortlists, strict=True):
        hid = str(hyp.get("id", ""))
        by_id = {str(r.get("paper_id", "")): (share, r) for share, r in shortlist}
        verdict = (judged or {}).get(hid) or {}
        matches = [by_id[pid] for pid in verdict.get("paper_ids", []) if pid in by_id]
        for share, r in matches:
            similar.append({**_paper_view(r, share), "hypothesis_id": hid,
                            "verdict": verdict["verdict"]})  # fmt: skip
        first = matches[0] if matches else (shortlist[0] if shortlist else None)
        per_hypothesis.append(
            {
                "hypothesis_id": hid,
                "verdict": verdict.get("verdict"),
                "reason": verdict.get("reason"),
                # The paper the verdict names first; the closest by overlap when it names none.
                "closest_paper": {"title": str(first[1].get("title", "")),
                                  "paper_id": str(first[1].get("paper_id", "")),
                                  "similarity": first[0]} if first else None,
                "papers_read": list(by_id),
            }
        )  # fmt: skip

    verdicts = [str(row["verdict"]) for row in per_hypothesis if row["verdict"]]
    if judged is None or len(verdicts) < len(per_hypothesis) or not verdicts:
        score, assessment, recommendation = None, "insufficient_data", "proceed_with_caution"
    else:
        score, assessment = _assess(verdicts)
        recommendation = _recommendation(verdicts, coverage)

    return {
        "kind": "novelty_assessment",
        "disclaimer": DISCLAIMER,
        "method": (
            "a model read the papers closest to each hypothesis by keyword overlap and judged "
            "whether one already tests its prediction"
        ),
        "topic": topic,
        "hypotheses_checked": len(hypotheses),
        "search_queries": queries,
        "papers_compared": len(papers),
        "similar_papers_found": len({p["paper_id"] for p in similar if p["verdict"] == "tested"}),
        "novelty_score": score,
        "assessment": assessment,
        "similar_papers": similar,
        "per_hypothesis": per_hypothesis,
        "recommendation": recommendation,
        "search_coverage": coverage,
        "total_papers_retrieved": retrieved,
        "search_errors": search_errors,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
    }
