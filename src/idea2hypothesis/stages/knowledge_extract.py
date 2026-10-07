"""Stage 6 - KNOWLEDGE_EXTRACT: one evidence card per shortlisted paper (abstract level)."""

from __future__ import annotations

from typing import Any

from idea2hypothesis.pipeline.contracts import CARD_FIELDS, Findings, card_id_for
from idea2hypothesis.pipeline.models import StageContext
from idea2hypothesis.stages.base import StageFailure, compact_json, gather_limited, request_json
from idea2hypothesis.storage.runs import utc_now

STAGE = 6
EVIDENCE_SCOPE = "abstract"


def check_card_fields(data: Any) -> Findings:
    f = Findings()
    if not f.require(isinstance(data, dict), "the card must be a JSON object"):
        return f
    for key in CARD_FIELDS:
        value = data.get(key)
        f.require(
            key in data and (value is None or isinstance(value, str)),
            f"{key} must be a string or null",
        )
    return f


def render_card_markdown(card: dict[str, Any]) -> str:
    lines = [
        f"# {card['title']}",
        "",
        f"- Card: `{card['card_id']}`",
        f"- Paper: `{card['paper_id']}`",
    ]
    lines += [
        f"- Cite key: `{card['cite_key']}`",
        f"- Evidence scope: {card['evidence_scope']}",
        "",
    ]
    for key in CARD_FIELDS:
        lines += [f"## {key.title()}", "", str(card[key]) if card[key] else "_unknown_", ""]
    return "\n".join(lines)


async def _extract(ctx: StageContext, paper: dict[str, Any]) -> dict[str, Any]:
    authors = [a.get("name", "") for a in paper.get("authors", [])[:5] if isinstance(a, dict)]
    view = {
        "paper_id": paper["paper_id"],
        "title": paper.get("title", ""),
        "authors": authors,
        "year": paper.get("year") or None,
        "venue": paper.get("venue") or None,
        "abstract": paper.get("abstract", ""),
    }
    prompt = ctx.prompts.render("knowledge_extract", topic=ctx.topic, paper_json=compact_json(view))
    data, _ = await request_json(
        ctx, prompt, label=f"knowledge_extract {paper['paper_id']}", validate=check_card_fields
    )
    return {
        "schema_version": 1,
        "card_id": card_id_for(str(paper["paper_id"])),
        "paper_id": paper["paper_id"],
        "title": paper.get("title", ""),
        "cite_key": paper.get("cite_key", ""),
        "year": paper.get("year") or None,
        "venue": paper.get("venue") or None,
        "doi": paper.get("doi") or None,
        "arxiv_id": paper.get("arxiv_id") or None,
        "url": paper.get("url") or None,
        "evidence_scope": EVIDENCE_SCOPE,
        **{k: (data[k].strip() or None) if isinstance(data[k], str) else None for k in CARD_FIELDS},
        "generated_at": utc_now(),
    }


async def run(ctx: StageContext) -> list[str]:
    shortlist = ctx.artifacts.read_jsonl(5, "shortlist.jsonl")
    if not shortlist:
        raise StageFailure("EMPTY_SHORTLIST", "stage 6 needs a non-empty shortlist")
    skipped: list[dict[str, str]] = []
    papers: list[dict[str, Any]] = []
    for paper in shortlist:
        if str(paper.get("abstract") or "").strip():
            papers.append(paper)
        else:
            skipped.append({"paper_id": str(paper["paper_id"]), "reason": "no abstract available"})

    cards = await gather_limited(
        [lambda p=p: _extract(ctx, p) for p in papers], ctx.config.runtime.concurrency
    )
    if not cards:
        raise StageFailure("NO_CARDS", "no shortlisted paper has an abstract to extract from")
    for card in cards:
        ctx.artifacts.write_json(STAGE, f"cards/{card['card_id']}.json", card)
        ctx.artifacts.write_text(STAGE, f"cards/{card['card_id']}.md", render_card_markdown(card))
    ctx.artifacts.write_json(
        STAGE,
        "knowledge_meta.json",
        {
            "schema_version": 1,
            "shortlist_size": len(shortlist),
            "cards": len(cards),
            "evidence_scope": EVIDENCE_SCOPE,
            "skipped": skipped,
            "generated_at": utc_now(),
        },
    )
    return [f"{s['paper_id']}: {s['reason']}" for s in skipped]
