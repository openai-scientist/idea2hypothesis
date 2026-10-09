"""Stage 6 - KNOWLEDGE_EXTRACT: one evidence card per shortlisted paper (abstract level).

Every filled field of a card is backed by passages copied from the abstract (``quotes``); the
passages are checked word for word, so a card cannot state what its abstract does not. A paper
whose card still fails that check after the repair rounds gets no card and is listed in
``knowledge_meta.json`` with the reason.
"""

from __future__ import annotations

from typing import Any

from idea2hypothesis.pipeline.contracts import (
    CARD_FIELDS,
    QUOTED_CARD_SCHEMA,
    Findings,
    card_id_for,
    check_card_quotes,
)
from idea2hypothesis.pipeline.models import StageContext
from idea2hypothesis.stages.base import StageFailure, compact_json, gather_limited, request_json
from idea2hypothesis.storage.runs import utc_now

STAGE = 6
EVIDENCE_SCOPE = "abstract"


def check_card_fields(data: Any, abstract: str | None = None) -> Findings:
    """The card's fields are text or null and, given the abstract, quoted from it."""
    f = Findings()
    if not f.require(isinstance(data, dict), "the card must be a JSON object"):
        return f
    for key in CARD_FIELDS:
        value = data.get(key)
        f.require(
            key in data and (value is None or isinstance(value, str)),
            f"{key} must be a string or null",
        )
    if abstract is not None and f.ok:
        f.extend(check_card_quotes(data, abstract))
    return f


def _quotes(data: dict[str, Any]) -> dict[str, list[str]]:
    """Quotes of the filled fields only, in field order."""
    raw = data.get("quotes") if isinstance(data.get("quotes"), dict) else {}
    out: dict[str, list[str]] = {}
    for key in CARD_FIELDS:
        if isinstance(data.get(key), str) and data[key].strip():
            out[key] = [str(q).strip() for q in raw.get(key) or [] if str(q).strip()]
    return out


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
    quotes = card.get("quotes") or {}
    for key in CARD_FIELDS:
        lines += [f"## {key.title()}", "", str(card[key]) if card[key] else "_unknown_", ""]
        lines += [f"> {q}" for q in quotes.get(key) or []]
        if quotes.get(key):
            lines.append("")
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
    abstract = str(paper.get("abstract") or "")
    data, _ = await request_json(
        ctx,
        prompt,
        label=f"knowledge_extract {paper['paper_id']}",
        validate=lambda d: check_card_fields(d, abstract),
    )
    return {
        "schema_version": QUOTED_CARD_SCHEMA,
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
        "quotes": _quotes(data),
        "generated_at": utc_now(),
    }


def _partial_name(paper: dict[str, Any]) -> str:
    return f"{card_id_for(str(paper['paper_id']))}.json"


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

    # Cards written before a pause or a retried try of this attempt are reused, not re-extracted.
    cached: dict[str, dict[str, Any]] = {}
    for paper in papers:
        card = ctx.artifacts.read_partial(STAGE, ctx.attempt, _partial_name(paper))
        if isinstance(card, dict) and card.get("paper_id") == paper["paper_id"]:
            cached[str(paper["paper_id"])] = card
    await ctx.progress("cards_plan", total=len(papers), cached=len(cached), skipped=len(skipped))
    done = 0

    async def extract(paper: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal done
        card = cached.get(str(paper["paper_id"]))
        if card is None:
            try:
                card = await _extract(ctx, paper)
            except StageFailure as exc:
                # The model could not back the card with the abstract's own words: no card is
                # better than an unverifiable one. The paper is listed with the reason.
                skipped.append({"paper_id": str(paper["paper_id"]), "reason": exc.message})
                return None
            ctx.artifacts.write_partial(STAGE, ctx.attempt, _partial_name(paper), card)
        done += 1
        await ctx.progress(
            "card", partial=_partial_name(paper), card_id=card["card_id"], index=done,
            total=len(papers),
        )  # fmt: skip
        return card

    ordered = sorted(papers, key=lambda p: str(p["paper_id"]) not in cached)
    results = await gather_limited(
        [lambda p=p: extract(p) for p in ordered], ctx.config.runtime.concurrency
    )
    cards = [card for card in results if card is not None]
    if not cards:
        raise StageFailure(
            "NO_CARDS",
            "no shortlisted paper gave a card"
            + (f" ({'; '.join(s['reason'] for s in skipped[:3])})" if skipped else ""),
        )
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
            # Every filled field of every card is backed by quotes checked against its abstract.
            "quoted": True,
            "skipped": skipped,
            "generated_at": utc_now(),
        },
    )
    return [f"{s['paper_id']}: {s['reason']}" for s in skipped]
