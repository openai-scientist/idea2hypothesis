"""Stage 7 - SYNTHESIS: clusters, tensions and evidence-linked research gaps."""

from __future__ import annotations

from typing import Any

from idea2hypothesis.pipeline.contracts import (
    CARD_FIELDS,
    SIDED_TENSION_SCHEMA,
    card_source_text,
    check_synthesis,
    sub_question_ids,
)
from idea2hypothesis.pipeline.models import StageContext
from idea2hypothesis.stages.base import compact_json, request_json
from idea2hypothesis.storage.runs import utc_now

STAGE = 7
MAX_CARDS_CHARS = 120_000
MAX_FIELD_CHARS = 600


def load_cards(ctx: StageContext) -> list[dict[str, Any]]:
    stage = 6
    names = [
        n for n in ctx.artifacts.list_files(stage) if n.startswith("cards/") and n.endswith(".json")
    ]
    return [ctx.artifacts.read_json(stage, n) for n in names]


def card_view(card: dict[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {
        "card_id": card["card_id"], "title": card["title"], "year": card.get("year"),
    }  # fmt: skip
    for key in CARD_FIELDS:
        value = card.get(key)
        if value:
            view[key] = str(value)[:MAX_FIELD_CHARS]
    return view


def render_synthesis_markdown(synthesis: dict[str, Any]) -> str:
    lines = ["# Synthesis", "", str(synthesis.get("overview") or ""), "", "## Clusters", ""]
    for c in synthesis["clusters"]:
        lines.append(f"### {c['id']}: {c['title']}")
        lines += [str(c.get("claim") or ""), f"Cards: {', '.join(c['card_ids'])}", ""]
    tensions = synthesis.get("tensions") or []
    if tensions:
        lines += ["## Tensions", ""]
        for t in tensions:
            head = f"{t['id']} " if t.get("id") else ""
            between = " vs ".join(map(str, t.get("between", [])))
            lines.append(f"- {head}{between}: {t.get('text', '')}")
            for i, side in enumerate(t.get("sides") or [], 1):
                cards = ", ".join(side.get("card_ids") or [])
                lines.append(f"  - Side {i}: {side.get('claim', '')} ({cards})")
        lines.append("")
    lines += ["## Research gaps", ""]
    for g in synthesis["gaps"]:
        lines.append(f"### {g['id']}: {g['text']}")
        lines.append(f"- Sub-questions: {', '.join(g['sub_question_ids'])}")
        lines.append(f"- Evidence: {', '.join(g['card_ids'])}")
        if g.get("why_prioritized"):
            lines.append(f"- Why prioritised: {g['why_prioritized']}")
        lines.append("")
    opportunities = synthesis.get("prioritized_opportunities") or []
    if opportunities:
        lines += ["## Prioritised opportunities", ""]
        lines += [f"- {o.get('gap_id', '')}: {o.get('direction', '')}" for o in opportunities]
        lines.append("")
    return "\n".join(lines)


async def run(ctx: StageContext) -> list[str]:
    cards = load_cards(ctx)
    tree = ctx.artifacts.read_json(2, "problem_tree.json")
    warnings: list[str] = []

    views = [card_view(c) for c in cards]
    while len(compact_json(views)) > MAX_CARDS_CHARS and len(views) > 1:
        views.pop()
        warnings.append("cards were truncated to fit the synthesis prompt budget")
    included = {v["card_id"] for v in views}
    kept_cards = [c for c in cards if c["card_id"] in included]

    known_sq = sub_question_ids(tree)
    prompt = ctx.prompts.render(
        "synthesis",
        topic=ctx.topic,
        problem_tree_json=compact_json(
            {"sub_questions": [{"id": q["id"], "text": q["text"]} for q in tree["sub_questions"]]}
        ),
        cards_json=compact_json(views),
    )
    source_text = card_source_text(kept_cards)
    data, soft = await request_json(
        ctx,
        prompt,
        label="synthesis",
        validate=lambda d: check_synthesis(
            d, known_sq, included, source_text, every_card=True, sided_tensions=True
        ),
    )
    synthesis = {
        "schema_version": SIDED_TENSION_SCHEMA,
        "topic": ctx.topic,
        **data,
        "generated_at": utc_now(),
    }
    ctx.artifacts.write_json(STAGE, "synthesis.json", synthesis)
    ctx.artifacts.write_text(STAGE, "synthesis.md", render_synthesis_markdown(synthesis))
    return [*dict.fromkeys(warnings), *soft]
