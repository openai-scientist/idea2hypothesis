"""Stage 9 - ARGUMENT_MAP: the run's records as one semantic graph and as a research canvas.

One model call judges what no earlier stage records: how each clustered card bears on its
cluster's claim, and which claims ground or challenge each hypothesis. Everything else in the
graph and the canvas is copied from the records of stages 1 to 8, and every relation says so
(``status``: ``stated`` by a record, ``unreviewed`` when it is the model's judgement).
"""

from __future__ import annotations

import re
from typing import Any

from idea2hypothesis.pipeline.contracts import check_argument_map, cluster_members
from idea2hypothesis.pipeline.models import StageContext
from idea2hypothesis.stages.base import compact_json, request_json
from idea2hypothesis.stages.synthesis import load_cards
from idea2hypothesis.storage.runs import utc_now

STAGE = 9
MAX_FIELD_CHARS = 500
ONTOLOGY = (
    "Scientific Research Canvas v1.0: 7 entity types (5 core, 2 optional), 11 directed "
    "relations, 3 presentation layers. A hypothesis is a prediction still to be tested, never a "
    "claim; layers group the display only."
)


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def _surname(author: dict[str, Any]) -> str:
    name = str(author.get("name") or "").strip()
    if "," in name:  # "Liu, Hongsheng"
        return name.split(",", 1)[0].strip()
    return name.split()[-1] if name else ""


def paper_labels(ctx: StageContext) -> dict[str, tuple[str, str]]:
    """``paper_id -> (citation, code)``: ``Wang et al., 2026`` and a short unique ``wang2026``."""
    rows = (
        ctx.artifacts.read_jsonl(5, "shortlist.jsonl")
        if ctx.artifacts.exists(5, "shortlist.jsonl")
        else []
    )
    out: dict[str, tuple[str, str]] = {}
    used: set[str] = set()
    for row in rows:
        authors = row.get("authors") or []
        name = _surname(authors[0]) if authors else ""
        year = row.get("year") or "n.d."
        if name:
            citation = f"{name} et al., {year}" if len(authors) > 1 else f"{name}, {year}"
            base = re.sub(r"[^a-z]", "", name.lower()) + str(year)
        else:
            citation = str(row.get("cite_key") or row.get("title") or row["paper_id"])[:60]
            base = str(row.get("cite_key") or row["paper_id"])[:16]
        code, n = base, 1
        while code in used:
            n += 1
            code = f"{base}{chr(ord('a') + n - 1)}"
        used.add(code)
        out[str(row["paper_id"])] = (citation, code)
    return out


# ---------------------------------------------------------------------------
# Semantic graph
# ---------------------------------------------------------------------------


class _Graph:
    def __init__(self) -> None:
        self.entities: list[dict[str, Any]] = []
        self.relations: list[dict[str, Any]] = []
        self._ids: set[str] = set()

    def has(self, entity_id: str) -> bool:
        return entity_id in self._ids

    def entity(self, entity_id: str, type_: str, code: str, text: str, stage: int, **extra: Any):
        self._ids.add(entity_id)
        row = {"id": entity_id, "type": type_, "code": code, "label": text, "text": text}
        row["stage"] = stage
        row.update({k: v for k, v in extra.items() if v not in (None, "", [])})
        self.entities.append(row)

    def relation(
        self,
        source: str,
        relation: str,
        target: str,
        status: str,
        rationale: str,
        provenance: str,
        polarity: str | None = None,
    ) -> None:
        if not (self.has(source) and self.has(target)):
            return
        row: dict[str, Any] = {
            "id": f"{source}|{relation}|{target}",
            "from": source,
            "to": target,
            "relation": relation,
        }
        if polarity:
            row["polarity"] = polarity
        row.update({"status": status, "rationale": rationale, "provenance": provenance})
        self.relations.append(row)


def _facts(*pairs: tuple[str, Any]) -> list[dict[str, str]]:
    return [{"label": k, "value": str(v)} for k, v in pairs if v not in (None, "", [])]


def _number(hypothesis_id: str) -> str:
    digits = re.sub(r"\D", "", hypothesis_id)
    return digits or hypothesis_id


def build_graph(
    *,
    topic: str,
    goal: dict[str, Any],
    tree: dict[str, Any],
    synthesis: dict[str, Any],
    cards: dict[str, dict[str, Any]],
    labels: dict[str, tuple[str, str]],
    hypotheses: list[dict[str, Any]],
    judged: dict[str, Any],
) -> dict[str, Any]:
    """Entities in the order the run made them, each relation after both its ends."""
    g = _Graph()
    topic_id = "RQ:topic"
    g.entity(
        topic_id,
        "question",
        "RQ",
        topic,
        1,
        facts=_facts(
            ("Problem", goal.get("problem")),
            ("Objective", goal.get("objective")),
            ("Novel angle", goal.get("novel_angle")),
        ),
    )
    risks = {str(r.get("sub_question_id")): r for r in tree.get("risks") or []}
    for q in tree.get("sub_questions", []):
        sq = f"RQ:{q['id']}"
        risk = risks.get(str(q["id"]))
        g.entity(
            sq,
            "question",
            str(q["id"]),
            str(q["text"]),
            2,
            facts=_facts(
                ("Tests", q.get("tests")),
                ("Priority", q.get("priority")),
                ("Goal link", q.get("goal_link")),
                (
                    f"Risk · {risk.get('level')}" if risk else "Risk",
                    risk.get("text") if risk else None,
                ),
            ),
        )
        g.relation(
            topic_id,
            "decomposes_into",
            sq,
            "stated",
            f"Linked to the goal: {q.get('goal_link') or q.get('tests') or q['text']}",
            "Stage 2 · problem tree",
        )

    code_of = {
        cid: labels.get(str(c["paper_id"]), (c.get("cite_key") or cid, cid))[1]
        for cid, c in cards.items()
    }
    links = [x for x in judged["evidence_links"] if x["relation"] != "unrelated"]
    for cluster in synthesis.get("clusters", []):
        claim = f"CL:{cluster['id']}"
        own = [x for x in links if x["claim_id"] == cluster["id"] and x["card_id"] in cards]
        for link in own:
            card = cards[link["card_id"]]
            evidence = f"EV:{code_of[card['card_id']]}"
            if g.has(evidence):
                continue
            g.entity(
                evidence,
                "evidence",
                code_of[card["card_id"]],
                str(card.get("findings") or ""),
                6,
                detail=card.get("data"),
                source=labels.get(str(card["paper_id"]), (card.get("title"), ""))[0],
                facts=_facts(
                    ("Method", card.get("method")),
                    ("Limitations", card.get("limitations")),
                    ("Card", card["card_id"]),
                ),
            )
        g.entity(
            claim,
            "claim",
            str(cluster["id"]),
            str(cluster.get("claim") or cluster["title"]),
            7,
            origin="literature",
            facts=_facts(
                ("School of thought", cluster.get("title")),
                ("Cards", len(cluster.get("card_ids", []))),
            ),
        )
        for link in own:
            g.relation(
                f"EV:{code_of[link['card_id']]}",
                link["relation"],
                claim,
                "unreviewed",
                link["rationale"],
                f"Stage 9 · {cluster['id']} judged against its cards",
            )
    for gap in synthesis.get("gaps", []):
        gid = f"GAP:{gap['id']}"
        papers = [cards[c] for c in gap.get("card_ids", []) if c in cards]
        g.entity(
            gid,
            "gap",
            str(gap["id"]),
            str(gap["text"]),
            7,
            provenance=[
                {
                    "source": labels.get(str(c["paper_id"]), (c.get("title"), ""))[0],
                    "text": str(c.get("limitations") or ""),
                }
                for c in papers
            ],
            facts=_facts(
                ("Why prioritised", gap.get("why_prioritized")), ("Priority", gap.get("priority"))
            ),
        )
        for sq in gap.get("sub_question_ids", []):
            g.relation(
                gid,
                "motivates",
                f"RQ:{sq}",
                "stated",
                str(gap.get("why_prioritized") or gap["text"]),
                f"Stage 7 · {gap['id']}",
            )

    rationales = judged["rationales"]
    for h in hypotheses:
        hid = f"H:{h['id']}"
        g.entity(
            hid,
            "hypothesis",
            str(h["id"]),
            str(h["statement"]),
            8,
            facts=_facts(
                ("Predicts", f"{h.get('exposure')} → {h.get('outcome')}, {h.get('prediction')}"),
                ("Test", f"{h.get('estimand')} by {h.get('method')}"),
                ("Wrong if", h.get("falsification_criteria")),
                ("Why", h.get("rationale")),
                ("Risk", h.get("risk")),
            ),
        )
        for sq in h.get("sub_question_ids") or []:
            g.relation(
                hid,
                "proposes_answer_to",
                f"RQ:{sq}",
                "stated",
                f"Tests {h.get('exposure')} against {h.get('outcome')}.",
                f"Stage 8 · {h['id']}",
            )
        for r in rationales:
            if r["hypothesis_id"] == h["id"]:
                g.relation(
                    f"CL:{r['claim_id']}",
                    "provides_rationale_for",
                    hid,
                    "unreviewed",
                    r["rationale"],
                    f"Stage 9 · {h['id']} rationale",
                    polarity=r["polarity"],
                )
        g.relation(
            hid,
            "addresses",
            f"GAP:{h['gap_id']}",
            "stated",
            str(h.get("novelty") or ""),
            f"Stage 8 · {h['id']}",
        )
        if h.get("conditions"):
            assumption = f"AS:{h['id']}"
            g.entity(
                assumption, "assumption", f"AS{_number(str(h['id']))}", str(h["conditions"]), 8
            )
            g.relation(
                hid,
                "depends_on",
                assumption,
                "stated",
                f"The conditions under which {h['id']} is claimed to hold.",
                f"Stage 8 · {h['id']}",
            )
    for h in hypotheses:
        ec = f"EC:{h['id']}"
        g.entity(
            ec,
            "contribution",
            f"EC{_number(str(h['id']))}",
            str(h.get("novelty") or ""),
            8,
            facts=_facts(("Status", f"Expected. Nothing is shown until {h['id']} is tested.")),
        )
        g.relation(
            f"H:{h['id']}",
            "informs",
            ec,
            "stated",
            f"What {h['id']} would add if it holds.",
            f"Stage 8 · {h['id']} novelty",
        )
        g.relation(
            ec,
            "targets",
            f"GAP:{h['gap_id']}",
            "stated",
            f"Would narrow {h['gap_id']}.",
            f"Stage 8 · {h['id']}",
        )
    return {
        "schema_version": 1,
        "topic": topic,
        "ontology": ONTOLOGY,
        "entities": g.entities,
        "relations": g.relations,
        "generated_at": utc_now(),
    }


# ---------------------------------------------------------------------------
# Research canvas (AMJ Management Research Canvas, Dorobantu et al., 2024)
# ---------------------------------------------------------------------------


def _item(text: Any, *refs: str) -> dict[str, Any]:
    item: dict[str, Any] = {"text": str(text)}
    if refs:
        item["refs"] = [r for r in refs if r]
    return item


def build_canvas(
    *,
    topic: str,
    goal: dict[str, Any],
    tree: dict[str, Any],
    synthesis: dict[str, Any],
    hypotheses: list[dict[str, Any]],
) -> dict[str, Any]:
    """The nine pieces, the puzzle first, each line with the records it comes from."""
    domains = goal.get("domains") or []
    success = goal.get("success_criteria") or []
    scope = goal.get("scope")
    conditions = list(
        dict.fromkeys(str(h["conditions"]) for h in hypotheses if h.get("conditions"))
    )
    limits = [(h["id"], x) for h in hypotheses for x in (h.get("limitations") or []) if x]
    pieces = [
        {
            "id": "puzzle",
            "status": "filled",
            "items": [_item(topic, "topic")]
            + ([_item(goal["problem"])] if goal.get("problem") else []),
        },
        {
            "id": "audience",
            "status": "filled",
            "note": "The audience is implied by the fields you listed; the prior research is the "
            "run's schools of thought.",
            "items": ([_item(f"Conversations in {', '.join(domains)}.")] if domains else [])
            + [
                _item(f"{c['title']}: {c.get('claim', '')}", str(c["id"]))
                for c in synthesis.get("clusters", [])
            ]
            + ([_item(synthesis["overview"])] if synthesis.get("overview") else []),
        },
        {
            "id": "question",
            "status": "filled",
            "items": ([_item(goal["objective"])] if goal.get("objective") else [])
            + [_item(q["text"], str(q["id"])) for q in tree.get("sub_questions", [])],
        },
        {
            "id": "theory",
            "status": "filled",
            "items": [
                _item(
                    f"{h.get('exposure')} → {h.get('outcome')}, expected {h.get('prediction')}. "
                    f"{h.get('rationale') or ''}".strip(),
                    str(h["id"]),
                )
                for h in hypotheses
            ]
            + [
                _item(t.get("text", ""), *map(str, t.get("between", [])))
                for t in synthesis.get("tensions") or []
                if t.get("text")
            ],
        },
        {
            "id": "setting",
            "status": "filled",
            "items": ([_item(scope, "topic")] if scope else [])
            + [
                _item(c, *[str(h["id"]) for h in hypotheses if h.get("conditions") == c])
                for c in conditions
            ],
        },
        {
            "id": "design",
            "status": "filled",
            "items": [
                _item(f"{h.get('estimand')} by {h.get('method')}.", str(h["id"]))
                for h in hypotheses
            ]
            + [_item(s) for s in success],
        },
        {
            "id": "findings",
            "status": "pending",
            "note": "Nothing has been tested yet. These are the results that would count against "
            "each hypothesis.",
            "items": [_item(h.get("falsification_criteria", ""), str(h["id"])) for h in hypotheses],
        },
        {
            "id": "contributions",
            "status": "filled",
            "note": "What each hypothesis would add if it holds.",
            "items": [
                _item(h.get("novelty", ""), str(h["id"]), str(h.get("gap_id") or ""))
                for h in hypotheses
            ],
        },
        {
            "id": "boundaries",
            "status": "filled",
            "items": [
                _item(r["text"], str(r.get("sub_question_id") or ""))
                for r in tree.get("risks") or []
                if r.get("text")
            ]
            + [_item(text, str(hid)) for hid, text in limits]
            + [
                _item(f"Set aside: {a.get('reason', '')}")
                for a in synthesis.get("set_aside") or []
                if a.get("reason")
            ],
        },
    ]
    # A piece the run has nothing for stays visibly empty rather than padded.
    for piece in pieces:
        if not piece["items"] and piece["status"] == "filled":
            piece["status"] = "pending"
            piece["note"] = "The run recorded nothing for this piece."
    return {"schema_version": 1, "topic": topic, "pieces": pieces, "generated_at": utc_now()}


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


def _clip(value: Any) -> str:
    return str(value or "")[:MAX_FIELD_CHARS]


async def run(ctx: StageContext) -> list[str]:
    goal = ctx.artifacts.read_json(1, "goal.json")
    tree = ctx.artifacts.read_json(2, "problem_tree.json")
    synthesis = ctx.artifacts.read_json(7, "synthesis.json")
    hypotheses = ctx.artifacts.read_json(8, "hypotheses.json")["hypotheses"]
    cards = {c["card_id"]: c for c in load_cards(ctx)}
    members = cluster_members(synthesis)
    ids = {str(h["id"]) for h in hypotheses}

    claims = [
        {
            "id": c["id"],
            "claim": c.get("claim") or c["title"],
            "cards": [
                {
                    "card_id": cid,
                    "findings": _clip(cards[cid].get("findings")),
                    "data": _clip(cards[cid].get("data")),
                }
                for cid in c.get("card_ids", [])
                if cid in cards
            ],
        }
        for c in synthesis.get("clusters", [])
    ]
    views = [
        {
            "id": h["id"],
            "statement": h["statement"],
            "prediction": h.get("prediction"),
            "rationale": h.get("rationale"),
            "gap_id": h.get("gap_id"),
            "evidence_refs": h.get("evidence_refs", []),
        }
        for h in hypotheses
    ]
    prompt = ctx.prompts.render(
        "argument_map",
        topic=ctx.topic,
        claims_json=compact_json(claims),
        hypotheses_json=compact_json(views),
    )
    data, soft = await request_json(
        ctx,
        prompt,
        label="argument_map",
        validate=lambda d: check_argument_map(d, members, ids),
    )
    judged = {
        "schema_version": 1,
        "topic": ctx.topic,
        "assessment": "model judgement, not reviewed",
        "evidence_links": data["evidence_links"],
        "rationales": data["rationales"],
        "generated_at": utc_now(),
    }
    ctx.artifacts.write_json(STAGE, "argument_map.json", judged)
    graph = build_graph(
        topic=ctx.topic,
        goal=goal,
        tree=tree,
        synthesis=synthesis,
        cards=cards,
        labels=paper_labels(ctx),
        hypotheses=hypotheses,
        judged=judged,
    )
    ctx.artifacts.write_json(STAGE, "semantic_graph.json", graph)
    ctx.artifacts.write_json(
        STAGE,
        "research_canvas.json",
        build_canvas(
            topic=ctx.topic, goal=goal, tree=tree, synthesis=synthesis, hypotheses=hypotheses
        ),
    )
    warnings = list(soft)
    unrelated = [x for x in judged["evidence_links"] if x["relation"] == "unrelated"]
    if unrelated:
        warnings.append(
            f"{len(unrelated)} clustered card(s) judged unrelated to their cluster's claim; "
            "they are left off the graph"
        )
    return warnings
