"""Translate core run events and stage artifacts into the Platform event stream.

Platform consumers (BE and the run studio) read ``{source_seq, type, stage_key, actor, payload}``
envelopes. Every envelope built here is derived from a core event that actually happened and
from the artifacts the stage really wrote; nothing is simulated. Envelopes are appended to
``runs/<id>/platform_events.jsonl`` (``source_seq`` contiguous from 1) before anything is
delivered, so they can be replayed through ``GET /runs/{id}/events`` and the webhook.

UI groups (``stage_key``): scope = stages 1-2, search = 3-4, screen = 5, read = 6,
synthesize = 7 and ``r1-hypothesize`` = 8 (the group key the run studio reads).

Stages announce parts of their result while they run (core ``stage.progress``): a query's hits,
a scored batch, a card, a perspective. Those parts are sent at once; when the stage completes
only what was not sent yet follows. ``agent.message`` lines say what a step is doing and, when
it ends, what it produced; their numbers and names come from the same records.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from idea2hypothesis.pipeline import events as ev
from idea2hypothesis.pipeline import gates
from idea2hypothesis.pipeline.events import Event
from idea2hypothesis.pipeline.models import Stage
from idea2hypothesis.storage.artifacts import ArtifactStore, write_bytes_atomic
from idea2hypothesis.storage.runs import RunStore

logger = logging.getLogger(__name__)

PLATFORM_FILE = "platform_events.jsonl"
HYPOTHESIZE_KEY = "r1-hypothesize"
MAP_KEY = "map"
_PRIVATE = ("_core_seq", "_batch")

# ---------------------------------------------------------------------------
# UI groups and plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Group:
    key: str
    stage: str
    title: str
    stages: tuple[int, ...]
    cast: tuple[str, ...]
    purpose: str
    reads: tuple[str, ...]

    @property
    def last(self) -> int:
        return self.stages[-1]


GROUPS: tuple[Group, ...] = (
    Group(
        "scope",
        "scope",
        "Scope the question",
        (1, 2),
        ("strategist", "pi"),
        "Your topic becomes a research goal and a tree of prioritised sub-questions, and the "
        "topic is rated before any search starts.",
        ("Your topic", "Your fields"),
    ),
    Group(
        "search",
        "search",
        "Search the literature",
        (3, 4),
        ("librarian",),
        "Queries derived from the sub-questions go to the configured scholarly sources and "
        "everything found is merged into one deduplicated list.",
        ("Sub-question tree",),
    ),
    Group(
        "screen",
        "screen",
        "Screen the papers",
        (5,),
        ("librarian", "pi"),
        "Every candidate is scored for relevance and quality; only papers that clear both "
        "bars are kept and every decision carries a reason.",
        ("Candidate list", "Sub-question tree"),
    ),
    Group(
        "read",
        "read",
        "Read and extract",
        (6,),
        ("librarian",),
        "Each shortlisted paper becomes one knowledge card; fields the abstract does not "
        "support stay empty.",
        ("Shortlist",),
    ),
    Group(
        "synthesize",
        "synthesize",
        "Find the gaps",
        (7,),
        ("theorist",),
        "Cards are grouped into schools of thought and research gaps are traced back to the "
        "cards behind them.",
        ("Knowledge cards", "Sub-question tree"),
    ),
    Group(
        HYPOTHESIZE_KEY,
        "hypothesize",
        "Hypothesize",
        (8,),
        ("theorist", "methodologist", "skeptic", "librarian", "pi"),
        "Several perspectives propose hypotheses; the final set links each one to a gap, to "
        "evidence and to the result that would prove it wrong.",
        ("Gap map", "Shortlist"),
    ),
    Group(
        MAP_KEY,
        "map",
        "Map the argument",
        (9,),
        ("strategist", "librarian", "theorist", "reporter"),
        "Everything the run found is drawn as one semantic graph, from the evidence to what each "
        "hypothesis would contribute, and laid out on the nine pieces of a research canvas.",
        ("Sub-question tree", "Knowledge cards", "Gap map", "Hypotheses"),
    ),
)

_BY_STAGE = {n: g for g in GROUPS for n in g.stages}

# step id -> (title, actor, pipeline stage, artifact, explanation)
_STEPS: dict[str, tuple[str, str, int, str | None, str]] = {
    "profile": (
        "Check the machine",
        "strategist",
        1,
        "Compute profile",
        "Reads the hardware of the machine that runs the engine.",
    ),
    "goal": (
        "Set the goal",
        "strategist",
        1,
        "Research goal",
        "Writes the goal: problem, objective, scope and success criteria.",
    ),
    "decompose": (
        "Split it into sub-questions",
        "strategist",
        2,
        "Sub-question tree",
        "Prioritised sub-questions, each linked to the goal, with their risks.",
    ),
    "evaluate": (
        "Rate the topic",
        "pi",
        2,
        "Topic score",
        "Scores novelty, specificity and feasibility before any search starts.",
    ),
    "scope_gate": (
        "Human check: scope",
        "pi",
        2,
        None,
        "Approve the goal and sub-questions, or ask for another scoping attempt.",
    ),
    "strategy": (
        "Plan the searches",
        "librarian",
        3,
        "Search plan",
        "Search strategies and queries, each tied to sub-questions.",
    ),
    "collect": (
        "Collect candidates",
        "librarian",
        4,
        "Candidate list",
        "Every query goes to every configured source; duplicates are merged.",
    ),
    "score": (
        "Score every paper",
        "librarian",
        5,
        "Screened scores",
        "Relevance and quality scores from 0 to 1.",
    ),
    "reject": (
        "Reject wrong-field matches",
        "librarian",
        5,
        None,
        "Papers that share words but not the field are excluded with a reason.",
    ),
    "shortlist": (
        "Keep the shortlist",
        "librarian",
        5,
        "Shortlist",
        "What clears both bars is kept with a one-line reason.",
    ),
    "screen_gate": (
        "Human check: shortlist",
        "pi",
        5,
        None,
        "Approve the papers, remove ones you do not trust, or search again.",
    ),
    "extract": (
        "Extract a card per paper",
        "librarian",
        6,
        "Knowledge cards",
        "Problem, method, data, metrics, findings and limitations from the abstract.",
    ),
    "cluster": (
        "Group into schools of thought",
        "theorist",
        7,
        None,
        "Cards are grouped by approach.",
    ),
    "overview": ("Sum up the field", "theorist", 7, None, "Where the evidence stands."),
    "tension": (
        "Find where they disagree",
        "theorist",
        7,
        None,
        "Where two groups pull in different directions.",
    ),
    "gaps": (
        "Name the gaps",
        "theorist",
        7,
        "Gap map",
        "Each gap is traced to the cards that point to it.",
    ),
    "rank": ("Rank the opportunities", "theorist", 7, None, "Gaps ordered by priority."),
    "debate": (
        "Compare the perspectives",
        "theorist",
        8,
        "Perspectives",
        "Each perspective proposes hypotheses; in debate rounds they challenge each other, answer "
        "every challenge, and review the answers.",
    ),
    "write": (
        "Write each hypothesis",
        "theorist",
        8,
        "Hypotheses",
        "Statement, gap, evidence, prediction and the result that would falsify it.",
    ),
    "check": (
        "Check novelty",
        "librarian",
        8,
        "Novelty check",
        "Heuristic comparison with the papers already retrieved.",
    ),
    "select": ("Pick the set", "pi", 8, None, "The final hypotheses of this run."),
    "hypotheses_gate": (
        "Human check: hypotheses",
        "pi",
        8,
        None,
        "Approve the set, drop hypotheses, keep a held-back one, or ask for a new set.",
    ),
    "questions": (
        "Lay out the questions",
        "strategist",
        9,
        None,
        "Your topic and the sub-questions it decomposes into, from stages 1 and 2.",
    ),
    "foundation": (
        "Lay the research foundation",
        "librarian",
        9,
        None,
        "The findings from stage 6, the claims they support or contradict, and the gaps from "
        "stage 7.",
    ),
    "reasoning": (
        "Tie in the hypotheses",
        "theorist",
        9,
        "Semantic graph",
        "Each hypothesis from stage 8: the sub-question it answers, the gap it addresses, and the "
        "claims that give it a rationale or challenge it.",
    ),
    "contribution": (
        "State what each would contribute",
        "theorist",
        9,
        None,
        "What each hypothesis would add if it holds. Expected only: nothing has been tested.",
    ),
    "canvas": (
        "Fill the research canvas",
        "reporter",
        9,
        "Research canvas",
        "The nine pieces of the AMJ Management Research Canvas, filled from the run. Findings "
        "stay open until something is tested.",
    ),
}

_STAGE_STEPS: dict[int, tuple[str, ...]] = {
    1: ("profile", "goal"),
    2: ("decompose", "evaluate"),
    3: ("strategy",),
    4: ("collect",),
    5: ("score", "reject", "shortlist"),
    6: ("extract",),
    7: ("cluster", "overview", "tension", "gaps", "rank"),
    8: ("debate", "write", "check", "select"),
    9: ("questions", "foundation", "reasoning", "contribution", "canvas"),
}


#: What a step is doing while it runs (no result yet). Steps that report progress replace it
#: with a line built from what they report.
_DOING: dict[str, str] = {
    "goal": "Turning your topic into a research goal: problem, objective, scope and success.",
    "decompose": "Splitting the goal into sub-questions and noting the risks of each.",
    "strategy": "Planning search angles and short queries for each sub-question.",
    "collect": "Sending the queries to each scholarly source.",
    "score": "Scoring every candidate for relevance and quality.",
    "extract": "Reading each shortlisted abstract into a knowledge card.",
    "cluster": "Grouping the knowledge cards into schools of thought.",
    "debate": "Each perspective is proposing hypotheses from the gaps.",
    "write": "Merging the perspectives into one set of testable hypotheses.",
    "check": "Comparing each hypothesis with the literature for novelty.",
    "questions": "Judging how each card bears on its claim and what grounds each hypothesis.",
}


@dataclass(frozen=True)
class Flags:
    mode: str
    hardware_advisory: bool
    novelty_check: bool

    def steps(self, stage: int) -> tuple[str, ...]:
        steps = _STAGE_STEPS[stage]
        if stage == 1 and not self.hardware_advisory:
            steps = tuple(s for s in steps if s != "profile")
        if stage == 8 and not self.novelty_check:
            steps = tuple(s for s in steps if s != "check")
        return steps

    def gate_step(self, group: Group) -> str | None:
        spec = gates.gate_after(self.mode, Stage(group.last))
        return None if spec is None else f"{spec.kind}_gate"


def _step_plan(step_id: str) -> dict[str, Any]:
    title, actor, pipeline_stage, artifact, explain = _STEPS[step_id]
    step: dict[str, Any] = {
        "id": step_id,
        "title": title,
        "actor": actor,
        "explain": explain,
        "pipeline_stage": pipeline_stage,
    }
    if artifact:
        step["artifact"] = artifact
    if step_id.endswith("_gate"):
        step["gate"] = step_id.removesuffix("_gate")
    return step


def planned_stage(group: Group, flags: Flags) -> dict[str, Any]:
    return {
        "key": group.key,
        "stage": group.stage,
        "title": group.title,
        "has_gate": flags.gate_step(group) is not None,
        "pipeline": list(group.stages),
    }


def stage_plan(group: Group, flags: Flags) -> dict[str, Any]:
    step_ids = [s for n in group.stages for s in flags.steps(n)]
    gate_step = flags.gate_step(group)
    if gate_step:
        step_ids.append(gate_step)
    return {
        **planned_stage(group, flags),
        "purpose": group.purpose,
        "reads": list(group.reads),
        "cast": list(group.cast),
        "steps": [_step_plan(s) for s in step_ids],
    }


# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------

Envelope = dict[str, Any]


def _env(
    type_: str,
    payload: dict[str, Any] | None = None,
    *,
    stage_key: str | None = None,
    actor: str | None = None,
) -> Envelope:
    return {"type": type_, "stage_key": stage_key, "actor": actor, "payload": payload or {}}


def _message(message_id: str, text: str, *, stage_key: str, actor: str, done: bool) -> Envelope:
    """One narration line; a later line with the same id replaces it, ``done`` closes it."""
    payload: dict[str, Any] = {"message_id": message_id, "text": text}
    if done:
        payload["done"] = True
    return _env("agent.message", payload, stage_key=stage_key, actor=actor)


def _message_id(stage_run: int, try_index: int, step_id: str) -> str:
    return f"m{stage_run}.{try_index}.{step_id}"


def public(row: dict[str, Any]) -> dict[str, Any]:
    """The Platform envelope without the private bookkeeping keys."""
    return {k: v for k, v in row.items() if k not in _PRIVATE}


def _cost_str(usage: dict[str, Any] | None) -> str | None:
    cost = (usage or {}).get("cost_usd")
    return None if cost is None else f"{float(cost):.4f}"


def _with_cost(payload: dict[str, Any], usage: dict[str, Any] | None) -> dict[str, Any]:
    cost = _cost_str(usage)
    if cost is not None:
        payload["cost_usd"] = cost
    return payload


def _short(text: str, limit: int = 80) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0] or text[:limit]
    return cut.rstrip(",;:.") + "..."


def _citation(row: dict[str, Any]) -> str:
    authors = row.get("authors") or []
    name = str(authors[0].get("name", "")).split()[-1] if authors and authors[0].get("name") else ""
    year = row.get("year") or "n.d."
    if not name:
        return str(row.get("cite_key") or row.get("title", ""))[:60]
    return f"{name} et al., {year}" if len(authors) > 1 else f"{name}, {year}"


# ---------------------------------------------------------------------------
# Per-stage content (only real artifacts)
# ---------------------------------------------------------------------------

StepEvents = dict[str, list[tuple[str, dict[str, Any]]]]


def _content_stage1(art: ArtifactStore, flags: Flags, domains: list[str]) -> StepEvents:
    out: StepEvents = {}
    if flags.hardware_advisory and art.exists(1, "hardware_profile.json"):
        hw = art.read_json(1, "hardware_profile.json")
        detail = (
            " · ".join(str(x) for x in (hw.get("tier"), hw.get("warning")) if x)
            or "hardware detected"
        )
        out["profile"] = [
            (
                "scope.profile",
                {
                    "domains": domains,
                    "compute": {"label": str(hw.get("gpu_name") or "CPU only"), "detail": detail},
                },
            ),
        ]
    goal = art.read_json(1, "goal.json")
    fields = (
        ("title", "Working title", goal.get("working_title")),
        ("problem", "Problem", goal.get("problem")),
        ("objective", "Objective", goal.get("objective")),
        ("scope", "Scope", goal.get("scope")),
        ("success", "Success criteria", "; ".join(goal.get("success_criteria") or [])),
    )
    out["goal"] = [
        ("scope.goal", {"field": f, "label": label, "value": str(value)})
        for f, label, value in fields
        if value
    ]
    return out


def _content_stage2(art: ArtifactStore) -> StepEvents:
    tree = art.read_json(2, "problem_tree.json")
    events: list[tuple[str, dict[str, Any]]] = []
    risks = tree.get("risks") or []
    for q in tree["sub_questions"]:
        events.append(
            (
                "problem.subquestion",
                {
                    "sub_question": {
                        "id": q["id"],
                        "text": q["text"],
                        "priority": q["priority"],
                        "tests": q.get("tests", ""),
                        "covers": q.get("covers", []),
                        "goal_link": q.get("goal_link"),
                    }
                },
            )
        )
        for r in risks:
            if r.get("sub_question_id") == q["id"]:
                events.append(
                    (
                        "problem.risk",
                        {
                            "risk": {
                                "id": r["id"],
                                "sq_id": r["sub_question_id"],
                                "text": r["text"],
                                "level": r["level"],
                            }
                        },
                    )
                )
    return {"decompose": events, "evaluate": [_topic_evaluated(art)]}


def _topic_evaluated(art: ArtifactStore) -> tuple[str, dict[str, Any]]:
    e = art.read_json(2, "topic_evaluation.json")
    return (
        "topic.evaluated",
        {
            "scores": {k: e[k] for k in ("novelty", "specificity", "feasibility")},
            "overall": e["overall"],
            "threshold": e.get("threshold"),
            "reasons": e.get("reasons") or {},
            "advice": e.get("suggestion", ""),
            "basis": e.get("basis") or "model judgement before any literature search",
        },
    )


def _content_stage3(art: ArtifactStore, delay_ms: int = 0) -> StepEvents:
    plan = _read_yaml(art, 3, "search_plan.yaml")
    queries = art.read_json(3, "queries.json")["queries"]
    sources = art.read_json(3, "sources.json")["sources"]
    events: list[tuple[str, dict[str, Any]]] = []
    ids = {s["name"]: f"S{i}" for i, s in enumerate(plan["search_strategies"], 1)}
    for s in plan["search_strategies"]:
        events.append(
            (
                "search.strategy",
                {
                    "strategy": {
                        "id": ids[s["name"]],
                        "title": s["name"],
                        "why": s.get("rationale", ""),
                    }
                },
            )
        )
        for q in queries:
            if q["strategy"] == s["name"]:
                events.append(
                    (
                        "search.query",
                        {
                            "query": {
                                "id": q["id"],
                                "strategy_id": ids[s["name"]],
                                "text": q["text"],
                            }
                        },
                    )
                )
    events.append(
        (
            "search.sources",
            {
                "sources": [{"id": s["id"], "name": s["name"]} for s in sources],
                "delay_ms": delay_ms,
            },
        )
    )
    return {"strategy": events}


def _read_yaml(art: ArtifactStore, stage: int, name: str) -> Any:
    import yaml

    return yaml.safe_load(art.read_text(stage, name))


def _merges(art: ArtifactStore, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Papers found as several records, each record as its source gave it. The first record is
    the one whose metadata was kept; the others filled its gaps."""
    names = {}
    if art.exists(3, "sources.json"):
        names = {s["id"]: s["name"] for s in art.read_json(3, "sources.json")["sources"]}
    out = []
    for paper in candidates:
        records = paper.get("source_records") or []
        # Records from before each kept its own citations say nothing about who was kept.
        if len(records) < 2 or any(r.get("citations") is None for r in records):
            continue
        out.append(
            {
                "title": paper["title"],
                "records": [
                    {
                        "source": names.get(r["provider"], r["provider"]),
                        "record_id": r["source_id"],
                        "citations": r["citations"],
                        "has_doi": bool(r.get("has_doi")),
                    }
                    for r in records
                ],
                "kept": names.get(records[0]["provider"], records[0]["provider"]),
                "kept_record": records[0]["source_id"],
            }
        )
    return out


def _content_stage4(art: ArtifactStore) -> StepEvents:
    meta = art.read_json(4, "search_meta.json")
    queries = {q["text"]: q["id"] for q in art.read_json(3, "queries.json")["queries"]}
    sources = [s["id"] for s in art.read_json(3, "sources.json")["sources"]]
    events: list[tuple[str, dict[str, Any]]] = []
    for text, per_source in meta.get("per_query", {}).items():
        if text not in queries:
            continue  # expansion queries have no id in the plan
        for source, hits in per_source.items():
            if source in sources:
                events.append(
                    ("literature.request", {"query_id": queries[text], "source_id": source})
                )
                events.append(
                    (
                        "literature.batch",
                        {"query_id": queries[text], "source_id": source, "hits": hits},
                    )
                )
    candidates = art.read_jsonl(4, "candidates.jsonl")
    merges = _merges(art, candidates)
    events += [("literature.merged", m) for m in merges]
    stamped = all(
        r.get("citations") is not None for c in candidates for r in c.get("source_records") or []
    )
    extra = [q["text"] for q in meta.get("queries_used") or [] if q.get("origin") == "expansion"]
    per_query = meta.get("per_query", {})
    events.append(
        (
            "literature.collected",
            {
                "raw": meta["raw"],
                "unique": meta["unique"],
                "duplicates": meta["duplicates"],
                # Duplicates that were not another source's record: a source returned the same
                # record to another query. Unknown for papers stored before records kept citations.
                "repeats": (
                    meta["duplicates"] - sum(len(m["records"]) - 1 for m in merges)
                    if stamped
                    else None
                ),
                # Queries the engine adds from the topic's words have no cell in the plan's grid.
                "expansion": {
                    "queries": len(extra),
                    "hits": sum(sum(per_query.get(q, {}).values()) for q in extra),
                },
                "files": [
                    {"name": "candidates.jsonl", "detail": f"{len(candidates)} papers"},
                    {"name": "references.bib", "detail": f"{len(candidates)} entries"},
                ],
                "collected_at": meta.get("ts"),
                "per_source": meta.get("per_source", {}),
                "errors": meta.get("errors", []),
            },
        )
    )
    return {"collect": events}


def _content_stage5(art: ArtifactStore) -> StepEvents:
    review = art.read_json(5, "review.json")
    decisions = review["decisions"]
    thresholds = review.get("thresholds", {})
    points = [
        {"id": d["paper_id"], "relevance": d["relevance_score"], "quality": d["quality_score"]}
        for d in decisions
        if d.get("relevance_score") is not None and d.get("quality_score") is not None
    ]
    score: list[tuple[str, dict[str, Any]]] = [
        (
            "screen.criteria",
            {
                "rules": review.get("rules", []),
                "relevance_min": thresholds.get("min_relevance"),
                "quality_min": thresholds.get("min_quality"),
                "candidates": len(decisions),
            },
        )
    ]
    score += [("screen.scored", {"points": points[i : i + 48]}) for i in range(0, len(points), 48)]
    reject = [_rejected(d) for d in decisions if d["decision"] in _NOT_KEPT]
    kept = []
    for row in art.read_jsonl(5, "shortlist.jsonl"):
        records = row.get("source_records") or []
        paper: dict[str, Any] = {
            "id": row["paper_id"],
            "citation": _citation(row),
            "title": row["title"],
            "venue": row.get("venue") or "",
            "year": row.get("year"),
            "relevance": row["relevance_score"],
            "quality": row["quality_score"],
            "reason": row["keep_reason"],
            "source": records[0]["provider"] if records else "",
            "citations": row.get("citation_count") or 0,
        }
        if row.get("doi"):
            paper["doi"] = row["doi"]
        kept.append(("screen.kept", {"paper": paper}))
    return {"score": score, "reject": reject, "shortlist": kept}


_NOT_KEPT = ("rejected", "below_cutoff", "unscored", "prefiltered")
_CARD_FIELDS = ("problem", "method", "data", "metrics", "findings", "limitations")


def _rejected(d: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return (
        "screen.rejected",
        {
            "paper": {
                "id": d["paper_id"],
                "title": d.get("title", ""),
                "venue": d.get("venue") or "",
                "false_friend": d.get("false_friend") or "",
                "reason": d["reason"],
                "decision": d["decision"],
            }
        },
    )


def _citations(art: ArtifactStore) -> dict[str, str]:
    """``Author et al., year`` for each shortlisted paper (cards keep no authors)."""
    if not art.exists(5, "shortlist.jsonl"):
        return {}
    return {str(r["paper_id"]): _citation(r) for r in art.read_jsonl(5, "shortlist.jsonl")}


def _card_extracted(
    card: dict[str, Any], citations: dict[str, str] | None = None
) -> tuple[str, dict[str, Any]]:
    cited = (citations or {}).get(str(card["paper_id"]))
    payload: dict[str, Any] = {
        "id": card["card_id"],
        "paper_id": card["paper_id"],
        "citation": cited or card.get("cite_key") or card.get("title", ""),
        "cite_key": card.get("cite_key"),
        "title": card.get("title"),
        "evidence_scope": card.get("evidence_scope"),
        "unknown_fields": [k for k in _CARD_FIELDS if card.get(k) is None],
    }
    payload.update({k: card.get(k) or "" for k in _CARD_FIELDS})
    # The abstract's own words behind each filled field (cards from schema 2 on).
    if isinstance(card.get("quotes"), dict):
        payload["quotes"] = {k: list(v) for k, v in card["quotes"].items() if v}
    return ("card.extracted", {"card": payload})


def _content_stage6(art: ArtifactStore) -> StepEvents:
    citations = _citations(art)
    return {"extract": [_card_extracted(art.read_json(6, name), citations) for name in _cards(art)]}


def _set_aside(syn: dict[str, Any]) -> list[dict[str, Any]]:
    """Cards left out of every cluster, with why; entries that name no card are skipped."""
    return [
        {
            "id": str(a.get("id") or f"A{i}"),
            "card_ids": a["card_ids"],
            "reason": a.get("reason", ""),
        }
        for i, a in enumerate(syn.get("set_aside") or [], 1)
        if isinstance(a, dict) and a.get("card_ids")
    ]


def _tension(t: dict[str, Any]) -> dict[str, Any]:
    """A point where cards disagree; from synthesis schema 2 on with an id and the cards on each
    of its two sides."""
    out: dict[str, Any] = {"between": t["between"], "text": t["text"]}
    if t.get("id"):
        out["id"] = t["id"]
    if t.get("sides"):
        out["sides"] = [{"claim": x["claim"], "card_ids": x["card_ids"]} for x in t["sides"]]
    return out


def _content_stage7(art: ArtifactStore) -> StepEvents:
    syn = art.read_json(7, "synthesis.json")
    out: StepEvents = {
        "cluster": [
            (
                "synthesis.cluster",
                {
                    "cluster": {
                        "id": c["id"],
                        "title": c["title"],
                        "claim": c.get("claim", ""),
                        "card_ids": c["card_ids"],
                    }
                },
            )
            for c in syn["clusters"]
        ]
        + [("synthesis.set_aside", {"aside": a}) for a in _set_aside(syn)],
        "overview": [("synthesis.overview", {"text": syn["overview"]})]
        if syn.get("overview")
        else [],
        "tension": [
            ("synthesis.tension", {"tension": _tension(t)}) for t in syn.get("tensions") or []
        ],
        "gaps": [
            (
                "synthesis.gap",
                {
                    "gap": {
                        "id": g["id"],
                        "text": g["text"],
                        "from": g["card_ids"],
                        "sub_question_ids": g["sub_question_ids"],
                        "why_prioritized": g.get("why_prioritized"),
                    }
                },
            )
            for g in syn["gaps"]
        ],
    }
    ranked = sorted(enumerate(syn["gaps"]), key=lambda t: (t[1].get("priority") or 10**6, t[0]))
    out["rank"] = [
        (
            "synthesis.ranked",
            {
                "ranking": [
                    {"gap_id": g["id"], "priority": i + 1, "text": g["text"]}
                    for i, (_, g) in enumerate(ranked)
                ]
            },
        )
    ]
    return out


#: Perspective roles of each prompt domain (prompts/hypothesis_roles.yaml) and the Platform agent
#: that speaks for them: the one who proposes, the one who asks how it is measured, the one who
#: looks for how it fails.
_ML_ROLES = {"innovator": "theorist", "pragmatist": "methodologist", "contrarian": "skeptic"}
_ROLE_ACTORS = {
    **_ML_ROLES,
    # hep
    "theorist": "theorist",
    "phenomenologist": "methodologist",
    "experimentalist": "skeptic",
    # biology (experimentalist as in hep)
    "model_builder": "theorist",
    "fba_analyst": "methodologist",
}
#: Letter of a perspective's own hypotheses in the debate: T1 is the Theorist's first.
_ACTOR_LETTERS = {"theorist": "T", "methodologist": "M", "skeptic": "S"}
# Only the ML role names are engine jargon; "experimentalist" or "theorist" in a sentence is a
# plain word and stays as written.
_ROLE_WORDS = re.compile(r"\b(" + "|".join(_ML_ROLES) + r")\b", re.IGNORECASE)
#: Thread of the judge's ranking, which is about the positions, not one hypothesis.
_VERDICT = "Verdict"
_ZONES: dict[str, list[int | None]] = {"> 0": [None, 0], "< 0": [0, None], "≠ 0": [0, 0]}


#: Order of a round's documents: the critiques come before the authors' answers to them. A
#: document of an older run holds both in one file and keeps the answer's place.
_PHASES = {"critique": 1, "answer": 2, "review": 3}


def _turns(art: ArtifactStore) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[int, int, str, dict[str, Any]]] = []
    for name in art.list_files(8):
        if not (name.startswith("perspectives/") and name.endswith(".json")):
            continue
        if name.endswith("debate_record.json"):
            continue
        doc = art.read_json(8, name)
        if "role" not in doc or not ("hypotheses" in doc or "responses" in doc or "reviews" in doc):
            continue
        phase = _PHASES.get(str(doc.get("phase")), 2)
        rows.append((int(doc.get("round", 0)), phase, str(doc["role"]), doc))
    rows.sort(key=lambda r: r[:3])
    turns = [t for *_, doc in rows for t in _turns_of(doc)]
    if art.exists(8, "perspectives/debate_record.json"):
        turns += _judge_turn(art.read_json(8, "perspectives/debate_record.json"))
    return turns


def _role_name(role: str) -> str:
    """The name a perspective has on the Platform: the innovator is the Theorist."""
    return _ROLE_ACTORS.get(role.lower(), role).title()


def _named(text: str) -> str:
    """Model text speaks of the engine's perspective names; the Platform shows its agents' names."""
    return _ROLE_WORDS.sub(lambda m: _role_name(m.group(1)), text)


def _idea(role: str, number: int) -> str:
    """Thread of one perspective's hypothesis, such as M2."""
    return f"{_ACTOR_LETTERS.get(_ROLE_ACTORS.get(role, ''), 'T')}{number}"


def _turn(
    turn_id: str,
    role: str,
    rnd: int,
    stance: str,
    text: str,
    *,
    about: str,
    reply_to: str | None = None,
    actor: str | None = None,
    phase: str | None = None,
    answers: list[str] | None = None,
    note: str | None = None,
    objection: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "id": turn_id,
        "actor": actor or _ROLE_ACTORS.get(role, "theorist"),
        "stance": stance,
        "text": _named(text),
        "about": about,
        "role": role,
        "round": rnd,
    }
    if reply_to:
        payload["reply_to"] = reply_to
    if phase:
        payload["phase"] = phase
    if answers:  # the challenges this turn answers
        payload["answers"] = answers
    if note:  # why the author rewrote the claim
        payload["note"] = _named(note)
    if objection and objection.get("severity"):  # how serious a challenge is
        payload["severity"] = objection["severity"]
        for key in ("flaw", "field", "card_id"):
            if objection.get(key):
                payload[key] = objection[key]
    return ("debate.turn", {"turn": payload})


def _challenge_id(rnd: int, answer: dict[str, Any]) -> str:
    """The turn of the challenge an answer is about: its critic's ``response``-th critique."""
    return f"{answer['from']}-r{rnd}-{int(answer['response'])}"


def _answer_id(rnd: int, objection: dict[str, Any]) -> str | None:
    """The author's turn that answered a challenge: its rewrite, or its defence or withdrawal."""
    answer = objection.get("answer") or {}
    if objection.get("response") is None or not answer:
        return None
    if answer.get("action") == "revise":
        return f"{objection['to']}-r{rnd}-h{int(objection['hypothesis'])}"
    return f"{objection['critic']}-r{rnd}-{int(objection['response'])}-answer"


def _review_turns(doc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """A critic's review: each answered challenge resolved (a concession) or standing (a
    challenge again, at the severity it keeps); then its critique of the added hypotheses."""
    rnd, critic = int(doc.get("round", 0)), str(doc["role"])
    turns = []
    for r in doc.get("reviews") or []:
        verdict = (r.get("review") or {}).get("verdict")
        text = (r.get("review") or {}).get("text") or ""
        cid = f"{critic}-r{rnd}-{int(r['response'])}" if r.get("response") is not None else None
        stands = verdict == "stands"
        turns.append(
            _turn(
                f"{critic}-r{rnd}-v{int(r['item'])}",
                critic,
                rnd,
                "challenge" if stands else "concede",
                text or ("The objection stands." if stands else "The answer resolves it."),
                about=_idea(str(r["to"]), int(r["hypothesis"])),
                reply_to=_answer_id(rnd, r) or cid,
                phase="review",
                answers=[cid] if cid else None,
                objection=r if stands else None,
            )  # fmt: skip
        )
    for a in doc.get("added") or []:
        turns.append(
            _turn(
                f"{critic}-r{rnd}-x{int(a['item'])}",
                critic,
                rnd,
                str(a["stance"]),
                str(a.get("text") or ""),
                about=_idea(str(a["to"]), int(a["hypothesis"])),
                reply_to=f"{a['to']}-r{rnd}-h{int(a['hypothesis'])}",
                phase="review",
                objection=a if a["stance"] == "challenge" else None,
            )  # fmt: skip
        )
    return turns


def _turns_of(doc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """One perspective's turns. Each hypothesis is a thread: a proposal starts it, the others'
    challenges and concessions reply to it, and its author answers each challenge by rewriting
    the claim, defending it or withdrawing it."""
    rnd, role = int(doc.get("round", 0)), str(doc["role"])
    hyps = doc.get("hypotheses") or []
    phase = doc.get("phase")

    def own(i: int, turn_id: str, stance: str, **extra: Any) -> tuple[str, dict[str, Any]]:
        statement = str(hyps[i - 1].get("statement", ""))
        return _turn(
            turn_id, role, rnd, stance, statement, about=_idea(role, i), phase=phase, **extra
        )

    if phase == "review":
        return _review_turns(doc)
    if rnd == 0:
        return [own(i, f"{role}-h{i}", "propose") for i in range(1, len(hyps) + 1)]
    turns = [
        _turn(
            f"{role}-r{rnd}-{k}", role, rnd, str(r["stance"]), str(r["text"]),
            about=_idea(str(r["to"]), int(r["hypothesis"])),
            reply_to=f"{r['to']}-h{int(r['hypothesis'])}",
            phase=phase,
            objection=r,
        )
        for k, r in enumerate(doc.get("responses") or [], 1)
    ]  # fmt: skip
    numbers = range(1, len(hyps) + 1)
    answers = [a for a in doc.get("answers") or [] if isinstance(a, dict)]
    revised = set(doc.get("revised") or [])
    withdrawn = set(doc.get("withdrawn") or [])
    for i in numbers:
        mine = [a for a in answers if a.get("hypothesis") == i]
        for a in mine:
            if a["action"] in ("defend", "withdraw"):
                cid = _challenge_id(rnd, a)
                stance = "defend" if a["action"] == "defend" else "concede"
                turns.append(
                    _turn(
                        f"{cid}-answer",
                        role,
                        rnd,
                        stance,
                        str(a["text"]),
                        about=_idea(role, i),
                        reply_to=cid,
                        phase=phase,
                        answers=[cid],
                    )  # fmt: skip
                )
        if i in withdrawn and not any(a["action"] == "withdraw" for a in mine):
            turns.append(
                _turn(
                    f"{role}-r{rnd}-w{i}",
                    role,
                    rnd,
                    "concede",
                    "Withdrew this hypothesis.",
                    about=_idea(role, i),
                    phase=phase,
                )  # fmt: skip
            )
        if i in revised:
            rewrites = [a for a in mine if a["action"] == "revise"]
            ids = [_challenge_id(rnd, a) for a in rewrites]
            turns.append(
                own(
                    i,
                    f"{role}-r{rnd}-h{i}",
                    "refine",
                    reply_to=ids[0] if ids else None,
                    answers=ids,
                    note=" ".join(str(a["text"]) for a in rewrites) or None,
                )  # fmt: skip
            )
    turns += [
        own(i, f"{role}-r{rnd}-h{i}", "propose") for i in doc.get("added") or [] if i in numbers
    ]
    return turns


def _judge_turn(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The independent judge's ranking, said by the PI (who picks the set)."""
    rankings = [r for r in record.get("rankings") or [] if isinstance(r, dict)]
    if not rankings:
        return []
    lines = []
    for i, r in enumerate(rankings, 1):
        name = _role_name(str(r.get("role", "")))
        score = f" ({r['score']}/10)" if r.get("score") is not None else ""
        lines.append(f"{i}. {name}{score}: {r.get('reason', '')}".rstrip(": "))
    rounds = int(record.get("rounds", 0))
    text = "\n".join(lines)
    return [_turn("judge", "judge", rounds, "test", text, about=_VERDICT, actor="pi")]


def _hypothesis_payload(h: dict[str, Any]) -> dict[str, Any]:
    sq = h.get("sub_question_ids") or []
    return {
        "id": h["id"],
        "statement": h["statement"],
        "short": _short(h["statement"]),
        "prediction": h["prediction"],
        "outcome": h["outcome"],
        "exposure": h.get("exposure") or "",
        "estimand": h.get("estimand") or "",
        "method": h.get("method") or "",
        "sub_question": sq[0] if sq else "",
        "gap": h["gap_id"],
        "novelty": h["novelty"],
        "rationale": h["rationale"],
        "falsify": {
            "text": h["falsification_criteria"],
            "zone": _ZONES.get(h["prediction"], [None, None]),
            "unit": h["outcome"],
        },
        "evidence_refs": h["evidence_refs"],
        "conditions": h.get("conditions"),
        "limitations": h.get("limitations"),
        "risk": h.get("risk"),
        "tension_ids": h.get("tension_ids") or [],
        # The debate candidates it is built from, as their threads (T1, M2).
        "from": [_candidate_thread(c) for c in h.get("from") or []],
        "contested": [_objection_payload(o) for o in h.get("contested") or []],
        "caveats": [_objection_payload(o) for o in h.get("caveats") or []],
        **({"kept_by_reviewer": h["kept_by_reviewer"]} if h.get("kept_by_reviewer") else {}),
    }


def _candidate_thread(candidate: str) -> str:
    """``innovator-2`` (a debate candidate) as its thread on the Platform, ``T2``."""
    role, _, number = candidate.rpartition("-")
    return _idea(role, int(number)) if number.isdigit() else candidate


def _objection_payload(o: dict[str, Any]) -> dict[str, Any]:
    return {
        "by": _ROLE_ACTORS.get(str(o.get("from")), "skeptic"),
        "about": _candidate_thread(str(o.get("candidate", ""))),
        "severity": o.get("severity"),
        "flaw": o.get("flaw"),
        "field": o.get("field"),
        "card_id": o.get("card_id"),
        "text": _named(str(o.get("text") or "")),
    }


def held_back_ids(art: ArtifactStore) -> dict[str, str]:
    """Thread id (T2) of each held-back candidate of the current set -> its engine id."""
    if not art.exists(8, "hypotheses.json"):
        return {}
    doc = art.read_json(8, "hypotheses.json")
    return {
        _candidate_thread(str(b["candidate"])): str(b["candidate"])
        for b in doc.get("held_back") or []
    }


def _held_back_events(doc: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Each held-back candidate as an idea set aside, with the objection that keeps it out."""
    out = []
    for b in doc.get("held_back") or []:
        first = (b.get("objections") or [{}])[0]
        who = _role_name(str(first.get("from", "")))
        flaw = str(first.get("flaw") or "fatal").replace("_", " ")
        out.append(
            (
                "idea.set_aside",
                {
                    "idea_id": _candidate_thread(str(b["candidate"])),
                    "statement": _named(str(b["hypothesis"].get("statement", ""))),
                    "reason": f"Held back: the {who}'s objection stands ({flaw}). "
                    + _named(str(first.get("text") or "")),
                },
            )
        )
    return out


def _content_stage8(art: ArtifactStore, flags: Flags) -> StepEvents:
    doc = art.read_json(8, "hypotheses.json")
    hypotheses = doc["hypotheses"]
    out: StepEvents = {
        "debate": _turns(art),
        "write": [
            ("hypothesis.drafted", {"hypothesis": _hypothesis_payload(h)}) for h in hypotheses
        ]
        + _held_back_events(doc)
        + [
            (
                "rule.checked",
                {
                    "rule": 5,
                    "state": "pass",
                    "detail": "every hypothesis states its falsification criteria",
                },
            )
        ],
        "select": [("hypothesis.selected", {"hypothesis_id": h["id"]}) for h in hypotheses],
    }
    if flags.novelty_check and art.exists(8, "novelty_report.json"):
        report = art.read_json(8, "novelty_report.json")
        threshold = float(report.get("similarity_threshold", 0.25))
        known = (
            {str(r["paper_id"]): r for r in art.read_jsonl(4, "candidates.jsonl")}
            if art.exists(4, "candidates.jsonl")
            else {}
        )
        checks = []
        for row in report.get("per_hypothesis", []):
            closest = row.get("closest_paper") or {}
            similarity = float(closest.get("similarity", 0.0))
            # "Author et al., year" when the paper is one the run collected; its title otherwise.
            found = known.get(str(closest.get("paper_id")))
            checks.append(
                (
                    "hypothesis.checked",
                    {
                        "hypothesis_id": row["hypothesis_id"],
                        "novelty": {
                            "novel": similarity < threshold,
                            "closest": _citation(found) if found else closest.get("title", ""),
                            "similarity": similarity,
                        },
                        "assessment": "heuristic, not proof of novelty",
                    },
                )
            )
        out["check"] = checks
    return out


#: entity type and relation -> the map step that draws it
_MAP_STEP = {
    "question": "questions",
    "decomposes_into": "questions",
    "evidence": "foundation",
    "claim": "foundation",
    "gap": "foundation",
    "supports": "foundation",
    "contradicts": "foundation",
    "motivates": "foundation",
    "hypothesis": "reasoning",
    "assumption": "reasoning",
    "proposes_answer_to": "reasoning",
    "provides_rationale_for": "reasoning",
    "addresses": "reasoning",
    "depends_on": "reasoning",
    "contribution": "contribution",
    "informs": "contribution",
    "targets": "contribution",
}


def _content_stage9(art: ArtifactStore) -> StepEvents:
    """Each step draws its entities, then the relations they complete."""
    graph = art.read_json(9, "semantic_graph.json")
    out: StepEvents = {s: [] for s in _STAGE_STEPS[9]}
    for node in graph["entities"]:
        out[_MAP_STEP[node["type"]]].append(("map.node", {"node": node}))
    for edge in graph["relations"]:
        out[_MAP_STEP[edge["relation"]]].append(("map.edge", {"edge": edge}))
    pieces = art.read_json(9, "research_canvas.json")["pieces"]
    out["canvas"] = [("canvas.piece", {"piece": p}) for p in pieces]
    return out


def stage_content(
    art: ArtifactStore, stage: int, flags: Flags, domains: list[str], delay_ms: int = 0
) -> StepEvents:
    """Events of every step of ``stage``, built from the artifacts on disk."""
    if stage == 1:
        return _content_stage1(art, flags, domains)
    if stage == 2:
        return _content_stage2(art)
    if stage == 3:
        return _content_stage3(art, delay_ms)
    if stage == 4:
        return _content_stage4(art)
    if stage == 5:
        return _content_stage5(art)
    if stage == 6:
        return _content_stage6(art)
    if stage == 7:
        return _content_stage7(art)
    if stage == 9:
        return _content_stage9(art)
    return _content_stage8(art, flags)


def stage_summary(art: ArtifactStore, group: Group) -> str:
    last = group.last
    if last == 2:
        tree = art.read_json(2, "problem_tree.json")
        evaluation = art.read_json(2, "topic_evaluation.json")
        return (
            f"Goal set • {len(tree['sub_questions'])} sub-questions, ranked • "
            f"topic scored {evaluation['overall']} of 10"
        )
    if last == 4:
        meta = art.read_json(4, "search_meta.json")
        return (
            f"{meta['raw']} hits • {meta['unique']} unique papers • "
            f"{meta['duplicates']} duplicates merged"
        )
    if last == 5:
        summary = art.read_json(5, "review.json")["summary"]
        return (
            f"{summary['candidates']} candidates • {summary['kept']} kept • "
            f"{summary['candidates'] - summary['kept']} excluded"
        )
    if last == 6:
        return f"{len(_cards(art))} knowledge cards"
    if last == 7:
        syn = art.read_json(7, "synthesis.json")
        return (
            f"{_plural(len(syn['clusters']), 'group')} • "
            f"{_plural(len(syn.get('tensions') or []), 'tension')} • "
            f"{_plural(len(syn['gaps']), 'gap')}"
        )
    if last == 9:
        graph = art.read_json(9, "semantic_graph.json")
        pieces = art.read_json(9, "research_canvas.json")["pieces"]
        filled = sum(1 for p in pieces if p["status"] == "filled")
        return (
            f"{_plural(len(graph['entities']), 'entity', 'entities')} • "
            f"{_plural(len(graph['relations']), 'relation')} • "
            f"{filled} of {len(pieces)} canvas pieces filled; findings wait for the experiment"
        )
    count = len(art.read_json(8, "hypotheses.json")["hypotheses"])
    return f"{count} hypotheses to test, each with a way to be wrong"


def _cards(art: ArtifactStore) -> list[str]:
    return [n for n in art.list_files(6) if n.startswith("cards/") and n.endswith(".json")]


def _plural(count: int, noun: str, plural: str | None = None) -> str:
    return f"{count} {noun if count == 1 else plural or noun + 's'}"


def _source_names(art: ArtifactStore) -> list[str]:
    if not art.exists(3, "sources.json"):
        return []
    return [str(s["name"]) for s in art.read_json(3, "sources.json")["sources"]]


def _join(names: list[str]) -> str:
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def step_note(art: ArtifactStore, step_id: str) -> str | None:
    """What a finished step produced, in one line built from its artifacts (or None)."""
    if step_id == "goal":
        title = art.read_json(1, "goal.json").get("working_title")
        return f"Goal set: {title}." if title else None
    if step_id == "decompose":
        tree = art.read_json(2, "problem_tree.json")
        risks = tree.get("risks") or []
        return (
            f"{_plural(len(tree['sub_questions']), 'sub-question')}, ranked by priority, "
            f"with {_plural(len(risks), 'risk')} noted."
        )
    if step_id == "evaluate":
        e = art.read_json(2, "topic_evaluation.json")
        bar = f" against a bar of {e['threshold']}" if e.get("threshold") is not None else ""
        return f"Topic scored {e['overall']} of 10{bar}."
    if step_id == "strategy":
        plan = _read_yaml(art, 3, "search_plan.yaml")
        queries = art.read_json(3, "queries.json")["queries"]
        return (
            f"{_plural(len(plan['search_strategies']), 'search angle')} with "
            f"{_plural(len(queries), 'query', 'queries')} for {_join(_source_names(art))}."
        )
    if step_id == "collect":
        meta = art.read_json(4, "search_meta.json")
        failed = len(meta.get("errors") or [])
        tail = f" {_plural(failed, 'request')} failed." if failed else ""
        return (
            f"{meta['raw']} hits; {meta['unique']} unique papers after merging "
            f"{_plural(meta['duplicates'], 'duplicate')}.{tail}"
        )
    if step_id in ("score", "reject", "shortlist"):
        return _screen_note(art, step_id)
    if step_id == "extract":
        meta = art.read_json(6, "knowledge_meta.json")
        skipped = meta.get("skipped") or []
        no_abstract = sum(1 for s in skipped if s.get("reason") == "no abstract available")
        unquoted = len(skipped) - no_abstract
        tail = f"; {_plural(no_abstract, 'paper')} had no abstract" if no_abstract else ""
        if unquoted:
            tail += (
                f"; {_plural(unquoted, 'paper')} gave no card the abstract's own words could back"
            )
        quoted = " quoted from the abstracts" if meta.get("quoted") else " from abstracts"
        return f"{_plural(meta['cards'], 'knowledge card')}{quoted}{tail}."
    if step_id == "cluster":
        syn = art.read_json(7, "synthesis.json")
        aside = sum(len(a["card_ids"]) for a in _set_aside(syn))
        tail = f"; {_plural(aside, 'card')} set aside, each with a reason" if aside else ""
        return f"{_plural(len(syn['clusters']), 'school')} of thought{tail}."
    if step_id == "gaps":
        syn = art.read_json(7, "synthesis.json")
        return f"{_plural(len(syn['gaps']), 'research gap')}, each traced to its cards."
    if step_id == "debate":
        turns = [t["turn"] for _, t in _turns(art)]
        proposals = [t for t in turns if t["round"] == 0]
        roles = {t["role"] for t in proposals}
        rounds = {t["round"] for t in turns if t["round"] > 0 and t["role"] != "judge"}
        challenged = sum(
            1 for t in turns if t["stance"] == "challenge" and t.get("phase") != "review"
        )
        answered = len(
            {c for t in turns if t.get("phase") == "answer" for c in t.get("answers") or []}
        )
        reviews = [t for t in turns if t.get("phase") == "review" and t.get("answers")]
        resolved = sum(1 for t in reviews if t["stance"] == "concede")
        tally = f" ({challenged} challenged, {answered} answered" if challenged else ""
        if reviews:
            tally += f", {resolved} resolved on review, {len(reviews) - resolved} still standing"
        tally += ")" if challenged else ""
        debated = (
            f", then answered each other in {_plural(len(rounds), 'round')}{tally}"
            if rounds
            else ""
        )
        return (
            f"{_plural(len(roles), 'perspective')} proposed "
            f"{_plural(len(proposals), 'hypothesis', 'hypotheses')}{debated}."
        )
    if step_id == "write":
        doc = art.read_json(8, "hypotheses.json")
        hyps = doc["hypotheses"]
        settled = sorted({t for h in hyps for t in h.get("tension_ids") or []})
        still = doc.get("open_tensions") or []
        tail = ""
        contested = [h["id"] for h in hyps if h.get("contested")]
        if contested:
            tail += f"; contested: {', '.join(contested)}"
        if doc.get("held_back"):
            tail += (
                f"; {_plural(len(doc['held_back']), 'candidate')} held back by a fatal objection"
            )
        if settled:
            names = ", ".join(settled)
            tail += f"; {_plural(len(settled), 'tension')} of the literature settled ({names})"
        if still:
            tail += f"; left open: {', '.join(still)}"
        return (
            f"{_plural(len(hyps), 'hypothesis', 'hypotheses')}, each with a result that would "
            f"prove it wrong{tail}."
        )
    if step_id in ("questions", "foundation", "reasoning", "contribution"):
        return _map_note(art, step_id)
    if step_id == "check" and art.exists(8, "novelty_report.json"):
        report = art.read_json(8, "novelty_report.json")
        threshold = float(report.get("similarity_threshold", 0.25))
        rows = report.get("per_hypothesis", [])
        similar = [float((r.get("closest_paper") or {}).get("similarity", 0)) for r in rows]
        novel = sum(1 for value in similar if value < threshold)
        return (
            f"{novel} of {len(rows)} have no close match among the papers found "
            "(word overlap, a heuristic, not proof of novelty)."
        )
    return None


def _map_note(art: ArtifactStore, step_id: str) -> str | None:
    graph = art.read_json(9, "semantic_graph.json")
    n = {
        t: sum(1 for e in graph["entities"] if e["type"] == t)
        for t in ("question", "evidence", "claim", "gap", "hypothesis", "contribution")
    }
    if step_id == "questions":
        return f"Your question and its {_plural(n['question'] - 1, 'sub-question')}."
    if step_id == "foundation":
        return (
            f"{_plural(n['evidence'], 'finding')} behind {_plural(n['claim'], 'claim')}, and "
            f"{_plural(n['gap'], 'gap')} still open."
        )
    if step_id == "reasoning":
        return (
            f"{_plural(n['hypothesis'], 'hypothesis', 'hypotheses')}, each tied to its question, "
            "gap and claims."
        )
    return f"{_plural(n['contribution'], 'expected contribution')}, none shown yet."


def _screen_note(art: ArtifactStore, step_id: str) -> str | None:
    review = art.read_json(5, "review.json")
    summary = review["summary"]
    if step_id == "score":
        scored = [d for d in review["decisions"] if d.get("relevance_score") is not None]
        passed = ("kept", "below_cutoff", "dropped_by_reviewer")
        clear = sum(1 for d in scored if d["decision"] in passed)
        return f"Scored {_plural(len(scored), 'paper')}; {clear} clear both bars."
    if step_id == "reject":
        aside = summary.get("prefiltered", 0) + summary.get("unscored", 0)
        tail = f"; {aside} set aside unscored" if aside else ""
        return f"{_plural(summary.get('rejected', 0), 'paper')} rejected with a reason{tail}."
    cut = summary.get("below_cutoff", 0)
    tail = f"; {cut} more cleared both bars but ranked lower" if cut else ""
    return f"Kept {summary['kept']} of {summary['candidates']} papers{tail}."


# ---------------------------------------------------------------------------
# Gate payloads
# ---------------------------------------------------------------------------


def _gate_position(mode: str, kind: str) -> tuple[int, int]:
    specs = gates.gates_for(mode)
    index = next(i for i, s in enumerate(specs, 1) if s.kind == kind)
    return index, len(specs)


def gate_spec(mode: str, data: dict[str, Any], art: ArtifactStore) -> dict[str, Any]:
    """Gate description for the run studio, built from the gate data of the core event."""
    kind = data["kind"]
    index, total = _gate_position(mode, kind)
    base: dict[str, Any] = {
        "id": data["gate_id"],
        "gate_id": data["gate_id"],
        "kind": kind,
        "stop_index": index,
        "stop_total": total,
    }
    if kind == gates.SCREEN:
        shortlist = data.get("shortlist", [])
        summary = data.get("summary", {})
        meta = art.read_json(4, "search_meta.json") if art.exists(4, "search_meta.json") else {}
        base.update(
            {
                "title": "Approve the shortlist",
                "why": (
                    f"In {mode.title()} mode you check the reading list, because every gap "
                    "and hypothesis is built on it."
                ),
                "summary": [
                    f"{len(shortlist)} of {meta.get('unique', summary.get('candidates', '?'))} "
                    "papers kept, each with a reason",
                    *(
                        [
                            f"{summary['below_cutoff']} more cleared both bars but ranked below "
                            "the best-scored papers kept"
                        ]
                        if summary.get("below_cutoff")
                        else []
                    ),
                    f"{summary.get('rejected', 0)} papers rejected, "
                    f"{summary.get('unscored', 0) + summary.get('prefiltered', 0)} "
                    "excluded unscored",
                ],
                "options": [
                    {
                        "id": "approve",
                        "label": f"Read all {len(shortlist)}",
                        "description": "Approve the shortlist as screened.",
                        "leads_to": "Each paper becomes a knowledge card",
                        "confirm_label": "Approve the shortlist",
                        "recommended": True,
                    },
                    {
                        "id": "drop",
                        "label": "Remove some first",
                        "description": "Click papers in the shortlist to leave them out.",
                        "leads_to": "Removed papers are not read or cited",
                        "confirm_label": "Choose what to remove",
                    },
                    {
                        "id": "reject",
                        "label": "Search again",
                        "description": "Reject the shortlist and redo the search strategy.",
                        "leads_to": "Stages 3 to 8 run again with your note as feedback",
                        "confirm_label": "Reject and search again",
                    },
                ],
                "droppable": [r["paper_id"] for r in shortlist],
                "shortlist": shortlist,
            }
        )
        return base
    if kind == gates.HYPOTHESES:
        return {**base, **_hypotheses_gate(mode, data)}
    goal = data.get("goal", {})
    questions = data.get("sub_questions", [])
    evaluation = data.get("topic_evaluation", {})
    base.update(
        {
            "title": "Approve the scope",
            "why": (
                f"In {mode.title()} mode you check the goal and sub-questions before any "
                "search starts."
            ),
            "summary": [
                f"Goal: {goal.get('working_title')}",
                f"{len(questions)} sub-questions",
                f"Topic scored {evaluation.get('overall')} of 10",
            ],
            "options": [
                {
                    "id": "approve",
                    "label": "Approve the scope",
                    "description": "The goal and sub-questions are good to search with.",
                    "leads_to": "Search strategy and literature collection",
                    "confirm_label": "Approve the scope",
                    "recommended": True,
                },
                {
                    "id": "reject",
                    "label": "Scope again",
                    "description": "Reject the scope; stages 1 and 2 run again with your note.",
                    "leads_to": "A new goal and sub-question tree",
                    "confirm_label": "Reject and rescope",
                },
            ],
            "droppable": [],
            "goal": goal,
            "sub_questions": questions,
            "topic_evaluation": evaluation,
        }
    )
    return base


def _hypotheses_gate(mode: str, data: dict[str, Any]) -> dict[str, Any]:
    hypotheses = data.get("hypotheses", [])
    held = data.get("held_back", [])
    contested = [h["id"] for h in hypotheses if h.get("contested")]
    summary = [f"{_plural(len(hypotheses), 'hypothesis', 'hypotheses')} in the set, each built "
               "from debate candidates that survived the challenges"]  # fmt: skip
    if contested:
        summary.append(
            f"{', '.join(contested)} contested: a fatal objection to what "
            f"{'it is' if len(contested) == 1 else 'they are'} built from still stands"
        )
    if held:
        summary.append(
            f"{_plural(len(held), 'candidate')} held back by a fatal objection; you can keep one "
            "anyway, and the objection stays on record"
        )
    if data.get("open_tensions"):
        summary.append(f"Tensions left open: {', '.join(data['open_tensions'])}")
    return {
        "title": "Approve the hypotheses",
        "why": (
            f"In {mode.title()} mode you decide what goes forward: the debate can flag a claim "
            "wrongly, and every hypothesis is mapped and tested from here."
        ),
        "summary": summary,
        "options": [
            {
                "id": "approve",
                "label": f"Keep all {len(hypotheses)}",
                "description": "Approve the set as the debate left it.",
                "leads_to": "The argument map is drawn from these hypotheses",
                "confirm_label": "Approve the hypotheses",
                "recommended": True,
            },
            {
                "id": "drop",
                "label": "Change the set first" if held else "Remove some first",
                "description": (
                    "Click hypotheses to leave them out, or held-back ones to keep them."
                    if held
                    else "Click hypotheses to leave them out."
                ),
                "leads_to": "Removed ones are not mapped; kept ones stay marked contested",
                "confirm_label": "Choose what to change",
            },
            {
                "id": "reject",
                "label": "Write a new set",
                "description": "Reject the set; the perspectives debate again with your note.",
                "leads_to": "Stage 8 runs again with your note as feedback",
                "confirm_label": "Reject and hypothesize again",
            },
        ],
        "droppable": [h["id"] for h in hypotheses],
        "keepable": [_candidate_thread(str(b["candidate"])) for b in held],
        "hypotheses": hypotheses,
        "held_back": [{**b, "thread": _candidate_thread(str(b["candidate"]))} for b in held],
    }


def option_for(answer: dict[str, Any]) -> str:
    if answer.get("decision") == gates.REJECT:
        return "reject"
    return "drop" if answer.get("dropped") or answer.get("kept") else "approve"


# ---------------------------------------------------------------------------
# Core event -> Platform envelopes
# ---------------------------------------------------------------------------


@dataclass
class RunView:
    """Read-only view of a run used while mapping its events."""

    run_id: str
    record: dict[str, Any]
    flags: Flags
    artifacts: ArtifactStore
    store: RunStore | None = None
    #: Pause between literature queries, as configured for this run.
    query_delay_ms: int = 0

    @property
    def domains(self) -> list[str]:
        return list(self.record.get("domains") or [])

    @property
    def usage(self) -> dict[str, Any]:
        return self.record.get("usage") or {}


def load_view(store: RunStore, run_id: str) -> RunView:
    record = store.read_run(run_id)
    try:
        snapshot = store.read_snapshot(run_id, "config")
    except (OSError, ValueError):
        snapshot = {}
    research = snapshot.get("research", {})
    delay = float((snapshot.get("literature") or {}).get("inter_query_delay_sec", 0) or 0)
    flags = Flags(
        mode=record["review_mode"],
        hardware_advisory=bool(research.get("hardware_advisory", False)),
        novelty_check=bool(research.get("novelty_check", True)),
    )
    return RunView(
        run_id, record, flags, store.artifacts(run_id), store, query_delay_ms=round(delay * 1000)
    )


def _stage_events(
    group: Group,
    stage: int,
    view: RunView,
    content: StepEvents,
    *,
    started: str,
    done: frozenset[str] | set[str] = frozenset(),
    stage_run: int = 0,
    try_index: int = 0,
) -> list[Envelope]:
    """Step events of one completed core stage.

    ``started`` is the step already started (by ``stage.started`` or by progress); steps in
    ``done`` were already completed by progress and are skipped.
    """
    out: list[Envelope] = []
    for step_id in view.flags.steps(stage):
        if step_id in done:
            continue
        actor = _STEPS[step_id][1]
        if step_id != started:
            out.append(_env("step.started", {"step_id": step_id}, stage_key=group.key, actor=actor))
        out += [_env(t, p, stage_key=group.key, actor=actor) for t, p in content.get(step_id, [])]
        note = _note(view.artifacts, step_id)
        if note:
            message_id = _message_id(stage_run, try_index, step_id)
            out.append(_message(message_id, note, stage_key=group.key, actor=actor, done=True))
        out.append(_env("step.completed", {"step_id": step_id}, stage_key=group.key, actor=actor))
    return out


def _note(art: ArtifactStore, step_id: str) -> str | None:
    try:
        return step_note(art, step_id)
    except (KeyError, ValueError, OSError, TypeError):  # a note never stops the stream
        logger.exception("could not build the note of step %s", step_id)
        return None


def _doing(view: RunView, step_id: str) -> str | None:
    """The live line of a step that has just started."""
    if step_id == "cluster":
        cards = len(_cards(view.artifacts))
        return f"Grouping {_plural(cards, 'knowledge card')} into schools of thought."
    return _DOING.get(step_id)


def _on_stage_started(view: RunView, event: Event) -> list[Envelope]:
    stage = int(event.stage or 0)
    group = _BY_STAGE[stage]
    out: list[Envelope] = []
    if stage == group.stages[0]:
        out.append(
            _env(
                "stage.started",
                {"plan": stage_plan(group, view.flags)},
                stage_key=group.key,
                actor=group.cast[0],
            )
        )
    first = view.flags.steps(stage)[0]
    actor = _STEPS[first][1]
    out.append(_env("step.started", {"step_id": first}, stage_key=group.key, actor=actor))
    doing = _doing(view, first)
    if doing:
        message_id = _message_id(event.seq, 0, first)
        out.append(_message(message_id, doing, stage_key=group.key, actor=actor, done=False))
    return out


def _on_stage_completed(view: RunView, event: Event) -> list[Envelope]:
    stage = int(event.stage or 0)
    group = _BY_STAGE[stage]
    streamed = _streamed(view, stage, event.seq)
    content = stage_content(
        view.artifacts, stage, view.flags, view.domains, delay_ms=view.query_delay_ms
    )
    started, done = _trim(stage, content, streamed, view.flags.steps(stage))
    out = _stage_events(
        group, stage, view, content, started=started, done=done,
        stage_run=streamed.stage_run, try_index=streamed.try_index,
    )  # fmt: skip
    if stage == group.last and view.flags.gate_step(group) is None:
        out.append(_stage_done(view, group, event))
    return out


# ---------------------------------------------------------------------------
# Stage progress: parts of a result sent while the stage runs
# ---------------------------------------------------------------------------

#: progress kind -> the step it belongs to
_PROGRESS_STEP = {
    "queries": "collect",
    "query": "collect",
    "screen_plan": "score",
    "screen_batch": "score",
    "cards_plan": "extract",
    "card": "extract",
    "perspectives_plan": "debate",
    "perspective": "debate",
    "critique": "debate",
    "review": "debate",
    "judged": "debate",
    "merge": "write",
    "novelty": "check",
}


@dataclass
class _Streamed:
    """What the current try of a stage's execution has already sent through progress."""

    stage_run: int = 0
    try_index: int = 0
    kinds: set[str] = field(default_factory=set)
    #: ``point:<paper>``, ``rejected:<paper>``, ``card:<card>`` and ``turn:<role>-r<round>``
    ids: set[str] = field(default_factory=set)


def _streamed(view: RunView, stage: int, until_seq: int) -> _Streamed:
    out = _Streamed()
    if view.store is None:
        return out
    for e in view.store.read_events(view.run_id):
        if e.seq >= until_seq:
            break
        if e.stage != stage:
            continue
        if e.type == ev.STAGE_STARTED:
            out = _Streamed(stage_run=e.seq)
        elif e.type == ev.STAGE_PROGRESS:
            kind = str(e.data.get("kind"))
            if kind == "restart":  # what the failed try sent was replaced
                out = _Streamed(stage_run=out.stage_run, try_index=int(e.data.get("try", 0)))
                continue
            out.try_index = int(e.data.get("try", out.try_index))
            out.kinds.add(kind)
            out.ids.update(_streamed_ids(e.data))
    return out


def _streamed_ids(data: dict[str, Any]) -> list[str]:
    kind = data.get("kind")
    if kind == "screen_plan":
        return [f"rejected:{d['paper_id']}" for d in data.get("prefiltered") or []]
    if kind == "screen_batch":
        ids = []
        for d in data.get("decisions") or []:
            if d.get("relevance_score") is not None and d.get("quality_score") is not None:
                ids.append(f"point:{d['paper_id']}")
            if d.get("decision") in ("rejected", "unscored"):
                ids.append(f"rejected:{d['paper_id']}")
        return ids
    if kind == "card":
        return [f"card:{data['card_id']}"]
    if kind == "perspective":  # every turn of that perspective in that round
        return [f"turns:{data['role']}-r{int(data.get('round', 0))}"]
    if kind in ("critique", "review"):  # its critiques of the others, or its review, that round
        return [f"turns:{data['role']}-r{int(data.get('round', 0))}-{kind}"]
    if kind == "judged":
        return ["turn:judge"]
    return []


def _trim(
    stage: int, content: StepEvents, streamed: _Streamed, steps: tuple[str, ...]
) -> tuple[str, set[str]]:
    """Remove from ``content`` what progress already sent; returns (started step, done steps)."""
    started, done = steps[0], set[str]()
    kinds, ids = streamed.kinds, streamed.ids
    if stage == 4 and "query" in kinds:
        sent = ("literature.request", "literature.batch")
        content["collect"] = [e for e in content.get("collect", []) if e[0] not in sent]
    elif stage == 5:
        score = content.get("score", [])
        if "screen_plan" in kinds:
            score = [e for e in score if e[0] != "screen.criteria"]
        points = [
            p
            for t, payload in score
            if t == "screen.scored"
            for p in payload["points"]
            if f"point:{p['id']}" not in ids
        ]
        score = [e for e in score if e[0] != "screen.scored"]
        chunks = range(0, len(points), 48)
        score += [("screen.scored", {"points": points[i : i + 48]}) for i in chunks]
        content["score"] = score
        content["reject"] = [
            e for e in content.get("reject", []) if f"rejected:{e[1]['paper']['id']}" not in ids
        ]
    elif stage == 6:
        content["extract"] = [
            e for e in content.get("extract", []) if f"card:{e[1]['card']['id']}" not in ids
        ]
    elif stage == 8:
        content["debate"] = [
            e for e in content.get("debate", []) if not ({_turn_key(e[1]["turn"])} & ids)
        ]
        if "merge" in kinds:
            done.add("debate")
            started = "write"
        if "novelty" in kinds:
            done.add("write")
            started = "check"
    return started, done


def _turn_key(turn: dict[str, Any]) -> str:
    """The progress id that sent a turn: its own id (the judge) or its perspective's round and,
    for a critique, the phase."""
    if turn["id"] == "judge":
        return "turn:judge"
    phase = f"-{turn['phase']}" if turn.get("phase") in ("critique", "review") else ""
    return f"turns:{turn['role']}-r{turn['round']}{phase}"


def _on_stage_progress(view: RunView, event: Event) -> list[Envelope]:
    data = event.data
    kind = str(data.get("kind"))
    stage = int(event.stage or 0)
    group = _BY_STAGE[stage]
    stage_run, try_index = int(data.get("stage_run", 0)), int(data.get("try", 0))
    if kind == "restart":
        return _on_restart(view, group, stage, stage_run, try_index)
    step = _PROGRESS_STEP.get(kind)
    if step is None:
        return []
    progress = _Progress(view, group, event, step, stage_run, try_index)
    handler = _PROGRESS_HANDLERS.get(kind)
    return handler(progress, data) if handler else []


def _on_restart(
    view: RunView, group: Group, stage: int, stage_run: int, try_index: int
) -> list[Envelope]:
    """A transient error restarted the stage: start its UI group over (never another stage's)."""
    if len(group.stages) != 1:
        return []  # keyed content (search cells) is simply sent again
    first = view.flags.steps(stage)[0]
    actor = _STEPS[first][1]
    out = [
        _env("stage.started", {"plan": stage_plan(group, view.flags)}, stage_key=group.key,
             actor=group.cast[0]),
        _env("step.started", {"step_id": first}, stage_key=group.key, actor=actor),
    ]  # fmt: skip
    doing = _doing(view, first)
    if doing:
        message_id = _message_id(stage_run, try_index, first)
        out.append(_message(message_id, doing, stage_key=group.key, actor=actor, done=False))
    return out


@dataclass
class _Progress:
    """Envelope helpers for one progress event of one step."""

    view: RunView
    group: Group
    event: Event
    step: str
    stage_run: int
    try_index: int

    @property
    def art(self) -> ArtifactStore:
        return self.view.artifacts

    def env(self, type_: str, payload: dict[str, Any], step: str | None = None) -> Envelope:
        actor = _STEPS[step or self.step][1]
        return _env(type_, payload, stage_key=self.group.key, actor=actor)

    def say(self, text: str, *, step: str | None = None, done: bool = False) -> Envelope:
        step = step or self.step
        return _message(
            _message_id(self.stage_run, self.try_index, step),
            text,
            stage_key=self.group.key,
            actor=_STEPS[step][1],
            done=done,
        )

    def close(self, step: str) -> list[Envelope]:
        """End ``step``: its note (closing its live line) and ``step.completed``."""
        out = []
        note = _note(self.art, step)
        if note:
            out.append(self.say(note, step=step, done=True))
        out.append(self.env("step.completed", {"step_id": step}, step))
        return out

    def open(self, step: str) -> list[Envelope]:
        out = [self.env("step.started", {"step_id": step}, step)]
        doing = _doing(self.view, step)
        if doing:
            out.append(self.say(doing, step=step))
        return out


def _progress_queries(p: _Progress, data: dict[str, Any]) -> list[Envelope]:
    total = int(data.get("planned", 0)) + int(data.get("expanded", 0))
    delay = p.view.query_delay_ms / 1000
    pace = f", {delay:g} s apart to respect each source" if delay else ""
    names = _join(_source_names(p.art)) or "the sources"
    return [p.say(f"Sending {_plural(total, 'query', 'queries')} to {names}{pace}.")]


def _progress_query(p: _Progress, data: dict[str, Any]) -> list[Envelope]:
    queries = {q["text"]: q["id"] for q in p.art.read_json(3, "queries.json")["queries"]}
    sources = {s["id"] for s in p.art.read_json(3, "sources.json")["sources"]}
    hits: dict[str, int] = data.get("hits") or {}
    out: list[Envelope] = []
    query_id = queries.get(data["text"])
    if query_id:  # expansion queries have no id in the plan, so no cell
        for source, count in hits.items():
            if source in sources:
                cell = {"query_id": query_id, "source_id": source}
                out.append(p.env("literature.request", cell))
                out.append(p.env("literature.batch", {**cell, "hits": count}))
    found = sum(int(v) for v in hits.values())
    out.append(
        p.say(
            f"Query {data['index']} of {data['total']}: “{_short(data['text'], 60)}”, "
            f"{_plural(found, 'hit')} ({data['raw']} so far)."
        )
    )
    return out


def _progress_screen_plan(p: _Progress, data: dict[str, Any]) -> list[Envelope]:
    criteria = {
        "rules": data.get("rules", []),
        "relevance_min": data.get("min_relevance"),
        "quality_min": data.get("min_quality"),
        "candidates": data.get("candidates"),
    }
    prefiltered = data.get("prefiltered") or []
    out = [p.env("screen.criteria", criteria)]
    out += [p.env(*_rejected(d)) for d in prefiltered]
    aside = (
        f"; {len(prefiltered)} share no keyword with the topic and are set aside"
        if prefiltered
        else ""
    )
    out.append(
        p.say(
            f"Scoring {_plural(int(data['to_screen']), 'paper')} in "
            f"{_plural(int(data['batches']), 'batch', 'batches')} against the relevance and "
            f"quality bars{aside}."
        )
    )
    return out


def _progress_screen_batch(p: _Progress, data: dict[str, Any]) -> list[Envelope]:
    rows = data.get("decisions") or []
    points = [
        {"id": d["paper_id"], "relevance": d["relevance_score"], "quality": d["quality_score"]}
        for d in rows
        if d.get("relevance_score") is not None and d.get("quality_score") is not None
    ]
    out = [p.env("screen.scored", {"points": points})] if points else []
    out += [p.env(*_rejected(d)) for d in rows if d.get("decision") in ("rejected", "unscored")]
    clear = sum(1 for d in rows if d.get("decision") == "kept")
    out.append(
        p.say(
            f"Batch {data['index']} of {data['total']} scored: {clear} of {len(rows)} clear "
            "both bars."
        )
    )
    return out


def _progress_cards_plan(p: _Progress, data: dict[str, Any]) -> list[Envelope]:
    cached = int(data.get("cached", 0))
    again = f" ({cached} already read)" if cached else ""
    total = int(data.get("total", 0))
    return [p.say(f"Reading {_plural(total, 'abstract')}, one knowledge card each{again}.")]


def _progress_card(p: _Progress, data: dict[str, Any]) -> list[Envelope]:
    card = p.art.read_partial(6, p.event.attempt, str(data.get("partial", "")))
    if not isinstance(card, dict):
        return []
    title = _short(str(card.get("title") or card.get("cite_key") or ""), 70)
    return [
        p.env(*_card_extracted(card, _citations(p.art))),
        p.say(f"Card {data['index']} of {data['total']}: {title}"),
    ]


def _progress_perspectives_plan(p: _Progress, data: dict[str, Any]) -> list[Envelope]:
    roles = [_role_name(str(r)) for r in data.get("roles") or []]
    who = f"The {_join(roles)} are" if roles else "Each perspective is"
    return [p.say(f"{who} proposing hypotheses from the gaps.")]


def _progress_critique(p: _Progress, data: dict[str, Any]) -> list[Envelope]:
    name = str(data.get("file", ""))
    if not p.art.exists(8, name):
        return []
    doc = p.art.read_json(8, name)
    stances = [r.get("stance") for r in doc.get("responses") or []]
    who = _role_name(str(doc["role"]))
    said = (
        f"challenged {_plural(stances.count('challenge'), 'hypothesis', 'hypotheses')} and "
        f"conceded {stances.count('concede')} in round {int(doc.get('round', 0))}"
    )
    return [*(p.env(*t) for t in _turns_of(doc)), p.say(f"The {who} {said}.")]


def _progress_review(p: _Progress, data: dict[str, Any]) -> list[Envelope]:
    name = str(data.get("file", ""))
    if not p.art.exists(8, name):
        return []
    doc = p.art.read_json(8, name)
    verdicts = [(r.get("review") or {}).get("verdict") for r in doc.get("reviews") or []]
    fatal = sum(
        1
        for r in doc.get("reviews") or []
        if (r.get("review") or {}).get("verdict") == "stands" and r.get("severity") == "fatal"
    )
    who = _role_name(str(doc["role"]))
    said = (
        f"reviewed {_plural(len(verdicts), 'answer')}: {verdicts.count('resolved')} resolved, "
        f"{verdicts.count('stands')} still standing ({fatal} fatal)"
    )
    if doc.get("added"):
        said += f", and examined {_plural(len(doc['added']), 'new hypothesis', 'new hypotheses')}"
    return [*(p.env(*t) for t in _turns_of(doc)), p.say(f"The {who} {said}.")]


def _progress_perspective(p: _Progress, data: dict[str, Any]) -> list[Envelope]:
    name = str(data.get("file", ""))
    if not p.art.exists(8, name):
        return []
    doc = p.art.read_json(8, name)
    rnd = int(doc.get("round", 0))
    if rnd == 0:
        what = f"proposed {_plural(len(doc.get('hypotheses') or []), 'hypothesis', 'hypotheses')}"
    elif doc.get("phase") == "answer":
        actions = [a.get("action") for a in doc.get("answers") or []]
        what = (
            f"answered {_plural(len(actions), 'challenge')} in round {rnd}: "
            f"{actions.count('revise')} revised, {actions.count('defend')} defended, "
            f"{actions.count('withdraw')} withdrawn"
        )
    else:
        stances = [r.get("stance") for r in doc.get("responses") or []]
        parts = [
            f"challenged {_plural(stances.count('challenge'), 'hypothesis', 'hypotheses')}",
            f"conceded {stances.count('concede')}",
            f"revised {len(doc.get('revised') or [])} of its own",
        ]
        what = f"{', '.join(parts)} in round {rnd}"
    name = _role_name(str(doc["role"]))
    return [*(p.env(*t) for t in _turns_of(doc)), p.say(f"The {name} {what}.")]


def _progress_judged(p: _Progress, data: dict[str, Any]) -> list[Envelope]:
    name = str(data.get("file", ""))
    if not p.art.exists(8, name):
        return []
    record = p.art.read_json(8, name)
    turns = _judge_turn(record)
    if not turns:
        return []
    who = (
        "An independent reviewer model"
        if record.get("independent_judge")
        else "A judge (the same model as the perspectives)"
    )
    return [*(p.env(*t) for t in turns), p.say(f"{who} ranked the positions.")]


def _progress_merge(p: _Progress, data: dict[str, Any]) -> list[Envelope]:
    return [*p.close("debate"), *p.open("write")]


def _progress_novelty(p: _Progress, data: dict[str, Any]) -> list[Envelope]:
    """The final set is written (only the novelty check is left): send it now."""
    content = _content_stage8(p.art, p.view.flags)
    out = [p.env(t, payload, "write") for t, payload in content.get("write", [])]
    out += p.close("write")
    out += p.open("check")
    return out


_PROGRESS_HANDLERS: dict[str, Callable[[_Progress, dict[str, Any]], list[Envelope]]] = {
    "queries": _progress_queries,
    "query": _progress_query,
    "screen_plan": _progress_screen_plan,
    "screen_batch": _progress_screen_batch,
    "cards_plan": _progress_cards_plan,
    "card": _progress_card,
    "perspectives_plan": _progress_perspectives_plan,
    "perspective": _progress_perspective,
    "critique": _progress_critique,
    "review": _progress_review,
    "judged": _progress_judged,
    "merge": _progress_merge,
    "novelty": _progress_novelty,
}


def _stage_done(view: RunView, group: Group, event: Event) -> Envelope:
    payload: dict[str, Any] = {"summary": stage_summary(view.artifacts, group)}
    if event.data.get("warnings"):
        payload["warnings"] = event.data["warnings"]
    if event.data.get("advisories"):
        payload["advisories"] = event.data["advisories"]
    return _env(
        "stage.completed", _with_cost(payload, view.usage), stage_key=group.key, actor=group.cast[0]
    )


def _on_stage_failed(view: RunView, event: Event) -> list[Envelope]:
    """A failed stage ends the run (``run.failed`` follows); keep the topic rating if it exists."""
    stage = int(event.stage or 0)
    if stage == 2 and view.artifacts.exists(2, "topic_evaluation.json"):
        try:
            kind, payload = _topic_evaluated(view.artifacts)
        except (KeyError, ValueError, OSError):
            return []
        return [_env(kind, payload, stage_key="scope", actor="pi")]
    return []


def gate_summary(kind: str, option: str, dropped: list[str], kept: list[str]) -> str:
    """The run studio's one-line record of a gate answer."""
    if option == "reject":
        return {
            gates.SCREEN: "Asked for a new search.",
            gates.HYPOTHESES: "Asked for a new set of hypotheses.",
        }.get(kind, "Asked for new scoping.")
    noun = {gates.SCREEN: "shortlist", gates.HYPOTHESES: "hypotheses"}.get(kind, "scope")
    item = "paper" if kind == gates.SCREEN else "hypothesis"
    parts = []
    if dropped:
        many = "papers" if kind == gates.SCREEN else "hypotheses"
        parts.append(f"without {_plural(len(dropped), item, many)}")
    if kept:
        parts.append(f"keeping {', '.join(kept)} despite the objection")
    return f"Approved the {noun}{' ' + ' and '.join(parts) if parts else ''}."


def _on_gate_opened(view: RunView, event: Event) -> list[Envelope]:
    spec = gate_spec(view.flags.mode, event.data, view.artifacts)
    group = _BY_STAGE[int(event.stage or 0)]
    spec["stage_key"] = group.key  # where the platform files the answer
    step = f"{spec['kind']}_gate"
    return [
        _env(
            "run.status", _with_cost({"status": "awaiting_review"}, view.usage), stage_key=group.key
        ),
        _env("step.started", {"step_id": step}, stage_key=group.key, actor="pi"),
        _env("gate.opened", {**spec, "gate": spec}, stage_key=group.key, actor="pi"),
    ]


def _on_gate_resolved(view: RunView, event: Event) -> list[Envelope]:
    group = _BY_STAGE[int(event.stage or 0)]
    kind = event.data["kind"]
    option = option_for(event.data)
    dropped = list(event.data.get("dropped") or [])
    kept = list(event.data.get("kept") or [])
    summary = gate_summary(kind, option, dropped, [_candidate_thread(k) for k in kept])
    answer = {
        "option_id": option,
        "dropped": dropped,
        "kept": [_candidate_thread(k) for k in kept],
        "note": event.data.get("note") or None,
    }
    out = [
        _env(
            "gate.resolved",
            {"gate_id": event.data["gate_id"], "kind": kind, "answer": answer, "summary": summary},
            stage_key=group.key,
            actor="pi",
        )
    ]
    if option == "reject":
        return out
    if kind == gates.HYPOTHESES and kept and view.artifacts.exists(8, "hypotheses.json"):
        doc = view.artifacts.read_json(8, "hypotheses.json")
        for h in doc["hypotheses"]:
            if h.get("kept_by_reviewer"):
                out += [
                    _env("hypothesis.drafted", {"hypothesis": _hypothesis_payload(h)},
                         stage_key=group.key, actor="pi"),
                    _env("hypothesis.selected", {"hypothesis_id": h["id"],
                         "override_note": h["kept_by_reviewer"]}, stage_key=group.key, actor="pi"),
                ]  # fmt: skip
    if kind == gates.SCOPE:
        out.append(
            _env(
                "scope.approved",
                {
                    "at": event.timestamp,
                    "note": event.data.get("note") or "Approved by the reviewer.",
                },
                stage_key=group.key,
                actor="pi",
            )
        )
    out.append(_env("step.completed", {"step_id": f"{kind}_gate"}, stage_key=group.key, actor="pi"))
    out.append(_stage_done(view, group, event))
    return out


def map_core_event(view: RunView, event: Event) -> list[Envelope]:
    """Platform envelopes for one core event (possibly none)."""
    kind = event.type
    if kind == ev.RUN_STARTED:
        flags = view.flags
        return [
            _env(
                "run.started",
                {"mode": flags.mode, "topic": view.record["topic"], "domains": view.domains},
            ),
            _env("run.plan", {"stages": [planned_stage(g, flags) for g in GROUPS]}),
        ]
    if kind == ev.STAGE_STARTED:
        return _on_stage_started(view, event)
    if kind == ev.STAGE_COMPLETED:
        return _on_stage_completed(view, event)
    if kind == ev.STAGE_PROGRESS:
        return _on_stage_progress(view, event)
    if kind == ev.STAGE_FAILED:
        return _on_stage_failed(view, event)
    if kind == ev.GATE_OPENED:
        return _on_gate_opened(view, event)
    if kind == ev.GATE_RESOLVED:
        return _on_gate_resolved(view, event)
    if kind == ev.RUN_PAUSED:
        return [
            _env(
                "run.status",
                _with_cost(
                    {"status": "paused", "reason": event.data.get("reason", "")}, view.usage
                ),
            )
        ]
    if kind == ev.RUN_RESUMED:
        return [_env("run.status", {"status": "running"})]
    if kind == ev.RUN_CANCELLED:
        reason = event.data.get("reason") or "user"
        return [
            _env(
                "run.status",
                _with_cost({"status": "failed", "reason": f"cancelled: {reason}"}, view.usage),
            )
        ]
    if kind == ev.RUN_FAILED:
        reason = f"{event.data.get('code', 'FAILED')}: {event.data.get('message', '')}".strip()
        return [_env("run.status", _with_cost({"status": "failed", "reason": reason}, view.usage))]
    if kind == ev.RUN_COMPLETED:
        payload = _with_cost({"usage": event.data.get("usage") or view.usage}, view.usage)
        return [_env("run.completed", payload)]
    return []


# ---------------------------------------------------------------------------
# Persistence and projection
# ---------------------------------------------------------------------------


@dataclass
class _LogState:
    last_seq: int = 0
    last_core: int = 0
    memory_core: int = 0


class PlatformEventLog:
    """Append-only ``platform_events.jsonl`` per run with contiguous ``source_seq``."""

    def __init__(self, store: RunStore) -> None:
        self.store = store
        self._state: dict[str, _LogState] = {}

    def path(self, run_id: str) -> Path:
        return self.store.run_dir(run_id) / PLATFORM_FILE

    def _load(self, run_id: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            lines = self.path(run_id).read_text(encoding="utf-8").splitlines()
        except OSError:
            return rows
        for line in lines:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                break  # torn tail after a crash
        return rows

    def _heal(self, run_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop a trailing core-event batch that was only partly written."""
        if not rows:
            return rows
        core = rows[-1].get("_core_seq")
        tail = [r for r in rows if r.get("_core_seq") == core]
        size = int(rows[-1].get("_batch", [0, 0])[1] or 0)
        complete = len(tail) == size
        text_ok = self._file_ends_cleanly(run_id)
        if complete and text_ok:
            return rows
        kept = rows if complete else [r for r in rows if r.get("_core_seq") != core]
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
        write_bytes_atomic(self.path(run_id), body.encode("utf-8"))
        return kept

    def _file_ends_cleanly(self, run_id: str) -> bool:
        try:
            data = self.path(run_id).read_bytes()
        except OSError:
            return True
        return not data or data.endswith(b"\n")

    def state(self, run_id: str) -> _LogState:
        with self.store.lock(run_id):
            if run_id not in self._state:
                rows = self._heal(run_id, self._load(run_id))
                last = rows[-1] if rows else {}
                self._state[run_id] = _LogState(
                    last_seq=int(last.get("source_seq", 0)),
                    last_core=int(last.get("_core_seq", 0)),
                    memory_core=int(last.get("_core_seq", 0)),
                )
            return self._state[run_id]

    def append(self, run_id: str, core_seq: int, envelopes: list[Envelope]) -> int:
        """Persist the envelopes of one core event; returns the last ``source_seq``."""
        with self.store.lock(run_id):
            st = self.state(run_id)
            seq = st.last_seq
            lines = []
            for i, env in enumerate(envelopes):
                seq += 1
                row = {
                    "source_seq": seq,
                    **env,
                    "_core_seq": core_seq,
                    "_batch": [i, len(envelopes)],
                }
                lines.append(json.dumps(row, ensure_ascii=False) + "\n")
            if lines:
                path = self.path(run_id)
                prefix = b""
                if path.exists() and path.stat().st_size > 0:
                    with path.open("rb") as probe:
                        probe.seek(-1, os.SEEK_END)
                        if probe.read(1) != b"\n":
                            prefix = b"\n"
                with path.open("ab") as handle:
                    handle.write(prefix + "".join(lines).encode("utf-8"))
                    handle.flush()
                    os.fsync(handle.fileno())
                st.last_seq = seq
                st.last_core = core_seq
            st.memory_core = max(st.memory_core, core_seq)
            return seq

    def last_seq(self, run_id: str) -> int:
        return self.state(run_id).last_seq

    def read(self, run_id: str, after: int = 0, limit: int | None = None) -> list[dict[str, Any]]:
        """Public envelopes with ``source_seq > after`` in order."""
        with self.store.lock(run_id):
            self.state(run_id)  # heals a torn tail before reading
            rows = [public(r) for r in self._load(run_id) if int(r["source_seq"]) > after]
        return rows[:limit] if limit is not None else rows


class PlatformProjector:
    """Core :class:`~idea2hypothesis.pipeline.ports.EventSink` that persists Platform events."""

    def __init__(self, store: RunStore, log: PlatformEventLog | None = None) -> None:
        self.store = store
        self.log = log or PlatformEventLog(store)
        self.listeners: list[Callable[[str], None]] = []

    async def emit(self, event: Event) -> None:
        self.project(event.run_id)

    def project(self, run_id: str) -> int:
        """Persist envelopes for every core event not yet projected; returns how many."""
        with self.store.lock(run_id):
            st = self.log.state(run_id)
            events = self.store.read_events(run_id, after_seq=st.memory_core)
            if not events:
                return 0
            view = load_view(self.store, run_id)
            added = 0
            for event in events:
                try:
                    envelopes = map_core_event(view, event)
                except Exception:  # noqa: BLE001 - a mapping problem must not stop the run
                    logger.exception("could not map %s of run %s", event.type, run_id)
                    envelopes = []
                self.log.append(run_id, event.seq, envelopes)
                added += len(envelopes)
        if added:
            for listener in self.listeners:
                listener(run_id)
        return added
