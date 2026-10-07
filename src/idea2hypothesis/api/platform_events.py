"""Translate core run events and stage artifacts into the Platform event stream.

Platform consumers (BE and the run studio) read ``{source_seq, type, stage_key, actor, payload}``
envelopes. Every envelope built here is derived from a core event that actually happened and
from the artifacts the stage really wrote; nothing is simulated. Envelopes are appended to
``runs/<id>/platform_events.jsonl`` (``source_seq`` contiguous from 1) before anything is
delivered, so they can be replayed through ``GET /runs/{id}/events`` and the webhook.

UI groups (``stage_key``): scope = stages 1-2, search = 3-4, screen = 5, read = 6,
synthesize = 7 and ``r1-hypothesize`` = 8 (the group key the run studio reads).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
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
        "Each perspective proposes hypotheses (and rebuts when debate rounds are on).",
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
            "advice": e.get("suggestion", ""),
        },
    )


def _content_stage3(art: ArtifactStore) -> StepEvents:
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
            {"sources": [{"id": s["id"], "name": s["name"]} for s in sources], "delay_ms": 0},
        )
    )
    return {"strategy": events}


def _read_yaml(art: ArtifactStore, stage: int, name: str) -> Any:
    import yaml

    return yaml.safe_load(art.read_text(stage, name))


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
    events.append(
        (
            "literature.collected",
            {
                "raw": meta["raw"],
                "unique": meta["unique"],
                "duplicates": meta["duplicates"],
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
            },
        )
    ]
    score += [("screen.scored", {"points": points[i : i + 48]}) for i in range(0, len(points), 48)]
    reject = [
        (
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
        for d in decisions
        if d["decision"] in ("rejected", "unscored", "prefiltered")
    ]
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


def _content_stage6(art: ArtifactStore) -> StepEvents:
    events: list[tuple[str, dict[str, Any]]] = []
    for name in art.list_files(6):
        if not (name.startswith("cards/") and name.endswith(".json")):
            continue
        card = art.read_json(6, name)
        fields = ("problem", "method", "data", "metrics", "findings", "limitations")
        card_payload: dict[str, Any] = {
            "id": card["card_id"],
            "paper_id": card["paper_id"],
            "citation": card.get("cite_key") or card.get("title", ""),
            "cite_key": card.get("cite_key"),
            "title": card.get("title"),
            "evidence_scope": card.get("evidence_scope"),
            "unknown_fields": [k for k in fields if card.get(k) is None],
        }
        card_payload.update({k: card.get(k) or "" for k in fields})
        events.append(("card.extracted", {"card": card_payload}))
    return {"extract": events}


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
        ],
        "overview": [("synthesis.overview", {"text": syn["overview"]})]
        if syn.get("overview")
        else [],
        "tension": [
            ("synthesis.tension", {"tension": {"between": t["between"], "text": t["text"]}})
            for t in syn.get("tensions") or []
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


_ROLE_ACTORS = {
    "innovator": "theorist",
    "pragmatist": "methodologist",
    "contrarian": "skeptic",
}
_ZONES: dict[str, list[int | None]] = {"> 0": [None, 0], "< 0": [0, None], "≠ 0": [0, 0]}


def _turns(art: ArtifactStore) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[int, str, dict[str, Any]]] = []
    for name in art.list_files(8):
        if not (name.startswith("perspectives/") and name.endswith(".json")):
            continue
        if name.endswith("debate_record.json"):
            continue
        doc = art.read_json(8, name)
        if "role" not in doc or "hypotheses" not in doc:
            continue
        rows.append((int(doc.get("round", 0)), str(doc["role"]), doc))
    rows.sort(key=lambda r: (r[0], r[1]))
    turns = []
    for rnd, role, doc in rows:
        text = "\n".join(
            f"{i}. {h.get('statement', '')}" for i, h in enumerate(doc["hypotheses"], 1)
        )
        turns.append(
            (
                "debate.turn",
                {
                    "turn": {
                        "id": f"{role}-r{rnd}",
                        "actor": _ROLE_ACTORS.get(role, "theorist"),
                        "stance": "propose" if rnd == 0 else "refine",
                        "text": text,
                        "role": role,
                        "round": rnd,
                    }
                },
            )
        )
    return turns


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
    }


def _content_stage8(art: ArtifactStore, flags: Flags) -> StepEvents:
    doc = art.read_json(8, "hypotheses.json")
    hypotheses = doc["hypotheses"]
    out: StepEvents = {
        "debate": _turns(art),
        "write": [
            ("hypothesis.drafted", {"hypothesis": _hypothesis_payload(h)}) for h in hypotheses
        ]
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
        checks = []
        for row in report.get("per_hypothesis", []):
            closest = row.get("closest_paper") or {}
            similarity = float(closest.get("similarity", 0.0))
            checks.append(
                (
                    "hypothesis.checked",
                    {
                        "hypothesis_id": row["hypothesis_id"],
                        "novelty": {
                            "novel": similarity < threshold,
                            "closest": closest.get("title", ""),
                            "similarity": similarity,
                        },
                        "assessment": "heuristic, not proof of novelty",
                    },
                )
            )
        out["check"] = checks
    return out


def stage_content(art: ArtifactStore, stage: int, flags: Flags, domains: list[str]) -> StepEvents:
    """Events of every step of ``stage``, built from the artifacts on disk."""
    if stage == 1:
        return _content_stage1(art, flags, domains)
    if stage == 2:
        return _content_stage2(art)
    if stage == 3:
        return _content_stage3(art)
    if stage == 4:
        return _content_stage4(art)
    if stage == 5:
        return _content_stage5(art)
    if stage == 6:
        return _content_stage6(art)
    if stage == 7:
        return _content_stage7(art)
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
            f"{len(syn['clusters'])} groups • {len(syn.get('tensions') or [])} tensions • "
            f"{len(syn['gaps'])} gaps"
        )
    count = len(art.read_json(8, "hypotheses.json")["hypotheses"])
    return f"{count} hypotheses to test, each with a way to be wrong"


def _cards(art: ArtifactStore) -> list[str]:
    return [n for n in art.list_files(6) if n.startswith("cards/") and n.endswith(".json")]


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


def option_for(answer: dict[str, Any]) -> str:
    if answer.get("decision") == gates.REJECT:
        return "reject"
    return "drop" if answer.get("dropped") else "approve"


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

    @property
    def domains(self) -> list[str]:
        return list(self.record.get("domains") or [])

    @property
    def usage(self) -> dict[str, Any]:
        return self.record.get("usage") or {}


def load_view(store: RunStore, run_id: str) -> RunView:
    record = store.read_run(run_id)
    try:
        research = store.read_snapshot(run_id, "config").get("research", {})
    except (OSError, ValueError):
        research = {}
    flags = Flags(
        mode=record["review_mode"],
        hardware_advisory=bool(research.get("hardware_advisory", False)),
        novelty_check=bool(research.get("novelty_check", True)),
    )
    return RunView(run_id, record, flags, store.artifacts(run_id))


def _stage_events(
    group: Group, stage: int, view: RunView, content: StepEvents, *, first_started: bool
) -> list[Envelope]:
    """Step events of one completed core stage (its first step may already be started)."""
    out: list[Envelope] = []
    for i, step_id in enumerate(view.flags.steps(stage)):
        actor = _STEPS[step_id][1]
        if not (i == 0 and first_started):
            out.append(_env("step.started", {"step_id": step_id}, stage_key=group.key, actor=actor))
        out += [_env(t, p, stage_key=group.key, actor=actor) for t, p in content.get(step_id, [])]
        out.append(_env("step.completed", {"step_id": step_id}, stage_key=group.key, actor=actor))
    return out


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
    out.append(
        _env("step.started", {"step_id": first}, stage_key=group.key, actor=_STEPS[first][1])
    )
    return out


def _on_stage_completed(view: RunView, event: Event) -> list[Envelope]:
    stage = int(event.stage or 0)
    group = _BY_STAGE[stage]
    content = stage_content(view.artifacts, stage, view.flags, view.domains)
    out = _stage_events(group, stage, view, content, first_started=True)
    if stage == group.last and view.flags.gate_step(group) is None:
        out.append(_stage_done(view, group, event))
    return out


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


def _on_gate_opened(view: RunView, event: Event) -> list[Envelope]:
    spec = gate_spec(view.flags.mode, event.data, view.artifacts)
    group = _BY_STAGE[int(event.stage or 0)]
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
    noun = "shortlist" if kind == gates.SCREEN else "scope"
    if option == "reject":
        summary = "Asked for a new search." if kind == gates.SCREEN else "Asked for new scoping."
    elif dropped:
        summary = f"Approved the {noun} without {len(dropped)} papers."
    else:
        summary = f"Approved the {noun}."
    answer = {"option_id": option, "dropped": dropped, "note": event.data.get("note") or None}
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
