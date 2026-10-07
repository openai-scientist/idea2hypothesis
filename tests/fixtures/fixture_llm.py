"""FixtureLLM: a scripted LLMPort that answers each stage prompt with schema-valid JSON.

It reads the ids it needs (sub-question, paper, card, gap ids) from the prompt text, so the
canned answers always reference records that really exist in the run. This is a test double: it
checks orchestration, not research quality.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from idea2hypothesis.llm.models import ChatMessage, LLMResponse
from tests.fixtures.fixture_literature import OFFTOPIC_MARK

_MARKERS: list[tuple[str, str]] = [
    ("Create a SMART research goal as JSON", "topic_init"),
    ("Evaluate this research topic and its goal", "topic_evaluation"),
    ("Decompose this research problem", "problem_decompose"),
    ("Create a search strategy for this topic", "search_strategy"),
    ("Screen every paper below", "literature_screen"),
    ("Extract a structured knowledge card", "knowledge_extract"),
    ("Produce a synthesis (topic clusters", "synthesis"),
    ("Write the final set of 2-4 hypotheses", "hypothesis_gen"),
    ("Write your updated position", "debate_rebuttal"),
    ("Score each perspective 1-10", "debate_judge"),
    ("Allowed evidence references", "perspective"),
]


_MECHANISMS = [
    "Prefrontal synaptic plasticity during slow-wave sleep",
    "Suprachiasmatic nucleus desynchrony after irregular schedules",
    "Adenosine accumulation that blocks neural transmission",
    "Hippocampal replay that consolidates examined material",
]
_GAPS = [
    "A within-subject design around exam weeks",
    "Objective actigraphy rather than self-report",
    "The interaction between sleep debt and exam timing",
    "Mediation through working memory capacity",
]


@dataclass
class PromptInfo:
    key: str
    system: str
    user: str
    index: int  # 0-based call number for this key

    @property
    def topic(self) -> str:
        match = re.search(r"^Topic: (.+)$", self.user, re.MULTILINE)
        return match.group(1).strip() if match else ""

    def section_json(self, marker: str, end: str | None = None) -> Any:
        text = self.user.split(marker, 1)[1]
        if end and end in text:
            text = text.split(end, 1)[0]
        start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=0)
        return json.JSONDecoder().raw_decode(text[start:])[0]

    def ids(self, pattern: str) -> list[str]:
        return list(dict.fromkeys(re.findall(pattern, self.user)))

    def list_after(self, label: str) -> list[str]:
        match = re.search(rf"{re.escape(label)}\s*(.+)", self.user)
        return [s.strip() for s in match.group(1).split(",")] if match else []


@dataclass
class FixtureLLM:
    """Scripted chat model. ``overrides[key]`` may be a value, a callable or a list of items
    consumed one per call (exceptions in the list are raised); defaults answer the rest."""

    overrides: dict[str, Any] = field(default_factory=dict)
    keep_title: Callable[[str], bool] = lambda title: OFFTOPIC_MARK not in title
    cost_per_call: float | None = None
    on_call: Callable[[PromptInfo], Awaitable[None] | None] | None = None
    calls: list[PromptInfo] = field(default_factory=list)
    _counts: dict[str, int] = field(default_factory=dict)
    _queues: dict[str, list[Any]] = field(default_factory=dict)

    # -- LLMPort ------------------------------------------------------------

    async def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        user = "\n".join(m.content for m in messages if m.role == "user")
        if len(messages) > 1:  # a repair round: classify by the original prompt
            user = messages[0].content
        info = self._classify(system or "", user)
        self.calls.append(info)
        if self.on_call is not None:
            result = self.on_call(info)
            if inspect.isawaitable(result):
                await result
        payload = self._payload(info)
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMResponse(
            text=text,
            model="fixture-model",
            prompt_tokens=max(1, len(user) // 4),
            completion_tokens=max(1, len(text) // 4),
            cost_usd=self.cost_per_call,
        )

    # -- helpers ------------------------------------------------------------

    def count(self, key: str) -> int:
        return sum(1 for c in self.calls if c.key == key)

    def _classify(self, system: str, user: str) -> PromptInfo:
        key = next((k for marker, k in _MARKERS if marker in user), "unknown")
        index = self._counts.get(key, 0)
        self._counts[key] = index + 1
        return PromptInfo(key, system, user, index)

    def _payload(self, info: PromptInfo) -> Any:
        if info.key in self.overrides:
            override = self.overrides[info.key]
            if isinstance(override, list):
                queue = self._queues.setdefault(info.key, list(override))
                if queue:
                    return self._resolve(queue.pop(0), info)
            else:
                return self._resolve(override, info)
        return getattr(self, f"_default_{info.key}")(info)

    def _resolve(self, item: Any, info: PromptInfo) -> Any:
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(info)
        return item

    # -- default answers ----------------------------------------------------

    def _default_topic_init(self, info: PromptInfo) -> dict[str, Any]:
        return {
            "researchable": True,
            "rejection_reason": None,
            "working_title": "Sleep duration and exam performance",
            "problem": "It is unclear how habitual sleep duration affects exam scores.",
            "objective": "Estimate the effect of sleep duration on standardized exam scores.",
            "novel_angle": (
                "Within-subject variation around exam weeks rather than between-student contrasts."
            ),
            "scope": "University students in a single semester.",
            "smart_goal": {
                "specific": "x",
                "measurable": "y",
                "achievable": "z",
                "relevant": "r",
                "time_bound": "one semester",
            },  # fmt: skip
            "constraints": ["no new data collection"],
            "success_criteria": ["Report an effect size with a 95% interval"],
            "benchmark": None,
        }

    def _default_problem_decompose(self, info: PromptInfo) -> dict[str, Any]:
        questions = [
            {
                "id": f"SQ{i}",
                "text": f"Sub-question {i} about sleep and exam scores?",
                "priority": i,
                "goal_link": "objective",
                "tests": "published cohort studies",
                "covers": ["sleep"],
            }
            for i in range(1, 5)
        ]
        return {
            "sub_questions": questions,
            "priority_ranking": [q["id"] for q in questions],
            "risks": [
                {"id": "R1", "sub_question_id": "SQ1", "text": "confounding", "level": "medium"}
            ],
        }

    def _default_topic_evaluation(self, info: PromptInfo) -> dict[str, Any]:
        return {"novelty": 8, "specificity": 7, "feasibility": 7, "overall": 7.3, "suggestion": ""}

    def _default_search_strategy(self, info: PromptInfo) -> dict[str, Any]:
        sq = info.ids(r"\"id\": \"(SQ\d+)\"")
        names = ["core_topic", "mechanisms", "measurement"]
        strategies = []
        for i, name in enumerate(names):
            strategies.append(
                {
                    "name": name,
                    "rationale": f"covers {name}",
                    "sub_question_ids": [sq[i % len(sq)], sq[(i + 1) % len(sq)]],
                    "queries": [f"sleep exam query {i}{j} students" for j in range(3)],
                }
            )
        return {"search_strategies": strategies, "filters": {"min_year": 2015}}

    def _default_literature_screen(self, info: PromptInfo) -> dict[str, Any]:
        papers = info.section_json("Papers:\n")
        screened = []
        for p in papers:
            keep = self.keep_title(p["title"])
            screened.append(
                {
                    "paper_id": p["paper_id"],
                    "decision": "keep" if keep else "reject",
                    "relevance_score": 0.9 if keep else 0.75,
                    "quality_score": 0.8,
                    "reason": "directly studies sleep and exam outcomes"
                    if keep
                    else "different field that shares the word sleep",
                    "false_friend": None if keep else "sleep",
                }
            )
        return {"screened": screened}

    def _default_knowledge_extract(self, info: PromptInfo) -> dict[str, Any]:
        paper = info.section_json("Paper:\n")
        return {
            "problem": f"Addresses: {paper['title']}",
            "method": "Controlled study with participants",
            "data": None,
            "metrics": None,
            "findings": "Reports effects on outcomes",
            "limitations": "Abstract does not state limitations in detail"
            if info.index % 2
            else None,
        }

    def _default_synthesis(self, info: PromptInfo) -> dict[str, Any]:
        cards = info.section_json("Cards:\n")
        card_ids = [c["card_id"] for c in cards]
        sq = info.ids(r"\"id\": \"(SQ\d+)\"")
        half = max(1, len(card_ids) // 2)
        return {
            "overview": "Evidence converges on a modest sleep effect with measurement gaps.",
            "clusters": [
                {
                    "id": "C1",
                    "title": "Duration effects",
                    "claim": "More sleep helps",
                    "card_ids": card_ids[:half],
                },
                {
                    "id": "C2",
                    "title": "Timing effects",
                    "claim": "Regularity matters",
                    "card_ids": card_ids[half:] or card_ids[:1],
                },
            ],  # fmt: skip
            "tensions": [{"between": ["C1", "C2"], "text": "duration versus regularity"}],
            "gaps": [
                {
                    "id": "G1",
                    "text": "Few within-subject designs",
                    "sub_question_ids": sq[:2],
                    "card_ids": card_ids[:2],
                    "why_prioritized": "limits causal reading",
                    "priority": 1,
                },
                {
                    "id": "G2",
                    "text": "Weak objective sleep measurement",
                    "sub_question_ids": sq[2:4],
                    "card_ids": card_ids[-2:],
                    "why_prioritized": "self-report bias",
                    "priority": 2,
                },
            ],  # fmt: skip
            "prioritized_opportunities": [{"gap_id": "G1", "direction": "within-subject cohort"}],
            "set_aside": [],
        }

    def _hypotheses(self, info: PromptInfo, tag: str, count: int) -> list[dict[str, Any]]:
        gaps = info.list_after("Allowed gap ids:")
        match = re.search(r"Allowed evidence references[^:\n]*:\s*(.+)", info.user)
        refs = [r.strip() for r in match.group(1).split(",")] if match else []
        predictions = ["> 0", "< 0", "≠ 0", "> 0"]
        items = []
        for i in range(count):
            items.append(
                {
                    "id": f"H{i + 1}",
                    "statement": (
                        f"[{tag}] Hypothesis {i + 1}: sleep exposure {i + 1} changes exam score"
                    ),
                    "gap_id": gaps[i % len(gaps)],
                    "sub_question_ids": ["SQ1"],
                    "evidence_refs": [refs[i % len(refs)], refs[(i + 1) % len(refs)]],
                    "exposure": f"hours of sleep per night (variant {i + 1})",
                    "outcome": "standardized exam score in SD units",
                    "estimand": "within-subject regression coefficient",
                    "method": "fixed-effects panel regression",
                    "conditions": "university students",
                    "prediction": predictions[i % len(predictions)],
                    "falsification_criteria": (
                        f"Wrong if the 95% confidence interval of the coefficient includes zero "
                        f"in variant {i + 1}"
                    ),
                    "limitations": ["observational data"],
                    "rationale": (
                        f"{_MECHANISMS[i % len(_MECHANISMS)]} explains the effect "
                        f"in variant {tag}{i}."
                    ),
                    "novelty": (
                        f"{_GAPS[i % len(_GAPS)]} has not been tested in the {tag}{i} setting."
                    ),
                    "risk": "medium",
                }
            )
        return items

    def _default_perspective(self, info: PromptInfo) -> dict[str, Any]:
        return {"hypotheses": self._hypotheses(info, f"role{info.index}", 2)}

    def _default_debate_rebuttal(self, info: PromptInfo) -> dict[str, Any]:
        return {"hypotheses": self._hypotheses(info, f"rebut{info.index}", 2), "concessions": ["x"]}

    def _default_debate_judge(self, info: PromptInfo) -> dict[str, Any]:
        return {"rankings": [{"role": "any", "score": 7, "reason": "solid"}]}

    def _default_hypothesis_gen(self, info: PromptInfo) -> dict[str, Any]:
        return {"hypotheses": self._hypotheses(info, "final", 3), "disagreements": ["effect size"]}

    def _default_unknown(self, info: PromptInfo) -> Any:
        raise AssertionError(f"FixtureLLM got an unrecognised prompt: {info.user[:200]!r}")
