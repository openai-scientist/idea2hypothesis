"""Per-stage input/output contracts and validators.

Validators come in two forms that share the same rules:

* ``check_*`` functions validate the parsed JSON of one artifact (used by stages to ask the
  model to repair an invalid answer before anything is written);
* ``validate_stage`` validates the artifacts of a stage on disk (used after a stage runs, on
  resume, and when an artifact is edited through the API).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import yaml

from idea2hypothesis.pipeline.models import Stage
from idea2hypothesis.storage.artifacts import ArtifactStore

SCHEMA_VERSION = 1
#: The expected effect: positive, negative, non-zero, or negligible (within ± equivalence_margin).
PREDICTIONS = ("> 0", "< 0", "≠ 0", "≈ 0")
EQUIVALENCE = "≈ 0"
EVIDENCE_SCOPES = ("abstract", "full_text")
DECISIONS = (
    "kept", "rejected", "below_cutoff", "unscored", "prefiltered", "dropped_by_reviewer"
)  # fmt: skip
CARD_FIELDS = ("problem", "method", "data", "metrics", "findings", "limitations")
CARD_CONTENT_FIELDS = ("problem", "method", "findings", "limitations")
#: Cards from this schema on back every filled field with quotes from the abstract.
QUOTED_CARD_SCHEMA = 2
#: An abstract may name each dataset or metric in its own sentence; joining them is not verbatim.
MAX_QUOTES_PER_FIELD = 6
MIN_QUOTE_WORDS = 4
MIN_SUB_QUESTIONS = 3
MIN_STRATEGIES = 2
MIN_GAPS = 2
MIN_HYPOTHESES = 2
#: Words in a novelty search query (fewest, most).
NOVELTY_QUERY_WORDS = (2, 7)
NOVELTY_VERDICTS = ("tested", "related", "new")
_QUERY_SYNTAX = re.compile(r"\"|\b[a-z]{2,3}:|\b(?:AND|OR|NOT)\b")
#: Syntheses from this schema on give each tension an id and the cards on each of its two sides.
SIDED_TENSION_SCHEMA = 2
TENSION_ID = re.compile(r"^X\d+$")
_CONDITION_WORDS = re.compile(
    r"\b(if|when|unless|exceed\w*|below|above|less|greater|fail\w*|interval|threshold|bound|limit"
    r"|zero|ci)\b|[<>=≥≤≠]|\d",
    re.IGNORECASE,
)


@dataclass
class Findings:
    """Collected problems: ``errors`` fail the stage, ``warnings`` are advisory."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def require(self, condition: object, message: str) -> bool:
        if not condition:
            self.errors.append(message)
        return bool(condition)

    def extend(self, other: Findings) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    @property
    def ok(self) -> bool:
        return not self.errors


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _str_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(_text(v) for v in value)


def _normalise_text(value: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", value.lower())).strip()


def card_id_for(paper_id: str) -> str:
    return f"card-{paper_id}"


_QUOTE_FOLD = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
        "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u2212": "-", "\u00a0": " ",
    }
)  # fmt: skip


def normalise_quote(text: str) -> str:
    """The form a quote and its abstract are compared in.

    Only differences that carry no meaning are removed: Unicode width and ligature forms, curly
    quotes and dash variants, HTML/JATS tags some sources leave in abstracts, letter case, runs of
    white space, and quote marks or ellipses wrapped around the passage. Words, numbers and their
    order must match exactly.
    """
    text = unicodedata.normalize("NFKC", text).translate(_QUOTE_FOLD)
    text = re.sub(r"<[^>]{1,40}>", " ", text)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text.strip(" \"'.…").strip()


def quote_problems(field_name: str, quotes: Any, abstract: str) -> list[str]:
    """Why the quotes given for one card field do not back it (empty when they do)."""
    if not isinstance(quotes, list) or not quotes:
        return [f"{field_name} is filled but has no quote from the abstract"]
    if len(quotes) > MAX_QUOTES_PER_FIELD:
        return [f"{field_name} has {len(quotes)} quotes; give at most {MAX_QUOTES_PER_FIELD}"]
    source = normalise_quote(abstract)
    problems: list[str] = []
    for quote in quotes:
        if not _text(quote):
            problems.append(f"{field_name} has an empty quote")
            continue
        norm = normalise_quote(quote)
        if len(norm.split()) < MIN_QUOTE_WORDS:
            problems.append(
                f"{field_name} quote {quote[:60]!r} is shorter than {MIN_QUOTE_WORDS} words"
            )
        elif norm not in source:
            problems.append(
                f"{field_name} quote {quote[:80]!r} is not in the abstract word for word"
            )
    return problems


def check_card_quotes(data: Any, abstract: str) -> Findings:
    """Every filled card field is backed by quotes found in the abstract; null fields have none."""
    f = Findings()
    if not isinstance(data, dict):
        f.error("the card must be a JSON object")
        return f
    quotes = data.get("quotes")
    if not f.require(isinstance(quotes, dict), "the card needs a 'quotes' object"):
        return f
    for key in quotes:
        f.require(key in CARD_FIELDS, f"quotes has an unknown field {key!r}")
    for key in CARD_FIELDS:
        if _text(data.get(key)):
            for problem in quote_problems(key, quotes.get(key), abstract):
                f.error(problem)
        elif quotes.get(key):
            f.error(f"{key} is null but has quotes; fill it or drop its quotes")
    return f


# ---------------------------------------------------------------------------
# JSON-level checks (shared by stages and file validation)
# ---------------------------------------------------------------------------


def check_goal(goal: Any) -> Findings:
    f = Findings()
    if not f.require(isinstance(goal, dict), "goal must be a JSON object"):
        return f
    if not f.require(
        isinstance(goal.get("researchable"), bool), "goal.researchable must be a boolean"
    ):
        return f
    if not goal["researchable"]:
        f.require(_text(goal.get("rejection_reason")), "a rejected topic needs a rejection_reason")
        return f
    for key in ("working_title", "problem", "objective", "scope"):
        f.require(_text(goal.get(key)), f"goal.{key} is required")
    f.require(
        _str_list(goal.get("success_criteria")), "goal.success_criteria needs at least one entry"
    )
    f.require(
        isinstance(goal.get("constraints", []), list), "goal.constraints must be a list of strings"
    )
    return f


def check_problem_tree(tree: Any) -> Findings:
    f = Findings()
    if not f.require(isinstance(tree, dict), "problem tree must be a JSON object"):
        return f
    questions = tree.get("sub_questions")
    if not f.require(
        isinstance(questions, list) and len(questions) >= MIN_SUB_QUESTIONS,
        f"at least {MIN_SUB_QUESTIONS} prioritised sub-questions are required",
    ):
        return f
    seen: set[str] = set()
    for i, q in enumerate(questions):
        if not f.require(isinstance(q, dict), f"sub_questions[{i}] must be an object"):
            continue
        qid = q.get("id")
        if f.require(_text(qid), f"sub_questions[{i}].id is required"):
            f.require(qid not in seen, f"duplicate sub-question id {qid}")
            seen.add(str(qid))
        f.require(_text(q.get("text")), f"sub_questions[{i}].text is required")
        f.require(_text(q.get("goal_link")), f"sub_questions[{i}].goal_link is required")
        f.require(
            isinstance(q.get("priority"), int) and not isinstance(q.get("priority"), bool),
            f"sub_questions[{i}].priority must be an integer",
        )
    return f


def check_topic_evaluation(evaluation: Any, *, require_reasons: bool = False) -> Findings:
    """``require_reasons``: the model's answer must explain each score (runs stored before
    reasons existed are not re-judged)."""
    f = Findings()
    if not f.require(isinstance(evaluation, dict), "topic evaluation must be a JSON object"):
        return f
    for key in ("novelty", "specificity", "feasibility", "overall"):
        value = evaluation.get(key)
        f.require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 10,
            f"topic_evaluation.{key} must be a number between 0 and 10",
        )
    if require_reasons:
        reasons = evaluation.get("reasons")
        for key in ("novelty", "specificity", "feasibility"):
            f.require(
                isinstance(reasons, dict) and _text(reasons.get(key)),
                f"topic_evaluation.reasons.{key} must say what earned the score",
            )
        f.require(_text(evaluation.get("suggestion")), "topic_evaluation.suggestion is required")
    return f


def sub_question_ids(tree: dict[str, Any]) -> set[str]:
    return {
        str(q["id"]) for q in tree.get("sub_questions", []) if isinstance(q, dict) and "id" in q
    }


def check_search_plan(plan: Any, known_sq: set[str]) -> Findings:
    f = Findings()
    if not f.require(isinstance(plan, dict), "search plan must be an object"):
        return f
    strategies = plan.get("search_strategies")
    if not f.require(
        isinstance(strategies, list) and len(strategies) >= MIN_STRATEGIES,
        f"at least {MIN_STRATEGIES} search strategies are required",
    ):
        return f
    for i, strategy in enumerate(strategies):
        if not f.require(isinstance(strategy, dict), f"search_strategies[{i}] must be an object"):
            continue
        f.require(_text(strategy.get("name")), f"search_strategies[{i}].name is required")
        f.require(
            _str_list(strategy.get("queries")), f"search_strategies[{i}] needs non-empty queries"
        )
        ids = strategy.get("sub_question_ids")
        if f.require(_str_list(ids), f"search_strategies[{i}] must list sub_question_ids"):
            unknown = sorted(set(ids) - known_sq)
            f.require(
                not unknown, f"search_strategies[{i}] references unknown sub-questions {unknown}"
            )
    return f


def check_queries(queries: Any, known_sq: set[str]) -> Findings:
    f = Findings()
    if not f.require(isinstance(queries, dict), "queries.json must be an object"):
        return f
    rows = queries.get("queries")
    if not f.require(isinstance(rows, list) and rows, "queries.json needs at least one query"):
        return f
    for i, row in enumerate(rows):
        if not f.require(isinstance(row, dict), f"queries[{i}] must be an object"):
            continue
        f.require(_text(row.get("text")), f"queries[{i}].text is required")
        f.require(_text(row.get("strategy")), f"queries[{i}].strategy is required")
        ids = row.get("sub_question_ids")
        if f.require(_str_list(ids), f"queries[{i}] must be linked to sub-questions"):
            f.require(set(ids) <= known_sq, f"queries[{i}] references unknown sub-questions")
    return f


def check_candidates(rows: list[dict[str, Any]]) -> Findings:
    f = Findings()
    if not f.require(rows, "no candidate papers were collected"):
        return f
    ids: set[str] = set()
    identities: set[str] = set()
    for i, row in enumerate(rows):
        pid = row.get("paper_id")
        if f.require(_text(pid), f"candidates[{i}].paper_id is required"):
            f.require(pid not in ids, f"duplicate paper_id {pid}")
            ids.add(str(pid))
        f.require(_text(row.get("title")), f"candidates[{i}].title is required")
        records = row.get("source_records")
        if f.require(
            isinstance(records, list) and records,
            f"candidates[{i}] has no source_records (provenance)",
        ):
            for record in records:
                f.require(
                    isinstance(record, dict)
                    and _text(record.get("provider"))
                    and _text(record.get("source_id"))
                    and _text(record.get("retrieved_at")),
                    f"candidates[{i}] has an incomplete source record",
                )
        for key in (
            str(row.get("doi") or "").lower(),
            str(row.get("arxiv_id") or ""),
            _normalise_text(str(row.get("title", ""))),
        ):
            if key:
                f.require(key not in identities, f"candidates contain a duplicate of {key[:60]!r}")
                identities.add(key)
    return f


def check_screen(
    candidates: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    review: Any,
) -> Findings:
    f = Findings()
    candidate_ids = {str(c["paper_id"]) for c in candidates}
    f.require(shortlist, "the shortlist is empty")
    shortlist_ids = [str(r.get("paper_id")) for r in shortlist]
    f.require(len(set(shortlist_ids)) == len(shortlist_ids), "shortlist contains duplicate papers")
    for row in shortlist:
        pid = row.get("paper_id")
        f.require(pid in candidate_ids, f"shortlisted paper {pid} is not a collected candidate")
        for key in ("relevance_score", "quality_score"):
            value = row.get(key)
            f.require(
                isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1,
                f"shortlist paper {pid}: {key} must be in [0, 1]",
            )
        f.require(_text(row.get("keep_reason")), f"shortlist paper {pid}: keep_reason is required")
    if not f.require(isinstance(review, dict), "review.json must be an object"):
        return f
    decisions = review.get("decisions")
    if not f.require(isinstance(decisions, list), "review.json needs a decisions list"):
        return f
    by_id = {str(d.get("paper_id")): d for d in decisions if isinstance(d, dict)}
    f.require(candidate_ids <= set(by_id), "review.json does not account for every candidate")
    kept = {pid for pid, d in by_id.items() if d.get("decision") == "kept"}
    f.require(kept == set(shortlist_ids), "review.json kept papers differ from the shortlist")
    for pid, d in by_id.items():
        f.require(d.get("decision") in DECISIONS, f"review decision for {pid} is invalid")
        f.require(_text(d.get("reason")), f"review decision for {pid} has no reason")
        if d.get("decision") in ("unscored", "prefiltered"):
            f.require(
                d.get("relevance_score") is None and d.get("quality_score") is None,
                f"{pid} was not scored and must not carry scores",
            )
    return f


def check_card(card: Any, shortlist_ids: set[str], abstract: str | None = None) -> Findings:
    """One stored card. From schema 2 on, its quotes must be found in ``abstract``."""
    f = Findings()
    if not f.require(isinstance(card, dict), "card must be an object"):
        return f
    cid = card.get("card_id")
    pid = card.get("paper_id")
    f.require(
        _text(cid) and cid == card_id_for(str(pid)), f"card {cid}: card_id must be card-<paper_id>"
    )
    f.require(pid in shortlist_ids, f"card {cid}: paper {pid} is not in the shortlist")
    f.require(
        card.get("evidence_scope") in EVIDENCE_SCOPES,
        f"card {cid}: evidence_scope must be one of {list(EVIDENCE_SCOPES)}",
    )
    for key in CARD_FIELDS:
        present = key in card and (card[key] is None or _text(card[key]))
        f.require(present, f"card {cid}: field {key} must be text or null")
    if int(card.get("schema_version") or 1) >= QUOTED_CARD_SCHEMA:
        if abstract is None:
            f.error(f"card {cid}: the abstract of paper {pid} is not available to check quotes")
        else:
            for problem in check_card_quotes(card, abstract).errors:
                f.error(f"card {cid}: {problem}")
    return f


def _set_aside_ids(f: Findings, raw: Any, known_cards: set[str]) -> list[str]:
    """Cards the synthesis set aside, each entry with its reason."""
    if raw is None or raw == []:
        return []
    if not f.require(isinstance(raw, list), "set_aside must be a list"):
        return []
    ids: list[str] = []
    for i, item in enumerate(raw):
        ok = (
            isinstance(item, dict) and _str_list(item.get("card_ids")) and _text(item.get("reason"))
        )
        if f.require(ok, f"set_aside[{i}] needs card_ids and a reason"):
            f.require(
                set(item["card_ids"]) <= known_cards, f"set_aside[{i}] references unknown cards"
            )
            ids += item["card_ids"]
    return ids


def check_tensions(raw: Any, known_cards: set[str], cluster_ids: set[str]) -> Findings:
    """Each tension has an id, the clusters it lies between, and two sides that each state a
    finding and list the cards behind it; no card sits on both sides."""
    f = Findings()
    if raw is None or raw == []:
        return f
    if not f.require(isinstance(raw, list), "tensions must be a list"):
        return f
    seen: set[str] = set()
    for i, tension in enumerate(raw):
        if not f.require(isinstance(tension, dict), f"tensions[{i}] must be an object"):
            continue
        tid = tension.get("id")
        if f.require(
            isinstance(tid, str) and TENSION_ID.match(tid), f"tensions[{i}].id must look like X1"
        ):
            f.require(tid not in seen, f"duplicate tension id {tid}")
            seen.add(str(tid))
        tag = f"tension {tid}" if isinstance(tid, str) else f"tensions[{i}]"
        f.require(_text(tension.get("text")), f"{tag}: text is required")
        between = tension.get("between")
        if f.require(_str_list(between), f"{tag}: between must list the clusters it lies between"):
            unknown = sorted(set(between) - cluster_ids)
            f.require(not unknown, f"{tag}: between references unknown clusters {unknown}")
        sides = tension.get("sides")
        if not f.require(
            isinstance(sides, list) and len(sides) == 2, f"{tag}: sides must hold exactly two sides"
        ):
            continue
        cards: list[set[str]] = []
        for j, side in enumerate(sides, 1):
            ok = (
                isinstance(side, dict)
                and _text(side.get("claim"))
                and _str_list(side.get("card_ids"))
            )
            if not f.require(ok, f"{tag}: side {j} needs a claim and at least one card"):
                cards.append(set())
                continue
            unknown = sorted(set(side["card_ids"]) - known_cards)
            f.require(not unknown, f"{tag}: side {j} references unknown cards {unknown}")
            cards.append(set(side["card_ids"]))
        both = sorted(cards[0] & cards[1])
        f.require(not both, f"{tag}: cards {both} are on both sides")
    return f


def tension_ids(synthesis: dict[str, Any]) -> set[str]:
    """Ids of the synthesis tensions a hypothesis may settle (none in syntheses before ids)."""
    return {
        str(t["id"])
        for t in synthesis.get("tensions") or []
        if isinstance(t, dict) and isinstance(t.get("id"), str) and TENSION_ID.match(t["id"])
    }


def check_synthesis(
    synthesis: Any,
    known_sq: set[str],
    known_cards: set[str],
    source_text: str = "",
    *,
    every_card: bool = False,
    sided_tensions: bool = False,
) -> Findings:
    """``every_card``: each known card is in a cluster or set aside with a reason (stage 7
    accounts for every card it sent). ``sided_tensions``: tensions carry ids and two sides of
    cards (see :func:`check_tensions`)."""
    f = Findings()
    if not f.require(isinstance(synthesis, dict), "synthesis must be an object"):
        return f
    clusters = synthesis.get("clusters")
    placed: list[str] = []
    cluster_ids: set[str] = set()
    if f.require(isinstance(clusters, list) and clusters, "at least one cluster is required"):
        for i, cluster in enumerate(clusters):
            ok = (
                isinstance(cluster, dict)
                and _text(cluster.get("id"))
                and _text(cluster.get("title"))
            )
            if not f.require(ok, f"clusters[{i}] needs id and title"):
                continue
            cluster_ids.add(str(cluster["id"]))
            ids = cluster.get("card_ids")
            if f.require(_str_list(ids), f"clusters[{i}] needs card_ids"):
                f.require(set(ids) <= known_cards, f"clusters[{i}] references unknown cards")
                placed += ids
    if every_card:
        placed += _set_aside_ids(f, synthesis.get("set_aside"), known_cards)
        missing = sorted(known_cards - set(placed))
        if missing:
            f.error(
                f"{len(missing)} cards are in no cluster and not set aside: {', '.join(missing)}; "
                "put each card in exactly one cluster, or in set_aside with a reason"
            )
    twice = sorted({c for c in placed if placed.count(c) > 1})
    if twice:
        f.warn(f"cards placed more than once: {', '.join(twice)}")
    if sided_tensions:
        f.extend(check_tensions(synthesis.get("tensions"), known_cards, cluster_ids))
    gaps = synthesis.get("gaps")
    if f.require(
        isinstance(gaps, list) and len(gaps) >= MIN_GAPS,
        f"at least {MIN_GAPS} research gaps are required",
    ):
        seen: set[str] = set()
        for i, gap in enumerate(gaps):
            if not f.require(isinstance(gap, dict), f"gaps[{i}] must be an object"):
                continue
            if f.require(_text(gap.get("id")), f"gaps[{i}].id is required"):
                f.require(gap["id"] not in seen, f"duplicate gap id {gap['id']}")
                seen.add(str(gap["id"]))
            f.require(_text(gap.get("text")), f"gaps[{i}].text is required")
            sq = gap.get("sub_question_ids")
            if f.require(_str_list(sq), f"gaps[{i}] must reference sub-questions"):
                f.require(set(sq) <= known_sq, f"gaps[{i}] references unknown sub-questions")
            ids = gap.get("card_ids")
            if f.require(_str_list(ids), f"gaps[{i}] must be connected to evidence cards"):
                f.require(set(ids) <= known_cards, f"gaps[{i}] references unknown cards")
    if source_text:
        for number in unsupported_numbers(written_text(synthesis), source_text)[:5]:
            f.warn(f"synthesis mentions {number!r} which does not appear in any card")
    return f


_LEADING_NUMBER = re.compile(r"^\s*[±+]?\s*(\d+(?:\.\d+)?|\.\d+)(?![\d.])")


def normalise_margin(value: Any) -> Any:
    """An equivalence margin written as text that starts with its number (``"0.10 SD units"``,
    ``"±0.1"``) becomes that number; anything else is returned as given for the contract."""
    if isinstance(value, str) and (m := _LEADING_NUMBER.match(value)):
        return float(m.group(1))
    return value


def equivalence_problem(h: dict[str, Any], tag: str) -> str | None:
    """Why a negligible-effect hypothesis (``≈ 0``) cannot be tested as written: it needs a
    positive ``equivalence_margin``. None when it can, or when it predicts something else."""
    if normalise_prediction(h.get("prediction")) != EQUIVALENCE:
        return None
    margin = normalise_margin(h.get("equivalence_margin"))
    if isinstance(margin, int | float) and not isinstance(margin, bool) and margin > 0:
        return None
    return (
        f"{tag}: a negligible-effect prediction (≈ 0) needs equivalence_margin, a positive "
        "number in the outcome's unit written as a bare JSON number such as 0.1 (no text); "
        "without one it cannot be tested"
    )


def normalise_prediction(value: Any) -> Any:
    """Map common spellings of the allowed predictions to their canonical form."""
    if not isinstance(value, str):
        return value
    compact = re.sub(r"\s+", "", value)
    mapping = {">0": "> 0", "<0": "< 0", "≠0": "≠ 0", "!=0": "≠ 0", "=/=0": "≠ 0",
               "≈0": "≈ 0", "~0": "≈ 0", "~=0": "≈ 0"}  # fmt: skip
    return mapping.get(compact, value)


def check_hypotheses(
    data: Any,
    valid_gaps: set[str],
    valid_refs: set[str],
    valid_tensions: set[str] | None = None,
) -> Findings:
    """Validate the hypothesis list (``data`` is the parsed ``hypotheses.json`` object).

    ``valid_tensions``: the synthesis tension ids; each ``tension_ids`` entry must be one of them,
    and when there are any, at least one hypothesis must settle one.
    """
    tensions = valid_tensions or set()
    settled: set[str] = set()
    f = Findings()
    if not f.require(isinstance(data, dict), "hypotheses output must be an object"):
        return f
    items = data.get("hypotheses")
    if not f.require(
        isinstance(items, list) and len(items) >= MIN_HYPOTHESES,
        f"at least {MIN_HYPOTHESES} hypotheses are required",
    ):
        return f
    ids: set[str] = set()
    for i, h in enumerate(items):
        if not f.require(isinstance(h, dict), f"hypotheses[{i}] must be an object"):
            continue
        hid = h.get("id")
        if f.require(_text(hid), f"hypotheses[{i}].id is required"):
            f.require(hid not in ids, f"duplicate hypothesis id {hid}")
            ids.add(str(hid))
        tag = f"hypothesis {hid or i}"
        for key in ("statement", "outcome", "rationale", "novelty"):
            f.require(_text(h.get(key)), f"{tag}: {key} is required")
        f.require(
            h.get("gap_id") in valid_gaps, f"{tag}: gap_id must be one of {sorted(valid_gaps)}"
        )
        refs = h.get("evidence_refs")
        if f.require(_str_list(refs), f"{tag}: evidence_refs needs at least one reference"):
            unknown = sorted(set(refs) - valid_refs)
            f.require(not unknown, f"{tag}: evidence_refs {unknown} do not resolve")
        f.require(
            h.get("prediction") in PREDICTIONS,
            f"{tag}: prediction must be one of {list(PREDICTIONS)}",
        )
        problem = equivalence_problem(h, tag)
        if problem:
            f.error(problem)
        crit = h.get("falsification_criteria")
        if f.require(_text(crit), f"{tag}: falsification_criteria is required"):
            f.require(
                len(crit.strip()) >= 25 and _CONDITION_WORDS.search(crit),
                f"{tag}: falsification_criteria must state a concrete failing observation",
            )
        lim = h.get("limitations")
        f.require(_text(lim) or _str_list(lim), f"{tag}: limitations are required")
        named = h.get("tension_ids")
        if named not in (None, []) and f.require(
            _str_list(named), f"{tag}: tension_ids must be a list of tension ids"
        ):
            unknown = sorted(set(named) - tensions)
            f.require(
                not unknown,
                f"{tag}: tension_ids {unknown} are not tensions of the synthesis "
                f"(allowed: {sorted(tensions) or 'none'})",
            )
            settled |= set(named) & tensions
    if tensions and not settled:
        f.error(
            f"the synthesis lists tensions {sorted(tensions)}; at least one hypothesis must settle "
            "one of them and name it in tension_ids"
        )
    _check_distinct(f, items, "novelty")
    _check_distinct(f, items, "rationale")
    # The predicted directions follow the evidence; a set that all points one way is not a fault.
    return f


def _check_distinct(f: Findings, items: list[Any], key: str) -> None:
    texts = [
        (h.get("id"), _normalise_text(str(h.get(key, "")))) for h in items if isinstance(h, dict)
    ]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (a_id, a), (b_id, b) = texts[i], texts[j]
            if a and b and (a == b or SequenceMatcher(None, a, b).ratio() >= 0.92):
                f.error(f"hypotheses {a_id} and {b_id} repeat the same {key} text")


#: Stored fields the model did not write: a stamp's digits are not claims about the evidence.
_NOT_WRITTEN = frozenset({"schema_version", "topic", "generated_at"})


def written_text(value: Any) -> str:
    """The prose of a model answer: its strings, without ids, references or stamps."""
    if isinstance(value, dict):
        return " ".join(
            written_text(v)
            for k, v in value.items()
            if k not in _NOT_WRITTEN and k != "id" and not k.endswith(("_id", "_ids"))
        )
    if isinstance(value, list):
        return " ".join(written_text(v) for v in value)
    return value if isinstance(value, str) else ""


def unsupported_numbers(text: str, source_text: str) -> list[str]:
    """Numeric tokens in ``text`` (other than single digits) absent from ``source_text``."""
    pattern = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?(?![\w])")
    known = set(pattern.findall(source_text))
    seen: list[str] = []
    for token in pattern.findall(text):
        bare = token.rstrip("%")
        if token in known or bare in known or (bare.isdigit() and len(bare) == 1):
            continue
        if token not in seen:
            seen.append(token)
    return seen


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


# --- Stage 9: argument map -------------------------------------------------------------------

#: Each relation of the argument map, read from ``from`` to ``to``, with its domain and range.
MAP_VOCABULARY: dict[str, tuple[str, str]] = {
    "supports": ("evidence", "claim"),
    "contradicts": ("evidence", "claim"),
    "substantiates": ("claim", "gap"),
    "motivates": ("gap", "question"),
    "decomposes_into": ("question", "question"),
    "proposes_answer_to": ("hypothesis", "question"),
    "provides_rationale_for": ("claim", "hypothesis"),
    "depends_on": ("hypothesis", "assumption"),
    "addresses": ("hypothesis", "gap"),
    "informs": ("hypothesis", "contribution"),
    "targets": ("contribution", "gap"),
}
MAP_ENTITY_TYPES = (
    "gap", "question", "hypothesis", "claim", "evidence", "assumption", "contribution"
)  # fmt: skip
#: ``stated`` by a record of the run, ``derived`` from other relations, or a model judgement
#: nobody has reviewed.
RELATION_STATUSES = ("stated", "derived", "unreviewed")
EVIDENCE_RELATIONS = ("supports", "contradicts", "unrelated")
POLARITIES = ("supports", "challenges")
CANVAS_PIECES = (
    "puzzle", "audience", "question", "theory", "setting", "design", "findings", "contributions",
    "boundaries",
)  # fmt: skip


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def check_argument_map(
    data: Any, members: dict[str, set[str]], hypothesis_ids: set[str]
) -> Findings:
    """The model's judgements for the argument map.

    ``members`` maps each cluster (claim) id to its card ids. Every card is judged once against
    its own cluster's claim; every hypothesis gets at least one claim as its rationale.
    """
    f = Findings()
    if not f.require(isinstance(data, dict), "the argument map must be an object"):
        return f
    links = data.get("evidence_links")
    judged: set[tuple[str, str]] = set()
    if f.require(isinstance(links, list), "evidence_links must be a list"):
        for i, link in enumerate(links):
            if not isinstance(link, dict):
                f.error(f"evidence_links[{i}] must be an object")
                continue
            card, claim = str(link.get("card_id")), str(link.get("claim_id"))
            if card not in members.get(claim, set()):
                f.error(f"evidence_links[{i}]: {card} is not a card of cluster {claim}")
                continue
            if (card, claim) in judged:
                f.error(f"evidence_links[{i}]: {card} is judged twice against {claim}")
            judged.add((card, claim))
            f.require(
                link.get("relation") in EVIDENCE_RELATIONS,
                f"evidence_links[{i}].relation must be one of {', '.join(EVIDENCE_RELATIONS)}",
            )
            f.require(_text(link.get("rationale")), f"evidence_links[{i}].rationale is empty")
        missing = sorted(
            f"{card} in {claim}"
            for claim, cards in members.items()
            for card in cards
            if (card, claim) not in judged
        )
        f.require(not missing, f"cards not judged against their claim: {', '.join(missing)}")
    rationales = data.get("rationales")
    grounded: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    if f.require(isinstance(rationales, list), "rationales must be a list"):
        for i, r in enumerate(rationales):
            if not isinstance(r, dict):
                f.error(f"rationales[{i}] must be an object")
                continue
            claim, hyp = str(r.get("claim_id")), str(r.get("hypothesis_id"))
            known_claim = f.require(claim in members, f"rationales[{i}]: unknown claim {claim}")
            known_hyp = f.require(
                hyp in hypothesis_ids, f"rationales[{i}]: unknown hypothesis {hyp}"
            )
            if not (known_claim and known_hyp):
                continue
            if (claim, hyp) in pairs:
                f.error(f"rationales[{i}]: {claim} is tied to {hyp} twice")
            pairs.add((claim, hyp))
            grounded.add(hyp)
            f.require(
                r.get("polarity") in POLARITIES,
                f"rationales[{i}].polarity must be supports or challenges",
            )
            f.require(_text(r.get("rationale")), f"rationales[{i}].rationale is empty")
        ungrounded = sorted(hypothesis_ids - grounded)
        f.require(not ungrounded, f"hypotheses with no claim as rationale: {', '.join(ungrounded)}")
    return f


def check_novelty_queries(data: Any, hypothesis_ids: set[str]) -> Findings:
    """One keyword query per hypothesis for the novelty search.

    arXiv returns only papers holding every word of a query, so a query is a few plain words.
    """
    f = Findings()
    if not f.require(isinstance(data, dict), "the novelty queries must be an object"):
        return f
    rows = data.get("queries")
    if not f.require(isinstance(rows, list), "queries must be a list"):
        return f
    seen: set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            f.error(f"queries[{i}] must be an object")
            continue
        hyp, query = str(row.get("hypothesis_id")), row.get("query")
        if not f.require(hyp in hypothesis_ids, f"queries[{i}]: unknown hypothesis {hyp}"):
            continue
        if hyp in seen:
            f.error(f"queries[{i}]: {hyp} has more than one query")
        seen.add(hyp)
        if not f.require(_text(query), f"queries[{i}].query is empty"):
            continue
        words = str(query).split()
        f.require(
            NOVELTY_QUERY_WORDS[0] <= len(words) <= NOVELTY_QUERY_WORDS[1],
            f"queries[{i}]: '{query}' has {len(words)} words; write "
            f"{NOVELTY_QUERY_WORDS[0]}-{NOVELTY_QUERY_WORDS[1]} keywords",
        )
        f.require(
            not _QUERY_SYNTAX.search(str(query)),
            f"queries[{i}]: '{query}' must be plain words, without double quotes, field "
            "prefixes or AND/OR/NOT",
        )
    missing = sorted(hypothesis_ids - seen)
    f.require(not missing, f"hypotheses with no query: {', '.join(missing)}")
    return f


def check_novelty_judgements(data: Any, papers: dict[str, set[str]]) -> Findings:
    """The judge's verdict on each hypothesis against the papers it was given.

    ``papers`` maps each hypothesis id to the ids of the papers given for it. "tested" and
    "related" name at least one of those papers; "new" names none.
    """
    f = Findings()
    if not f.require(isinstance(data, dict), "the novelty judgements must be an object"):
        return f
    rows = data.get("judgements")
    if not f.require(isinstance(rows, list), "judgements must be a list"):
        return f
    seen: set[str] = set()
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            f.error(f"judgements[{i}] must be an object")
            continue
        hyp = str(row.get("hypothesis_id"))
        if not f.require(hyp in papers, f"judgements[{i}]: unknown hypothesis {hyp}"):
            continue
        if hyp in seen:
            f.error(f"judgements[{i}]: {hyp} is judged twice")
        seen.add(hyp)
        verdict = row.get("verdict")
        f.require(
            verdict in NOVELTY_VERDICTS,
            f"judgements[{i}].verdict must be one of {', '.join(NOVELTY_VERDICTS)}",
        )
        ids = row.get("paper_ids")
        if not f.require(
            isinstance(ids, list) and all(isinstance(p, str) for p in ids),
            f"judgements[{i}].paper_ids must be a list of paper ids",
        ):
            continue
        unknown = sorted(set(ids) - papers[hyp])
        f.require(
            not unknown,
            f"judgements[{i}]: {', '.join(unknown)} were not given for {hyp}; name only "
            f"{', '.join(sorted(papers[hyp]))}",
        )
        if verdict == "new":
            f.require(not ids, f"judgements[{i}]: a 'new' verdict names no paper")
        elif verdict in NOVELTY_VERDICTS:
            f.require(ids, f"judgements[{i}]: a '{verdict}' verdict names the paper(s)")
        f.require(_text(row.get("reason")), f"judgements[{i}].reason is empty")
    missing = sorted(set(papers) - seen)
    f.require(not missing, f"hypotheses not judged: {', '.join(missing)}")
    return f


def check_semantic_graph(graph: Any) -> Findings:
    """Entities and relations keep to the vocabulary; every relation carries its reasons."""
    f = Findings()
    if not f.require(isinstance(graph, dict), "the semantic graph must be an object"):
        return f
    entities = graph.get("entities")
    relations = graph.get("relations")
    if not f.require(isinstance(entities, list) and entities, "entities must be a non-empty list"):
        return f
    if not f.require(isinstance(relations, list), "relations must be a list"):
        return f
    types: dict[str, str] = {}
    for e in entities:
        if not isinstance(e, dict) or not _text(e.get("id")):
            f.error("every entity needs an id")
            continue
        if e["id"] in types:
            f.error(f"entity {e['id']} appears twice")
        f.require(e.get("type") in MAP_ENTITY_TYPES, f"entity {e['id']}: unknown type")
        types[e["id"]] = str(e.get("type"))
    seen: set[str] = set()
    for r in relations:
        if not isinstance(r, dict):
            f.error("every relation must be an object")
            continue
        rid = str(r.get("id"))
        if rid in seen:
            f.error(f"relation {rid} appears twice")
        seen.add(rid)
        rule = MAP_VOCABULARY.get(str(r.get("relation")))
        if rule is None:
            f.error(f"relation {rid}: {r.get('relation')!r} is not in the vocabulary")
            continue
        ends = (types.get(str(r.get("from"))), types.get(str(r.get("to"))))
        f.require(ends == rule, f"relation {rid} must read from {rule[0]} to {rule[1]}")
        f.require(r.get("status") in RELATION_STATUSES, f"relation {rid}: unknown status")
        f.require(
            _text(r.get("rationale")) and _text(r.get("provenance")),
            f"relation {rid} needs a rationale and a provenance",
        )
    return f


def check_research_canvas(canvas: Any) -> Findings:
    f = Findings()
    pieces = canvas.get("pieces") if isinstance(canvas, dict) else None
    if not f.require(isinstance(pieces, list), "the research canvas must list its pieces"):
        return f
    ids = [p.get("id") for p in pieces if isinstance(p, dict)]
    f.require(
        sorted(map(str, ids)) == sorted(CANVAS_PIECES),
        "the research canvas must hold each of its nine pieces once",
    )
    for p in pieces:
        if isinstance(p, dict) and p.get("status") == "filled":
            f.require(p.get("items"), f"canvas piece {p.get('id')} is filled but empty")
    return f


@dataclass(frozen=True)
class StageContract:
    stage: Stage
    inputs: tuple[tuple[Stage, str], ...]
    outputs: tuple[str, ...]
    dod: str
    error_code: str


CONTRACTS: dict[Stage, StageContract] = {
    Stage.TOPIC_INIT: StageContract(
        Stage.TOPIC_INIT,
        (),
        ("goal.json", "goal.md"),
        "Problem, objective, scope and success criteria; topic judged researchable",
        "INVALID_GOAL",
    ),
    Stage.PROBLEM_DECOMPOSE: StageContract(
        Stage.PROBLEM_DECOMPOSE,
        ((Stage.TOPIC_INIT, "goal.json"),),
        ("problem_tree.json", "problem_tree.md", "topic_evaluation.json"),
        ">=3 prioritised sub-questions linked to the goal; valid topic scores",
        "INVALID_PROBLEM_TREE",
    ),
    Stage.SEARCH_STRATEGY: StageContract(
        Stage.SEARCH_STRATEGY,
        ((Stage.PROBLEM_DECOMPOSE, "problem_tree.json"),),
        ("search_plan.yaml", "queries.json", "sources.json"),
        ">=2 strategies with non-empty queries linked to sub-questions",
        "INVALID_SEARCH_PLAN",
    ),
    Stage.LITERATURE_COLLECT: StageContract(
        Stage.LITERATURE_COLLECT,
        ((Stage.SEARCH_STRATEGY, "queries.json"),),
        ("candidates.jsonl", "references.bib", "search_meta.json"),
        "Real, deduplicated papers with provenance; per-source errors recorded",
        "NO_LITERATURE",
    ),
    Stage.LITERATURE_SCREEN: StageContract(
        Stage.LITERATURE_SCREEN,
        ((Stage.LITERATURE_COLLECT, "candidates.jsonl"),),
        ("shortlist.jsonl", "screen_meta.json", "review.json"),
        "Every paper has a decision, reason and (when scored) relevance/quality; "
        "non-empty shortlist",
        "EMPTY_SHORTLIST",
    ),
    Stage.KNOWLEDGE_EXTRACT: StageContract(
        Stage.KNOWLEDGE_EXTRACT,
        ((Stage.LITERATURE_SCREEN, "shortlist.jsonl"),),
        ("knowledge_meta.json",),
        "One evidence card per shortlisted paper with unknown fields marked null",
        "INVALID_CARDS",
    ),
    Stage.SYNTHESIS: StageContract(
        Stage.SYNTHESIS,
        ((Stage.KNOWLEDGE_EXTRACT, "knowledge_meta.json"),),
        ("synthesis.json", "synthesis.md"),
        "Clusters and >=2 gaps linked to sub-questions and cards",
        "INVALID_SYNTHESIS",
    ),
    Stage.HYPOTHESIS_GEN: StageContract(
        Stage.HYPOTHESIS_GEN,
        ((Stage.SYNTHESIS, "synthesis.json"),),
        ("hypotheses.json", "hypotheses.md"),
        ">=2 falsifiable hypotheses with gap, evidence references and falsification criteria",
        "INVALID_HYPOTHESES",
    ),
    Stage.ARGUMENT_MAP: StageContract(
        Stage.ARGUMENT_MAP,
        ((Stage.HYPOTHESIS_GEN, "hypotheses.json"), (Stage.SYNTHESIS, "synthesis.json")),
        ("argument_map.json", "semantic_graph.json", "research_canvas.json"),
        "Every clustered card judged against its claim and every hypothesis grounded in a claim; "
        "relations within the vocabulary, each with rationale, provenance and status; nine "
        "canvas pieces",
        "INVALID_ARGUMENT_MAP",
    ),
}


def missing_inputs(stage: Stage, artifacts: ArtifactStore) -> list[str]:
    """Upstream artifacts the stage needs that are absent."""
    return [
        f"stage-{int(src):02d}/{name}"
        for src, name in CONTRACTS[stage].inputs
        if not artifacts.exists(int(src), name)
    ]


# ---------------------------------------------------------------------------
# Loading helpers for on-disk validation
# ---------------------------------------------------------------------------


class _Loader:
    def __init__(self, art: ArtifactStore, f: Findings) -> None:
        self.art = art
        self.f = f

    def json(self, stage: Stage, name: str) -> Any:
        try:
            return self.art.read_json(int(stage), name)
        except (OSError, ValueError):
            self.f.error(f"stage {int(stage)}: {name} is missing or not valid JSON")
            return None

    def jsonl(self, stage: Stage, name: str) -> list[dict[str, Any]]:
        try:
            return self.art.read_jsonl(int(stage), name)
        except (OSError, ValueError):
            self.f.error(f"stage {int(stage)}: {name} is missing or not valid JSONL")
            return []

    def exists(self, stage: Stage, name: str) -> bool:
        ok = self.art.exists(int(stage), name)
        if not ok:
            self.f.error(f"stage {int(stage)}: {name} is missing")
        return ok


def _tree_ids(ld: _Loader) -> set[str]:
    tree = ld.json(Stage.PROBLEM_DECOMPOSE, "problem_tree.json")
    return sub_question_ids(tree) if isinstance(tree, dict) else set()


def _card_rows(ld: _Loader) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage = int(Stage.KNOWLEDGE_EXTRACT)
    for name in ld.art.list_files(stage):
        if name.startswith("cards/") and name.endswith(".json"):
            try:
                rows.append(ld.art.read_json(stage, name))
            except (OSError, ValueError):
                ld.f.error(f"stage {stage}: {name} is not valid JSON")
    return rows


# ---------------------------------------------------------------------------
# On-disk validation per stage
# ---------------------------------------------------------------------------


def _v_topic_init(ld: _Loader, f: Findings) -> None:
    goal = ld.json(Stage.TOPIC_INIT, "goal.json")
    if goal is not None:
        f.extend(check_goal(goal))
    ld.exists(Stage.TOPIC_INIT, "goal.md")
    if ld.art.exists(1, "hardware_profile.json"):
        f.require(
            isinstance(ld.json(Stage.TOPIC_INIT, "hardware_profile.json"), dict),
            "hardware_profile.json must be an object",
        )


def _v_problem_decompose(ld: _Loader, f: Findings) -> None:
    tree = ld.json(Stage.PROBLEM_DECOMPOSE, "problem_tree.json")
    if tree is not None:
        f.extend(check_problem_tree(tree))
    evaluation = ld.json(Stage.PROBLEM_DECOMPOSE, "topic_evaluation.json")
    if evaluation is not None:
        f.extend(check_topic_evaluation(evaluation))
    ld.exists(Stage.PROBLEM_DECOMPOSE, "problem_tree.md")


def _v_search_strategy(ld: _Loader, f: Findings) -> None:
    known = _tree_ids(ld)
    try:
        plan = yaml.safe_load(ld.art.read_text(int(Stage.SEARCH_STRATEGY), "search_plan.yaml"))
    except (OSError, yaml.YAMLError):
        f.error("stage 3: search_plan.yaml is missing or invalid YAML")
        plan = None
    if plan is not None:
        f.extend(check_search_plan(plan, known))
    queries = ld.json(Stage.SEARCH_STRATEGY, "queries.json")
    if queries is not None:
        f.extend(check_queries(queries, known))
    sources = ld.json(Stage.SEARCH_STRATEGY, "sources.json")
    if sources is not None:
        f.require(
            isinstance(sources, dict)
            and isinstance(sources.get("sources"), list)
            and sources["sources"],
            "sources.json needs a non-empty sources list",
        )


def _v_literature_collect(ld: _Loader, f: Findings) -> None:
    rows = ld.jsonl(Stage.LITERATURE_COLLECT, "candidates.jsonl")
    f.extend(check_candidates(rows))
    meta = ld.json(Stage.LITERATURE_COLLECT, "search_meta.json")
    if meta is not None:
        f.require(
            isinstance(meta, dict) and isinstance(meta.get("per_source"), dict),
            "search_meta.json must record per-source results",
        )
    if ld.exists(Stage.LITERATURE_COLLECT, "references.bib") and rows:
        bib = ld.art.read_text(int(Stage.LITERATURE_COLLECT), "references.bib")
        f.require(
            len(re.findall(r"^@\w+\{", bib, re.MULTILINE)) == len(rows),
            "references.bib entries do not match the candidates",
        )


def _v_literature_screen(ld: _Loader, f: Findings) -> None:
    candidates = ld.jsonl(Stage.LITERATURE_COLLECT, "candidates.jsonl")
    shortlist = ld.jsonl(Stage.LITERATURE_SCREEN, "shortlist.jsonl")
    review = ld.json(Stage.LITERATURE_SCREEN, "review.json")
    ld.json(Stage.LITERATURE_SCREEN, "screen_meta.json")
    if review is not None:
        f.extend(check_screen(candidates, shortlist, review))


def _v_knowledge_extract(ld: _Loader, f: Findings) -> None:
    shortlist = ld.jsonl(Stage.LITERATURE_SCREEN, "shortlist.jsonl")
    ids = {str(r["paper_id"]) for r in shortlist}
    abstracts = {str(r["paper_id"]): str(r.get("abstract") or "") for r in shortlist}
    cards = _card_rows(ld)
    if not f.require(cards, "no knowledge cards were produced"):
        return
    seen_papers: set[str] = set()
    for card in cards:
        f.extend(check_card(card, ids, abstracts.get(str(card.get("paper_id")))))
        pid = str(card.get("paper_id"))
        f.require(pid not in seen_papers, f"more than one card for paper {pid}")
        seen_papers.add(pid)
        f.require(
            ld.art.exists(int(Stage.KNOWLEDGE_EXTRACT), f"cards/{card.get('card_id')}.md"),
            f"card {card.get('card_id')} has no markdown rendering",
        )
    ld.json(Stage.KNOWLEDGE_EXTRACT, "knowledge_meta.json")


def card_source_text(cards: Iterable[dict[str, Any]]) -> str:
    return " ".join(str(v) for c in cards for k, v in c.items() if k in CARD_FIELDS and v)


def _v_synthesis(ld: _Loader, f: Findings) -> None:
    synthesis = ld.json(Stage.SYNTHESIS, "synthesis.json")
    ld.exists(Stage.SYNTHESIS, "synthesis.md")
    if synthesis is None:
        return
    cards = _card_rows(ld)
    card_ids = {str(c.get("card_id")) for c in cards}
    sided = int(synthesis.get("schema_version") or 1) >= SIDED_TENSION_SCHEMA
    f.extend(
        check_synthesis(
            synthesis, _tree_ids(ld), card_ids, card_source_text(cards), sided_tensions=sided
        )
    )


def hypothesis_reference_sets(
    synthesis: dict[str, Any], cards: Iterable[dict[str, Any]]
) -> tuple[set[str], set[str]]:
    """``(valid_gap_ids, valid_evidence_refs)``: gap ids, card ids and shortlisted paper ids."""
    gaps = {str(g["id"]) for g in synthesis.get("gaps", []) if isinstance(g, dict) and "id" in g}
    refs: set[str] = set()
    for card in cards:
        refs.add(str(card.get("card_id")))
        refs.add(str(card.get("paper_id")))
    return gaps, refs


def _v_hypothesis_gen(ld: _Loader, f: Findings) -> None:
    synthesis = ld.json(Stage.SYNTHESIS, "synthesis.json")
    data = ld.json(Stage.HYPOTHESIS_GEN, "hypotheses.json")
    ld.exists(Stage.HYPOTHESIS_GEN, "hypotheses.md")
    stage = int(Stage.HYPOTHESIS_GEN)
    f.require(
        any(n.startswith("perspectives/") for n in ld.art.list_files(stage)),
        "perspectives/ contains no perspective outputs",
    )
    if synthesis is None or data is None:
        return
    gaps, refs = hypothesis_reference_sets(synthesis, _card_rows(ld))
    f.extend(check_hypotheses(data, gaps, refs, tension_ids(synthesis)))
    if ld.art.exists(stage, "novelty_report.json"):
        report = ld.json(Stage.HYPOTHESIS_GEN, "novelty_report.json")
        f.require(
            isinstance(report, dict) and report.get("kind") == "novelty_assessment",
            "novelty_report.json must be labelled as a novelty assessment",
        )


def cluster_members(synthesis: dict[str, Any]) -> dict[str, set[str]]:
    return {
        str(c["id"]): {str(x) for x in c.get("card_ids", [])}
        for c in synthesis.get("clusters", [])
        if isinstance(c, dict) and "id" in c
    }


def _v_argument_map(ld: _Loader, f: Findings) -> None:
    synthesis = ld.json(Stage.SYNTHESIS, "synthesis.json")
    hypotheses = ld.json(Stage.HYPOTHESIS_GEN, "hypotheses.json")
    judged = ld.json(Stage.ARGUMENT_MAP, "argument_map.json")
    if isinstance(synthesis, dict) and isinstance(hypotheses, dict) and judged is not None:
        ids = {str(h.get("id")) for h in hypotheses.get("hypotheses", []) if isinstance(h, dict)}
        f.extend(check_argument_map(judged, cluster_members(synthesis), ids))
    graph = ld.json(Stage.ARGUMENT_MAP, "semantic_graph.json")
    if graph is not None:
        f.extend(check_semantic_graph(graph))
    canvas = ld.json(Stage.ARGUMENT_MAP, "research_canvas.json")
    if canvas is not None:
        f.extend(check_research_canvas(canvas))


_VALIDATORS: dict[Stage, Callable[[_Loader, Findings], None]] = {
    Stage.TOPIC_INIT: _v_topic_init,
    Stage.PROBLEM_DECOMPOSE: _v_problem_decompose,
    Stage.SEARCH_STRATEGY: _v_search_strategy,
    Stage.LITERATURE_COLLECT: _v_literature_collect,
    Stage.LITERATURE_SCREEN: _v_literature_screen,
    Stage.KNOWLEDGE_EXTRACT: _v_knowledge_extract,
    Stage.SYNTHESIS: _v_synthesis,
    Stage.HYPOTHESIS_GEN: _v_hypothesis_gen,
    Stage.ARGUMENT_MAP: _v_argument_map,
}


def validate_stage(stage: Stage, artifacts: ArtifactStore) -> Findings:
    """Validate the on-disk artifacts of ``stage`` against its contract."""
    findings = Findings()
    loader = _Loader(artifacts, findings)
    for name in CONTRACTS[stage].outputs:
        loader.exists(stage, name)
    if findings.ok:
        _VALIDATORS[stage](loader, findings)
    return findings
