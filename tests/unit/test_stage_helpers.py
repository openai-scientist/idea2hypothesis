from __future__ import annotations

import asyncio

import pytest

from idea2hypothesis.api.platform_events import _hypothesis_payload, _set_aside_events
from idea2hypothesis.pipeline.contracts import (
    check_card_quotes,
    check_hypotheses,
    check_novelty_judgements,
    check_novelty_queries,
    check_synthesis,
    check_tensions,
    normalise_margin,
    normalise_prediction,
    normalise_quote,
    unsupported_numbers,
)
from idea2hypothesis.resources.hardware import HardwareProfile, detect_hardware
from idea2hypothesis.stages import literature_screen as screen
from idea2hypothesis.stages.base import gather_limited
from idea2hypothesis.stages.hypothesis_debate import (
    Candidate,
    Objection,
    check_answers,
    check_critique,
    check_perspective,
    check_review,
)
from idea2hypothesis.stages.hypothesis_gen import check_merge
from idea2hypothesis.stages.literature_collect import expand_queries
from idea2hypothesis.stages.search_strategy import sanitize_queries, shorten_query
from idea2hypothesis.stages.synthesis import card_view


def test_shorten_query_keeps_keywords_and_suffix() -> None:
    long = (
        "A comprehensive empirical study of the effects of sleep duration "
        "on university exam performance benchmark"
    )
    short = shorten_query(long)
    assert short.endswith("benchmark") and len(short.split()) <= 7
    assert "the" not in short.lower().split()


def test_sanitize_queries_dedups_and_shortens() -> None:
    wordy = " ".join(f"keyword{i}" for i in range(20))
    out = sanitize_queries(["sleep exams", "Sleep Exams", "  ", wordy, "grades students"])
    assert out[0] == "sleep exams" and "Sleep Exams" not in out
    assert "grades students" in out and len(out) == 3
    assert len(out[1].split()) == 6  # shortened to six keywords


def test_expand_queries_adds_broader_variants_without_repeating_planned_ones() -> None:
    topic = "effect of sleep duration on exam performance in university students"
    extra = expand_queries(["effect of sleep duration"], topic)
    assert "effect sleep duration exam survey" in extra
    assert "duration exam performance university students" in extra
    assert len({q.lower() for q in extra}) == len(extra)
    assert expand_queries(["a"], "short topic")[0] == "short topic survey"


def test_novelty_queries_are_a_few_plain_words_for_every_hypothesis() -> None:
    ids = {"H1", "H2"}
    good = {"queries": [
        {"hypothesis_id": "H1", "query": "retrieval-augmented generation conflicting evidence"},
        {"hypothesis_id": "H2", "query": "Alzheimer's disease RAG"},
    ]}  # fmt: skip
    assert check_novelty_queries(good, ids).ok
    sentence = "On PubMedQA the retrieval effect will be greater for highly relevant passages"
    bad = {"queries": [
        {"hypothesis_id": "H1", "query": sentence},
        {"hypothesis_id": "H1", "query": "ti:rag AND medical"},
        {"hypothesis_id": "H9", "query": "sleep exams"},
    ]}  # fmt: skip
    errors = " ".join(check_novelty_queries(bad, ids).errors)
    assert "write 2-7 keywords" in errors and "AND/OR/NOT" in errors
    assert "H1 has more than one query" in errors and "unknown hypothesis H9" in errors
    assert "hypotheses with no query: H2" in errors


def test_novelty_judgements_name_only_the_papers_given() -> None:
    papers = {"H1": {"p-1", "p-2"}, "H2": {"p-3"}}
    good = {"judgements": [
        {"hypothesis_id": "H1", "verdict": "tested", "paper_ids": ["p-2"], "reason": "r"},
        {"hypothesis_id": "H2", "verdict": "new", "paper_ids": [], "reason": "r"},
    ]}  # fmt: skip
    assert check_novelty_judgements(good, papers).ok
    bad = {"judgements": [
        {"hypothesis_id": "H1", "verdict": "tested", "paper_ids": ["p-3"], "reason": "r"},
        {"hypothesis_id": "H1", "verdict": "related", "paper_ids": [], "reason": ""},
        {"hypothesis_id": "H9", "verdict": "new", "paper_ids": [], "reason": "r"},
    ]}  # fmt: skip
    errors = " ".join(check_novelty_judgements(bad, papers).errors)
    assert "p-3 were not given for H1" in errors and "H1 is judged twice" in errors
    assert "a 'related' verdict names the paper(s)" in errors and "reason is empty" in errors
    assert "unknown hypothesis H9" in errors and "hypotheses not judged: H2" in errors
    new_with_paper = {"judgements": [
        {"hypothesis_id": "H2", "verdict": "new", "paper_ids": ["p-3"], "reason": "r"},
    ]}  # fmt: skip
    assert "a 'new' verdict names no paper" in " ".join(
        check_novelty_judgements(new_with_paper, {"H2": {"p-3"}}).errors
    )


def test_expand_queries_leaves_out_the_words_of_a_question() -> None:
    topic = "Does retrieval-augmented generation reduce hallucinations, and when does it fail?"
    extra = expand_queries([], topic)
    assert "retrieval-augmented generation reduce hallucinations survey" in extra
    assert not any(w in q.lower().split() for q in extra for w in ("does", "and", "when", "it"))
    assert expand_queries([], "What is it?") == []


def rows(n: int, abstract: str = "sleep exam study") -> list[dict]:
    return [
        {"paper_id": f"p-{i}", "title": f"Paper {i} sleep", "abstract": abstract, "year": 2020}
        for i in range(n)
    ]


def test_prefilter_splits_by_keyword_overlap_and_sorts_by_overlap() -> None:
    candidates = [
        {"paper_id": "a", "title": "Quark physics", "abstract": "gluons"},
        {"paper_id": "b", "title": "Sleep", "abstract": "short"},
        {"paper_id": "c", "title": "Sleep and exam", "abstract": "exam sleep students"},
    ]
    keep, dropped = screen.prefilter(candidates, ["sleep", "exam", "students"])
    assert [r["paper_id"] for r in keep] == ["c", "b"]
    assert [r["paper_id"] for r in dropped] == ["a"]


def test_the_reviewer_sees_unknown_citations_as_null_not_zero() -> None:
    def row(*providers: str) -> dict:
        return {
            "paper_id": "p-1", "title": "t", "abstract": "a", "citation_count": 0,
            "source_records": [{"provider": p, "source_id": "x"} for p in providers],
        }  # fmt: skip

    assert screen._prompt_view(row("arxiv"))["citation_count"] is None
    assert screen._prompt_view(row("arxiv", "openalex"))["citation_count"] == 0
    assert screen._prompt_view(row("semantic_scholar"))["citation_count"] == 0
    legacy = {"paper_id": "p-2", "title": "t", "abstract": "a", "citation_count": 7}
    assert screen._prompt_view(legacy)["citation_count"] == 7


def test_prefilter_sets_aside_papers_without_an_abstract() -> None:
    candidates = [
        {"paper_id": "a", "title": "Sleep and exam meta-analysis", "abstract": None},
        {"paper_id": "b", "title": "Sleep and exam", "abstract": "  "},
        {"paper_id": "c", "title": "Sleep", "abstract": "exam sleep students"},
    ]
    keep, dropped = screen.prefilter(candidates, ["sleep", "exam"])
    assert [r["paper_id"] for r in keep] == ["c"]
    assert [r["paper_id"] for r in dropped] == ["a", "b"]
    assert "no abstract" in screen._prefiltered(dropped[0])["reason"]
    assert "keyword" in screen._prefiltered({"paper_id": "d", "abstract": "x"})["reason"]


def test_topic_keywords_include_domain_terms() -> None:
    keywords = screen.topic_keywords("Sleep and exams", ("Education", "psychology"))
    assert {"sleep", "exams", "education", "psychology"} <= set(keywords)


def test_batches_cover_every_paper_without_truncation() -> None:
    papers = rows(95, abstract="x" * 900)
    batches = screen.make_batches(papers)
    assert sum(len(b) for b in batches) == 95
    assert len(batches) > 1 and all(len(b) <= screen.MAX_BATCH_PAPERS for b in batches)
    assert [r["paper_id"] for b in batches for r in b] == [r["paper_id"] for r in papers]


def test_screen_batch_check_tolerates_a_few_missing_papers_but_not_many() -> None:
    expected = {f"p-{i}" for i in range(10)}

    def entry(pid: str) -> dict:
        return {
            "paper_id": pid,
            "decision": "keep",
            "relevance_score": 0.9,
            "quality_score": 0.8,
            "reason": "r",
        }

    nine = {"screened": [entry(f"p-{i}") for i in range(9)]}
    result = screen.check_screen_batch(nine, expected)
    assert result.ok and result.warnings
    half = {"screened": [entry(f"p-{i}") for i in range(5)]}
    assert not screen.check_screen_batch(half, expected).ok
    bad = {
        "screened": [{**entry("p-0"), "relevance_score": 1.5}]
        + [entry(f"p-{i}") for i in range(1, 10)]
    }
    assert any("relevance_score" in e for e in screen.check_screen_batch(bad, expected).errors)
    assert not screen.check_screen_batch({"screened": "nope"}, expected).ok


def test_unscored_papers_get_no_default_scores() -> None:
    decision = screen._decision({"paper_id": "p-1", "title": "T"}, None, 0.7, 0.5)
    assert decision["decision"] == "unscored"
    assert decision["relevance_score"] is None and decision["quality_score"] is None


def test_threshold_demotes_a_keep_and_records_why() -> None:
    entry = {"decision": "keep", "relevance_score": 0.6, "quality_score": 0.9, "reason": "ok"}
    decision = screen._decision({"paper_id": "p-1", "title": "T"}, entry, 0.7, 0.5)
    assert decision["decision"] == "rejected" and "below thresholds" in decision["reason"]


def test_card_view_drops_null_fields_and_truncates() -> None:
    card = {
        "card_id": "card-p-1",
        "title": "T",
        "year": 2020,
        "problem": "x" * 2000,
        "method": None,
        "data": None,
        "metrics": None,
        "findings": "f",
        "limitations": None,
    }
    view = card_view(card)
    assert set(view) == {"card_id", "title", "year", "problem", "findings"}
    assert len(view["problem"]) == 600


def test_unsupported_numbers_ignores_single_digits_and_numbers_in_sources() -> None:
    assert unsupported_numbers(
        "effect of 35% and 12.5 over 3 groups in 2020", "we saw 35% in 2020"
    ) == ["12.5"]


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        (">0", "> 0"),
        ("< 0", "< 0"),
        ("!= 0", "≠ 0"),
        ("≠0", "≠ 0"),
        ("~ 0", "≈ 0"),
        ("≈0", "≈ 0"),
        ("positive", "positive"),
    ],  # fmt: skip
)
def test_prediction_normalisation(raw: str, canonical: str) -> None:
    assert normalise_prediction(raw) == canonical


_MECHANISMS = {
    1: "Prefrontal synaptic plasticity during slow-wave sleep restores attention",
    2: "Circadian desynchrony weakens memory encoding after irregular schedules",
    3: "Adenosine accumulation impairs neural transmission during late study hours",
}
_GAPS = {
    1: "No within-subject design around exam weeks has been published",
    2: "Objective actigraphy has not replaced self-report in these cohorts",
    3: "The interaction of sleep debt and exam timing is untested",
}


def hypothesis(i: int, **overrides: object) -> dict:
    base = {
        "id": f"H{i}", "statement": f"claim {i}", "gap_id": "G1", "evidence_refs": ["card-p-1"],
        "outcome": "score in SD units", "prediction": "> 0",
        "falsification_criteria": "Wrong if the 95% confidence interval includes zero",
        "limitations": ["observational"], "rationale": _MECHANISMS[i],
        "novelty": _GAPS[i],
    }  # fmt: skip
    return {**base, **overrides}


def test_hypothesis_contract_accepts_a_good_set_and_flags_missing_pieces() -> None:
    gaps, refs = {"G1"}, {"card-p-1", "p-1"}
    good = {"hypotheses": [hypothesis(1), hypothesis(2, prediction="< 0")]}
    assert check_hypotheses(good, gaps, refs).ok

    cases = {
        "gap_id": hypothesis(2, gap_id="G9"),
        "do not resolve": hypothesis(2, evidence_refs=["card-nope"]),
        "falsification_criteria": hypothesis(2, falsification_criteria="results differ"),
        "prediction": hypothesis(2, prediction="bigger"),
        "limitations": hypothesis(2, limitations=[]),
        "outcome": hypothesis(2, outcome=""),
    }
    for needle, bad in cases.items():
        result = check_hypotheses({"hypotheses": [hypothesis(1), bad]}, gaps, refs)
        assert any(needle in e for e in result.errors), needle
    assert not check_hypotheses({"hypotheses": [hypothesis(1)]}, gaps, refs).ok


def test_hypotheses_must_have_distinct_novelty_and_rationale() -> None:
    same = {"novelty": "Isolates causal mechanisms under controlled conditions.",
            "rationale": "Derived from literature limitations."}  # fmt: skip
    result = check_hypotheses(
        {"hypotheses": [hypothesis(1, **same), hypothesis(2, **same)]}, {"G1"}, {"card-p-1"}
    )
    assert any("repeat the same novelty" in e for e in result.errors)
    assert any("repeat the same rationale" in e for e in result.errors)


def test_synthesis_must_account_for_every_card_it_was_given() -> None:
    gap = {"text": "t", "sub_question_ids": ["SQ1"], "card_ids": ["a"]}
    gaps = [{"id": "G1", **gap}, {"id": "G2", **gap}]
    syn: dict = {
        "clusters": [
            {"id": "C1", "title": "x", "card_ids": ["a", "b"]},
            {"id": "C2", "title": "y", "card_ids": ["b"]},
        ],
        "gaps": gaps,
    }
    cards = {"a", "b", "c", "d"}
    left = check_synthesis(syn, {"SQ1"}, cards, every_card=True)
    assert any("2 cards are in no cluster and not set aside: c, d" in e for e in left.errors)
    assert any("placed more than once: b" in w for w in left.warnings)
    assert not check_synthesis(syn, {"SQ1"}, cards).errors  # a finished run is not re-judged
    syn["set_aside"] = [{"id": "A1", "card_ids": ["d"]}]
    assert any(
        "set_aside[0] needs card_ids and a reason" in e
        for e in check_synthesis(syn, {"SQ1"}, cards, every_card=True).errors
    )
    syn["clusters"].append({"id": "C3", "title": "z", "card_ids": ["c"]})
    syn["set_aside"] = [{"id": "A1", "card_ids": ["d"], "reason": "studies another task"}]
    assert not check_synthesis(syn, {"SQ1"}, cards, every_card=True).errors


def tension(**overrides: object) -> dict:
    base = {
        "id": "X1", "between": ["C1", "C2"], "text": "do retrieved passages help small models?",
        "sides": [
            {"claim": "retrieval raises accuracy", "card_ids": ["a"]},
            {"claim": "retrieved context distracts small models", "card_ids": ["b"]},
        ],
    }  # fmt: skip
    return {**base, **overrides}


def test_a_tension_names_the_cards_on_each_of_its_two_sides() -> None:
    cards, clusters = {"a", "b", "c"}, {"C1", "C2"}
    assert check_tensions([tension()], cards, clusters).ok
    assert check_tensions([], cards, clusters).ok  # cards that agree have no tension
    one_side = [{"claim": "x", "card_ids": ["a"]}]
    cases = {
        "must look like X1": tension(id="T1"),
        "unknown clusters ['C9']": tension(between=["C1", "C9"]),
        "exactly two sides": tension(sides=one_side),
        "side 2 needs a claim and at least one card": tension(
            sides=[*one_side, {"claim": "y", "card_ids": []}]
        ),
        "side 2 references unknown cards ['z']": tension(
            sides=[*one_side, {"claim": "y", "card_ids": ["z"]}]
        ),
        "cards ['a'] are on both sides": tension(
            sides=[*one_side, {"claim": "y", "card_ids": ["a", "b"]}]
        ),
    }
    for needle, bad in cases.items():
        assert any(needle in e for e in check_tensions([bad], cards, clusters).errors), needle
    twice = check_tensions([tension(), tension()], cards, clusters)
    assert any("duplicate tension id X1" in e for e in twice.errors)


def test_synthesis_checks_tensions_only_when_asked() -> None:
    gap = {"text": "t", "sub_question_ids": ["SQ1"], "card_ids": ["a"]}
    syn = {
        "clusters": [{"id": "C1", "title": "x", "card_ids": ["a"]}],
        "gaps": [{"id": "G1", **gap}, {"id": "G2", **gap}],
        "tensions": [{"between": ["C1", "C2"], "text": "older runs have no ids"}],
    }
    assert check_synthesis(syn, {"SQ1"}, {"a", "b"}).ok
    sided = check_synthesis(syn, {"SQ1"}, {"a", "b"}, sided_tensions=True)
    assert any("must look like X1" in e for e in sided.errors)


def test_when_the_cards_disagree_a_hypothesis_must_settle_a_tension() -> None:
    gaps, refs = {"G1"}, {"card-p-1"}
    plain = {"hypotheses": [hypothesis(1), hypothesis(2)]}
    assert check_hypotheses(plain, gaps, refs).ok  # no tension, nothing to settle
    unsettled = check_hypotheses(plain, gaps, refs, {"X1", "X2"})
    assert any("at least one hypothesis must settle one" in e for e in unsettled.errors)
    settles = {"hypotheses": [hypothesis(1, tension_ids=["X1"]), hypothesis(2, tension_ids=[])]}
    assert check_hypotheses(settles, gaps, refs, {"X1", "X2"}).ok
    made_up = {"hypotheses": [hypothesis(1, tension_ids=["X7"]), hypothesis(2)]}
    result = check_hypotheses(made_up, gaps, refs, {"X1"})
    assert any("tension_ids ['X7'] are not tensions" in e for e in result.errors)


def test_every_challenge_gets_an_answer_that_does_what_it_says() -> None:
    before = [{"statement": "retrieval cuts errors"}, {"statement": "abstention rises"}]
    challenges = [
        {"from": "pragmatist", "response": 1, "hypothesis": 1, "text": "confounded"},
        {"from": "contrarian", "response": 2, "hypothesis": 2, "text": "cannot fail"},
    ]
    revised = [{"statement": "retrieval cuts unsupported answers"}, before[1]]

    def answers(*items: tuple[int, str]) -> list[dict]:
        return [{"challenge": n, "action": a, "text": "because"} for n, a in items]

    good = {"answers": answers((1, "revise"), (2, "withdraw")), "hypotheses": revised,
            "withdrawn": [2]}  # fmt: skip
    assert check_answers(good, before, challenges).ok
    # a revision may change only the field the challenge is about
    retested = [{**before[0], "falsification_criteria": "wrong if the CI includes 0"}, before[1]]
    assert check_answers({**good, "hypotheses": retested}, before, challenges).ok
    cases = {
        "challenges [2] have no answer": {"answers": answers((1, "revise")),
                                          "hypotheses": revised},
        "you revise hypothesis 1 but it is unchanged": {
            "answers": answers((1, "revise"), (2, "defend")), "hypotheses": before},
        "list 2 in withdrawn": {"answers": answers((1, "defend"), (2, "withdraw")),
                                "hypotheses": before},
        "answered more than once": {"answers": answers((1, "defend"), (1, "defend"), (2, "defend")),
                                    "hypotheses": before},
        "keep all 2 hypotheses": {"answers": answers((1, "defend"), (2, "defend")),
                                  "hypotheses": before[:1]},
        "action must be": {"answers": answers((1, "ignore"), (2, "defend")), "hypotheses": before},
    }  # fmt: skip
    for needle, bad in cases.items():
        assert any(needle in e for e in check_answers(bad, before, challenges).errors), needle


def test_a_fatal_challenge_names_its_flaw_and_where_it_is() -> None:
    previous = {"innovator": [{"statement": "a"}], "pragmatist": [{"statement": "b"}]}
    refs = {"card-p-1"}

    def critique(**challenge: object) -> list[str]:
        item = {
            "to": "pragmatist",
            "hypothesis": 1,
            "stance": "challenge",
            "text": "why",
            **challenge,
        }
        data = {"responses": [item]}
        return check_critique(data, previous, "innovator", {}, refs).errors

    assert not critique(severity="caveat")
    assert not critique(severity="fatal", flaw="unfalsifiable", field="falsification_criteria")
    assert any("severity must be one of" in e for e in critique())
    assert any("names its flaw" in e for e in critique(severity="fatal", field="statement"))
    assert any(
        "names the hypothesis field" in e for e in critique(severity="fatal", flaw="unsupported")
    )
    made_up = critique(severity="fatal", flaw="already_established", field="statement", card_id="x")
    assert any("already_established needs card_id" in e for e in made_up)
    assert not critique(
        severity="fatal", flaw="already_established", field="statement", card_id="card-p-1"
    )


def test_a_review_judges_every_answer_and_never_raises_a_caveat() -> None:
    caveat = Objection("innovator", 1, "contrarian", 1, "caveat", "small sample")
    fatal = Objection("innovator", 2, "contrarian", 1, "fatal", "no card", flaw="unsupported")
    items, refs = [caveat, fatal], {"card-p-1"}

    def review(*reviews: dict, added: list | None = None, new: list | None = None) -> list[str]:
        data = {"reviews": list(reviews), "added": added or []}
        return check_review(data, items, new or [], refs).errors

    ok = {"item": 1, "verdict": "resolved"}
    assert not review(ok, {"item": 2, "verdict": "stands", "text": "still no card"})
    assert not review(ok, {"item": 2, "verdict": "stands", "severity": "caveat", "text": "lower"})
    assert any("items [2] have no review" in e for e in review(ok))
    raised = review(
        {"item": 1, "verdict": "stands", "severity": "fatal", "text": "x"}, ok | {"item": 2}
    )
    assert any("cannot be raised to fatal" in e for e in raised)
    assert any("leaves unaddressed" in e for e in review(ok, {"item": 2, "verdict": "stands"}))
    added = [{"item": 1, "stance": "challenge", "severity": "fatal", "text": "no card"}]
    both = review(ok, ok | {"item": 2}, added=added, new=[("pragmatist", 4)])
    assert any("added item 1: a fatal challenge names its flaw" in e for e in both)


def test_the_final_set_uses_a_fatal_candidate_only_to_reach_the_minimum() -> None:
    blocked = Objection("innovator", 1, "contrarian", 1, "fatal", "no card", flaw="unsupported")
    candidates = {
        "innovator-1": Candidate("innovator-1", "innovator", 1, {}, fatal=[blocked]),
        "innovator-2": Candidate("innovator-2", "innovator", 2, {}),
        "pragmatist-1": Candidate("pragmatist-1", "pragmatist", 1, {}),
    }

    def merge(*sources: list[str], minimum: int = 2) -> list[str]:
        data = {"hypotheses": [{"id": f"H{i}", "from": s} for i, s in enumerate(sources, 1)]}
        return check_merge(data, candidates, minimum, 6).errors

    assert not merge(["innovator-2"], ["pragmatist-1"])
    leaked = merge(["innovator-2"], ["pragmatist-1"], ["innovator-1"])
    assert any("build on candidates whose fatal objection stands" in e for e in leaked)
    # with a minimum of 3 only 2 cleared candidates exist, so the fatal one may fill the set
    assert not merge(["innovator-2"], ["pragmatist-1"], ["innovator-1"], minimum=3)
    assert any(
        "from ['nobody-1'] are not candidates" in e for e in merge(["nobody-1"], ["innovator-2"])
    )
    assert any("write between 2 and 6" in e for e in merge(["innovator-2"]))
    assert any("must list the candidate ids" in e for e in merge([], ["innovator-2"]))


def test_a_merge_keeps_each_claim_and_every_cleared_candidate_left_out_says_why() -> None:
    def candidate(cid: str, prediction: str) -> Candidate:
        role, _, n = cid.partition("-")
        return Candidate(cid, role, int(n), {"prediction": prediction})

    candidates = {
        c.id: c
        for c in (
            candidate("pragmatist-1", "> 0"),
            candidate("contrarian-1", "< 0"),
            candidate("innovator-1", ">0"),
            candidate("innovator-2", "> 0"),
        )
    }

    def errors(hyps: list[dict], not_used: list[dict] | None = None, maximum: int = 6) -> str:
        data = {"hypotheses": hyps, "not_used": not_used or []}
        return " | ".join(check_merge(data, candidates, 2, maximum).errors)

    note = "pragmatist-1 gives the claim; innovator-1 gives the test"
    one = {
        "id": "H1",
        "from": ["pragmatist-1", "innovator-1"],
        "merge_note": note,
        "prediction": "> 0",
    }
    rest = [{"id": "H2", "from": ["contrarian-1"], "prediction": "< 0"},
            {"id": "H3", "from": ["innovator-2"], "prediction": "> 0"}]  # fmt: skip
    assert not errors([one, *rest])
    assert "say in merge_note what it takes from each" in errors(
        [{**one, "merge_note": " "}, *rest]
    )
    clash = {"id": "H1", "from": ["pragmatist-1", "contrarian-1"], "merge_note": note}
    assert "predict different effects" in errors(
        [clash, rest[1], {"id": "H3", "from": ["innovator-1"]}]
    )
    # innovator-2 is cleared and unused: it needs a reason
    assert "candidates ['innovator-2'] are not used" in errors([one, rest[0]])
    dup = {"candidate": "innovator-2", "reason": "duplicate", "of": "H1", "text": "same claim"}
    assert not errors([one, rest[0]], [dup])
    assert "names in 'of' the final hypothesis" in errors([one, rest[0]], [{**dup, "of": "H9"}])
    full = {**dup, "reason": "over_limit", "of": None}
    assert "there is room for innovator-2" in errors([one, rest[0]], [full])
    assert not errors([one, rest[0]], [full], maximum=2)
    assert "is used by the set" in errors([one, *rest], [{**dup, "candidate": "pragmatist-1"}])
    assert "say why in text" in errors([one, rest[0]], [{**dup, "text": ""}])
    # a duplicate repeats the claim of the hypothesis it names
    assert "it is not a duplicate" in errors([one, rest[0]], [{**dup, "of": "H2"}])


def test_a_benefit_and_a_negligible_effect_are_never_merged() -> None:
    candidates = {
        "pragmatist-3": Candidate("pragmatist-3", "pragmatist", 3, {"prediction": "> 0"}),
        "contrarian-3": Candidate("contrarian-3", "contrarian", 3, {"prediction": "≈0"}),
    }
    merged = {"id": "H1", "from": ["pragmatist-3", "contrarian-3"], "merge_note": "both"}
    data = {"hypotheses": [merged, {"id": "H2", "from": ["pragmatist-3"]}]}
    errors = check_merge(data, candidates, 2, 6).errors
    assert any("predict different effects" in e for e in errors)


def test_a_negligible_effect_keeps_its_margin_through_the_debate() -> None:
    before = [{"statement": "no lasting gain", "prediction": "≈ 0", "equivalence_margin": 0.1}]
    challenges = [{"hypothesis": 1, "text": "the cards do not fix 0.1"}]
    answer = {"answers": [{"challenge": 1, "action": "revise", "text": "removed the margin"}]}
    dropped = [{**before[0], "equivalence_margin": None}]
    errors = check_answers({**answer, "hypotheses": dropped}, before, challenges).errors
    assert any("without one it cannot be tested" in e for e in errors)
    widened = [{**before[0], "equivalence_margin": "0.2 SD"}]
    assert check_answers({**answer, "hypotheses": widened}, before, challenges).ok
    assert check_perspective({"hypotheses": [{"statement": "x", "prediction": "≈0"}]}).errors
    assert check_perspective({"hypotheses": [{"statement": "WITHDRAWN", "prediction": "≈ 0"}]}).ok


def test_negligible_effects_with_different_margins_are_never_merged() -> None:
    def null(cid: str, margin: object) -> Candidate:
        role, _, n = cid.partition("-")
        return Candidate(cid, role, int(n), {"prediction": "≈ 0", "equivalence_margin": margin})

    def errors(*margins: object) -> list[str]:
        ids = [f"role{i}-1" for i in range(len(margins))]
        candidates = {c: null(c, m) for c, m in zip(ids, margins, strict=True)}
        merged = {"id": "H1", "from": ids, "merge_note": "both"}
        data = {"hypotheses": [merged, {"id": "H2", "from": ids[:1]}]}
        return check_merge(data, candidates, 2, 6).errors

    assert any("different equivalence margins" in e for e in errors(0.1, 0.2))
    # the same margin, also when one is written as text
    assert not any("equivalence margins" in e for e in errors(0.1, "0.10 SD"))


def test_a_negligible_effect_with_another_margin_is_left_out_as_a_duplicate() -> None:
    candidates = {
        "innovator-2": Candidate("innovator-2", "innovator", 2, {"prediction": "≈ 0",
                                                                "equivalence_margin": 0.1}),
        "pragmatist-2": Candidate("pragmatist-2", "pragmatist", 2, {"prediction": "≈ 0",
                                                                  "equivalence_margin": "0.2 SD"}),
        "pragmatist-1": Candidate("pragmatist-1", "pragmatist", 1, {"prediction": "> 0"}),
    }  # fmt: skip
    final = [
        {"id": "H1", "from": ["innovator-2"], "prediction": "≈ 0", "equivalence_margin": 0.1},
        {"id": "H2", "from": ["pragmatist-1"], "prediction": "> 0"},
    ]
    dup = {"candidate": "pragmatist-2", "reason": "duplicate", "of": "H1",
           "text": "H1 tests the same null within ±0.1 rather than ±0.2"}  # fmt: skip
    assert check_merge({"hypotheses": final, "not_used": [dup]}, candidates, 2, 6).ok
    not_used = [{**dup, "hypothesis": candidates["pragmatist-2"].hypothesis}]
    events = _set_aside_events({"hypotheses": final, "not_used": not_used})
    assert events[0][1]["reason"].startswith(
        "Not used: H1 tests the same negligible effect within ±0.1 rather than ±0.2."
    )


def test_a_negligible_effect_prediction_needs_its_equivalence_margin() -> None:
    null = hypothesis(2, prediction="≈ 0", equivalence_margin=0.1)
    assert check_hypotheses({"hypotheses": [hypothesis(1), null]}, {"G1"}, {"card-p-1"}).ok
    for margin in (None, 0, -0.1, "about 0.1", True):
        bad = {**null, "equivalence_margin": margin}
        errors = check_hypotheses({"hypotheses": [hypothesis(1), bad]}, {"G1"}, {"card-p-1"}).errors
        assert any("needs equivalence_margin" in e for e in errors), margin


@pytest.mark.parametrize(
    ("raw", "margin"),
    [("0.10 baseline-SD units; effects within ±0.10 count", 0.1), ("±0.2", 0.2), (" .5 SD", 0.5),
     (0.1, 0.1), ("about 0.1", "about 0.1"), ("1.2.3", "1.2.3")],
)  # fmt: skip
def test_an_equivalence_margin_written_as_text_keeps_its_number(
    raw: object, margin: object
) -> None:
    assert normalise_margin(raw) == margin


def test_a_negligible_effect_is_wrong_outside_its_margin_on_the_platform() -> None:
    null = hypothesis(2, prediction="≈ 0", equivalence_margin=0.1, sub_question_ids=["SQ1"])
    falsify = _hypothesis_payload(null)["falsify"]
    assert falsify["zone"] == [None, None] and falsify["within"] == [-0.1, 0.1]
    assert "within" not in _hypothesis_payload(hypothesis(1))["falsify"]


def test_hypotheses_may_all_predict_the_same_direction() -> None:
    # The direction follows the evidence; a set that all points one way is not flagged.
    hyps = [hypothesis(i) for i in (1, 2, 3)]
    result = check_hypotheses({"hypotheses": hyps}, {"G1"}, {"card-p-1"})
    assert result.ok and not any("direction" in w for w in result.warnings)


async def test_gather_limited_bounds_concurrency_and_keeps_order() -> None:
    running = {"now": 0, "max": 0}

    async def job(i: int) -> int:
        running["now"] += 1
        running["max"] = max(running["max"], running["now"])
        await asyncio.sleep(0.01)
        running["now"] -= 1
        return i

    results = await gather_limited([lambda i=i: job(i) for i in range(8)], limit=3)
    assert results == list(range(8)) and running["max"] == 3


async def test_gather_limited_propagates_the_first_error_and_cancels_the_rest() -> None:
    finished: list[int] = []

    async def job(i: int) -> int:
        if i == 1:
            raise ValueError("boom")
        await asyncio.sleep(0.2)
        finished.append(i)
        return i

    with pytest.raises(ValueError, match="boom"):
        await gather_limited([lambda i=i: job(i) for i in range(4)], limit=4)
    assert finished == []


def test_hardware_profile_is_read_only_and_serialisable(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def no_gpu(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", no_gpu)
    monkeypatch.setattr("idea2hypothesis.resources.hardware.platform.system", lambda: "Linux")
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile) and profile.gpu_type == "cpu"
    assert profile.to_dict()["tier"] == "cpu_only"


def test_hardware_nvidia_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    class Result:
        returncode = 0
        stdout = "NVIDIA RTX 4090, 24564\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
    profile = detect_hardware()
    assert (profile.gpu_type, profile.vram_mb, profile.tier) == ("cuda", 24564, "high")


ABSTRACT = (
    "Label smoothing (LS) enhances model calibration by introducing <i>entropy</i> regularization "
    "during training. On CIFAR-10 it lowers the expected calibration error by 2.1 points."
)


def test_quotes_match_the_abstract_up_to_meaningless_differences() -> None:
    assert normalise_quote("“Label  smoothing (LS) enhances…”") == "label smoothing (ls) enhances"
    card = {
        "problem": None, "method": "Entropy regularization during training", "data": None,
        "metrics": None, "findings": "Lower ECE on CIFAR-10", "limitations": None,
        "quotes": {
            "method": ["introducing entropy regularization during training"],  # tags dropped
            "findings": ["On CIFAR-10 it lowers the expected calibration error by 2.1 points."],
        },
    }  # fmt: skip
    assert check_card_quotes(card, ABSTRACT).ok


def test_paraphrased_short_or_missing_quotes_reject_the_card() -> None:
    card = {
        "problem": "Calibration", "method": "Entropy regularization", "data": None,
        "metrics": None, "findings": "ECE drops by 3 points", "limitations": None,
        "quotes": {
            "method": ["improves calibration through entropy regularization"],
            "findings": ["lowers the error"],
            "data": ["On CIFAR-10 it lowers the expected calibration error"],
        },
    }  # fmt: skip
    errors = check_card_quotes(card, ABSTRACT).errors
    assert any("problem is filled but has no quote" in e for e in errors)
    assert any("method quote" in e and "word for word" in e for e in errors)
    assert any("findings quote" in e and "shorter than 4 words" in e for e in errors)
    assert any("data is null but has quotes" in e for e in errors)
    # A changed number is a different claim, not a meaningless difference.
    card = {**card, "problem": None, "data": None, "method": None}
    card["quotes"] = {"findings": ["lowers the expected calibration error by 3.1 points"]}
    assert not check_card_quotes(card, ABSTRACT).ok


def test_a_field_naming_four_benchmarks_apart_quotes_each_place() -> None:
    abstract = (
        "We evaluate on RoleBench in terms of role-dependent responses. We also report "
        "CharacterBench with regards to character customization performance. PersonaGym "
        "regarding persona-agent behavior is used, and a customized MPI test for personality."
    )
    quotes = [
        "RoleBench in terms of role-dependent responses",
        "CharacterBench with regards to character customization performance",
        "PersonaGym regarding persona-agent behavior",
        "a customized MPI test for personality",
    ]
    card = {
        "problem": None, "method": None, "data": "RoleBench, CharacterBench, PersonaGym, MPI",
        "metrics": None, "findings": None, "limitations": None, "quotes": {"data": quotes},
    }  # fmt: skip
    assert check_card_quotes(card, abstract).ok
    card["quotes"] = {"data": quotes * 2}
    assert any("give at most 6" in e for e in check_card_quotes(card, abstract).errors)
