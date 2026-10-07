from __future__ import annotations

import asyncio

import pytest

from idea2hypothesis.pipeline.contracts import (
    check_hypotheses,
    normalise_prediction,
    unsupported_numbers,
)
from idea2hypothesis.resources.hardware import HardwareProfile, detect_hardware
from idea2hypothesis.stages import literature_screen as screen
from idea2hypothesis.stages.base import gather_limited
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
    assert "effect of sleep duration survey" in extra
    assert "exam performance in university students" in extra
    assert len({q.lower() for q in extra}) == len(extra)
    assert expand_queries(["a"], "short topic")[0] == "short topic survey"


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
    [(">0", "> 0"), ("< 0", "< 0"), ("!= 0", "≠ 0"), ("≠0", "≠ 0"), ("positive", "positive")],
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


def test_same_direction_portfolio_is_only_a_warning() -> None:
    hyps = [hypothesis(i) for i in (1, 2, 3)]
    result = check_hypotheses({"hypotheses": hyps}, {"G1"}, {"card-p-1"})
    assert result.ok and any("same direction" in w for w in result.warnings)


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
