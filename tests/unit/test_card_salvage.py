from __future__ import annotations

from pathlib import Path

from idea2hypothesis.pipeline.models import RunStatus
from idea2hypothesis.pipeline.runner import run_pipeline
from idea2hypothesis.stages.knowledge_extract import _validate_card, salvage_card
from tests.conftest import make_services, request
from tests.fixtures import FixtureLLM

ABSTRACT = "We study sleep using a controlled design and report effects on exam outcomes."


def card(**over) -> dict:
    base = {
        "problem": "Sleep and exams", "method": "Controlled design", "data": None,
        "metrics": None, "findings": "Effects on exams", "limitations": None,
        "quotes": {
            "problem": ["We study sleep using a controlled design"],
            "method": ["using a controlled design"],
            "findings": ["report effects on exam outcomes"],
        },
    }  # fmt: skip
    base.update(over)
    return base


def test_a_short_metrics_quote_is_dropped_and_the_field_becomes_null() -> None:
    data = card(metrics="AUC", quotes={**card()["quotes"], "metrics": ["AUC"]})
    f = _validate_card(data, ABSTRACT)
    assert f.ok and data["metrics"] is None and "metrics" not in data["quotes"]
    assert len(f.warnings) == 1 and "metrics set to null" in f.warnings[0]


def test_an_unverified_quote_is_removed_but_a_verified_one_keeps_the_field() -> None:
    quotes = {
        **card()["quotes"],
        "findings": ["report effects on exam outcomes", "made up words here ok"],
    }
    data = card(quotes=quotes)
    f = _validate_card(data, ABSTRACT)
    assert f.ok and data["findings"] == "Effects on exams"
    assert data["quotes"]["findings"] == ["report effects on exam outcomes"]
    assert "1 unverified quote(s) removed" in f.warnings[0]


def test_quotes_of_a_null_field_are_dropped() -> None:
    data = card(quotes={**card()["quotes"], "data": ["a controlled design"]})
    f = _validate_card(data, ABSTRACT)
    assert f.ok and "data" not in data["quotes"] and len(f.warnings) == 1


def test_a_core_field_without_a_verified_quote_is_not_salvaged() -> None:
    data = card(quotes={**card()["quotes"], "findings": ["words the abstract never had"]})
    assert salvage_card(data, ABSTRACT) == [] or data["findings"] == "Effects on exams"
    f = _validate_card(data, ABSTRACT)
    assert not f.ok and data["findings"] == "Effects on exams"
    assert any("word for word" in e for e in f.errors)


async def test_a_salvaged_card_costs_no_retry_and_shows_the_warning(tmp_path: Path) -> None:
    def answer(info):
        base = FixtureLLM()._default_knowledge_extract(info)
        base["metrics"] = "AUC"
        base["quotes"]["metrics"] = ["AUC"]
        return base

    llm = FixtureLLM(overrides={"knowledge_extract": answer})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED, result.error
    art = services.store.artifacts(result.run_id)
    calls = art.read_llm_calls(6, 1)
    assert all(c["outcome"] == "accepted" for c in calls)
    cards = [
        art.read_json(6, n)
        for n in art.list_files(6)
        if n.startswith("cards/") and n.endswith(".json")
    ]
    assert cards and all(c["metrics"] is None and "metrics" not in c["quotes"] for c in cards)
    assert all("metrics set to null" in c["warnings"][0] for c in cards)
