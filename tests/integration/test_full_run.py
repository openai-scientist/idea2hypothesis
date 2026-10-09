"""Full run 1 -> 9 with fixture LLM and literature."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idea2hypothesis.pipeline.models import RunStatus
from idea2hypothesis.pipeline.runner import run_pipeline
from tests.conftest import make_services, request
from tests.fixtures import FixtureLLM

SPEC_ARTIFACTS = {
    1: ["goal.json", "goal.md", "hardware_profile.json"],
    2: ["problem_tree.json", "problem_tree.md", "topic_evaluation.json"],
    3: ["search_plan.yaml", "queries.json", "sources.json"],
    4: ["candidates.jsonl", "references.bib", "search_meta.json"],
    5: ["shortlist.jsonl", "screen_meta.json", "review.json"],
    6: ["knowledge_meta.json"],
    7: ["synthesis.json", "synthesis.md"],
    8: ["hypotheses.json", "hypotheses.md", "novelty_report.json"],
    9: ["argument_map.json", "semantic_graph.json", "research_canvas.json"],
}


@pytest.fixture
def auto_services(tmp_path: Path):
    return make_services(tmp_path, review={"mode": "auto"})


async def test_full_run_produces_all_spec_artifacts(auto_services) -> None:
    result = await run_pipeline(request(), auto_services)

    assert result.status is RunStatus.COMPLETED, result.error
    assert result.completed_stages == (1, 2, 3, 4, 5, 6, 7, 8, 9)

    run_dir = auto_services.store.run_dir(result.run_id)
    for name in (
        "run.json",
        "config.snapshot.json",
        "prompts.snapshot.json",
        "checkpoint.json",
        "events.jsonl",
    ):
        assert (run_dir / name).is_file(), name
    for stage, names in SPEC_ARTIFACTS.items():
        for name in names:
            assert (run_dir / f"stage-{stage:02d}" / name).is_file(), f"stage {stage}: {name}"
    assert list((run_dir / "stage-06" / "cards").glob("*.json"))
    assert list((run_dir / "stage-06" / "cards").glob("*.md"))
    assert list((run_dir / "stage-08" / "perspectives").glob("*.json"))
    assert (run_dir / "stage-08" / "manifest.json").is_file()


async def test_hypotheses_reference_real_gaps_and_cards(auto_services) -> None:
    result = await run_pipeline(request(), auto_services)
    art = auto_services.store.artifacts(result.run_id)
    hypotheses = art.read_json(8, "hypotheses.json")["hypotheses"]
    gaps = {g["id"] for g in art.read_json(7, "synthesis.json")["gaps"]}
    cards = {c.stem for c in (art.stage_dir(6) / "cards").glob("*.json")}
    assert len(hypotheses) >= 2
    for h in hypotheses:
        assert h["gap_id"] in gaps
        assert h["falsification_criteria"]
        assert set(h["evidence_refs"]) & cards


async def test_off_topic_papers_are_rejected_with_reasons(auto_services) -> None:
    result = await run_pipeline(request(), auto_services)
    art = auto_services.store.artifacts(result.run_id)
    review = art.read_json(5, "review.json")
    rejected = [d for d in review["decisions"] if d["decision"] == "rejected"]
    assert len(rejected) == 3
    assert all(d["reason"] and d["false_friend"] == "sleep" for d in rejected)
    shortlist_ids = {r["paper_id"] for r in art.read_jsonl(5, "shortlist.jsonl")}
    assert not shortlist_ids & {d["paper_id"] for d in rejected}


async def test_events_are_sequenced_and_complete(auto_services) -> None:
    result = await run_pipeline(request(), auto_services)
    events = auto_services.store.read_events(result.run_id)
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    types = [e.type for e in events]
    assert types[0] == "run.started" and types[-1] == "run.completed"
    assert types.count("stage.started") == 9 and types.count("stage.completed") == 9
    raw = [
        json.loads(line)
        for line in (auto_services.store.run_dir(result.run_id) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {
        "schema_version",
        "run_id",
        "seq",
        "stage",
        "type",
        "timestamp",
        "attempt",
        "data",
    } <= set(raw[0])


async def test_usage_is_tracked_and_cost_absent_without_pricing(auto_services) -> None:
    result = await run_pipeline(request(), auto_services)
    assert result.usage["calls"] > 5
    assert result.usage["prompt_tokens"] > 0
    assert result.usage["cost_usd"] is None


async def test_cost_is_reported_when_responses_are_priced(tmp_path: Path) -> None:
    from tests.fixtures import FixtureLLM

    services = make_services(tmp_path, llm=FixtureLLM(cost_per_call=0.01), review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.usage["cost_usd"] == pytest.approx(0.01 * result.usage["calls"])


async def test_every_model_call_is_logged_as_sent_and_as_answered(auto_services) -> None:
    result = await run_pipeline(request(), auto_services)
    art = auto_services.store.artifacts(result.run_id)
    for stage in (1, 2, 3, 5, 6, 7, 8, 9):
        calls = art.read_llm_calls(stage, 1)
        assert calls, f"stage {stage} has no call log"
        for call in calls:
            assert call["outcome"] == "accepted" and call["stage"] == stage
            assert call["request"]["messages"][0]["role"] == "user"
            assert call["response"]["text"] and call["response"]["model"] == "fixture-model"
    screen = art.read_llm_calls(5, 1)[0]
    assert screen["request"]["temperature"] == 0 and screen["label"].startswith("literature_screen")
    final = [c for c in art.read_llm_calls(8, 1) if c["label"] == "hypothesis_gen"]
    assert final[0]["request"]["temperature"] == auto_services.config.llm.temperature


async def test_extraction_and_judgement_reach_the_model_at_temperature_zero(tmp_path: Path) -> None:
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    await run_pipeline(request(), services)
    by_key: dict[str, set] = {}
    for call in llm.calls:
        by_key.setdefault(call.key, set()).add(call.temperature)
    for key in ("literature_screen", "knowledge_extract", "synthesis", "argument_map"):
        assert by_key[key] == {0}, key
    assert by_key["hypothesis_gen"] == {None}  # the configured temperature


async def test_a_repaired_answer_logs_the_rejected_round_with_its_reasons(tmp_path: Path) -> None:
    bad = {"problem": "Invented", "method": None, "data": None, "metrics": None,
           "findings": None, "limitations": None,
           "quotes": {"problem": ["not in any abstract text"]}}  # fmt: skip
    llm = FixtureLLM(overrides={"knowledge_extract": [bad]})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED, result.error
    calls = services.store.artifacts(result.run_id).read_llm_calls(6, 1)
    rejected = [c for c in calls if c["outcome"] == "rejected"]
    assert len(rejected) == 1 and "word for word" in " ".join(rejected[0]["problems"])
    repair = next(c for c in calls if c["label"] == rejected[0]["label"] and c["round"] == 2)
    assert repair["outcome"] == "accepted" and len(repair["request"]["messages"]) == 3


async def test_a_card_its_abstract_cannot_back_is_left_out_with_the_reason(tmp_path: Path) -> None:
    def answer(info):
        paper = info.section_json("Paper:\n")
        if paper["title"].startswith("Sleep duration and academic"):
            return {"problem": "Something else", "method": None, "data": None, "metrics": None,
                    "findings": None, "limitations": None,
                    "quotes": {"problem": ["words the abstract never had"]}}  # fmt: skip
        return FixtureLLM()._default_knowledge_extract(info)

    llm = FixtureLLM(overrides={"knowledge_extract": answer})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED, result.error
    art = services.store.artifacts(result.run_id)
    meta = art.read_json(6, "knowledge_meta.json")
    skipped = [s for s in meta["skipped"] if "word for word" in s["reason"]]
    assert len(skipped) == 1 and meta["cards"] == meta["shortlist_size"] - 1
    assert not art.exists(6, f"cards/card-{skipped[0]['paper_id']}.json")


async def test_the_screen_reviewer_reads_whole_abstracts(tmp_path: Path) -> None:
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    art = services.store.artifacts(result.run_id)
    review = art.read_json(5, "review.json")
    assert review["reviewer_view"]["abstract_max_chars"] >= 4000
    prompts = " ".join(c.user for c in llm.calls if c.key == "literature_screen")
    for row in art.read_jsonl(4, "candidates.jsonl"):
        assert row["abstract"] in prompts
