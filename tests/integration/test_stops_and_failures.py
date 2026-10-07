"""Runs that must stop: no literature, nothing kept, bad topics, invalid or failing models."""

from __future__ import annotations

from pathlib import Path

from idea2hypothesis.llm.models import LLMResponseError, LLMTimeout
from idea2hypothesis.pipeline.models import RunStatus
from idea2hypothesis.pipeline.runner import run_pipeline
from tests.conftest import make_services, request
from tests.fixtures import FixtureLiterature, FixtureLLM


def _stage_dirs(services, run_id: str) -> set[str]:
    run_dir = services.store.run_dir(run_id)
    return {p.name for p in run_dir.glob("stage-*")}


async def test_empty_literature_stops_at_stage_4(tmp_path: Path) -> None:
    services = make_services(
        tmp_path,
        literature=FixtureLiterature(papers=[], source_errors={"openalex": "HTTP 503"}),
        review={"mode": "auto"},
    )
    result = await run_pipeline(request(), services)

    assert result.status is RunStatus.FAILED
    assert result.error is not None and result.error.code == "NO_LITERATURE"
    assert result.completed_stages == (1, 2, 3)
    dirs = _stage_dirs(services, result.run_id)
    assert "stage-05" not in dirs and "stage-06" not in dirs
    art = services.store.artifacts(result.run_id)
    assert not art.exists(4, "candidates.jsonl")  # no synthetic papers
    meta = art.read_json(4, "search_meta.json")
    assert meta["unique"] == 0 and meta["errors"]


async def test_rejecting_every_paper_stops_at_stage_5(tmp_path: Path) -> None:
    llm = FixtureLLM(keep_title=lambda title: False)
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)

    assert result.status is RunStatus.FAILED
    assert result.error is not None and result.error.code == "EMPTY_SHORTLIST"
    assert result.completed_stages == (1, 2, 3, 4)
    assert llm.count("knowledge_extract") == 0 and llm.count("synthesis") == 0
    assert "stage-06" not in _stage_dirs(services, result.run_id)
    review = services.store.artifacts(result.run_id).read_json(5, "review.json")
    assert review["summary"]["kept"] == 0
    assert all(d["reason"] for d in review["decisions"])


async def test_non_researchable_topic_stops_at_stage_1(tmp_path: Path) -> None:
    llm = FixtureLLM(
        overrides={
            "topic_init": {
                "researchable": False,
                "rejection_reason": "Casual conversation, not a research question.",
            }
        }
    )
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(topic="toi muon an com"), services)

    assert result.status is RunStatus.FAILED
    assert result.error is not None and result.error.code == "TOPIC_NOT_RESEARCHABLE"
    assert result.completed_stages == ()
    assert llm.count("problem_decompose") == 0
    goal = services.store.artifacts(result.run_id).read_json(1, "goal.json")
    assert goal["researchable"] is False


async def test_low_topic_score_stops_after_stage_2(tmp_path: Path) -> None:
    llm = FixtureLLM(
        overrides={
            "topic_evaluation": {
                "novelty": 2,
                "specificity": 3,
                "feasibility": 3,
                "overall": 9,
                "suggestion": "narrow it",
            }
        }
    )
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)

    assert result.status is RunStatus.FAILED
    assert result.error is not None and result.error.code == "TOPIC_BELOW_THRESHOLD"
    assert "narrow it" in result.error.message
    evaluation = services.store.artifacts(result.run_id).read_json(2, "topic_evaluation.json")
    assert evaluation["overall"] == 2.7  # recomputed from the three scores, not trusted


async def test_invalid_json_is_repaired_once_then_accepted(tmp_path: Path) -> None:
    llm = FixtureLLM(overrides={"problem_decompose": ["this is not json at all"]})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)

    assert result.status is RunStatus.COMPLETED
    assert llm.count("problem_decompose") == 2


async def test_persistently_invalid_json_fails_without_fabrication(tmp_path: Path) -> None:
    llm = FixtureLLM(overrides={"problem_decompose": ["nope", "still nope", "never json"]})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)

    assert result.status is RunStatus.FAILED
    assert result.error is not None and result.error.code == "LLM_OUTPUT_INVALID"
    assert not services.store.artifacts(result.run_id).exists(2, "problem_tree.json")


async def test_contract_violation_is_repaired_with_the_error_list(tmp_path: Path) -> None:
    too_few = {
        "sub_questions": [{"id": "SQ1", "text": "only one", "priority": 1, "goal_link": "x"}]
    }
    llm = FixtureLLM(overrides={"problem_decompose": [too_few]})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    assert llm.count("problem_decompose") == 2


async def test_transient_provider_error_retries_the_stage(tmp_path: Path) -> None:
    llm = FixtureLLM(overrides={"topic_init": [LLMTimeout("slow")]})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    assert llm.count("topic_init") == 2


async def test_provider_failure_after_retries_fails_the_run(tmp_path: Path) -> None:
    llm = FixtureLLM(overrides={"topic_init": [LLMTimeout("slow"), LLMTimeout("slow")]})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.FAILED
    assert result.error is not None and result.error.code == "LLM_TIMEOUT"


async def test_non_retryable_provider_error_is_not_retried(tmp_path: Path) -> None:
    llm = FixtureLLM(overrides={"topic_init": [LLMResponseError("HTTP 401", status=401)]})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.FAILED
    assert result.error is not None and result.error.code == "LLM_ERROR"
    assert llm.count("topic_init") == 1
