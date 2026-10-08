"""Review gates, rejection reruns, pause/cancel, resume and crash recovery."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from idea2hypothesis.pipeline.control import RunBusyError
from idea2hypothesis.pipeline.gates import GateError
from idea2hypothesis.pipeline.models import GateAnswer, RunStatus
from idea2hypothesis.pipeline.runner import (
    RunStateError,
    answer_gate,
    apply_gate_answer,
    cancel_run,
    pause_run,
    recover_interrupted,
    resume_pipeline,
    run_pipeline,
)
from tests.conftest import make_services, request
from tests.fixtures import FixtureLLM


def _types(services, run_id: str) -> list[str]:
    return [e.type for e in services.store.read_events(run_id)]


async def test_auto_mode_has_no_gate(tmp_path: Path) -> None:
    services = make_services(tmp_path, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    assert "gate.opened" not in _types(services, result.run_id)


async def test_light_mode_has_no_gate_but_reports_advisories(tmp_path: Path) -> None:
    services = make_services(tmp_path, review={"mode": "light"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    events = services.store.read_events(result.run_id)
    assert "gate.opened" not in [e.type for e in events]
    completed = [e for e in events if e.type == "stage.completed"]
    assert all("advisories" in e.data for e in completed)
    assert any(e.data["advisories"] for e in completed)  # e.g. fewer than 20 unique papers


async def test_copilot_opens_screen_gate_and_waits(tmp_path: Path) -> None:
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, review={"mode": "copilot"})
    result = await run_pipeline(request(), services)

    assert result.status is RunStatus.AWAITING_REVIEW
    assert result.gate is not None and result.gate["kind"] == "screen"
    assert result.completed_stages == (1, 2, 3, 4, 5)
    assert llm.count("knowledge_extract") == 0
    opened = [e for e in services.store.read_events(result.run_id) if e.type == "gate.opened"]
    assert len(opened) == 1
    assert opened[0].data["gate_id"] == result.gate["gate_id"]
    assert len(opened[0].data["shortlist"]) == 9


async def test_resume_while_gate_unanswered_stays_awaiting_review(tmp_path: Path) -> None:
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, review={"mode": "copilot"})
    first = await run_pipeline(request(), services)
    calls_before = len(llm.calls)
    again = await resume_pipeline(first.run_id, services)
    assert again.status is RunStatus.AWAITING_REVIEW
    assert len(llm.calls) == calls_before
    assert _types(services, first.run_id).count("gate.opened") == 1


async def test_approving_the_gate_continues_to_completion(tmp_path: Path) -> None:
    services = make_services(tmp_path, review={"mode": "copilot"})
    first = await run_pipeline(request(), services)
    result = await answer_gate(first.run_id, first.gate["gate_id"], GateAnswer("approve"), services)

    assert result.status is RunStatus.COMPLETED
    assert result.completed_stages == (1, 2, 3, 4, 5, 6, 7, 8, 9)
    types = _types(services, first.run_id)
    assert types.index("gate.resolved") < types.index("run.completed")


async def test_approving_with_dropped_papers_excludes_them(tmp_path: Path) -> None:
    services = make_services(tmp_path, review={"mode": "copilot"})
    first = await run_pipeline(request(), services)
    art = services.store.artifacts(first.run_id)
    shortlist = art.read_jsonl(5, "shortlist.jsonl")
    dropped = shortlist[0]["paper_id"]

    result = await answer_gate(
        first.run_id,
        first.gate["gate_id"],
        GateAnswer("approve", (dropped,), "off focus"),
        services,
    )

    assert result.status is RunStatus.COMPLETED
    assert dropped not in {r["paper_id"] for r in art.read_jsonl(5, "shortlist.jsonl")}
    assert dropped not in {
        c["paper_id"]
        for c in (
            art.read_json(6, f"cards/{p.name}") for p in (art.stage_dir(6) / "cards").glob("*.json")
        )
    }
    review = art.read_json(5, "review.json")
    assert review["human_review"]["dropped"] == [dropped]
    dropped_decision = next(d for d in review["decisions"] if d["paper_id"] == dropped)
    assert dropped_decision["decision"] == "dropped_by_reviewer"


async def test_dropping_every_paper_fails_the_run(tmp_path: Path) -> None:
    services = make_services(tmp_path, review={"mode": "copilot"})
    first = await run_pipeline(request(), services)
    art = services.store.artifacts(first.run_id)
    everyone = tuple(r["paper_id"] for r in art.read_jsonl(5, "shortlist.jsonl"))
    result = await answer_gate(
        first.run_id, first.gate["gate_id"], GateAnswer("approve", everyone), services
    )
    assert result.status is RunStatus.FAILED
    assert result.error is not None and result.error.code == "EMPTY_SHORTLIST"
    assert "stage-06" not in {p.name for p in services.store.run_dir(first.run_id).glob("stage-*")}


async def test_gate_answer_validation(tmp_path: Path) -> None:
    services = make_services(tmp_path, review={"mode": "copilot"})
    first = await run_pipeline(request(), services)
    gate_id = first.gate["gate_id"]
    with pytest.raises(GateError):
        await apply_gate_answer(first.run_id, "wrong-gate", GateAnswer("approve"), services)
    with pytest.raises(GateError):
        await apply_gate_answer(first.run_id, gate_id, GateAnswer("maybe"), services)
    with pytest.raises(GateError):
        await apply_gate_answer(
            first.run_id, gate_id, GateAnswer("approve", ("p-unknown",)), services
        )
    await answer_gate(first.run_id, gate_id, GateAnswer("approve"), services)
    with pytest.raises(GateError):
        await apply_gate_answer(first.run_id, gate_id, GateAnswer("approve"), services)


async def test_rejecting_the_screen_gate_reruns_from_stage_3_with_a_new_attempt(
    tmp_path: Path,
) -> None:
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, review={"mode": "copilot"})
    first = await run_pipeline(request(), services)
    old_queries = services.store.artifacts(first.run_id).read_json(3, "queries.json")

    result = await answer_gate(
        first.run_id,
        first.gate["gate_id"],
        GateAnswer("reject", note="too many unrelated fields"),
        services,
    )

    assert result.attempt == 2
    assert result.status is RunStatus.AWAITING_REVIEW  # a fresh gate for attempt 2
    assert result.gate["gate_id"].endswith("-a2")
    assert (
        llm.count("topic_init") == 1 and llm.count("problem_decompose") == 1
    )  # stages 1-2 not rerun
    assert llm.count("search_strategy") == 2
    run_dir = services.store.run_dir(first.run_id)
    assert (run_dir / "attempts" / "1" / "stage-05").is_dir()
    assert (run_dir / "attempts" / "1" / "stage-03").is_dir()
    assert not (run_dir / "stage-06").exists()
    # the second search_strategy prompt carries the reviewer feedback
    second_prompt = [c for c in llm.calls if c.key == "search_strategy"][1].user
    assert "too many unrelated fields" in second_prompt
    assert "Previous queries" in second_prompt
    archived = services.store.run_dir(first.run_id) / "attempts" / "1" / "stage-03" / "queries.json"
    assert json.loads(archived.read_text(encoding="utf-8")) == old_queries
    manifest = services.store.artifacts(first.run_id).read_manifest(5)
    assert manifest is not None and manifest["attempt"] == 2


async def test_full_mode_opens_scope_gate_after_stage_2(tmp_path: Path) -> None:
    services = make_services(tmp_path, review={"mode": "full"})
    first = await run_pipeline(request(), services)
    assert first.status is RunStatus.AWAITING_REVIEW
    assert first.gate["kind"] == "scope" and first.gate["stage"] == 2
    assert first.completed_stages == (1, 2)

    second = await answer_gate(first.run_id, first.gate["gate_id"], GateAnswer("approve"), services)
    assert second.status is RunStatus.AWAITING_REVIEW
    assert second.gate["kind"] == "screen"

    done = await answer_gate(second.run_id, second.gate["gate_id"], GateAnswer("approve"), services)
    assert done.status is RunStatus.COMPLETED
    assert _types(services, first.run_id).count("gate.opened") == 2


async def test_rejecting_the_scope_gate_reruns_from_stage_1(tmp_path: Path) -> None:
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, review={"mode": "full"})
    first = await run_pipeline(request(), services)
    result = await answer_gate(
        first.run_id,
        first.gate["gate_id"],
        GateAnswer("reject", note="narrow the population"),
        services,
    )
    assert result.attempt == 2
    assert result.gate["kind"] == "scope"
    assert llm.count("topic_init") == 2
    assert "narrow the population" in [c for c in llm.calls if c.key == "topic_init"][1].user
    assert (services.store.run_dir(first.run_id) / "attempts" / "1" / "stage-01").is_dir()


# -- crash, resume, pause, cancel ---------------------------------------------


async def test_resume_after_crash_does_not_rerun_completed_stages(tmp_path: Path) -> None:
    class Boom(BaseException):
        pass

    def crash(info) -> None:
        if info.key == "knowledge_extract" and info.index == 2:
            raise Boom

    llm = FixtureLLM(on_call=crash)
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    with pytest.raises(Boom):  # the process "dies" in the middle of stage 6
        await run_pipeline(request(), services)

    run_id = services.store.list_run_ids()[0]
    assert services.store.read_run(run_id)["status"] == "running"
    recovered = await recover_interrupted(services)
    assert recovered == [run_id]
    record = services.store.read_run(run_id)
    assert record["status"] == "paused" and record["pause_reason"] == "interrupted"

    before = {
        k: llm.count(k)
        for k in ("topic_init", "problem_decompose", "search_strategy", "literature_screen")
    }
    result = await resume_pipeline(run_id, services)
    assert result.status is RunStatus.COMPLETED
    for key, count in before.items():
        assert llm.count(key) == count  # stages 1-5 were not repeated
    events = [e.seq for e in services.store.read_events(run_id)]
    assert events == list(range(1, len(events) + 1))


async def test_resume_after_completion_is_a_no_op(tmp_path: Path) -> None:
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    done = await run_pipeline(request(), services)
    calls = len(llm.calls)
    events = len(services.store.read_events(done.run_id))

    again = await resume_pipeline(done.run_id, services)
    assert again.status is RunStatus.COMPLETED
    assert again.completed_stages == (1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert len(llm.calls) == calls
    assert len(services.store.read_events(done.run_id)) == events


async def test_resume_reruns_stage_whose_artifacts_were_tampered_with(tmp_path: Path) -> None:
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    done = await run_pipeline(request(), services)
    art = services.store.artifacts(done.run_id)
    art.path(7, "synthesis.md").write_text("tampered", encoding="utf-8")
    services.store.update_run(done.run_id, status="paused")

    result = await resume_pipeline(done.run_id, services)
    assert result.status is RunStatus.COMPLETED
    assert llm.count("synthesis") == 2 and llm.count("topic_init") == 1


async def test_pause_at_a_safe_point_and_resume(tmp_path: Path) -> None:
    holder: dict = {}

    def pause_during_stage_4(info) -> None:
        if info.key == "problem_decompose":
            holder["services"].control.request(holder["run_id"], "pause", "user")

    llm = FixtureLLM(on_call=pause_during_stage_4)
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    holder["services"] = services
    holder["run_id"] = "i2h-test-pause"
    result = await run_pipeline(request(run_id="i2h-test-pause"), services)

    assert result.status is RunStatus.PAUSED and result.pause_reason == "user"
    assert "run.paused" in _types(services, result.run_id)
    assert result.completed_stages in ((1,), (1, 2))  # the pause lands at the next safe point

    holder["services"].control.clear(result.run_id)
    llm.on_call = None
    resumed = await resume_pipeline(result.run_id, services)
    assert resumed.status is RunStatus.COMPLETED
    assert "run.resumed" in _types(services, result.run_id)


async def test_cancel_during_a_run_is_final(tmp_path: Path) -> None:
    holder: dict = {}

    def cancel(info) -> None:
        if info.key == "search_strategy":
            holder["services"].control.request("i2h-test-cancel", "cancel")

    services = make_services(tmp_path, llm=FixtureLLM(on_call=cancel), review={"mode": "auto"})
    holder["services"] = services
    result = await run_pipeline(request(run_id="i2h-test-cancel"), services)
    assert result.status is RunStatus.CANCELLED
    assert "run.cancelled" in _types(services, result.run_id)
    with pytest.raises(RunStateError):
        await resume_pipeline(result.run_id, services)
    with pytest.raises(RunStateError):
        await cancel_run(result.run_id, services)


async def test_pause_and_cancel_without_an_active_worker(tmp_path: Path) -> None:
    services = make_services(tmp_path, review={"mode": "copilot"})
    first = await run_pipeline(request(), services)
    paused = await pause_run(first.run_id, services)
    assert paused.status is RunStatus.PAUSED
    cancelled = await cancel_run(first.run_id, services)
    assert cancelled.status is RunStatus.CANCELLED


async def test_one_worker_per_run(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def block(info) -> None:
        if info.key == "topic_init":
            started.set()
            await release.wait()

    services = make_services(tmp_path, llm=FixtureLLM(on_call=block), review={"mode": "auto"})
    task = asyncio.create_task(run_pipeline(request(run_id="i2h-test-lock"), services))
    await started.wait()
    with pytest.raises(RunBusyError):
        await resume_pipeline("i2h-test-lock", services)
    release.set()
    assert (await task).status is RunStatus.COMPLETED


async def test_budget_exceeded_pauses_at_a_stage_boundary(tmp_path: Path) -> None:
    services = make_services(tmp_path, llm=FixtureLLM(cost_per_call=1.0), review={"mode": "auto"})
    result = await run_pipeline(request(budget_usd=2.5), services)
    assert result.status is RunStatus.PAUSED and result.pause_reason == "budget_exceeded"
    assert result.usage["cost_usd"] >= 2.5
    services.store.update_run(result.run_id, budget_usd=None)
    resumed = await resume_pipeline(result.run_id, services)
    assert resumed.status is RunStatus.COMPLETED


async def test_platform_id_claim_is_idempotent(tmp_path: Path) -> None:
    services = make_services(tmp_path, review={"mode": "auto"})
    store = services.store
    assert store.claim_platform_id("plat-1", "run-a") == "run-a"
    assert store.claim_platform_id("plat-1", "run-b") == "run-a"
    result = await run_pipeline(request(run_id="run-a", platform_run_id="plat-1"), services)
    assert store.find_by_platform("plat-1") == result.run_id
    # a new store over the same root (simulated restart) sees the same mapping
    from idea2hypothesis.storage.runs import RunStore

    assert RunStore(services.config.runs_root).find_by_platform("plat-1") == "run-a"
