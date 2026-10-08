"""Stage progress: parts of a result are announced as they are persisted and reused on resume."""

from __future__ import annotations

from pathlib import Path

from idea2hypothesis.pipeline import events as ev
from idea2hypothesis.pipeline.models import RunStatus
from idea2hypothesis.pipeline.runner import resume_pipeline, run_pipeline
from tests.conftest import make_services, request
from tests.fixtures import FixtureLLM


def _progress(services, run_id: str, stage: int) -> list[dict]:
    return [
        e.data
        for e in services.store.read_events(run_id)
        if e.type == ev.STAGE_PROGRESS and e.stage == stage
    ]


async def test_each_part_is_announced_before_the_stage_completes(tmp_path: Path) -> None:
    services = make_services(tmp_path, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    events = services.store.read_events(result.run_id)
    art = services.store.artifacts(result.run_id)
    completed = {e.stage: e.seq for e in events if e.type == ev.STAGE_COMPLETED}

    cards = [d for d in _progress(services, result.run_id, 6) if d["kind"] == "card"]
    names = [n for n in art.list_files(6) if n.startswith("cards/") and n.endswith(".json")]
    assert sorted(d["card_id"] for d in cards) == sorted(n[6:-5] for n in names)
    assert [d["index"] for d in cards] == list(range(1, len(cards) + 1))
    for e in events:  # every announcement comes before its stage completes
        if e.type == ev.STAGE_PROGRESS:
            assert e.seq < completed[e.stage]
            assert e.data["stage_run"] > 0 and e.data["try"] == 0

    batches = [d for d in _progress(services, result.run_id, 5) if d["kind"] == "screen_batch"]
    assert batches and {d["total"] for d in batches} == {len(batches)}
    screened = {d["paper_id"] for b in batches for d in b["decisions"]}
    review = art.read_json(5, "review.json")
    prefiltered = {d["paper_id"] for d in review["decisions"] if d["decision"] == "prefiltered"}
    assert screened | prefiltered == {d["paper_id"] for d in review["decisions"]}

    queries = [d for d in _progress(services, result.run_id, 4) if d["kind"] == "query"]
    used = art.read_json(4, "search_meta.json")["queries_used"]
    assert [d["text"] for d in queries] == [q["text"] for q in used]
    kinds = {d["kind"] for d in _progress(services, result.run_id, 8)}
    assert {"perspectives_plan", "perspective", "merge", "novelty"} <= kinds


async def test_shortlist_keeps_the_best_scored_papers_up_to_the_cap(tmp_path: Path) -> None:
    services = make_services(tmp_path, review={"mode": "auto"}, research={"max_shortlist": 2})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    art = services.store.artifacts(result.run_id)
    shortlist = art.read_jsonl(5, "shortlist.jsonl")
    review = art.read_json(5, "review.json")
    cut = [d for d in review["decisions"] if d["decision"] == "below_cutoff"]
    assert len(shortlist) == 2 and cut
    assert review["summary"]["below_cutoff"] == len(cut) and review["summary"]["kept"] == 2
    assert review["thresholds"]["max_shortlist"] == 2
    assert all("ranked below the 2 best-scored papers" in d["reason"] for d in cut)
    lowest_kept = min((r["relevance_score"], r["quality_score"]) for r in shortlist)
    assert all((d["relevance_score"], d["quality_score"]) <= lowest_kept for d in cut)
    cards = [n for n in art.list_files(6) if n.startswith("cards/") and n.endswith(".json")]
    assert len(cards) == 2  # only the shortlist is read


async def test_pause_while_reading_keeps_the_cards_already_written(tmp_path: Path) -> None:
    holder: dict = {}

    def pause_after_two_cards(info) -> None:
        if info.key == "knowledge_extract" and info.index == 1:
            holder["services"].control.request(holder["run_id"], "pause", "user")

    llm = FixtureLLM(on_call=pause_after_two_cards)
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"}, runtime={"concurrency": 1})
    holder.update(services=services, run_id="i2h-test-cards")
    paused = await run_pipeline(request(run_id="i2h-test-cards"), services)
    assert paused.status is RunStatus.PAUSED and 6 not in paused.completed_stages
    assert llm.count("knowledge_extract") == 2

    services.control.clear(paused.run_id)
    llm.on_call = None
    resumed = await resume_pipeline(paused.run_id, services)
    assert resumed.status is RunStatus.COMPLETED
    shortlist = services.store.artifacts(paused.run_id).read_jsonl(5, "shortlist.jsonl")
    assert llm.count("knowledge_extract") == len(shortlist)  # the two cards were not redone

    second = [
        d
        for d in _progress(services, paused.run_id, 6)
        if d["stage_run"] == max(x["stage_run"] for x in _progress(services, paused.run_id, 6))
    ]
    plan = next(d for d in second if d["kind"] == "cards_plan")
    assert plan["cached"] == 2
    assert len([d for d in second if d["kind"] == "card"]) == len(shortlist)


async def test_a_restarted_stage_announces_the_restart(tmp_path: Path) -> None:
    from idea2hypothesis.llm.models import LLMTimeout

    state = {"failed": False}

    def fail_once(info) -> None:
        if info.key == "synthesis" and not state["failed"]:
            state["failed"] = True
            raise LLMTimeout("transient")

    llm = FixtureLLM(on_call=fail_once)
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    restarts = [d for d in _progress(services, result.run_id, 7) if d["kind"] == "restart"]
    assert restarts == [{"kind": "restart", "stage_run": restarts[0]["stage_run"], "try": 1}]
