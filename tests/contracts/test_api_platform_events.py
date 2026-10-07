"""Platform event stream: built only from real artifacts, in the shape the consumers read."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from idea2hypothesis.api.platform_events import GROUPS, PlatformEventLog
from idea2hypothesis.storage.runs import RunStore
from tests.fixtures.api_helpers import SERVER_KEY, Harness, harness

#: Event types the engine can emit; everything else of the original full_runner is dropped.
EMITTED = {
    "run.started",
    "run.plan",
    "run.status",
    "run.completed",
    "stage.started",
    "stage.completed",
    "step.started",
    "step.completed",
    "rule.checked",
    "scope.profile",
    "scope.goal",
    "scope.approved",
    "problem.subquestion",
    "problem.risk",
    "topic.evaluated",
    "search.strategy",
    "search.query",
    "search.sources",
    "literature.request",
    "literature.batch",
    "literature.collected",
    "screen.criteria",
    "screen.scored",
    "screen.rejected",
    "screen.kept",
    "card.extracted",
    "synthesis.cluster",
    "synthesis.tension",
    "synthesis.gap",
    "synthesis.overview",
    "synthesis.ranked",
    "debate.turn",
    "hypothesis.drafted",
    "hypothesis.checked",
    "hypothesis.selected",
    "gate.opened",
    "gate.resolved",
}
DROPPED = {
    "agent.message",
    "skills.loaded",
    "scope.estimate",
    "scope.adjusted",
    "estimate.checked",
    "idea.set_aside",
    "literature.merged",
}
STAGE_KEYS = {"scope", "search", "screen", "read", "synthesize", "r1-hypothesize"}


@pytest.fixture(autouse=True)
def _server_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("I2H_SERVICE_KEY", SERVER_KEY)


async def _completed(h: Harness, mode: str = "auto") -> tuple[str, list[dict[str, Any]]]:
    run_id = (await h.start("p-1", review_mode=mode)).json()["popper_run_id"]
    await h.settle()
    return run_id, await h.events(run_id)


def _of(events: list[dict[str, Any]], type_: str) -> list[dict[str, Any]]:
    return [e for e in events if e["type"] == type_]


async def test_only_known_types_and_stage_keys_are_emitted(tmp_path: Path) -> None:
    async with harness(tmp_path, research={"hardware_advisory": True}) as h:
        _, events = await _completed(h, "full")
        for _ in range(2):  # scope and screen gates
            run_id = h.service.store.list_run_ids()[0]
            gate_id = _of(await h.events(run_id), "gate.opened")[-1]["payload"]["gate_id"]
            await h.client.post(f"/runs/{run_id}/gates/{gate_id}", json={"option_id": "approve"})
            await h.settle()
        events = await h.events(run_id)
    types = {e["type"] for e in events}
    assert types <= EMITTED, types - EMITTED
    assert not types & DROPPED
    assert {e["stage_key"] for e in events if e["stage_key"]} <= STAGE_KEYS
    assert {"scope", "r1-hypothesize"} <= {e["stage_key"] for e in events}
    for event in events:  # content events always carry their stage
        if event["type"].split(".")[0] in {"screen", "card", "synthesis", "hypothesis", "problem"}:
            assert event["stage_key"] in STAGE_KEYS


async def test_plan_and_stage_plans_use_the_ui_groups(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        _, events = await _completed(h, "copilot")
    plan = _of(events, "run.plan")[0]["payload"]["stages"]
    assert [(s["key"], s["stage"], s["pipeline"]) for s in plan] == [
        ("scope", "scope", [1, 2]),
        ("search", "search", [3, 4]),
        ("screen", "screen", [5]),
        ("read", "read", [6]),
        ("synthesize", "synthesize", [7]),
        ("r1-hypothesize", "hypothesize", [8]),
    ]
    assert [s["has_gate"] for s in plan] == [False, False, True, False, False, False]
    assert [g.key for g in GROUPS] == [s["key"] for s in plan]
    started = {e["stage_key"]: e["payload"]["plan"] for e in _of(events, "stage.started")}
    assert list(started) == ["scope", "search", "screen"]  # stops at the gate
    screen_steps = [s["id"] for s in started["screen"]["steps"]]
    assert screen_steps == ["score", "reject", "shortlist", "screen_gate"]
    assert started["screen"]["steps"][-1]["gate"] == "screen"
    for plan_payload in started.values():
        assert {"key", "stage", "title", "has_gate", "pipeline", "purpose", "cast", "steps"} <= set(
            plan_payload
        )
        for step in plan_payload["steps"]:
            assert {"id", "title", "actor", "explain"} <= set(step)


async def test_steps_are_balanced_and_stages_complete_in_order(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        _, events = await _completed(h)
    for key in STAGE_KEYS:
        started = [
            e["payload"]["step_id"]
            for e in events
            if e["type"] == "step.started" and e["stage_key"] == key
        ]
        done = [
            e["payload"]["step_id"]
            for e in events
            if e["type"] == "step.completed" and e["stage_key"] == key
        ]
        assert started == done and started, key
    order = [e["stage_key"] for e in events if e["type"] == "stage.completed"]
    assert order == ["scope", "search", "screen", "read", "synthesize", "r1-hypothesize"]
    assert events[-1]["type"] == "run.completed" and events[-1]["stage_key"] is None


async def test_content_comes_from_the_real_artifacts(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id, events = await _completed(h)
        art = h.service.store.artifacts(run_id)
        candidates = art.read_jsonl(4, "candidates.jsonl")
        shortlist = art.read_jsonl(5, "shortlist.jsonl")
        review = art.read_json(5, "review.json")
        synthesis = art.read_json(7, "synthesis.json")
        hypotheses = art.read_json(8, "hypotheses.json")["hypotheses"]
        tree = art.read_json(2, "problem_tree.json")
        meta = art.read_json(4, "search_meta.json")

    collected = _of(events, "literature.collected")[0]["payload"]
    assert (collected["raw"], collected["unique"], collected["duplicates"]) == (
        meta["raw"],
        meta["unique"],
        meta["duplicates"],
    )
    assert collected["unique"] == len(candidates) == 12

    # one screening point per scored paper, taken from review.json
    points = [p for e in _of(events, "screen.scored") for p in e["payload"]["points"]]
    assert sorted(p["id"] for p in points) == sorted(d["paper_id"] for d in review["decisions"])
    assert len({p["id"] for p in points}) == len(points)
    by_id = {d["paper_id"]: d for d in review["decisions"]}
    assert all(p["relevance"] == by_id[p["id"]]["relevance_score"] for p in points)

    # rejected papers carry their reason and the shared word that made them match
    rejected = [e["payload"]["paper"] for e in _of(events, "screen.rejected")]
    assert len(rejected) == 3 and all(
        r["reason"] and r["false_friend"] == "sleep" for r in rejected
    )
    kept = [e["payload"]["paper"] for e in _of(events, "screen.kept")]
    assert [k["id"] for k in kept] == [s["paper_id"] for s in shortlist]
    assert all(k["reason"] and 0 <= k["relevance"] <= 1 and k["citation"] for k in kept)

    questions = [e["payload"]["sub_question"] for e in _of(events, "problem.subquestion")]
    assert [q["id"] for q in questions] == [q["id"] for q in tree["sub_questions"]]
    evaluation = _of(events, "topic.evaluated")[0]["payload"]
    assert evaluation["overall"] == 7.3 and set(evaluation["scores"]) == {
        "novelty",
        "specificity",
        "feasibility",
    }

    cards = [e["payload"]["card"] for e in _of(events, "card.extracted")]
    assert sorted(c["paper_id"] for c in cards) == sorted(s["paper_id"] for s in shortlist)
    assert all(c["evidence_scope"] == "abstract" for c in cards)
    assert any(c["unknown_fields"] for c in cards)  # null stays unknown, never filled in

    gaps = [e["payload"]["gap"] for e in _of(events, "synthesis.gap")]
    assert [g["id"] for g in gaps] == [g["id"] for g in synthesis["gaps"]]
    card_ids = {c["id"] for c in cards}
    assert all(set(g["from"]) <= card_ids for g in gaps)
    ranking = _of(events, "synthesis.ranked")[0]["payload"]["ranking"]
    assert [r["priority"] for r in ranking] == [1, 2]

    drafted = [e["payload"]["hypothesis"] for e in _of(events, "hypothesis.drafted")]
    assert [h["id"] for h in drafted] == [h["id"] for h in hypotheses]
    assert len({h["novelty"] for h in drafted}) == len(drafted)  # distinct, not one template
    assert len({h["rationale"] for h in drafted}) == len(drafted)
    assert all(h["falsify"]["text"] and h["gap"] in {g["id"] for g in gaps} for h in drafted)
    zones = {h["prediction"]: h["falsify"]["zone"] for h in drafted}
    assert zones.get("> 0") == [None, 0] and zones.get("< 0") == [0, None]
    selected = [e["payload"]["hypothesis_id"] for e in _of(events, "hypothesis.selected")]
    assert selected == [h["id"] for h in hypotheses]
    checks = _of(events, "hypothesis.checked")
    assert checks and all("feasibility" not in c["payload"] for c in checks)  # never invented
    assert all(set(c["payload"]["novelty"]) == {"novel", "closest", "similarity"} for c in checks)
    assert _of(events, "rule.checked")[0]["payload"]["rule"] == 5
    assert _of(events, "debate.turn")  # one turn per real perspective output


async def test_topic_rejected_by_the_evaluation_reports_the_rating_and_stops(
    tmp_path: Path,
) -> None:
    from tests.fixtures import FixtureLLM

    llm = FixtureLLM(
        overrides={
            "topic_evaluation": {
                "novelty": 2,
                "specificity": 2,
                "feasibility": 3,
                "overall": 2.4,
                "suggestion": "narrow it",
            }
        }
    )
    async with harness(tmp_path, llm=llm) as h:
        run_id, events = await _completed(h)
        state = await h.state(run_id)
    assert state["status"] == "failed" and "TOPIC_BELOW_THRESHOLD" in state["message"]
    rated = _of(events, "topic.evaluated")[0]["payload"]
    assert rated["overall"] < rated["threshold"] and rated["advice"] == "narrow it"
    assert events[-1]["type"] == "run.status" and events[-1]["payload"]["status"] == "failed"
    assert {e["stage_key"] for e in _of(events, "stage.started")} == {"scope"}


async def test_gate_payload_matches_what_the_studio_reads(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        _, events = await _completed(h, "copilot")
    gate = _of(events, "gate.opened")[0]["payload"]
    for key in (
        "id",
        "gate_id",
        "kind",
        "title",
        "why",
        "summary",
        "options",
        "stop_index",
        "stop_total",
        "droppable",
        "gate",
    ):
        assert key in gate, key
    assert gate["gate"]["gate_id"] == gate["gate_id"]
    assert all(
        {"id", "label", "description", "leads_to", "confirm_label"} <= set(o)
        for o in gate["options"]
    )
    assert [o["recommended"] for o in gate["options"] if o.get("recommended")] == [True]
    shortlist_ids = {r["paper_id"] for r in gate["shortlist"]}
    assert set(gate["droppable"]) == shortlist_ids
    statuses = [e["payload"]["status"] for e in _of(events, "run.status")]
    assert statuses == ["awaiting_review"]


async def test_log_is_private_state_free_and_heals_a_torn_tail(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id, events = await _completed(h)
        path = h.service.store.run_dir(run_id) / "platform_events.jsonl"
    raw = path.read_text(encoding="utf-8").splitlines()
    assert all("_core_seq" not in json.dumps(e) for e in events)  # internals never leak
    assert json.loads(raw[-1])["_core_seq"] > 0

    # a crash that left half a line, then a partly written batch
    lines = raw[:-1]
    path.write_text("\n".join(lines) + "\n" + raw[-1][:20], encoding="utf-8")
    healed = PlatformEventLog(RunStore(tmp_path / "runs")).read(run_id)
    assert [e["source_seq"] for e in healed] == list(range(1, len(healed) + 1))
    assert len(healed) == len(events) - 1

    # the next start re-projects from the core log, so no event is lost or duplicated
    async with harness(tmp_path) as h2:
        again = await h2.events(run_id)
        assert [e["source_seq"] for e in again] == list(range(1, len(again) + 1))
        assert [e["type"] for e in again] == [e["type"] for e in events]
