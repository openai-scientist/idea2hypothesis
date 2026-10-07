"""Platform engine routes: idempotency, state, event replay, gates and run control."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.fixtures import FixtureLiterature, FixtureLLM
from tests.fixtures.api_helpers import SERVER_KEY, FakePlatform, Harness, harness, run_body

ENVELOPE_KEYS = {"source_seq", "type", "stage_key", "actor", "payload"}


@pytest.fixture(autouse=True)
def _server_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("I2H_SERVICE_KEY", SERVER_KEY)


def _gate_id(events: list[dict[str, Any]]) -> str:
    opened = [e for e in events if e["type"] == "gate.opened"]
    assert opened, "no gate.opened event"
    return str(opened[-1]["payload"]["gate_id"])


async def _answer(h: Harness, run_id: str, option: str = "approve", **body: Any) -> httpx.Response:
    gate_id = _gate_id(await h.events(run_id))
    response = await h.client.post(
        f"/runs/{run_id}/gates/{gate_id}", json={"option_id": option, **body}
    )
    await h.settle()
    return response


async def test_post_runs_is_idempotent_by_platform_run_id(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        first = await h.start("p-1")
        assert first.status_code == 201
        body = first.json()
        assert set(body) == {"popper_run_id", "status", "cost_usd", "message"}
        await h.settle()

        again = await h.start("p-1")
        assert again.status_code == 200
        assert again.json()["popper_run_id"] == body["popper_run_id"]
        assert len(h.service.store.list_run_ids()) == 1

        other = await h.start("p-2")
        assert other.json()["popper_run_id"] != body["popper_run_id"]


async def test_idempotency_survives_an_app_restart(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = (await h.start("p-1")).json()["popper_run_id"]
        await h.settle()
        before = await h.events(run_id)
    async with harness(tmp_path) as h2:
        again = await h2.start("p-1")
        assert again.status_code == 200
        assert again.json()["popper_run_id"] == run_id
        assert len(h2.service.store.list_run_ids()) == 1
        assert await h2.events(run_id) == before


async def test_state_lookup_by_either_id(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = (await h.start("plat-77")).json()["popper_run_id"]
        await h.settle()
        by_run = (await h.client.get(f"/runs/{run_id}")).json()
        by_platform = (await h.client.get("/runs/plat-77")).json()
        assert by_run == by_platform
        assert set(by_run) == {"popper_run_id", "status", "cost_usd", "message", "last_source_seq"}
        found = await h.client.get("/runs", params={"platform_run_id": "plat-77"})
        assert found.json()["popper_run_id"] == run_id
        assert (await h.client.get("/runs", params={"platform_run_id": "nope"})).status_code == 404
        assert (await h.client.get("/runs/unknown-id")).status_code == 404


async def test_events_are_contiguous_enveloped_and_replayable(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = (await h.start("p-1", review_mode="auto")).json()["popper_run_id"]
        await h.settle()
        events = await h.events(run_id)
        assert [e["source_seq"] for e in events] == list(range(1, len(events) + 1))
        assert all(set(e) == ENVELOPE_KEYS for e in events)
        assert events[0]["type"] == "run.started"
        assert events[-1]["type"] == "run.completed"

        tail = await h.events(run_id, after=len(events) - 3)
        assert [e["source_seq"] for e in tail] == [len(events) - 2, len(events) - 1, len(events)]
        limited = await h.client.get(f"/runs/{run_id}/events", params={"limit": 5})
        assert len(limited.json()["events"]) == 5
        # the persisted log, not a recomputation, is what is served
        path = h.service.store.run_dir(run_id) / "platform_events.jsonl"
        assert len(path.read_text(encoding="utf-8").splitlines()) == len(events)


async def test_copilot_gate_flow_runs_to_completion(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = (await h.start("p-1")).json()["popper_run_id"]
        await h.settle()
        assert (await h.state(run_id))["status"] == "awaiting_review"
        events = await h.events(run_id)
        gate = next(e for e in events if e["type"] == "gate.opened")
        assert gate["payload"]["kind"] == "screen"
        assert gate["payload"]["gate_id"] == gate["payload"]["gate"]["gate_id"]
        assert len(gate["payload"]["droppable"]) == 9
        assert {o["id"] for o in gate["payload"]["options"]} == {"approve", "drop", "reject"}
        assert events[-1]["type"] == "gate.opened"
        assert not any(e["type"] == "hypothesis.drafted" for e in events)

        response = await _answer(h, run_id)
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "message": "Gate answer accepted"}
        assert (await h.state(run_id))["status"] == "completed"
        events = await h.events(run_id)
        types = [e["type"] for e in events]
        assert "gate.resolved" in types and types[-1] == "run.completed"
        assert types.count("card.extracted") == 9

        # answering again is idempotent for the same option and a conflict otherwise
        gate_id = _gate_id(events)
        same = await h.client.post(f"/runs/{run_id}/gates/{gate_id}", json={"option_id": "approve"})
        assert same.status_code == 200 and "already" in same.json()["message"]
        other = await h.client.post(f"/runs/{run_id}/gates/{gate_id}", json={"option_id": "reject"})
        assert other.status_code == 409


async def test_gate_drop_excludes_papers_from_later_stages(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = (await h.start("p-1")).json()["popper_run_id"]
        await h.settle()
        gate = next(e for e in await h.events(run_id) if e["type"] == "gate.opened")
        victim = gate["payload"]["droppable"][0]
        await _answer(h, run_id, "drop", dropped=[victim], note="not relevant")
        events = await h.events(run_id)
        cards = [e["payload"]["card"] for e in events if e["type"] == "card.extracted"]
        assert len(cards) == 8 and victim not in {c["paper_id"] for c in cards}
        resolved = next(e for e in events if e["type"] == "gate.resolved")
        assert resolved["payload"]["answer"] == {
            "option_id": "drop",
            "dropped": [victim],
            "note": "not relevant",
        }


async def test_gate_errors(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = (await h.start("p-1")).json()["popper_run_id"]
        await h.settle()
        gate_id = _gate_id(await h.events(run_id))
        url = f"/runs/{run_id}/gates/{gate_id}"
        wrong = await h.client.post(f"/runs/{run_id}/gates/wrong", json={"option_id": "approve"})
        assert wrong.status_code == 404
        assert (await h.client.post(url, json={"option_id": "maybe"})).status_code == 422
        assert (await h.client.post(url, json={"option_id": "drop"})).status_code == 422
        bad = await h.client.post(url, json={"option_id": "drop", "dropped": ["not-a-paper"]})
        assert bad.status_code == 422
        assert (await h.state(run_id))["status"] == "awaiting_review"


async def test_full_mode_has_a_scope_gate_before_the_screen_gate(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = (await h.start("p-1", review_mode="full")).json()["popper_run_id"]
        await h.settle()
        events = await h.events(run_id)
        scope = next(e for e in events if e["type"] == "gate.opened")
        assert scope["payload"]["kind"] == "scope"
        assert (scope["payload"]["stop_index"], scope["payload"]["stop_total"]) == (1, 2)
        assert not any(e["type"] == "stage.started" and e["stage_key"] == "search" for e in events)
        plan = next(e for e in events if e["type"] == "run.plan")["payload"]["stages"]
        assert [s["has_gate"] for s in plan if s["key"] in ("scope", "screen")] == [True, True]

        await _answer(h, run_id)
        assert (await h.state(run_id))["status"] == "awaiting_review"
        events = await h.events(run_id)
        kinds = [e["payload"]["kind"] for e in events if e["type"] == "gate.opened"]
        assert kinds == ["scope", "screen"]
        assert any(e["type"] == "scope.approved" for e in events)
        assert any(e["type"] == "stage.completed" and e["stage_key"] == "scope" for e in events)

        await _answer(h, run_id)
        assert (await h.state(run_id))["status"] == "completed"


async def test_rejecting_the_screen_gate_searches_again(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = (await h.start("p-1")).json()["popper_run_id"]
        await h.settle()
        await _answer(h, run_id, "reject", note="queries too narrow")
        assert (await h.state(run_id))["status"] == "awaiting_review"
        events = await h.events(run_id)
        opened = [e["payload"]["gate_id"] for e in events if e["type"] == "gate.opened"]
        assert opened == ["gate-s05-a1", "gate-s05-a2"]
        searches = [
            e for e in events if e["type"] == "stage.started" and e["stage_key"] == "search"
        ]
        assert len(searches) == 2
        assert [e["source_seq"] for e in events] == list(range(1, len(events) + 1))


async def test_pause_resume_and_finished_run_conflicts(tmp_path: Path) -> None:
    holder: dict[str, Harness] = {}

    async def on_call(info: Any) -> None:  # pause while stage 3 is being generated
        if info.key == "search_strategy" and info.index == 0:
            h = holder["h"]
            run_id = h.service.store.list_run_ids()[0]
            assert (await h.client.post(f"/runs/{run_id}/pause")).status_code == 200

    llm = FixtureLLM(on_call=on_call)
    async with harness(tmp_path, llm=llm) as h:
        holder["h"] = h
        run_id = (await h.start("p-1", review_mode="auto")).json()["popper_run_id"]
        await h.settle()
        state = await h.state(run_id)
        assert state["status"] == "paused" and state["message"] == "user"
        events = await h.events(run_id)
        assert events[-1]["type"] == "run.status"
        assert events[-1]["payload"] == {"status": "paused", "reason": "user"}

        assert (await h.client.post(f"/runs/{run_id}/resume")).status_code == 200
        await h.settle()
        assert (await h.state(run_id))["status"] == "completed"
        assert llm.count("topic_init") == 1  # stages 1-2 were not repeated

        for action in ("pause", "resume", "cancel"):
            finished = await h.client.post(f"/runs/{run_id}/{action}")
            assert finished.status_code == 409, action
            assert finished.json()["detail"] == "RUN_FINISHED"


async def test_cancel_stops_the_run_and_reports_failed(tmp_path: Path) -> None:
    holder: dict[str, Harness] = {}

    async def on_call(info: Any) -> None:
        if info.key == "problem_decompose" and info.index == 0:
            h = holder["h"]
            run_id = h.service.store.list_run_ids()[0]
            assert (await h.client.post(f"/runs/{run_id}/cancel")).status_code == 200

    llm = FixtureLLM(on_call=on_call)
    async with harness(tmp_path, llm=llm) as h:
        holder["h"] = h
        run_id = (await h.start("p-1", review_mode="auto")).json()["popper_run_id"]
        await h.settle()
        state = await h.state(run_id)
        assert state["status"] == "failed" and state["message"] == "Cancelled by user"
        last = (await h.events(run_id))[-1]
        assert last["type"] == "run.status" and last["payload"]["status"] == "failed"
        assert llm.count("search_strategy") == 0
        assert (await h.client.post(f"/runs/{run_id}/cancel")).status_code == 200  # idempotent
        assert (await h.client.post(f"/runs/{run_id}/resume")).status_code == 409


async def test_interrupted_run_is_paused_on_startup_and_resumes(tmp_path: Path) -> None:
    reached = asyncio.Event()

    async def block(info: Any) -> None:  # simulate a process dying inside stage 3
        if info.key == "search_strategy":
            reached.set()
            await asyncio.Event().wait()

    async with harness(tmp_path, llm=FixtureLLM(on_call=block)) as h:
        run_id = (await h.start("p-1", review_mode="auto")).json()["popper_run_id"]
        await asyncio.wait_for(reached.wait(), 10)
        assert h.service.store.read_run(run_id)["status"] == "running"
        await h.service.shutdown()  # the "crash": workers are cancelled, files stay as they are
    stored = json.loads((tmp_path / "runs" / run_id / "run.json").read_text(encoding="utf-8"))
    assert stored["status"] == "running"

    llm = FixtureLLM()
    async with harness(tmp_path, llm=llm) as h2:
        state = await h2.state(run_id)
        assert state["status"] == "paused" and state["message"] == "interrupted"
        assert (await h2.client.post(f"/runs/{run_id}/resume")).status_code == 200
        await h2.settle()
        assert (await h2.state(run_id))["status"] == "completed"
        assert llm.count("topic_init") == 0  # completed stages were not repeated
        events = await h2.events(run_id)
        assert [e["source_seq"] for e in events] == list(range(1, len(events) + 1))
        statuses = [e["payload"]["status"] for e in events if e["type"] == "run.status"]
        assert statuses[:2] == ["paused", "running"]


async def test_cost_is_null_unless_priced(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = (await h.start("p-1", review_mode="auto")).json()["popper_run_id"]
        await h.settle()
        assert (await h.state(run_id))["cost_usd"] is None
        assert "cost_usd" not in (await h.events(run_id))[-1]["payload"]
    async with harness(tmp_path / "priced", llm=FixtureLLM(cost_per_call=0.01)) as h2:
        run_id = (await h2.start("p-1", review_mode="auto")).json()["popper_run_id"]
        await h2.settle()
        state = await h2.state(run_id)
        assert state["cost_usd"] == "0.1900"  # 19 priced calls
        assert (await h2.events(run_id))[-1]["payload"]["cost_usd"] == "0.1900"


async def test_budget_exceeded_pauses_a_priced_run(tmp_path: Path) -> None:
    async with harness(tmp_path, llm=FixtureLLM(cost_per_call=1.0)) as h:
        body = await h.start("p-1", review_mode="auto", budget_usd="2.5")
        await h.settle()
        state = await h.state(body.json()["popper_run_id"])
        assert state["status"] == "paused" and state["message"] == "budget_exceeded"


async def test_failed_run_reports_a_reason_and_stops_later_stages(tmp_path: Path) -> None:
    async with harness(tmp_path, literature=FixtureLiterature(papers=[])) as h:
        run_id = (await h.start("p-1", review_mode="auto")).json()["popper_run_id"]
        await h.settle()
        state = await h.state(run_id)
        assert state["status"] == "failed" and state["message"].startswith("NO_LITERATURE")
        events = await h.events(run_id)
        assert events[-1]["payload"]["status"] == "failed"
        assert not any(e["type"] in ("screen.kept", "card.extracted") for e in events)


async def test_invalid_requests_are_rejected(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        assert (await h.client.post("/runs", json=run_body(topic="short"))).status_code == 422
        assert (await h.client.post("/runs", json=run_body(review_mode="wild"))).status_code == 422
        assert (await h.client.post("/runs", json=run_body(budget_usd="0"))).status_code == 422
        assert h.service.store.list_run_ids() == []


async def test_delivery_failure_does_not_affect_the_run(tmp_path: Path) -> None:
    platform = FakePlatform(script=[500] * 50)
    async with harness(tmp_path, platform=platform) as h:
        run_id = (await h.start("p-1", review_mode="auto")).json()["popper_run_id"]
        await h.settle()
        assert (await h.state(run_id))["status"] == "completed"
        assert platform.received == []
        assert len(await h.events(run_id)) > 50  # still replayable
