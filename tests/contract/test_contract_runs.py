"""Contract runs: creation, state, lifecycle events and what stays secret."""

from __future__ import annotations

import json
import logging

import pytest

from tests.contract.fake_platform import validate
from tests.fixtures.contract_harness import (
    INTEGRATION_MARKER,
    OPENALEX_MARKER,
    SECRET_MARKER,
    contract_harness,
    run_body,
)

STAGE = "stage-run-1"


async def test_create_run_idempotent(tmp_path):
    async with contract_harness(tmp_path) as h:
        first = await h.create(STAGE)
        second = await h.create(STAGE)
        await h.settle()

        assert first.status_code == 201 and second.status_code == 200
        a, b = first.json(), second.json()
        assert a["engine_run_id"] == b["engine_run_id"]
        assert a["created"] is True and b["created"] is False
        validate("RunCreateResponse", a)
        assert len(h.service.store.list_run_ids()) == 1
        assert h.platform.phases(STAGE).count("drafting") == 1


async def test_light_run_reaches_awaiting_researcher(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE)

        validate("RunState", state)
        assert state["phase"] == "awaiting_researcher" and state["waiting_input"] is True
        assert state["step_key"] == "hypothesis_gen" and state["stage_run_id"] == STAGE
        gate = state["open_gate"]
        assert gate["kind"] == "artifact_review" and gate["step_key"] == "hypothesis_gen"
        assert h.platform.phases(STAGE) == ["drafting", "reviewing", "awaiting_researcher"]
        uploaded = h.platform.uploaded(STAGE, "i2h.hypotheses/v1")
        assert gate["artifact_ids"] == [uploaded[-1].artifact_id]
        assert uploaded[-1].meta["primary"] is True
        review = h.platform.events[STAGE][-3:]
        assert [e["type"] for e in review][-2:] == ["run.phase_changed", "run.gate_opened"]


async def test_events_contiguous_and_lifecycle_only(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE)
        events = h.platform.events[STAGE]

        assert [e["source_seq"] for e in events] == list(range(1, len(events) + 1))
        assert {e["class"] for e in events} == {"lifecycle"}
        assert "review.completed" in h.platform.types(STAGE)
        completed = next(e for e in events if e["type"] == "review.completed")
        assert completed["payload"]["artifact_id"] == (
            h.platform.uploaded(STAGE, "review.findings/v1")[-1].artifact_id
        )
        steps = [e["step_key"] for e in events if e["type"] == "run.step_changed"]
        assert steps[0] == "topic_init" and steps[-1] == "hypothesis_gen"
        page = await h.client.get(
            f"/v1/runs/{state['engine_run_id']}/events", params={"after": 2, "limit": 3}
        )
        validate("EventPage", page.json())
        assert [e["source_seq"] for e in page.json()["events"]] == [3, 4, 5]
        assert state["last_source_seq"] == len(events)


async def test_unknown_stage_key_rejected(tmp_path):
    async with contract_harness(tmp_path) as h:
        response = await h.create(STAGE, stage_key="experiment_plan")

        assert response.status_code == 422
        assert response.json()["code"] == "VALIDATION_ERROR"
        assert h.service.store.list_run_ids() == []


@pytest.mark.parametrize(
    "params", [{}, {"topic": "short"}, {"topic": "t" * 2001}, {"topic": "x" * 20, "domains": "ml"}]
)
async def test_invalid_params_rejected(tmp_path, params):
    async with contract_harness(tmp_path) as h:
        response = await h.create(STAGE, inputs={"params": params})

        assert response.status_code == 422 and response.json()["code"] == "VALIDATION_ERROR"


async def test_unknown_run_is_404_with_flat_body(tmp_path):
    async with contract_harness(tmp_path) as h:
        response = await h.client.get("/v1/runs/missing")

        assert response.status_code == 404
        assert set(response.json()) == {"code", "message"}
        assert response.json()["code"] == "RUN_NOT_FOUND"


async def test_validation_error_does_not_echo_api_key(tmp_path):
    async with contract_harness(tmp_path) as h:
        body = run_body(STAGE)
        body["unexpected"] = SECRET_MARKER
        body["models"]["stage"]["api_key"] = SECRET_MARKER
        body["models"]["stage"]["provider"] = SECRET_MARKER
        response = await h.client.post("/v1/runs", json=body)

        assert response.status_code == 422
        assert SECRET_MARKER not in response.text
        for detail in response.json()["details"]:
            assert set(detail) == {"loc", "msg", "type"}


async def test_review_model_must_differ_from_stage_model(tmp_path):
    async with contract_harness(tmp_path) as h:
        response = await h.create(STAGE, review_model="stage-model")

        assert response.status_code == 422


async def test_no_secret_leak(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    markers = (SECRET_MARKER, INTEGRATION_MARKER, OPENALEX_MARKER)
    integrations = {"semantic_scholar": INTEGRATION_MARKER, "openalex": OPENALEX_MARKER}
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE, integrations=integrations)
        run_id = state["engine_run_id"]
        gate_id = state["open_gate"]["gate_id"]
        approved = await h.client.post(
            f"/v1/runs/{run_id}/gates/{gate_id}", json={"action": "approve", "actor": "user"}
        )
        await h.settle()
        final = await h.client.get(f"/v1/runs/{run_id}")
        events = await h.client.get(f"/v1/runs/{run_id}/events")
        capabilities = await h.client.get("/v1/capabilities")
        files = [p for p in h.run_dir(run_id).rglob("*") if p.is_file()]
        assert any(p.name.endswith(".snapshot.json") for p in files) and files

        for marker in markers:
            found = [str(p) for p in files if marker.encode() in p.read_bytes()]
            assert found == [], f"{marker} was written to {found}"
            assert all(marker.encode() not in body for body in h.platform.bodies)
            responses = state, approved.text, final.text, events.text, capabilities.text
            assert marker not in json.dumps(responses) and marker not in caplog.text
        assert final.json()["phase"] == "published"


async def test_review_uses_review_model(tmp_path):
    async with contract_harness(tmp_path) as h:
        await h.reach(STAGE, stage_model="stage-x", review_model="review-x")
        calls = h.llm_calls()

        reviewing = [model for phase, model in calls if phase == "reviewing"]
        assert reviewing and set(reviewing) == {"review-x"}
        assert {model for phase, model in calls if phase == "drafting"} <= {"stage-x", "review-x"}
        assert "stage-x" in {model for _, model in calls}


async def test_event_gap_triggers_resend_not_cancel(tmp_path):
    async with contract_harness(tmp_path) as h:
        h.platform.gaps[STAGE] = 0
        state = await h.reach(STAGE)

        events = h.platform.events[STAGE]
        assert [e["source_seq"] for e in events] == list(range(1, len(events) + 1))
        assert state["phase"] == "awaiting_researcher"
        assert "run.cancelled" not in h.platform.types(STAGE)


async def test_stage_closed_cancels(tmp_path):
    async with contract_harness(tmp_path) as h:
        h.platform.closed.add(STAGE)
        response = await h.create(STAGE)
        await h.settle()

        state = await h.state(response.json()["engine_run_id"])
        assert state["phase"] == "cancelled"
