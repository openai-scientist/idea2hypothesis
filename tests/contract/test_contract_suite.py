"""The mandatory engine behaviours of the published contract (vendor/README.md).

Tests that carry a stable name from the README table are spread over the ``tests/contract``
modules; ``test_every_mandatory_behaviour_has_a_test`` keeps the list honest. The rest are here.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from idea2hypothesis.api.contract.capabilities import capabilities
from idea2hypothesis.api.contract.models import CONTRACT_VERSION
from tests.contract.fake_platform import validate
from tests.fixtures.contract_harness import contract_harness

HERE = Path(__file__).parent
STAGE = "stage-run-1"
#: Mandatory tests whose slice has not landed: they arrive with their feature (plan S4).
DEFERRED = {"test_telemetry_well_formed", "test_restart_fails_open_runs"}


def _mandatory_names() -> set[str]:
    readme = (HERE / "vendor" / "README.md").read_text(encoding="utf-8")
    table = readme.split("## Mandatory engine behavior tests", 1)[1]
    return set(re.findall(r"^\| `(test_\w+)` \|", table, re.MULTILINE))


def test_every_mandatory_behaviour_has_a_test():
    defined = {
        name
        for path in HERE.glob("test_*.py")
        for name in re.findall(r"^(?:async )?def (test_\w+)", path.read_text(encoding="utf-8"),
                               re.MULTILINE)
    }  # fmt: skip
    mandatory = _mandatory_names()
    assert len(mandatory) == 20
    assert sorted(mandatory - defined - DEFERRED) == []


def test_capabilities_match_the_contract():
    caps = capabilities()

    assert caps.engine_key == "i2h"
    assert caps.contract_version.split(".")[0] == CONTRACT_VERSION.split(".")[0]
    assert any(profile.is_base for profile in caps.profiles)
    assert sum(output.primary for output in caps.output_schemas) == 1


async def test_runs_with_base_profile(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE, profile={})

        assert state["phase"] == "awaiting_researcher"
        assert state["open_gate"]["kind"] == "artifact_review"
        assert h.platform.uploaded(STAGE, "i2h.hypotheses/v1")[-1].meta["primary"] is True


async def test_runs_require_service_key(tmp_path):
    async with contract_harness(tmp_path) as h:
        h.client.headers.pop("X-Service-Key")
        response = await h.create(STAGE)

        assert response.status_code == 401
        assert response.json()["code"] == "SERVICE_KEY_INVALID"
        assert h.service.store.list_run_ids() == []


async def test_reaches_review_gate(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE)

        assert state["phase"] == "awaiting_researcher" and state["waiting_input"] is True
        assert state["open_gate"]["kind"] == "artifact_review"
        validate("Gate", state["open_gate"])


async def test_lifecycle_events_ordered(tmp_path):
    async with contract_harness(tmp_path) as h:
        await h.reach(STAGE)
        events = h.platform.events[STAGE]
        phases = h.platform.phases(STAGE)

        assert [e["source_seq"] for e in events] == list(range(1, len(events) + 1))
        assert phases[0] == "drafting"
        assert phases.index("reviewing") < phases.index("awaiting_researcher")


async def test_activity_events_declared(tmp_path):
    async with contract_harness(tmp_path, review={"mode": "copilot"}) as h:
        await h.reach(STAGE, mode="copilot")
        declared = set(capabilities().activity_types)

        for event in (e for e in h.platform.events[STAGE] if e["class"] == "activity"):
            assert event["type"] in declared
            assert event["step_key"] and event["actor"] and event["summary"]


async def test_pause_then_resume_continues(tmp_path):
    holder: dict = {}

    async def hook(info):
        if info.key == "knowledge_extract" and "task" not in holder:
            run_id = holder["h"].service.store.list_run_ids()[-1]
            holder["task"] = asyncio.ensure_future(
                holder["h"].client.post(f"/v1/runs/{run_id}/pause")
            )
            await asyncio.sleep(0.2)  # the request is served while the call is pending

    async with contract_harness(tmp_path, on_call=hook) as h:
        holder["h"] = h
        created = await h.create(STAGE)
        run_id = created.json()["engine_run_id"]
        await h.settle()
        paused = (await holder["task"]).json()

        assert paused["phase"] == "paused" and paused["waiting_input"] is False
        checkpoint = h.service.store.read_checkpoint(run_id)["stages"]
        done = sorted(int(n) for n, e in checkpoint.items() if e["status"] == "completed")
        assert done and max(done) < 8
        art = h.service.store.artifacts(run_id)
        before = {n: art.read_manifest(n) for n in done}
        completed_before = [
            e.stage for e in h.service.store.read_events(run_id) if e.type == "stage.completed"
        ]

        resumed = await h.client.post(f"/v1/runs/{run_id}/resume")
        await h.settle()
        final = await h.state(run_id)

        assert resumed.status_code == 200
        assert final["phase"] == "awaiting_researcher" and final["open_gate"] is not None
        assert {n: art.read_manifest(n) for n in done} == before
        completed = [
            e.stage for e in h.service.store.read_events(run_id) if e.type == "stage.completed"
        ]
        assert completed[: len(completed_before)] == completed_before
        assert sorted(set(completed)) == completed and len(set(completed)) == len(completed)
        phases = h.platform.phases(STAGE)
        assert "paused" in phases and phases[-1] == "awaiting_researcher"


async def test_pause_and_resume_at_the_review_gate(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE)
        run_id = state["engine_run_id"]
        gate = state["open_gate"]

        paused = (await h.client.post(f"/v1/runs/{run_id}/pause")).json()
        resumed = (await h.client.post(f"/v1/runs/{run_id}/resume")).json()
        await h.settle()

        assert paused["phase"] == "paused"
        assert resumed["open_gate"] == gate and (await h.state(run_id))["phase"] == (
            "awaiting_researcher"
        )


async def test_cancel(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE)
        run_id = state["engine_run_id"]

        response = await h.client.post(f"/v1/runs/{run_id}/cancel")
        again = await h.client.post(f"/v1/runs/{run_id}/cancel")
        await h.settle()

        assert response.status_code == again.status_code == 200
        assert response.json()["phase"] == "cancelled" and response.json()["open_gate"] is None
        assert h.platform.types(STAGE)[-1] == "run.cancelled"
        assert h.platform.events[STAGE][-1]["phase"] == "cancelled"
        finished = await h.client.post(f"/v1/runs/{run_id}/pause")
        assert finished.status_code == 409 and finished.json()["code"] == "RUN_CLOSED"


async def test_cancel_while_drafting(tmp_path):
    holder: dict = {}

    async def hook(info):
        if info.key == "knowledge_extract" and "task" not in holder:
            run_id = holder["h"].service.store.list_run_ids()[-1]
            holder["task"] = asyncio.ensure_future(
                holder["h"].client.post(f"/v1/runs/{run_id}/cancel")
            )
            await asyncio.sleep(0.2)

    async with contract_harness(tmp_path, on_call=hook) as h:
        holder["h"] = h
        await h.create(STAGE)
        await h.settle()
        response = await holder["task"]

        assert response.json()["phase"] == "cancelled"
        assert h.platform.types(STAGE)[-1] == "run.cancelled"


@pytest.mark.parametrize("route", ["pause", "resume", "cancel"])
async def test_control_of_unknown_run_is_404(tmp_path, route):
    async with contract_harness(tmp_path) as h:
        response = await h.client.post(f"/v1/runs/none/{route}")

        assert response.status_code == 404 and response.json()["code"] == "RUN_NOT_FOUND"


async def test_contract_activity_never_uses_actor_pi(tmp_path):
    async with contract_harness(tmp_path, review={"mode": "copilot"}) as h:
        await h.reach(STAGE, mode="copilot")
        actors = {e["actor"] for e in h.platform.events[STAGE] if e["class"] == "activity"}

        assert actors and "pi" not in actors
        assert "chair" in actors
