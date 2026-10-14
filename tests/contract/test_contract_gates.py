"""Gate answers, the review loop and publishing."""

from __future__ import annotations

import hashlib
import json

from tests.contract.fake_platform import validate
from tests.fixtures.contract_harness import contract_harness

STAGE = "stage-run-1"
APPROVE = {"action": "approve", "actor": "user"}
FINDING = {
    "finding_id": "R1", "anchor": {"locator": "hypothesis:H1/prediction"}, "severity": "major",
    "issue": "name the comparison condition", "author": "researcher", "suggestion": "7 hours",
}  # fmt: skip


async def _answer(h, state, body, gate_id=None):
    gate_id = gate_id or state["open_gate"]["gate_id"]
    return await h.client.post(f"/v1/runs/{state['engine_run_id']}/gates/{gate_id}", json=body)


async def test_approve_publishes_primary_artifact(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE)
        response = await _answer(h, state, APPROVE)
        await h.settle()

        assert response.status_code == 200
        validate("RunState", response.json())
        final = await h.state(state["engine_run_id"])
        assert final["phase"] == "published" and final["open_gate"] is None
        primary = [a for a in h.platform.artifacts[STAGE] if a.meta["primary"]]
        assert [a.meta["schema"] for a in primary] == ["i2h.hypotheses/v1"]
        assert hashlib.sha256(primary[-1].content).hexdigest() == primary[-1].meta["sha256"]
        document = json.loads(primary[-1].content)
        assert document["schema"] == "i2h.hypotheses/v1"
        assert [h["id"] for h in document["hypotheses"]] == ["H1", "H2", "H3"]
        published = h.platform.events[STAGE][-1]
        assert published["type"] == "run.published" and published["phase"] == "published"
        assert published["payload"]["artifact_id"] == primary[-1].artifact_id
        assert h.platform.uploaded(STAGE, "i2h.argument_map/v1")
        assert h.platform.phases(STAGE)[-2:] == ["awaiting_researcher", "published"]
        closed = [e for e in h.platform.events[STAGE] if e["type"] == "run.gate_closed"]
        assert closed[-1]["payload"]["action"] == "approve"


async def test_approve_with_dropped_hypothesis_publishes_a_new_version(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE)
        first = h.platform.uploaded(STAGE, "i2h.hypotheses/v1")[-1]
        await _answer(h, state, {**APPROVE, "params": {"dropped": ["H3"]}})
        await h.settle()

        uploads = h.platform.uploaded(STAGE, "i2h.hypotheses/v1")
        assert [a.meta["version"] for a in uploads] == [1, 2]
        assert [x["id"] for x in json.loads(uploads[-1].content)["hypotheses"]] == ["H1", "H2"]
        published = h.platform.events[STAGE][-1]
        assert published["payload"]["artifact_id"] == uploads[-1].artifact_id != first.artifact_id


async def test_comment_returns_to_review_gate(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE)
        first_gate = state["open_gate"]
        body = {"action": "comment", "actor": "user", "text": "revise", "findings": [FINDING]}
        response = await _answer(h, state, body)
        await h.settle()

        assert response.status_code == 200
        assert h.platform.phases(STAGE) == [
            "drafting", "reviewing", "awaiting_researcher", "revising", "reviewing",
            "awaiting_researcher",
        ]  # fmt: skip
        final = await h.state(state["engine_run_id"])
        gate = final["open_gate"]
        assert gate["gate_id"] != first_gate["gate_id"] and gate["kind"] == "artifact_review"
        hypotheses = h.platform.uploaded(STAGE, "i2h.hypotheses/v1")
        assert gate["artifact_ids"] == [hypotheses[-1].artifact_id]
        answer = json.loads(h.platform.uploaded(STAGE, "revision.response/v1")[-1].content)
        validate("RevisionResponse", answer)
        assert answer["artifact_before"] == hypotheses[0].artifact_id
        assert answer["artifact_after"] == hypotheses[-1].artifact_id
        assert [(i["finding_id"], i["outcome"]) for i in answer["items"]] == [("R1", "fixed")]
        closed = [e for e in h.platform.events[STAGE] if e["type"] == "run.gate_closed"]
        assert closed[0]["payload"]["action"] == "comment"
        rounds = [a.meta["review_round"] for a in h.platform.uploaded(STAGE, "review.findings/v1")]
        assert rounds == [1, 2]


async def test_question_gate_answer_mapping(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE, mode="copilot")
        gate = state["open_gate"]

        assert state["phase"] == "drafting" and state["waiting_input"] is True
        assert gate["kind"] == "question" and gate["step_key"] == "literature_screen"
        assert [o["option_id"] for o in gate["options"]] == ["approve", "drop", "reject"]
        assert gate["default_option_id"] == "approve"

        reject = {"action": "answer", "actor": "user", "option_id": "reject"}
        again = await _answer(h, state, reject)
        await h.settle()
        assert again.status_code == 200
        second = await h.state(state["engine_run_id"])
        assert second["open_gate"]["gate_id"] != gate["gate_id"]
        steps = [e["step_key"] for e in h.platform.events[STAGE] if e["type"] == "run.step_changed"]
        assert steps.count("search_strategy") == 2

        await _answer(h, second, {"action": "answer", "actor": "user", "option_id": "approve"})
        await h.settle()
        third = await h.state(state["engine_run_id"])
        assert third["open_gate"]["kind"] == "artifact_review"
        assert third["phase"] == "awaiting_researcher"


async def test_invalid_gate_answer_422(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE, mode="copilot")
        question = {"action": "answer", "actor": "user"}

        wrong_action = await _answer(h, state, APPROVE)
        wrong_option = await _answer(h, state, {**question, "option_id": "maybe"})

        assert wrong_action.status_code == 422
        assert wrong_action.json()["code"] == "GATE_ACTION_INVALID"
        assert wrong_option.status_code == 422
        assert wrong_option.json()["code"] == "GATE_OPTION_INVALID"
        unchanged = await h.state(state["engine_run_id"])
        assert unchanged["open_gate"] == state["open_gate"]  # still open after the refusals

        await _answer(h, state, {**question, "option_id": "approve"})
        await h.settle()
        review = await h.state(state["engine_run_id"])
        no_findings = await _answer(h, review, {"action": "comment", "actor": "user"})
        unknown = await _answer(h, review, {**APPROVE, "params": {"dropped": ["H9"]}})
        bad_type = await _answer(h, review, {**APPROVE, "params": {"dropped": "H1"}})
        wrong_kind = await _answer(h, review, {**question, "option_id": "approve"})

        assert no_findings.status_code == unknown.status_code == bad_type.status_code == 422
        assert no_findings.json()["code"] == unknown.json()["code"] == "VALIDATION_ERROR"
        assert wrong_kind.json()["code"] == "GATE_ACTION_INVALID"
        assert (await h.state(state["engine_run_id"]))["open_gate"] is not None


async def test_error_body_shape(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE)
        response = await _answer(h, state, APPROVE, gate_id="no-such-gate")

        assert response.status_code == 409
        assert set(response.json()) == {"code", "message"}
        assert response.json()["code"] == "GATE_NOT_OPEN"


async def test_repeated_gate_answer_is_idempotent(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE)
        gate_id = state["open_gate"]["gate_id"]
        first = await _answer(h, state, APPROVE)
        again = await _answer(h, state, APPROVE, gate_id=gate_id)
        different = await _answer(h, state, {"action": "comment", "actor": "user",
                                             "findings": [FINDING]}, gate_id=gate_id)  # fmt: skip
        await h.settle()

        assert first.status_code == again.status_code == 200
        assert different.status_code == 409 and different.json()["code"] == "GATE_NOT_OPEN"
        assert h.platform.types(STAGE).count("run.published") == 1
        assert h.platform.types(STAGE).count("run.gate_closed") == 1
        assert len(h.platform.uploaded(STAGE, "i2h.argument_map/v1")) == 1
