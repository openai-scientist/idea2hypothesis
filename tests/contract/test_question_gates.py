"""Interaction modes and researcher decisions on scope and shortlisted papers."""

import pytest

from tests.contract.fake_platform import validate
from tests.contract.test_contract_gates import _answer
from tests.fixtures.contract_harness import contract_harness


async def test_full_mode_scope_then_screen_question_gates(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(mode="full")
        run_id = state["engine_run_id"]
        scope = state["open_gate"]
        assert scope["kind"] == "question" and scope["step_key"] == "problem_decompose"
        assert [o["option_id"] for o in scope["options"]] == ["approve", "reject"]
        assert scope["default_option_id"] == "approve"
        assert state["phase"] == "drafting" and state["waiting_input"]
        assert h.service.store.read_run(run_id)["review_mode"] == "full"
        await _answer(h, state, {"action": "answer", "option_id": "approve", "actor": "user"})
        await h.settle()
        screen_state = await h.state(run_id)
        screen = screen_state["open_gate"]
        validate("Gate", screen)
        assert screen["kind"] == "question" and screen["step_key"] == "literature_screen"
        assert [o["option_id"] for o in screen["options"]] == ["approve", "drop", "reject"]
        assert screen["default_option_id"] == "approve"
        shortlist = h.service.store.artifacts(run_id).read_jsonl(5, "shortlist.jsonl")
        assert shortlist
        assert all(p["paper_id"] in screen["prompt"] and p["title"] in screen["prompt"]
                   for p in shortlist)
        opened = [e for e in h.platform.events["stage-run-1"] if e["type"] == "run.gate_opened"]
        assert opened[-1]["payload"] == screen


async def test_screen_drop_excludes_papers(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(mode="copilot")
        run_id = state["engine_run_id"]
        art = h.service.store.artifacts(run_id)
        shortlist = art.read_jsonl(5, "shortlist.jsonl")
        victim = shortlist[0]["paper_id"]
        response = await _answer(h, state, {
            "action": "answer", "actor": "user", "option_id": "drop",
            "params": {"dropped": [victim]}, "text": "not relevant",
        })
        assert response.status_code == 200, response.text
        await h.settle()
        cards = [art.read_json(6, n) for n in art.list_files(6)
                 if n.startswith("cards/") and n.endswith(".json")]
        assert len(cards) == len(shortlist) - 1
        assert victim not in {c["paper_id"] for c in cards}
        assert victim not in {p["paper_id"] for p in art.read_jsonl(5, "shortlist.jsonl")}
        assert (await h.state(run_id))["open_gate"]["kind"] == "artifact_review"


@pytest.mark.parametrize("mode", ["auto", "light"])
async def test_auto_mode_opens_no_question_gate(tmp_path, mode):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(mode=mode)
        assert h.service.store.read_run(state["engine_run_id"])["review_mode"] == mode
        opened = [e["payload"] for e in h.platform.events["stage-run-1"]
                  if e["type"] == "run.gate_opened"]
        assert [g["kind"] for g in opened] == ["artifact_review"]
        assert state["phase"] == "awaiting_researcher"


@pytest.mark.parametrize("params", [{}, {"dropped": []}, {"dropped": "paper"}])
async def test_screen_drop_requires_paper_ids(tmp_path, params):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(mode="copilot")
        response = await _answer(h, state, {
            "action": "answer", "actor": "user", "option_id": "drop", "params": params,
        })
        assert response.status_code == 422 and response.json()["code"] == "VALIDATION_ERROR"
        assert (await h.state(state["engine_run_id"]))["open_gate"] == state["open_gate"]


async def test_scope_rejects_screen_only_drop_option(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(mode="full")
        response = await _answer(h, state, {
            "action": "answer", "actor": "user", "option_id": "drop",
            "params": {"dropped": ["paper"]},
        })
        assert response.status_code == 422 and response.json()["code"] == "GATE_OPTION_INVALID"
