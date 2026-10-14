"""Researcher timeline: live content, reviewer findings, and comment revisions."""

import json
from difflib import unified_diff

import pytest

from idea2hypothesis.llm.models import LLMTimeout
from tests.contract.fake_platform import validate
from tests.contract.test_contract_gates import APPROVE, FINDING, _answer
from tests.fixtures.contract_harness import contract_harness

STAGE = "stage-run-1"


async def test_activity_types_declared(tmp_path):
    async with contract_harness(tmp_path) as h:
        screen = await h.reach(mode="copilot")
        await _answer(h, screen, {"action": "answer", "option_id": "approve", "actor": "user"})
        await h.settle()
        state = await h.state(screen["engine_run_id"])
        await _answer(h, state, {"action": "comment", "actor": "user", "findings": [FINDING]})
        await h.settle()
        revised = await h.state(state["engine_run_id"])
        await _answer(h, revised, APPROVE)
        await h.settle()
        assert (await h.state(state["engine_run_id"]))["phase"] == "published"
        events = h.platform.events[STAGE]
        activities = [e for e in events if e["class"] == "activity"]
        assert activities
        declared = (await h.client.get("/v1/capabilities")).json()["activity_types"]
        assert {e["type"] for e in activities} <= set(declared)
        assert {"i2h.search.query", "i2h.screen.kept", "i2h.card.extracted",
                "i2h.synthesis.gap", "i2h.hypothesis.drafted", "i2h.debate.turn",
                "i2h.map.node", "i2h.map.edge", "i2h.review.finding",
                "i2h.agent.message"} <= {e["type"] for e in activities}
        assert not any(e["type"].startswith(("i2h.step.", "i2h.stage.", "i2h.gate.", "i2h.run."))
                       for e in activities)
        assert [e["source_seq"] for e in events] == list(range(1, len(events) + 1))
        findings = [e for e in activities if e["type"] == "i2h.review.finding"]
        reviews = h.platform.uploaded(STAGE, "review.findings/v1")
        expected = [f for a in reviews for f in json.loads(a.content)["findings"]]
        assert [e["payload"] for e in findings] == expected
        assert all(e["actor"] == "reviewer" and e["phase"] == "reviewing" for e in findings)
        for e in findings:
            assert e["payload"]["anchor"]["artifact_id"] in e["refs"]
            hypothesis_id = e["payload"]["anchor"]["locator"].split(":", 1)[1].split("/", 1)[0]
            assert hypothesis_id in e["refs"]
        cards = [e for e in activities if e["type"] == "i2h.card.extracted"]
        assert len(cards) == len({e["payload"]["card"]["id"] for e in cards}) == 9
        assert all(e["payload"]["card"]["paper_id"] in e["refs"] and
                   e["payload"]["card"]["id"] in e["refs"] for e in cards)
        scores = [e for e in activities if e["type"] == "i2h.screen.scored"]
        assert scores and all(p["id"] in e["refs"] for e in scores for p in e["payload"]["points"])
        gaps = [e for e in activities if e["type"] == "i2h.synthesis.gap"]
        assert gaps and all(c in e["refs"] for e in gaps for c in e["payload"]["gap"]["from"])
        before = len(events)
        h.service.contract.projector.forget(state["engine_run_id"])
        await h.state(state["engine_run_id"])
        await h.settle()
        assert len(h.platform.events[STAGE]) == before


async def test_activity_has_summary_actor_step(tmp_path):
    async with contract_harness(tmp_path) as h:
        await h.reach()
        activities = [e for e in h.platform.events[STAGE] if e["class"] == "activity"]
        assert activities
        steps = {s["step_key"] for s in (await h.client.get("/v1/capabilities")).json()["steps"]}
        for e in activities:
            validate("EventEnvelope", e)
            assert e["step_key"] in steps
            assert e["actor"].strip() and e["summary"].strip()
            assert "\n" not in e["summary"] and len(e["summary"]) <= 240


async def test_activity_has_no_prompt_text(tmp_path):
    prompts = []

    def capture(info):
        prompts.append(info.system)

    async with contract_harness(tmp_path, on_call=capture) as h:
        await h.reach()
        activities = [e for e in h.platform.events[STAGE] if e["class"] == "activity"]
        assert activities and prompts
        encoded = json.dumps(activities)
        assert all(p not in encoded for p in prompts if p)
        assert "hypothesis_review" not in encoded
        assert '"messages":' not in encoded and '"system":' not in encoded
        assert '"api_key":' not in encoded and '"raw_response":' not in encoded


async def test_card_activity_arrives_before_extraction_completes(tmp_path):
    holder = {}

    def capture(info):
        if info.key == "knowledge_extract":
            holder["calls"] = holder.get("calls", 0) + 1
            if holder["calls"] == 2:
                h = holder["h"]
                holder["live"] = [e for e in h.service.contract.log.read(holder["run_id"])
                                  if e["type"] == "i2h.card.extracted"]

    async with contract_harness(tmp_path, on_call=capture) as h:
        holder["h"] = h
        created = await h.create()
        holder["run_id"] = created.json()["engine_run_id"]
        await h.settle()
        assert holder["calls"] > 1
        assert len(holder["live"]) == 1


async def test_comment_uploads_markdown_revision_diff(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach()
        art = h.service.store.artifacts(state["engine_run_id"])
        before_text = art.read_text(8, "hypotheses.md")
        before_id = h.platform.uploaded(STAGE, "i2h.hypotheses/v1")[-1].artifact_id
        await _answer(h, state, {"action": "comment", "actor": "user", "findings": [FINDING]})
        await h.settle()
        diffs = h.platform.uploaded(STAGE, "revision.diff/v1")
        assert len(diffs) == 1
        diff = json.loads(diffs[0].content)
        validate("RevisionDiff", diff)
        assert diff["round"] == 1 and diff["format"] == "unified"
        assert diff["artifact_before"] == before_id
        after_id = h.platform.uploaded(STAGE, "i2h.hypotheses/v1")[-1].artifact_id
        assert diff["artifact_after"] == after_id
        expected = "".join(unified_diff(
            before_text.splitlines(keepends=True),
            art.read_text(8, "hypotheses.md").splitlines(keepends=True),
            fromfile="before/hypotheses.md", tofile="after/hypotheses.md",
        ))
        assert diff["patch"] == expected


@pytest.mark.parametrize("mode", ["auto", "light"])
async def test_optional_question_defaults_are_recorded_as_system_activity(tmp_path, mode):
    async with contract_harness(tmp_path) as h:
        await h.reach(mode=mode)
        defaults = [e for e in h.platform.events[STAGE]
                    if e["type"] == "i2h.question.defaulted"]
        assert [e["step_key"] for e in defaults] == ["problem_decompose", "literature_screen"]
        assert all(e["actor"] == "system" and e["payload"]["option_id"] == "approve"
                   for e in defaults)


async def test_retry_does_not_duplicate_cached_card_activity(tmp_path):
    def interrupt_second_card(info):
        if info.key == "knowledge_extract" and info.index == 1:
            raise LLMTimeout("fixture transient timeout")

    async with contract_harness(tmp_path, on_call=interrupt_second_card,
                                runtime={"concurrency": 1, "max_retries": 1,
                                         "retry_delay_sec": 0}) as h:
        state = await h.reach()
        assert state["phase"] == "awaiting_researcher"
        core = h.service.store.read_events(state["engine_run_id"])
        assert any(e.type == "stage.progress" and e.stage == 6 and e.data.get("kind") == "restart"
                   for e in core)
        cards = [e for e in h.platform.events[STAGE] if e["type"] == "i2h.card.extracted"]
        assert len(cards) == len({e["payload"]["card"]["id"] for e in cards}) == 9
