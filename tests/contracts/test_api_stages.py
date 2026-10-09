"""Stage review routes: same engine as /runs, artifact reads, validated edits, invalidation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from idea2hypothesis.pipeline.contracts import validate_stage
from idea2hypothesis.pipeline.models import Stage
from tests.conftest import TOPIC
from tests.fixtures.api_helpers import SERVER_KEY, Harness, harness


@pytest.fixture(autouse=True)
def _server_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("I2H_SERVICE_KEY", SERVER_KEY)


async def _full_run(h: Harness) -> str:
    response = await h.client.post("/api/phase1/start", json={"topic": TOPIC, "auto_approve": True})
    assert response.status_code == 200, response.text
    await h.settle()
    return str(response.json()["run_id"])


async def test_stage1_run_then_read_goal_and_hardware(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        response = await h.client.post("/api/stage1/run", json={"topic": TOPIC})
        assert response.status_code == 200, response.text
        result = response.json()
        run_id = result["run_id"]
        assert result["stage"] == 1 and result["stage_name"] == "TOPIC_INIT"
        assert result["status"] == "completed" and "goal.json" in result["artifacts"]

        goal = await h.client.get(f"/api/stage1/{run_id}/goal")
        assert goal.headers["content-type"].startswith("text/plain")
        assert goal.text.startswith("# Sleep duration and exam performance")
        as_json = (
            await h.client.get(f"/api/stage1/{run_id}/goal", params={"format": "json"})
        ).json()
        assert as_json["researchable"] is True and as_json["topic"] == TOPIC
        hardware = (await h.client.get(f"/api/stage1/{run_id}/hardware")).json()
        assert hardware["tier"] == "cpu_only"

        decision = (await h.client.get(f"/api/stage1/{run_id}/decision")).json()
        assert decision["stage"] == 1 and decision["status"] == "PASSED"
        health = (await h.client.get(f"/api/stage1/{run_id}/health")).json()
        assert health["stage_id"] == "01" and health["status"] == "completed"
        assert health["duration_sec"] is not None and "goal.json" in health["artifacts"]

        # not yet run: no data (the original answered 404 for missing files)
        assert (await h.client.get(f"/api/stage2/{run_id}/problem-tree")).status_code == 404
        assert (await h.client.get(f"/api/stage2/{run_id}/health")).status_code == 404


async def test_stages_run_one_at_a_time_and_check_their_inputs(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = (await h.client.post("/api/stage1/run", json={"topic": TOPIC})).json()["run_id"]
        skipped = await h.client.post(f"/api/stage4/{run_id}/run")
        assert skipped.status_code == 409 and skipped.json()["detail"]["code"] == "MISSING_INPUT"

        for n in (2, 3, 4):
            result = (await h.client.post(f"/api/stage{n}/{run_id}/run")).json()
            assert result["status"] == "completed", result
        tree = (await h.client.get(f"/api/stage2/{run_id}/problem-tree")).text
        assert "SQ1" in tree
        evaluation = (await h.client.get(f"/api/stage2/{run_id}/evaluation")).json()
        assert evaluation["overall"] == 7.3
        queries = (await h.client.get(f"/api/stage3/{run_id}/queries")).json()
        assert isinstance(queries, list) and queries and all(isinstance(q, str) for q in queries)
        assert "core_topic" in (await h.client.get(f"/api/stage3/{run_id}/plan")).text
        assert (await h.client.get(f"/api/stage3/{run_id}/sources")).json()["count"] == 3
        candidates = (
            await h.client.get(f"/api/stage4/{run_id}/candidates", params={"limit": 5})
        ).json()
        assert len(candidates) == 5 and candidates[0]["source_records"]
        stats = (await h.client.get(f"/api/stage4/{run_id}/stats")).json()
        assert stats["unique"] == 12

        bib = await h.client.get(f"/api/stage4/{run_id}/download-bibtex")
        assert bib.status_code == 200
        assert f"{run_id}_references.bib" in bib.headers["content-disposition"]
        assert bib.text.count("@") == 12
        text = (await h.client.get(f"/api/stage4/{run_id}/references-text")).text
        assert text == bib.text

        # the same engine: the run is visible through the Platform routes
        events = await h.events(run_id)
        assert [e["type"] for e in events if e["type"] == "stage.completed"] == [
            "stage.completed"
        ] * 2
        state = await h.state(run_id)
        assert state["status"] == "paused" and state["message"] == "stage_run"


async def test_phase1_run_management_routes(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = await _full_run(h)
        status = (await h.client.get("/api/phase1/status")).json()
        assert status["run_id"] == run_id and status["progress_percentage"] == 100
        assert [s["status"] for s in status["stages"]] == ["completed"] * 9
        assert status["is_running"] is False

        runs = (await h.client.get("/api/phase1/runs")).json()
        assert runs[0]["run_id"] == run_id and runs[0]["has_stage8_hypotheses"] is True
        summary = (await h.client.get(f"/api/phase1/runs/{run_id}/summary")).json()
        assert summary["stage_5_shortlist_count"] == 9
        assert summary["stage_6_knowledge_cards_count"] == 9
        assert summary["stage_8_novelty_report"]["kind"] == "novelty_assessment"
        assert summary["stage_8_hypotheses"].startswith("# Hypotheses")
        checkpoint = (await h.client.get(f"/api/phase1/runs/{run_id}/checkpoint")).json()
        assert sorted(checkpoint["stages"]) == [str(n) for n in range(1, 10)]
        overview = (await h.client.get(f"/api/phase1/runs/{run_id}/health-overview")).json()
        assert [row["stage_id"] for row in overview] == [f"{n:02d}" for n in range(1, 10)]

        # the same run is a Platform run too
        assert (await h.state(run_id))["status"] == "completed"
        # nothing is running, so there is nothing to stop
        assert (await h.client.post("/api/phase1/stop")).status_code == 404

        assert (await h.client.get("/api/stage1/nope-nope/goal")).status_code == 404
        assert (await h.client.get("/api/stage1/!bad/goal")).status_code == 400
        assert (await h.client.delete(f"/api/phase1/runs/{run_id}")).json()["status"] == "deleted"
        assert (await h.client.get("/api/phase1/runs")).json() == []


async def test_stage_artifacts_cards_and_perspectives(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = await _full_run(h)
        names = (await h.client.get(f"/api/stage6/{run_id}/cards")).json()
        assert len(names) == 9 and all(n.startswith("card-") and n.endswith(".json") for n in names)
        merged = (await h.client.get(f"/api/stage6/{run_id}/cards-merged")).json()
        assert [c["card_id"] for c in merged] == [n.removesuffix(".json") for n in names]
        detail = (await h.client.get(f"/api/stage6/{run_id}/cards/{names[0]}")).json()
        assert detail["evidence_scope"] == "abstract"
        assert (
            await h.client.get(f"/api/stage6/{run_id}/cards/..%2F..%2Frun.json")
        ).status_code == 404

        files = (await h.client.get(f"/api/stage8/{run_id}/perspectives")).json()
        assert "innovator.json" in files
        text = await h.client.get(f"/api/stage8/{run_id}/perspectives/innovator.md")
        assert text.status_code == 200 and "Perspective: innovator" in text.text
        assert (
            await h.client.get(f"/api/stage8/{run_id}/perspectives/..%2Frun.json")
        ).status_code == 404
        novelty = (await h.client.get(f"/api/stage8/{run_id}/novelty")).json()
        assert novelty["disclaimer"].startswith("heuristic")
        assert (
            "[final] Hypothesis 1" in (await h.client.get(f"/api/stage8/{run_id}/hypotheses")).text
        )
        assert "Synthesis" in (await h.client.get(f"/api/stage7/{run_id}/synthesis")).text


async def test_put_shortlist_validates_and_invalidates_downstream(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = await _full_run(h)
        shortlist = (await h.client.get(f"/api/stage5/{run_id}/shortlist")).json()
        assert len(shortlist) == 9
        art = h.service.store.artifacts(run_id)

        # an invalid edit (a paper that was never collected) changes nothing
        bad = await h.client.put(
            f"/api/stage5/{run_id}/shortlist", json=[*shortlist, {"paper_id": "p-invented"}]
        )
        assert bad.status_code == 422
        assert len(art.read_jsonl(5, "shortlist.jsonl")) == 9

        edited = await h.client.put(f"/api/stage5/{run_id}/shortlist", json=shortlist[:-2])
        assert edited.status_code == 200, edited.text
        assert edited.json()["downstream_invalidated"] is True
        assert len(art.read_jsonl(5, "shortlist.jsonl")) == 7
        assert validate_stage(Stage.LITERATURE_SCREEN, art).ok
        review = art.read_json(5, "review.json")
        dropped = [d for d in review["decisions"] if d["decision"] == "dropped_by_reviewer"]
        assert {d["paper_id"] for d in dropped} == {r["paper_id"] for r in shortlist[-2:]}
        # stages 6-8 are archived, not read: their endpoints have nothing and the run resumes
        assert (await h.client.get(f"/api/stage6/{run_id}/cards")).json() == []
        assert (await h.client.get(f"/api/stage8/{run_id}/hypotheses")).status_code == 404
        checkpoint = (await h.client.get(f"/api/phase1/runs/{run_id}/checkpoint")).json()
        assert (
            sorted(checkpoint["stages"]) == ["1", "2", "3", "4", "5"] and checkpoint["attempt"] == 2
        )
        assert (h.service.store.run_dir(run_id) / "attempts" / "1" / "stage-08").is_dir()

        assert (await h.client.post(f"/runs/{run_id}/resume")).status_code == 200
        await h.settle()
        assert (await h.state(run_id))["status"] == "completed"
        assert len((await h.client.get(f"/api/stage6/{run_id}/cards")).json()) == 7
        assert h.service.store.read_run(run_id)["human_edits"][0]["stage"] == 5


async def test_put_hypotheses_is_validated_against_the_contract(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = await _full_run(h)
        doc = (
            await h.client.get(f"/api/stage8/{run_id}/hypotheses", params={"format": "json"})
        ).json()
        art = h.service.store.artifacts(run_id)
        original = art.read_text(8, "hypotheses.json")

        broken = json.loads(json.dumps(doc))
        broken["hypotheses"][0]["falsification_criteria"] = "none"
        bad = await h.client.put(
            f"/api/stage8/{run_id}/hypotheses", json={"content": json.dumps(broken)}
        )
        assert bad.status_code == 422
        assert any("falsification_criteria" in e for e in bad.json()["detail"]["errors"])
        assert art.read_text(8, "hypotheses.json") == original

        not_json = await h.client.put(
            f"/api/stage8/{run_id}/hypotheses", json={"content": "# prose"}
        )
        assert not_json.status_code == 422

        good = json.loads(json.dumps(doc))
        good["hypotheses"][0]["statement"] = "Edited statement about sleep and exam scores"
        ok = await h.client.put(
            f"/api/stage8/{run_id}/hypotheses", json={"content": json.dumps(good)}
        )
        assert ok.status_code == 200, ok.text
        assert ok.json()["downstream_invalidated"] is True  # the argument map is redrawn
        assert "Edited statement" in (await h.client.get(f"/api/stage8/{run_id}/hypotheses")).text
        assert validate_stage(Stage.HYPOTHESIS_GEN, art).ok


async def test_put_goal_and_queries_invalidate_their_downstream(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = await _full_run(h)
        art = h.service.store.artifacts(run_id)

        edited = await h.client.put(
            f"/api/stage1/{run_id}/goal",
            json={"content": json.dumps({"objective": "Estimate the dose-response shape."})},
        )
        assert edited.status_code == 200 and edited.json()["downstream_invalidated"] is True
        assert "dose-response" in (await h.client.get(f"/api/stage1/{run_id}/goal")).text
        assert not art.exists(2, "problem_tree.json")  # stage 2 and later were archived

        # rebuild up to stage 3 and edit the queries
        for n in (2, 3):
            assert (await h.client.post(f"/api/stage{n}/{run_id}/run")).json()[
                "status"
            ] == "completed"
        queries = (await h.client.get(f"/api/stage3/{run_id}/queries")).json()
        new = [*queries[:4], "sleep restriction exam performance experiment"]
        put = await h.client.put(f"/api/stage3/{run_id}/queries", json=new)
        assert put.status_code == 200, put.text
        doc = (
            await h.client.get(f"/api/stage3/{run_id}/queries", params={"format": "json"})
        ).json()
        assert [q["text"] for q in doc["queries"]] == new
        assert doc["queries"][-1]["strategy"] == "manual" and doc["queries"][-1]["sub_question_ids"]
        assert validate_stage(Stage.SEARCH_STRATEGY, art).ok
        assert "manual" in (await h.client.get(f"/api/stage3/{run_id}/plan")).text

        invalid = await h.client.put(f"/api/stage3/{run_id}/plan", json={"content": "just: text"})
        assert invalid.status_code == 422


async def test_editing_requires_a_completed_stage_and_an_idle_run(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = (await h.client.post("/api/stage1/run", json={"topic": TOPIC})).json()["run_id"]
        edit = await h.client.put(f"/api/stage7/{run_id}/synthesis", json={"content": "{}"})
        assert edit.status_code == 404  # no synthesis.json yet
        late = await h.client.put(f"/api/stage5/{run_id}/shortlist", json=[])
        assert late.status_code in (404, 409, 422)


async def test_stage5_approve_answers_the_open_screen_gate(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        response = await h.client.post(
            "/api/phase1/start", json={"topic": TOPIC, "auto_approve": False}
        )
        run_id = response.json()["run_id"]
        await h.settle()
        assert (await h.state(run_id))["status"] == "awaiting_review"
        decision = (await h.client.get(f"/api/stage5/{run_id}/decision")).json()
        assert decision["status"] == "AWAITING_REVIEW" and decision["gate"]["kind"] == "screen"
        wrong = await h.client.post(f"/api/stage6/{run_id}/run")
        assert wrong.status_code == 409 and wrong.json()["detail"]["code"] == "GATE_OPEN"

        approved = await h.client.post(f"/api/stage5/{run_id}/approve", params={"reason": "ok"})
        assert approved.json() == {"status": "gate_approved", "run_id": run_id}
        await h.settle()
        # the run goes on to the hypotheses gate, the next stop in copilot mode
        assert (await h.state(run_id))["status"] == "awaiting_review"
        assert h.service.record(run_id)["gate"]["kind"] == "hypotheses"
        decision = (await h.client.get(f"/api/stage5/{run_id}/decision")).json()
        assert decision["status"] == "APPROVED" and decision["reason"] == "ok"
        again = await h.client.post(f"/api/stage5/{run_id}/approve")
        assert again.status_code == 409


async def test_stage_run_with_auto_approve_false_opens_the_gate_on_resume(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id = (await h.client.post("/api/stage1/run", json={"topic": TOPIC})).json()["run_id"]
        for n in (2, 3, 4):
            await h.client.post(f"/api/stage{n}/{run_id}/run")
        result = await h.client.post(f"/api/stage5/{run_id}/run", params={"auto_approve": False})
        assert result.json()["status"] == "completed"
        assert (await h.client.post(f"/runs/{run_id}/resume")).status_code == 200
        await h.settle()
        assert (await h.state(run_id))["status"] == "awaiting_review"


async def test_stage_failures_are_reported_not_hidden(tmp_path: Path) -> None:
    from tests.fixtures import FixtureLiterature

    async with harness(tmp_path, literature=FixtureLiterature(papers=[])) as h:
        run_id = (await h.client.post("/api/stage1/run", json={"topic": TOPIC})).json()["run_id"]
        for n in (2, 3):
            await h.client.post(f"/api/stage{n}/{run_id}/run")
        result = (await h.client.post(f"/api/stage4/{run_id}/run")).json()
        assert result["status"] == "failed" and result["error"]["code"] == "NO_LITERATURE"
        decision = (await h.client.get(f"/api/stage4/{run_id}/decision")).json()
        assert decision["status"] == "FAILED" and "no papers" in decision["reason"]
        assert (await h.state(run_id))["status"] == "failed"
        assert (await h.client.get(f"/api/stage4/{run_id}/candidates")).status_code == 404


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [r["paper_id"] for r in rows]
