"""The evidence artifact and send-back reuse over the contract."""

from __future__ import annotations

import json

from tests.conftest import TOPIC
from tests.contract.fake_platform import validate
from tests.fixtures.contract_harness import contract_harness, run_body

STAGE = "stage-run-1"
APPROVE = {"action": "approve", "actor": "user"}


def _hypotheses(h, stage: str = STAGE) -> dict:
    return json.loads(h.platform.uploaded(stage, "i2h.hypotheses/v1")[-1].content)


def _evidence(h, stage: str = STAGE) -> dict:
    return json.loads(h.platform.uploaded(stage, "i2h.evidence/v1")[-1].content)


async def test_capabilities_declare_evidence_and_reuse(tmp_path):
    async with contract_harness(tmp_path) as h:
        response = await h.client.get("/v1/capabilities")
        body = response.json()
        validate("Capabilities", body)

        outputs = {o["schema"]: o["primary"] for o in body["output_schemas"]}
        assert outputs["i2h.evidence/v1"] is False and outputs["i2h.hypotheses/v1"] is True
        assert body["reuses_prior_run"] is True


async def test_evidence_artifact_has_the_cards_the_hypotheses_cite(tmp_path):
    async with contract_harness(tmp_path) as h:
        await h.reach(STAGE)

        [upload] = h.platform.uploaded(STAGE, "i2h.evidence/v1")
        assert upload.meta["primary"] is False
        evidence = _evidence(h)
        assert evidence["schema_version"] == "i2h.evidence/v1"
        cited = {
            ref: {x["id"] for x in _hypotheses(h)["hypotheses"] if ref in x["evidence_refs"]}
            for x in _hypotheses(h)["hypotheses"]
            for ref in x["evidence_refs"]
        }
        assert {c["card_id"] for c in evidence["cards"]} == set(cited)
        for card in evidence["cards"]:
            assert card["title"] and set(card["hypothesis_ids"]) == cited[card["card_id"]]
            assert card["paper_id"] and "findings" in card and "limitations" in card


async def test_evidence_follows_the_published_set(tmp_path):
    async with contract_harness(tmp_path) as h:
        state = await h.reach(STAGE)
        gate_id = state["open_gate"]["gate_id"]
        body = {**APPROVE, "params": {"dropped": ["H3"]}}
        await h.client.post(f"/v1/runs/{state['engine_run_id']}/gates/{gate_id}", json=body)
        await h.settle()

        versions = h.platform.uploaded(STAGE, "i2h.evidence/v1")
        assert [u.meta["version"] for u in versions] == [1, 2]
        final = json.loads(versions[-1].content)
        assert all("H3" not in c["hypothesis_ids"] for c in final["cards"])
        assert {i for c in final["cards"] for i in c["hypothesis_ids"]} == {"H1", "H2"}


async def test_prior_engine_run_is_reused_and_announced(tmp_path):
    keys: list[str] = []
    async with contract_harness(tmp_path, on_call=lambda info: keys.append(info.key)) as h:
        first = await h.reach(STAGE)
        keys.clear()
        body = run_body("stage-run-2", platform_run_id="platform-run-1")
        body["inputs"] = {
            "params": {"topic": TOPIC, "domains": [], "guidance": ["Tighten H1"]},
            "prior_engine_run_id": first["engine_run_id"],
        }
        created = await h.client.post("/v1/runs", json=body)
        await h.settle()

        assert created.status_code == 201, created.text
        state = await h.state(created.json()["engine_run_id"])
        assert state["phase"] == "awaiting_researcher"
        assert "topic_init" not in keys and "knowledge_extract" not in keys
        assert "hypothesis_gen" in keys
        notes = [e for e in h.platform.events["stage-run-2"] if e["type"] == "i2h.agent.message"]
        assert any(f"Reused stages 1-7 of run {first['engine_run_id']}" in e["summary"]
                   for e in notes)  # fmt: skip
        assert [x["id"] for x in _hypotheses(h, "stage-run-2")["hypotheses"]] == [
            x["id"] for x in _hypotheses(h)["hypotheses"]
        ]


async def test_missing_prior_engine_run_runs_in_full_and_says_why(tmp_path):
    keys: list[str] = []
    async with contract_harness(tmp_path, on_call=lambda info: keys.append(info.key)) as h:
        body = run_body(STAGE)
        body["inputs"] = {
            "params": {"topic": TOPIC, "domains": []}, "prior_engine_run_id": "i2h-missing",
        }
        created = await h.client.post("/v1/runs", json=body)
        await h.settle()

        assert created.status_code == 201, created.text
        assert keys.count("topic_init") == 1
        notes = [e for e in h.platform.events[STAGE] if e["type"] == "i2h.agent.message"]
        assert any("was not reused" in e["summary"] for e in notes)


async def _perspective_systems(tmp_path, *, domains, profile=None) -> set[str]:
    systems: set[str] = set()

    def record(info) -> None:
        if info.key == "perspective":
            systems.add(info.system)

    async with contract_harness(tmp_path, on_call=record) as h:
        body = run_body(STAGE)
        body["inputs"] = {"params": {"topic": TOPIC, "domains": domains}}
        if profile:
            body["profile"] = {"domain": profile}
        created = await h.client.post("/v1/runs", json=body)
        await h.settle()
        assert created.status_code == 201, created.text
    return systems


async def test_run_domain_selects_the_hypothesis_roles(tmp_path):
    systems = await _perspective_systems(tmp_path, domains=["education", "se"])

    assert systems and all("software engineering" in s for s in systems)


async def test_profile_domain_selects_roles_when_the_params_name_none(tmp_path):
    systems = await _perspective_systems(tmp_path, domains=["education"], profile="se")

    assert systems and all("software engineering" in s for s in systems)


async def test_unknown_domains_keep_the_default_roles(tmp_path):
    systems = await _perspective_systems(tmp_path, domains=["education", "psychology"])

    assert systems and not any("software engineering" in s for s in systems)
