"""Contract runs: an independent review of the final set precedes the hypotheses gate."""

from __future__ import annotations

import hashlib
from pathlib import Path

from idea2hypothesis.pipeline.models import GateAnswer, RunStatus
from idea2hypothesis.pipeline.runner import answer_gate, run_pipeline
from tests.conftest import make_config, make_services, request
from tests.fixtures import FixtureLLM


def _services(
    tmp_path: Path,
    *,
    reviewer: FixtureLLM | None,
    llm: FixtureLLM | None = None,
    mode: str = "light",
    **sections,
):
    cfg = make_config(tmp_path, review={"mode": mode}, llm={"debate_rounds": 1}, **sections)
    services = make_services(tmp_path, llm=llm or FixtureLLM(), config=cfg)
    services.reviewer = reviewer
    return services


def _hashes(services, run_id: str, stages=range(1, 8)) -> dict[str, str]:
    art = services.store.artifacts(run_id)
    return {
        f"{n}/{name}": hashlib.sha256(art.path(n, name).read_bytes()).hexdigest()
        for n in stages
        for name in art.list_files(n)
    }


def _phases(services, run_id: str) -> list[str]:
    return [
        e.data["phase"] for e in services.store.read_events(run_id) if e.type == "phase.changed"
    ]


async def test_review_phase_opens_hypotheses_gate_in_light_mode(tmp_path: Path) -> None:
    services = _services(tmp_path, reviewer=FixtureLLM())
    result = await run_pipeline(request(review_phase=True), services)

    assert result.status is RunStatus.AWAITING_REVIEW, result.error
    assert result.gate is not None and result.gate["kind"] == "hypotheses"
    art = services.store.artifacts(result.run_id)
    review = art.read_json(8, "review.json")
    assert review["round"] == 1 and review["passed"] is True and review["findings"]
    assert review["findings"][0]["category"] == "baseline"
    assert not art.stage_dir(9).exists()
    types = [e.type for e in services.store.read_events(result.run_id)]
    assert types.index("review.completed") < types.index("gate.opened")
    assert _phases(services, result.run_id) == ["reviewing"]


async def test_review_goes_to_reviewer_only(tmp_path: Path) -> None:
    llm, reviewer = FixtureLLM(), FixtureLLM()
    services = _services(tmp_path, reviewer=reviewer, llm=llm)
    await run_pipeline(request(review_phase=True), services)

    assert reviewer.count("hypothesis_review") == 1 and llm.count("hypothesis_review") == 0
    assert reviewer.count("debate_judge") == 1 and llm.count("debate_judge") == 0
    assert llm.count("hypothesis_gen") >= 1 and reviewer.count("hypothesis_gen") == 0


async def test_missing_reviewer_fails_run(tmp_path: Path) -> None:
    services = _services(tmp_path, reviewer=None)
    result = await run_pipeline(request(review_phase=True), services)

    assert result.status is RunStatus.FAILED
    assert result.error is not None and result.error.code == "REVIEWER_NOT_CONFIGURED"
    art = services.store.artifacts(result.run_id)
    assert not art.exists(8, "review.json") and not art.exists(8, "hypotheses.json")


def _contested_set(info: object) -> dict:
    """H1 is built from the candidate with a standing fatal objection."""
    data = FixtureLLM()._default_hypothesis_gen(info)  # type: ignore[arg-type]
    first = data["hypotheses"][0]
    freed = first["from"][0]
    first["from"] = ["innovator-1"]
    data["not_used"].append(
        {"candidate": freed, "reason": "duplicate", "of": f"H{freed.rsplit('-', 1)[1]}",
         "text": "same claim"}
    )
    return data


def _fatal_review(info: object) -> dict:
    data = FixtureLLM()._default_debate_review(info)  # type: ignore[arg-type]
    if "as the pragmatist perspective" in info.system:  # type: ignore[attr-defined]
        data["reviews"] = [
            {"item": r["item"], "verdict": "stands", "severity": "fatal", "flaw": "unsupported",
             "text": "no card supports it"}
            for r in data["reviews"]
        ]  # fmt: skip
    return data


async def test_contested_hypothesis_gets_blocking_finding(tmp_path: Path) -> None:
    llm = FixtureLLM(overrides={"debate_review": _fatal_review, "hypothesis_gen": _contested_set})
    services = _services(tmp_path, reviewer=FixtureLLM(), llm=llm)
    result = await run_pipeline(request(review_phase=True), services)

    assert result.status is RunStatus.AWAITING_REVIEW, result.error
    art = services.store.artifacts(result.run_id)
    doc = art.read_json(8, "hypotheses.json")
    assert doc["hypotheses"][0]["contested"]
    review = art.read_json(8, "review.json")
    blocking = [f for f in review["findings"] if f["severity"] == "blocking"]
    assert [f["hypothesis_id"] for f in blocking] == ["H1"]
    assert blocking[0]["category"] == "evidence" and "no card supports it" in blocking[0]["issue"]
    assert review["passed"] is False


async def test_invalid_review_answer_opens_gate_with_error(tmp_path: Path) -> None:
    reviewer = FixtureLLM(overrides={"hypothesis_review": {"findings": "none"}})
    services = _services(tmp_path, reviewer=reviewer)
    result = await run_pipeline(request(review_phase=True), services)

    assert result.status is RunStatus.AWAITING_REVIEW
    review = services.store.artifacts(result.run_id).read_json(8, "review.json")
    assert review["passed"] is False and review["error"]
    opened = [e for e in services.store.read_events(result.run_id) if e.type == "gate.opened"]
    assert opened[-1].data["review_error"] == review["error"]


async def test_reject_reruns_only_hypothesis_gen(tmp_path: Path) -> None:
    services = _services(tmp_path, reviewer=FixtureLLM())
    first = await run_pipeline(request(review_phase=True), services)
    before = _hashes(services, first.run_id)
    art = services.store.artifacts(first.run_id)
    art.write_run_json(
        "researcher_findings.r1.json",
        [{
            "finding_id": "F1", "anchor": {"locator": "hypothesis:H1/prediction"},
            "issue": "name the baseline", "suggestion": "compare with 7 hours",
        }],
    )  # fmt: skip

    second = await answer_gate(
        first.run_id, first.gate["gate_id"], GateAnswer("reject", note="see findings"), services
    )

    assert second.status is RunStatus.AWAITING_REVIEW and second.attempt == 2
    assert second.gate["kind"] == "hypotheses" and second.gate["gate_id"].endswith("a2")
    assert _hashes(services, first.run_id) == before
    responses = art.read_json(8, "revision_responses.json")["items"]
    assert [r["finding_id"] for r in responses] == ["F1"]
    assert art.read_json(8, "review.json")["round"] == 2
    assert _phases(services, first.run_id) == ["reviewing", "reviewing"]


async def test_approve_with_dropped_and_kept_then_argument_map(tmp_path: Path) -> None:
    services = _services(tmp_path, reviewer=FixtureLLM())
    first = await run_pipeline(request(review_phase=True), services)
    art = services.store.artifacts(first.run_id)
    doc = art.read_json(8, "hypotheses.json")
    used = {c for h in doc["hypotheses"] for c in h["from"]}
    pick = next(u["candidate"] for u in doc["not_used"] if u["candidate"] not in used)

    done = await answer_gate(
        first.run_id,
        first.gate["gate_id"],
        GateAnswer("approve", dropped=("H3",), kept=(pick,), note="keep it"),
        services,
    )

    assert done.status is RunStatus.COMPLETED, done.error
    final = art.read_json(8, "hypotheses.json")
    assert [h["id"] for h in final["hypotheses"]] == ["H1", "H2", "H4"]
    mapped = {r["hypothesis_id"] for r in art.read_json(9, "argument_map.json")["rationales"]}
    assert mapped == {"H1", "H2", "H4"}


async def test_without_review_phase_behaviour_unchanged(tmp_path: Path) -> None:
    llm = FixtureLLM()
    services = _services(tmp_path, reviewer=None, llm=llm)
    result = await run_pipeline(request(), services)

    assert result.status is RunStatus.COMPLETED
    art = services.store.artifacts(result.run_id)
    assert not art.exists(8, "review.json") and llm.count("hypothesis_review") == 0
    types = [e.type for e in services.store.read_events(result.run_id)]
    assert "gate.opened" not in types and "phase.changed" not in types
