"""Contract runs: an independent review of the final set precedes the hypotheses gate."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from idea2hypothesis.pipeline.models import GateAnswer, RunStatus
from idea2hypothesis.pipeline.runner import answer_gate, run_pipeline
from idea2hypothesis.stages.hypothesis_gen import _check_resolutions, changed_hypotheses
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
        {
            "candidate": freed,
            "reason": "duplicate",
            "of": f"H{freed.rsplit('-', 1)[1]}",
            "text": "same claim",
        }
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


def _finding(hypothesis_id: str, severity: str, due: str = "now", issue: str = "x") -> dict:
    return {
        "hypothesis_id": hypothesis_id, "field": "prediction", "severity": severity,
        "category": "baseline", "kind": "fix", "due": due, "issue": issue, "suggestion": None,
    }  # fmt: skip


def _round_two(info: object) -> bool:
    return bool(re.search(r'"changed_hypotheses"\s*:\s*\[', info.user))  # type: ignore[attr-defined]


def _two_round_reviewer(info: object) -> dict:
    if not _round_two(info):
        return {"findings": [
            _finding("H1", "blocking", issue="no baseline"),
            _finding("H2", "blocking", issue="leaks"),
            {**_finding("H3", "blocking", due="later", issue="pin the seed"),
             "category": "reproducibility"},
        ]}  # fmt: skip
    return {
        "findings": [
            _finding("H1", "blocking", issue="still no control"),
            _finding("H2", "blocking", issue="new objection"),
        ],
        "resolutions": [
            {"finding_id": "F1-1", "status": "resolved", "reason": "baseline added"},
            {"finding_id": "F1-2", "status": "maintained", "reason": "leak stands"},
        ],
    }  # fmt: skip


def _revise_h1(info: object) -> dict:
    data = FixtureLLM()._default_hypothesis_gen(info)  # type: ignore[arg-type]
    if re.search(r"^- finding ", info.user, re.MULTILINE):  # type: ignore[attr-defined]
        data["hypotheses"][0]["statement"] += " (with the baseline)"
    return data


async def test_later_findings_are_minor_and_marked(tmp_path: Path) -> None:
    reviewer = FixtureLLM(overrides={"hypothesis_review": _two_round_reviewer})
    services = _services(tmp_path, reviewer=reviewer)
    result = await run_pipeline(request(review_phase=True), services)

    review = services.store.artifacts(result.run_id).read_json(8, "review.json")
    later = [f for f in review["findings"] if f["hypothesis_id"] == "H3"]
    assert later[0]["severity"] == "minor"
    assert later[0]["issue"] == "[for experiment design] pin the seed"
    assert [f["severity"] for f in review["findings"] if f["hypothesis_id"] != "H3"] == [
        "blocking",
        "blocking",
    ]
    assert "resolutions" not in review


async def test_a_design_flaw_marked_later_stays_blocking(tmp_path: Path) -> None:
    def reviewer_answer(info: object) -> dict:
        return {"findings": [{**_finding("H1", "blocking", due="later", issue="leaks"),
                              "category": "leakage"}]}  # fmt: skip

    reviewer = FixtureLLM(overrides={"hypothesis_review": reviewer_answer})
    services = _services(tmp_path, reviewer=reviewer)
    result = await run_pipeline(request(review_phase=True), services)

    review = services.store.artifacts(result.run_id).read_json(8, "review.json")
    assert [(f["issue"], f["severity"]) for f in review["findings"]] == [("leaks", "blocking")]
    assert review["passed"] is False


async def test_a_finding_answered_twice_is_invalid() -> None:
    owed = {"F1-1": {"hypothesis_id": "H1"}}
    data = {"resolutions": [
        {"finding_id": "F1-1", "status": "resolved", "reason": "fixed"},
        {"finding_id": "F1-1", "status": "maintained", "reason": "still open"},
    ]}  # fmt: skip
    assert any("answered twice" in e for e in _check_resolutions(data, owed, {"H1"}).errors)


async def test_missing_due_means_now_and_needs_evidence_is_never_later(tmp_path: Path) -> None:
    def answer(info: object) -> dict:
        no_due = _finding("H1", "blocking", issue="no baseline")
        del no_due["due"]
        later_evidence = {
            **_finding("H2", "blocking", due="later", issue="no source"),
            "kind": "needs_evidence",
        }
        return {"findings": [no_due, later_evidence]}

    reviewer = FixtureLLM(overrides={"hypothesis_review": answer})
    services = _services(tmp_path, reviewer=reviewer)
    result = await run_pipeline(request(review_phase=True), services)

    review = services.store.artifacts(result.run_id).read_json(8, "review.json")
    assert "error" not in review
    assert [(f["issue"], f["severity"]) for f in review["findings"]] == [
        ("no baseline", "blocking"), ("no source", "blocking")
    ]  # fmt: skip


async def test_second_round_checks_the_fixes_and_does_not_move_the_target(tmp_path: Path) -> None:
    llm = FixtureLLM(overrides={"hypothesis_gen": _revise_h1})
    reviewer = FixtureLLM(overrides={"hypothesis_review": _two_round_reviewer})
    services = _services(tmp_path, reviewer=reviewer, llm=llm)
    first = await run_pipeline(request(review_phase=True), services)
    art = services.store.artifacts(first.run_id)
    art.write_run_json(
        "researcher_findings.r1.json",
        [{"finding_id": "F1-1", "anchor": {"locator": "hypothesis:H1/prediction"},
          "issue": "no baseline"}],
    )  # fmt: skip

    second = await answer_gate(
        first.run_id, first.gate["gate_id"], GateAnswer("reject", note="see findings"), services
    )

    assert second.status is RunStatus.AWAITING_REVIEW, second.error
    review = art.read_json(8, "review.json")
    assert review["round"] == 2 and review["passed"] is False
    by_issue = {f["issue"]: f for f in review["findings"]}
    # H1 was changed by the revision: a new blocking finding stays blocking.
    assert by_issue["still no control"]["severity"] == "blocking"
    # H2 was not changed: a new blocking finding there is only major.
    assert by_issue["[unchanged] new objection"]["severity"] == "major"
    # A maintained earlier blocking finding stays blocking and names the earlier id.
    maintained = by_issue["[maintains F1-2] leak stands"]
    assert maintained["hypothesis_id"] == "H2" and maintained["severity"] == "blocking"
    assert not any("F1-1" in f["issue"] for f in review["findings"])
    assert {r["finding_id"]: r["status"] for r in review["resolutions"]} == {
        "F1-1": "resolved", "F1-2": "maintained"
    }  # fmt: skip


async def test_second_round_without_resolutions_is_invalid(tmp_path: Path) -> None:
    def reviewer_answer(info: object) -> dict:
        data = _two_round_reviewer(info)
        data.pop("resolutions", None)
        return data

    reviewer = FixtureLLM(overrides={"hypothesis_review": reviewer_answer})
    services = _services(tmp_path, reviewer=reviewer)
    first = await run_pipeline(request(review_phase=True), services)
    art = services.store.artifacts(first.run_id)
    art.write_run_json("researcher_findings.r1.json", [])

    second = await answer_gate(
        first.run_id, first.gate["gate_id"], GateAnswer("reject", note="again"), services
    )

    review = art.read_json(8, "review.json")
    assert second.status is RunStatus.AWAITING_REVIEW
    assert review["passed"] is False and "resolutions" in review["error"]


def test_changed_hypotheses_compares_content_not_derived_fields() -> None:
    base = {"id": "H1", "statement": "a", "prediction": "> 0", "limitations": ["x"]}
    same = {**base, "limitations": ["x", "caveat"], "contested": [{"from": "y"}]}
    edited = {**base, "statement": "b"}
    new = {"id": "H9", "statement": "c"}

    assert changed_hypotheses([base], [same]) == set()
    assert changed_hypotheses([base], [edited, new]) == {"H1", "H9"}
