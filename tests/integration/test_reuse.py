"""Send-back reuse: a run with a prior run starts at stage 8 and revises the earlier set."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from idea2hypothesis.pipeline import reuse
from idea2hypothesis.pipeline.models import GateAnswer, RunStatus
from idea2hypothesis.pipeline.runner import answer_gate, run_pipeline
from idea2hypothesis.stages import hypothesis_gen
from tests.conftest import make_config, make_services, request
from tests.fixtures import FixtureLLM


def _services(tmp_path: Path, mode: str = "light"):
    cfg = make_config(tmp_path, review={"mode": mode}, llm={"debate_rounds": 1})
    services = make_services(tmp_path, llm=FixtureLLM(), config=cfg)
    services.reviewer = FixtureLLM()
    return services


def _hashes(services, run_id: str, stages=range(1, 8)) -> dict[str, str]:
    art = services.store.artifacts(run_id)
    return {
        f"{n}/{name}": hashlib.sha256(art.path(n, name).read_bytes()).hexdigest()
        for n in stages
        for name in art.list_files(n)
    }


def _notices(services, run_id: str) -> list[str]:
    return [
        e.data["text"]
        for e in services.store.read_events(run_id)
        if e.type == "stage.progress" and e.data.get("kind") == "notice"
    ]


async def test_prior_run_stages_are_reused_and_its_set_is_revised(tmp_path: Path) -> None:
    services = _services(tmp_path)
    first = await run_pipeline(request(review_phase=True), services)
    services.llm = services.reviewer = fresh = FixtureLLM()

    second = await run_pipeline(
        request(
            review_phase=True,
            prior_run_id=first.run_id,
            constraints=("Research lead: Handoff: tighten the baseline",),
        ),
        services,
    )

    assert second.status is RunStatus.AWAITING_REVIEW, second.error
    assert fresh.count("topic_init") == 0 and fresh.count("knowledge_extract") == 0
    assert fresh.count("synthesis") == 0 and fresh.count("hypothesis_gen") >= 1
    assert _hashes(services, second.run_id) == _hashes(services, first.run_id)
    art = services.store.artifacts(second.run_id)
    assert art.verify_manifest(3, run_id=second.run_id)
    checkpoint = services.store.read_checkpoint(second.run_id)
    assert {checkpoint["stages"][str(n)]["reused_from"] for n in range(1, 8)} == {first.run_id}
    assert services.store.read_run(second.run_id)["reused_from"] == first.run_id
    merge = next(c for c in fresh.calls if c.key == "hypothesis_gen")
    assert "Previous hypotheses: H1:" in merge.user and "same id, same statement" in merge.user
    assert "Research lead: Handoff: tighten the baseline" in merge.user
    old = services.store.artifacts(first.run_id).read_json(8, "hypotheses.json")
    new = art.read_json(8, "hypotheses.json")
    assert [h["id"] for h in new["hypotheses"]] == [h["id"] for h in old["hypotheses"]]
    assert _notices(services, second.run_id) == [
        f"Reused stages 1-7 of run {first.run_id}; revising its hypotheses."
    ]


async def test_unknown_prior_run_runs_every_stage_and_says_why(tmp_path: Path) -> None:
    services = _services(tmp_path)
    fresh = services.llm = FixtureLLM()
    services.reviewer = FixtureLLM()

    result = await run_pipeline(request(review_phase=True, prior_run_id="i2h-missing"), services)

    assert result.status is RunStatus.AWAITING_REVIEW, result.error
    assert fresh.count("topic_init") == 1
    assert _notices(services, result.run_id) == [
        "Run i2h-missing was not reused (run i2h-missing is not in the local store); "
        "running every stage."
    ]
    assert "reused_from" not in services.store.read_run(result.run_id)


async def test_tampered_prior_stage_is_not_reused(tmp_path: Path) -> None:
    services = _services(tmp_path)
    first = await run_pipeline(request(review_phase=True), services)
    art = services.store.artifacts(first.run_id)
    target = art.path(3, art.list_files(3)[0])
    target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    services.llm = services.reviewer = fresh = FixtureLLM()

    second = await run_pipeline(request(review_phase=True, prior_run_id=first.run_id), services)

    assert second.status is RunStatus.AWAITING_REVIEW, second.error
    assert fresh.count("topic_init") == 1
    [notice] = _notices(services, second.run_id)
    assert "stage 3" in notice and "no valid artifacts" in notice


async def test_prior_run_with_other_topic_is_not_reused(tmp_path: Path) -> None:
    services = _services(tmp_path)
    first = await run_pipeline(request(review_phase=True), services)
    services.llm = services.reviewer = fresh = FixtureLLM()

    second = await run_pipeline(
        request(
            review_phase=True,
            prior_run_id=first.run_id,
            topic="Effect of caffeine intake on exam performance in university students",
        ),
        services,
    )

    assert second.status is RunStatus.AWAITING_REVIEW, second.error
    assert fresh.count("topic_init") == 1
    assert "different topic" in _notices(services, second.run_id)[0]


async def test_reused_run_does_not_ask_the_earlier_questions_again(tmp_path: Path) -> None:
    services = _services(tmp_path, mode="copilot")
    first = await run_pipeline(request(review_phase=True), services)
    assert first.gate is not None and first.gate["kind"] == "screen"
    first = await answer_gate(first.run_id, first.gate["gate_id"], GateAnswer("approve"), services)
    assert first.gate is not None and first.gate["kind"] == "hypotheses"

    second = await run_pipeline(request(review_phase=True, prior_run_id=first.run_id), services)

    assert second.status is RunStatus.AWAITING_REVIEW, second.error
    assert second.gate is not None and second.gate["kind"] == "hypotheses"
    opened = [e.data["kind"] for e in services.store.read_events(second.run_id)
              if e.type == "gate.opened"]  # fmt: skip
    assert opened == ["hypotheses"]


async def _second_run(tmp_path: Path, first_kwargs: dict, second_kwargs: dict):
    services = _services(tmp_path)
    first = await run_pipeline(request(review_phase=True, **first_kwargs), services)
    services.llm = services.reviewer = fresh = FixtureLLM()
    second = await run_pipeline(
        request(review_phase=True, prior_run_id=first.run_id, **second_kwargs), services
    )
    assert second.status is RunStatus.AWAITING_REVIEW, second.error
    return services, fresh, second, first


async def test_prior_run_with_other_domains_is_not_reused(tmp_path: Path) -> None:
    services, fresh, second, _ = await _second_run(tmp_path, {}, {"domains": ("physics",)})

    assert fresh.count("topic_init") == 1
    assert "different domains" in _notices(services, second.run_id)[0]


async def test_prior_run_with_other_framing_is_not_reused(tmp_path: Path) -> None:
    services, fresh, second, _ = await _second_run(
        tmp_path,
        {"constraints": ("Research lead: Research question: does sleep help?",)},
        {"constraints": ("Research lead: Research question: does caffeine help?",)},
    )

    assert fresh.count("topic_init") == 1
    assert "different framing" in _notices(services, second.run_id)[0]


async def test_new_directive_or_handoff_does_not_block_reuse(tmp_path: Path) -> None:
    question = "Research lead: Research question: does sleep help?"
    services, fresh, second, _ = await _second_run(
        tmp_path,
        {"constraints": (question,)},
        {
            "constraints": (
                question,
                "Research lead: Directive: public data only",
                "Research lead: Handoff: tighten H1",
            )
        },
    )

    assert fresh.count("topic_init") == 0
    assert "Reused stages 1-7" in _notices(services, second.run_id)[0]


async def test_failure_while_adopting_runs_every_stage_with_a_generic_notice(
    tmp_path: Path, monkeypatch
) -> None:
    def broken(prior):
        raise KeyError("/secret/local/path")

    monkeypatch.setattr(reuse, "_revision_text", broken)
    services, fresh, second, first = await _second_run(tmp_path, {}, {})

    assert fresh.count("topic_init") == 1
    assert _notices(services, second.run_id) == [
        f"Run {first.run_id} was not reused (the prior run's files could not be copied); "
        "running every stage."
    ]
    assert "reused_from" not in services.store.read_run(second.run_id)


def _convergence_reviewer(info: object) -> dict:
    finding = {
        "hypothesis_id": "H2", "field": "prediction", "severity": "blocking",
        "category": "baseline", "kind": "fix", "due": "now", "issue": "leaks",
        "suggestion": None,
    }  # fmt: skip
    if not re.search(r'"changed_hypotheses"\s*:\s*\[', info.user):  # type: ignore[attr-defined]
        return {"findings": [finding]}
    return {
        "findings": [{**finding, "issue": "new objection"}],
        "resolutions": [{"finding_id": "P1-1", "status": "maintained", "reason": "leak stands"}],
    }


async def test_reused_run_reviews_against_the_prior_runs_final_review(tmp_path: Path) -> None:
    services = _services(tmp_path)
    services.reviewer = FixtureLLM(overrides={"hypothesis_review": _convergence_reviewer})
    first = await run_pipeline(request(review_phase=True), services)

    second = await run_pipeline(request(review_phase=True, prior_run_id=first.run_id), services)

    assert second.status is RunStatus.AWAITING_REVIEW, second.error
    review = services.store.artifacts(second.run_id).read_json(8, "review.json")
    assert "error" not in review
    by_issue = {f["issue"]: f for f in review["findings"]}
    assert by_issue["[unchanged] new objection"]["severity"] == "major"
    assert by_issue["[maintains P1-1] leak stands"]["severity"] == "blocking"
    assert [r["finding_id"] for r in review["resolutions"]] == ["P1-1"]


def _edited(statements: dict[str, str]):
    """A hypothesis_gen answer: the fixture's set with the given statements replaced."""

    def answer(info):
        data = FixtureLLM()._default_hypothesis_gen(info)
        for h in data["hypotheses"]:
            h["statement"] = statements.get(h["id"], h["statement"])
        return data

    return answer


async def _revise(tmp_path: Path, handoff: str, answers: list):
    services = _services(tmp_path)
    first = await run_pipeline(request(review_phase=True), services)
    old = services.store.artifacts(first.run_id).read_json(8, "hypotheses.json")["hypotheses"]
    services.llm = services.reviewer = fresh = FixtureLLM(overrides={"hypothesis_gen": answers})
    second = await run_pipeline(
        request(
            review_phase=True,
            prior_run_id=first.run_id,
            constraints=(f"Research lead: Handoff: {handoff}",),
        ),
        services,
    )
    assert second.status is RunStatus.AWAITING_REVIEW, second.error
    new = services.store.artifacts(second.run_id).read_json(8, "hypotheses.json")["hypotheses"]
    return fresh, old, new


async def test_revision_that_renumbers_is_retried_until_ids_keep_their_claims(
    tmp_path: Path,
) -> None:
    renumbered = _edited({"H2": "A claim that was H3 before."})
    fresh, old, new = await _revise(
        tmp_path, "Make H1's prediction a number", [renumbered, _edited({})]
    )

    assert fresh.count("hypothesis_gen") == 2
    assert {h["id"]: h["statement"] for h in new} == {h["id"]: h["statement"] for h in old}


async def test_named_hypothesis_may_change_its_statement(tmp_path: Path) -> None:
    fresh, old, new = await _revise(
        tmp_path, "Make H1's prediction a number", [_edited({"H1": "H1 with a number."})]
    )

    assert fresh.count("hypothesis_gen") == 1
    assert [h["id"] for h in new] == [h["id"] for h in old]
    assert new[0]["statement"] == "H1 with a number."
    assert [h["statement"] for h in new[1:]] == [h["statement"] for h in old[1:]]


def test_new_hypothesis_with_a_previous_id_is_rejected_by_name() -> None:
    prior = {"H1": "Sleep helps", "H2": "Caffeine hurts"}
    reused_id = {"hypotheses": [
        {"id": "H1", "statement": "Sleep  helps"},
        {"id": "H2", "statement": "A different claim."},
    ]}  # fmt: skip

    errors = hypothesis_gen._check_ids_bound(reused_id, prior, named=set()).errors
    assert len(errors) == 1 and "H2" in errors[0] and "id that no previous" in errors[0]

    dropped = {"hypotheses": [{"id": "H1", "statement": "Sleep helps."}]}
    errors = hypothesis_gen._check_ids_bound(dropped, prior, named=set()).errors
    assert len(errors) == 1 and "H2 is missing" in errors[0]

    assert hypothesis_gen._check_ids_bound(reused_id, prior, named={"H2"}).ok
