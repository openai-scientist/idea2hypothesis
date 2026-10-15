"""Vendored examples parse with the engine's own models and round-trip against the schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from idea2hypothesis.api.contract import models
from tests.contract.fake_platform import validate

EXAMPLES = Path(__file__).parent / "vendor" / "examples"

#: Examples of wire models this engine has no use for yet (later slices add them).
NOT_YET = {
    "DatasetDescriptor", "LlmCall", "PipelineRecommendation", "RunCost",
    "TelemetryBatch", "TelemetryRecord", "TokenUsage", "EmptyPayload", "PhaseChanged",
    "StepChanged", "ReviewCompleted", "GateClosed", "RunPublished", "ArtifactDelivered",
}  # fmt: skip


def _names() -> list[str]:
    return sorted(p.stem for p in EXAMPLES.glob("*.json") if p.stem not in NOT_YET)


@pytest.mark.parametrize("name", _names())
def test_example_parses_with_engine_model(name):
    document = json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))
    if name.startswith("event-"):
        model = models.EventEnvelope
    else:
        model = getattr(models, name, None)
        assert model is not None, f"the engine has no model for {name}"
    model.model_validate(document)


def test_outgoing_run_state_matches_the_schema():
    gate = models.Gate(
        gate_id="g1", kind="artifact_review", step_key="hypothesis_gen", prompt="p",
        opened_at="2026-10-14T00:00:00Z", artifact_ids=["a1"],
    )  # fmt: skip
    state = models.RunState(
        engine_run_id="e1", stage_run_id="s1", phase="awaiting_researcher",
        step_key="hypothesis_gen", waiting_input=True, open_gate=gate,
    )  # fmt: skip
    validate("RunState", state.model_dump(mode="json"))


def test_major_version_mismatch_is_rejected():
    document = json.loads((EXAMPLES / "RunCreateRequest.json").read_text(encoding="utf-8"))
    document["contract_version"] = "2.0.0"
    with pytest.raises(ValueError, match="major version"):
        models.RunCreateRequest.model_validate(document)
    document["contract_version"] = "1.7.3"
    models.RunCreateRequest.model_validate(document)


def _finding(severity):
    return models.Finding(
        finding_id="F1-1", anchor=models.Anchor(locator="hypothesis:H1/statement"),
        severity=severity, issue="i", author="reviewer",
    )  # fmt: skip


def test_review_findings_passed_must_match_blocking():
    base = {"round": 1, "reviewed_artifact_id": "a", "reviewer_model": "m"}
    models.ReviewFindings(**base, findings=[_finding("blocking")], passed=False)
    models.ReviewFindings(**base, findings=[_finding("minor")], passed=True)
    with pytest.raises(ValueError, match="passed"):
        models.ReviewFindings(**base, findings=[_finding("blocking")], passed=True)
    with pytest.raises(ValueError, match="passed"):
        models.ReviewFindings(**base, findings=[], passed=False)


def test_a_failed_review_is_reported_as_an_incomplete_review_not_a_pass():
    from idea2hypothesis.api.contract.review import review_findings

    doc = {"round": 2, "reviewer_model": "m", "findings": [], "passed": False,
           "error": "invalid JSON"}  # fmt: skip
    report = review_findings(doc, artifact_id="a1")
    assert report.passed is False
    [finding] = report.findings
    assert (finding.severity, finding.kind) == ("blocking", "needs_evidence")
    assert finding.anchor.locator == "document:hypotheses" and "invalid JSON" in finding.issue


def test_revision_item_needs_a_note_unless_fixed():
    models.RevisionItem(finding_id="F1-1", outcome="fixed")
    models.RevisionItem(finding_id="F1-1", outcome="rebutted", note="because")
    for note in (None, "  "):
        with pytest.raises(ValueError, match="note"):
            models.RevisionItem(finding_id="F1-1", outcome="not_fixed", note=note)


def test_revision_response_from_a_reviser_without_notes_is_valid():
    from idea2hypothesis.api.contract.review import revision_response

    answer = revision_response(
        {"items": [{"finding_id": "F1-1", "outcome": "rebutted"},
                   {"finding_id": "F1-2", "outcome": "fixed"}]},
        round_=1, before="a", after="b",
    )  # fmt: skip
    assert answer.items[0].note == "No reason given by the reviser."
    assert answer.items[1].note is None


def test_gate_rules():
    opts = [models.GateOption(option_id=i, label=i) for i in ("a", "b")]
    base = {"gate_id": "g", "step_key": None, "prompt": "p", "opened_at": "t"}
    models.Gate(**base, kind="question", options=opts, default_option_id="a")
    with pytest.raises(ValueError, match="question"):
        models.Gate(**base, kind="question", options=opts, default_option_id="z")
    with pytest.raises(ValueError, match="question"):
        models.Gate(**base, kind="question", options=opts[:1], default_option_id="a")
    with pytest.raises(ValueError, match="artifact_review"):
        models.Gate(**base, kind="artifact_review", artifact_ids=[])
    with pytest.raises(ValueError, match="artifact_review"):
        models.Gate(**base, kind="artifact_review", artifact_ids=["x"], options=opts)


def test_run_state_failure_exactly_when_failed():
    base = {"engine_run_id": "e", "stage_run_id": "s", "step_key": None,
            "waiting_input": False, "open_gate": None}  # fmt: skip
    failure = models.FailureInfo(code="c", message="m")
    models.RunState(**base, phase="failed", failure=failure)
    with pytest.raises(ValueError, match="failure"):
        models.RunState(**base, phase="failed")
    with pytest.raises(ValueError, match="failure"):
        models.RunState(**base, phase="drafting", failure=failure)


def test_run_state_hides_the_error_until_the_failed_phase_is_projected():
    from idea2hypothesis.api.contract.mapping import Timeline, run_state

    record = {"status": "failed", "error": {"code": "c", "message": "m"}}
    meta = {"stage_run_id": "s"}
    early = run_state("e", record, meta, Timeline(phase="drafting"), 1, None)
    assert early.failure is None
    late = run_state("e", record, meta, Timeline(phase="failed"), 2, None)
    assert late.failure is not None


def test_a_review_gate_shows_only_once_its_hypotheses_were_pushed():
    from idea2hypothesis.api.contract.mapping import Timeline, run_state
    from idea2hypothesis.pipeline import gates

    gate = {"gate_id": "g", "kind": gates.HYPOTHESES, "stage": 8, "status": "open",
            "opened_at": "2026-10-16T00:00:00Z"}  # fmt: skip
    record = {"status": "awaiting_review", "gate": gate}
    meta = {"stage_run_id": "s"}
    assert run_state("e", record, meta, Timeline(), 1, None).open_gate is None
    shown = run_state("e", record, meta, Timeline(), 1, "a1").open_gate
    assert shown is not None and shown.artifact_ids == ["a1"]


def _event(**kw):
    base = {"source_seq": 1, "type": "run.phase_changed", "class": "lifecycle",
            "occurred_at": "2026-10-16T00:00:00Z", "phase": "drafting"}  # fmt: skip
    return models.EventEnvelope.model_validate({**base, **kw})


def test_event_envelope_rules():
    _event()
    act = {"type": "i2h.card.extracted", "class": "activity", "step_key": "s", "actor": "a",
           "summary": "x."}  # fmt: skip
    _event(**act)
    for bad in (
        {"type": "run.cost"},  # a lifecycle type sent as activity
        {"type": "extracted"},  # no engine prefix
        {"step_key": None},
        {"actor": " "},
        {"summary": None},
    ):
        with pytest.raises(ValueError, match="activity"):
            _event(**{**act, **bad})
    with pytest.raises(ValueError, match="unknown lifecycle"):
        _event(type="run.bogus")
    with pytest.raises(ValueError, match="phase is required"):
        _event(phase=None)
    with pytest.raises(ValueError, match="step_key"):
        _event(type="run.step_changed")
    with pytest.raises(ValueError, match="terminal"):
        _event(type="run.failed", phase="drafting", payload={"code": "c", "message": "m"})
    with pytest.raises(ValueError):
        _event(type="run.failed", phase="failed", payload={})
    with pytest.raises(ValueError, match="question"):
        _event(type="run.gate_opened", payload={
            "gate_id": "g", "kind": "question", "step_key": None, "prompt": "p",
            "opened_at": "t"})  # fmt: skip
