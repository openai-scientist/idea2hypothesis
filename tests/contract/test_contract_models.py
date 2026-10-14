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
