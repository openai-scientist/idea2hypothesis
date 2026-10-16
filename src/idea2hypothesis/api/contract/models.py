"""Engine-owned wire models; synchronized through published JSON Schema."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

CONTRACT_VERSION = "1.2.0"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class ResearchProfile(ContractModel):
    domain: str | None = None
    research_type: str | None = None

    @property
    def is_base(self) -> bool:
        return self.domain is None and self.research_type is None


class StepInfo(ContractModel):
    step_key: str
    label: str
    order: int = Field(ge=1)


class OutputDecl(ContractModel):
    schema_id: str = Field(alias="schema")
    primary: bool = False


class Capabilities(ContractModel):
    engine_key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    engine_version: str
    contract_version: str
    stage_keys: list[str] = Field(min_length=1)
    input_schemas: list[str] = Field(default_factory=list)
    output_schemas: list[OutputDecl]
    steps: list[StepInfo]
    question_gates: dict[Literal["auto", "light", "copilot", "full"], list[str]] = Field(
        default_factory=dict
    )
    activity_types: list[str] = Field(default_factory=list)
    integrations: list[str] = Field(default_factory=list)
    optional_integrations: list[str] = Field(default_factory=list)
    providers: list[Literal["openai_compatible", "bedrock"]] = Field(
        default_factory=lambda: ["openai_compatible"]
    )
    auto_revision: bool
    feedback_schemas: list[str] = Field(default_factory=list)
    reuses_prior_run: bool = False
    profiles: list[ResearchProfile]

    @model_validator(mode="after")
    def validate_declarations(self) -> Capabilities:
        if sum(output.primary for output in self.output_schemas) != 1:
            raise ValueError("exactly one primary output is required")
        if not any(profile.is_base for profile in self.profiles):
            raise ValueError("base profile is required")
        if any(not kind.startswith(self.engine_key + ".") for kind in self.activity_types):
            raise ValueError("activity types require the engine prefix")
        if set(self.integrations) & set(self.optional_integrations):
            raise ValueError("integration declarations must not overlap")
        return self


class Health(ContractModel):
    status: Literal["ok", "degraded", "down"]
    engine_version: str
    contract_version: str
    active_runs: int = Field(ge=0)
    checks: dict[str, bool] | None = None
    last_error: str | None = None


class ModelConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    provider: Literal["openai_compatible", "bedrock"]
    base_url: str | None = None
    model: str = Field(min_length=1)
    api_key: SecretStr | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def identity(self) -> tuple[str, str]:
        return self.provider, self.model


class Models(ContractModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    stage: ModelConfig
    review: ModelConfig

    @model_validator(mode="after")
    def distinct_models(self) -> Models:
        if self.stage.identity == self.review.identity:
            raise ValueError("review model must differ from stage model")
        return self


Phase = Literal[
    "drafting", "reviewing", "revising", "awaiting_researcher",
    "published", "failed", "cancelled", "paused",
]  # fmt: skip
InteractionMode = Literal["auto", "light", "copilot", "full"]
Severity = Literal["blocking", "major", "minor"]

LIFECYCLE_TYPES = frozenset(
    {
        "run.phase_changed",
        "run.step_changed",
        "run.gate_opened",
        "run.gate_closed",
        "review.completed",
        "artifact.delivered",
        "run.published",
        "run.failed",
        "run.cancelled",
        "run.cost",
    }
)
FindingKind = Literal["fix", "needs_evidence"]
FindingCategory = Literal[
    "leakage", "baseline", "control", "statistics", "reproducibility", "evidence", "other"
]  # fmt: skip


class ArtifactRef(ContractModel):
    artifact_id: str
    schema_id: str = Field(alias="schema")
    version: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    download_url: str
    primary: bool = False


class PipelineFeedback(ContractModel):
    source_stage_key: str
    note: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)


class Decisions(ContractModel):
    actor: Literal["user", "system"]
    decided_by: str | None = None
    note: str | None = None
    selection: dict[str, list[str]] = Field(default_factory=dict)


class Inputs(ContractModel):
    params: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    decisions: Decisions | None = None
    feedback: list[PipelineFeedback] = Field(default_factory=list)
    history: list[ArtifactRef] = Field(default_factory=list)
    prior_engine_run_id: str | None = None


class ReviewPolicy(ContractModel):
    max_auto_revisions: int = Field(default=2, ge=0)


class RunCreateRequest(ContractModel):
    """Carries credentials, so unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    contract_version: str
    platform_run_id: str
    stage_run_id: str
    stage_key: str
    interaction_mode: InteractionMode
    models: Models
    budget_usd: float | str
    callback_url: str
    iteration: int = Field(default=1, ge=1)
    profile: ResearchProfile = Field(default_factory=ResearchProfile)
    inputs: Inputs = Field(default_factory=Inputs)
    integrations: dict[str, SecretStr] = Field(default_factory=dict)
    review_policy: ReviewPolicy = Field(default_factory=ReviewPolicy)

    @field_validator("contract_version")
    @classmethod
    def same_major(cls, value: str) -> str:
        if value.split(".")[0] != CONTRACT_VERSION.split(".")[0]:
            raise ValueError("unsupported contract major version")
        return value

    @field_validator("budget_usd")
    @classmethod
    def positive_budget(cls, value: float | str) -> float | str:
        try:
            amount = float(value)
        except ValueError as exc:
            raise ValueError("budget_usd must be a positive number") from exc
        if not amount > 0:
            raise ValueError("budget_usd must be a positive number")
        return value


class RunCreateResponse(ContractModel):
    engine_run_id: str
    phase: Phase
    created: bool


class GateOption(ContractModel):
    option_id: str
    label: str


class Gate(ContractModel):
    gate_id: str
    kind: Literal["question", "artifact_review"]
    step_key: str | None
    prompt: str
    opened_at: str
    options: list[GateOption] = Field(default_factory=list)
    default_option_id: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)

    # The Platform enforces this rule and JSON Schema cannot express it.
    @model_validator(mode="after")
    def validate_gate(self) -> Gate:
        if self.kind == "question":
            ids = [option.option_id for option in self.options]
            if not 2 <= len(ids) <= 4 or self.default_option_id not in ids:
                raise ValueError("question requires 2-4 options and a default in those options")
        elif self.options or not self.artifact_ids:
            raise ValueError("artifact_review requires artifact_ids and no options")
        return self


class FailureInfo(ContractModel):
    code: str
    message: str


class RunState(ContractModel):
    engine_run_id: str
    stage_run_id: str
    phase: Phase
    step_key: str | None
    waiting_input: bool
    open_gate: Gate | None
    failure: FailureInfo | None = None
    cost_usd: float | str = "0"
    last_source_seq: int = Field(default=0, ge=0)

    # The Platform enforces this rule and JSON Schema cannot express it.
    @model_validator(mode="after")
    def validate_failure(self) -> RunState:
        if (self.phase == "failed") != (self.failure is not None):
            raise ValueError("failure is required exactly when phase is failed")
        return self


class Anchor(ContractModel):
    locator: str = Field(min_length=1)
    artifact_id: str | None = None


class Finding(ContractModel):
    finding_id: str
    anchor: Anchor
    severity: Severity
    issue: str = Field(min_length=1)
    author: Literal["reviewer", "researcher"]
    category: FindingCategory = "other"
    kind: FindingKind = "fix"
    suggestion: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)


class GateAnswer(ContractModel):
    action: Literal["answer", "approve", "comment"]
    actor: Literal["user", "system"]
    option_id: str | None = None
    text: str | None = None
    findings: list[Finding] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    decided_by: str | None = None


class ArtifactMeta(ContractModel):
    schema_id: str = Field(alias="schema")
    version: int = Field(ge=1)
    filename: str
    media_type: str
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    primary: bool = False
    review_round: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewFindings(ContractModel):
    round: int = Field(ge=1)
    reviewed_artifact_id: str
    reviewer_model: str
    findings: list[Finding]
    passed: bool

    # The Platform enforces this rule and JSON Schema cannot express it.
    @model_validator(mode="after")
    def validate_passed(self) -> ReviewFindings:
        if self.passed != (not any(f.severity == "blocking" for f in self.findings)):
            raise ValueError("passed must match absence of blocking findings")
        return self


class RevisionItem(ContractModel):
    finding_id: str
    outcome: Literal["fixed", "rebutted", "not_fixed"]
    note: str | None = None

    # The Platform enforces this rule and JSON Schema cannot express it.
    @model_validator(mode="after")
    def validate_note(self) -> RevisionItem:
        if self.outcome != "fixed" and not (self.note and self.note.strip()):
            raise ValueError("rebutted and not_fixed require a note")
        return self


class RevisionResponse(ContractModel):
    round: int = Field(ge=1)
    artifact_before: str
    artifact_after: str
    items: list[RevisionItem]


class RevisionDiff(ContractModel):
    round: int = Field(ge=1)
    artifact_before: str
    artifact_after: str
    format: Literal["unified"] = "unified"
    patch: str


class EventEnvelope(ContractModel):
    source_seq: int = Field(ge=1)
    type: str
    class_: Literal["lifecycle", "activity"] = Field(alias="class")
    occurred_at: str
    phase: Phase | None = None
    step_key: str | None = None
    actor: str | None = None
    summary: str | None = None
    refs: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)

    # The Platform enforces this rule and JSON Schema cannot express it.
    @model_validator(mode="after")
    def validate_event(self) -> EventEnvelope:
        if self.class_ == "activity":
            if self.type in LIFECYCLE_TYPES or not re.fullmatch(
                r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_.]*", self.type
            ):
                raise ValueError("activity type requires an engine prefix")
            if not all(v and v.strip() for v in (self.step_key, self.actor, self.summary)):
                raise ValueError("activity requires step_key, actor and summary")
            return self
        if self.type not in LIFECYCLE_TYPES:
            raise ValueError("unknown lifecycle type")
        if self.type in {"run.phase_changed", "run.gate_opened"} and self.phase is None:
            raise ValueError("phase is required")
        if self.type == "run.step_changed" and not self.step_key:
            raise ValueError("step_key is required")
        terminal = {
            "run.published": "published",
            "run.failed": "failed",
            "run.cancelled": "cancelled",
        }
        if self.type in terminal and self.phase != terminal[self.type]:
            raise ValueError("terminal event phase does not match type")
        if self.type == "run.gate_opened":
            Gate.model_validate(self.payload)
        elif self.type == "run.failed":
            FailureInfo.model_validate(self.payload)
        return self


class EventBatch(ContractModel):
    events: list[EventEnvelope] = Field(min_length=1)


class EventPage(ContractModel):
    events: list[EventEnvelope]
