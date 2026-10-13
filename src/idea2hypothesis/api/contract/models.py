"""Engine-owned wire models; synchronized through published JSON Schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "1.0.0"


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
