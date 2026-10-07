"""Request and response models of the HTTP adapter (wire contract for Platform BE and Swagger)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ApiStatus = Literal["running", "paused", "awaiting_review", "completed", "failed"]

# -- engine runs (Platform BE) ------------------------------------------------


class RunCreateRequest(BaseModel):
    platform_run_id: str = Field(
        min_length=1, max_length=128, description="Run id from Platform BE (idempotency key)"
    )
    topic: str = Field(min_length=12, max_length=1000, description="Research topic")
    domains: list[str] = Field(default_factory=list, max_length=6, description="Academic domains")
    review_mode: Literal["auto", "light", "copilot", "full"] = Field(
        default="copilot", description="Review mode controlling human gates"
    )
    budget_usd: Decimal = Field(default=Decimal("5.00"), gt=0, description="Budget in USD")
    callback_url: str = Field(description="Webhook base URL for event delivery")

    @field_validator("topic")
    @classmethod
    def clean_topic(cls, value: str) -> str:
        text = value.strip()
        if len(text) < 12:
            raise ValueError("Topic must be at least 12 characters")
        return text

    @field_validator("domains")
    @classmethod
    def clean_domains(cls, value: list[str]) -> list[str]:
        return [d.strip() for d in value if d.strip()][:6]


class RunCreateResponse(BaseModel):
    popper_run_id: str
    status: ApiStatus
    cost_usd: str | None = Field(default=None, description="Decimal string; null when unpriced")
    message: str | None = None


class RunStateResponse(BaseModel):
    popper_run_id: str
    status: ApiStatus
    cost_usd: str | None = Field(default=None, description="Decimal string; null when unpriced")
    message: str | None = None
    last_source_seq: int


class GateAnswerRequest(BaseModel):
    option_id: str
    dropped: list[str] = Field(default_factory=list)
    note: str | None = None


class EventItem(BaseModel):
    source_seq: int
    type: str
    stage_key: str | None = None
    actor: str | None = None
    payload: dict[str, Any]


class EventsBatchResponse(BaseModel):
    events: list[EventItem]


class OkResponse(BaseModel):
    status: str = "ok"
    message: str = ""


# -- stage routes (Swagger / review) -----------------------------------------


class Phase1StartRequest(BaseModel):
    topic: str = Field(description="Research topic")
    domains: list[str] = Field(default_factory=list)
    llm_provider: str | None = Field(
        default=None, description="'bedrock' or 'openai' (default: configured provider)"
    )
    model: str | None = Field(default=None, description="Model id override")
    quality_threshold: float = Field(default=4.0, description="Accepted for compatibility; unused")
    auto_approve: bool = Field(
        default=True, description="true = review mode 'auto', false = 'copilot' (screen gate)"
    )


class Stage1RunRequest(BaseModel):
    topic: str = Field(description="Research topic")
    run_id: str | None = Field(default=None, description="Optional run id; generated otherwise")
    llm_provider: str | None = Field(default=None)
    model: str | None = Field(default=None)


class TextContentUpdate(BaseModel):
    content: str = Field(description="New artifact content (JSON text for JSON artifacts)")
