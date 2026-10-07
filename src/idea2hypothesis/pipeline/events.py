"""Core run events.

Events are persisted by the run store before any observer sees them. They only describe
things that actually happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1

RUN_STARTED = "run.started"
STAGE_STARTED = "stage.started"
STAGE_COMPLETED = "stage.completed"
STAGE_FAILED = "stage.failed"
GATE_OPENED = "gate.opened"
GATE_RESOLVED = "gate.resolved"
RUN_PAUSED = "run.paused"
RUN_RESUMED = "run.resumed"
RUN_CANCELLED = "run.cancelled"
RUN_FAILED = "run.failed"
RUN_COMPLETED = "run.completed"

EVENT_TYPES = (
    RUN_STARTED,
    STAGE_STARTED,
    STAGE_COMPLETED,
    STAGE_FAILED,
    GATE_OPENED,
    GATE_RESOLVED,
    RUN_PAUSED,
    RUN_RESUMED,
    RUN_CANCELLED,
    RUN_FAILED,
    RUN_COMPLETED,
)


@dataclass(frozen=True)
class Event:
    run_id: str
    seq: int
    type: str
    timestamp: str
    attempt: int = 1
    stage: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "seq": self.seq,
            "stage": self.stage,
            "type": self.type,
            "timestamp": self.timestamp,
            "attempt": self.attempt,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Event:
        return cls(
            schema_version=int(raw.get("schema_version", SCHEMA_VERSION)),
            run_id=str(raw["run_id"]),
            seq=int(raw["seq"]),
            stage=raw.get("stage"),
            type=str(raw["type"]),
            timestamp=str(raw["timestamp"]),
            attempt=int(raw.get("attempt", 1)),
            data=dict(raw.get("data") or {}),
        )
