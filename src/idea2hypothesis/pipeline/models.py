"""Core pipeline types: stages, statuses, requests, results and the service bundle."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Any

from idea2hypothesis.config import Config
from idea2hypothesis.literature.models import LiteraturePort
from idea2hypothesis.llm.models import LLMPort
from idea2hypothesis.pipeline.control import RunControl
from idea2hypothesis.pipeline.ports import EventSink
from idea2hypothesis.prompts.loader import PromptLoader
from idea2hypothesis.storage.artifacts import ArtifactStore
from idea2hypothesis.storage.runs import RunStore

if TYPE_CHECKING:
    from idea2hypothesis.memory.ideation import IdeationMemory
    from idea2hypothesis.resources.hardware import HardwareProfile


class Stage(IntEnum):
    """The idea to hypothesis stages: eight that build the hypotheses, then the argument map."""

    TOPIC_INIT = 1
    PROBLEM_DECOMPOSE = 2
    SEARCH_STRATEGY = 3
    LITERATURE_COLLECT = 4
    LITERATURE_SCREEN = 5
    KNOWLEDGE_EXTRACT = 6
    SYNTHESIS = 7
    HYPOTHESIS_GEN = 8
    ARGUMENT_MAP = 9


STAGE_SEQUENCE: tuple[Stage, ...] = tuple(Stage)


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    PAUSED = "paused"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStatus(StrEnum):
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


@dataclass(frozen=True)
class StageError:
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass
class StageResult:
    stage: Stage
    status: StageStatus
    artifacts: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: StageError | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": int(self.stage),
            "status": str(self.status),
            "artifacts": list(self.artifacts),
            "evidence_refs": list(self.evidence_refs),
            "warnings": list(self.warnings),
            "error": self.error.to_dict() if self.error else None,
        }


@dataclass(frozen=True)
class RunRequest:
    topic: str
    domains: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    review_mode: str | None = None
    run_id: str | None = None
    platform_run_id: str | None = None
    budget_usd: float | None = None


@dataclass(frozen=True)
class GateAnswer:
    """Reviewer decision: ``approve`` or ``reject``. ``dropped`` lists ids to exclude (papers at
    the screening gate, hypotheses at the hypotheses gate); ``kept`` lists held-back hypothesis
    candidates the reviewer keeps despite a standing objection (hypotheses gate only)."""

    decision: str
    dropped: tuple[str, ...] = ()
    note: str = ""
    kept: tuple[str, ...] = ()


@dataclass
class RunResult:
    run_id: str
    status: RunStatus
    attempt: int
    completed_stages: tuple[int, ...] = ()
    error: StageError | None = None
    gate: dict[str, Any] | None = None
    pause_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


Sleep = Callable[[float], Awaitable[None]]
Report = Callable[[str, dict[str, Any]], Awaitable[None]]


@dataclass
class Services:
    """Everything the engine needs, injected by the caller."""

    config: Config
    llm: LLMPort
    literature: LiteraturePort
    prompts: PromptLoader
    store: RunStore
    events: EventSink | None = None
    reviewer: LLMPort | None = None
    memory: IdeationMemory | None = None
    control: RunControl = field(default_factory=RunControl)
    hardware: Callable[[], HardwareProfile] | None = None
    sleep: Sleep = asyncio.sleep


@dataclass
class StageContext:
    """Inputs of one stage execution; stages never reach outside this object."""

    run_id: str
    attempt: int
    stage: Stage
    topic: str
    domains: tuple[str, ...]
    constraints: tuple[str, ...]
    config: Config
    llm: LLMPort
    literature: LiteraturePort
    prompts: PromptLoader
    artifacts: ArtifactStore
    reviewer: LLMPort | None = None
    memory: IdeationMemory | None = None
    feedback: str = ""
    checkpoint: Callable[[], Awaitable[None]] | None = None
    hardware: Callable[[], HardwareProfile] | None = None
    sleep: Sleep = asyncio.sleep
    report: Report | None = None
    #: Which execution of the stage this is (0 first; a transient LLM error starts another).
    try_index: int = 0

    async def safe_point(self) -> None:
        """Raise :class:`RunInterrupted` if a pause or cancel was requested."""
        if self.checkpoint is not None:
            await self.checkpoint()

    async def progress(self, kind: str, **data: Any) -> None:
        """Announce part of the result as soon as it is persisted (a ``stage.progress`` event)."""
        if self.report is not None:
            await self.report(kind, {"try": self.try_index, **data})
