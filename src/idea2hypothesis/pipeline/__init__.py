"""Pipeline orchestration. Heavy names are imported lazily to keep ``import`` cheap."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "run_pipeline": "idea2hypothesis.pipeline.runner",
    "resume_pipeline": "idea2hypothesis.pipeline.runner",
    "execute_stage": "idea2hypothesis.pipeline.runner",
    "answer_gate": "idea2hypothesis.pipeline.runner",
    "build_services": "idea2hypothesis.pipeline.services",
    "RunRequest": "idea2hypothesis.pipeline.models",
    "RunResult": "idea2hypothesis.pipeline.models",
    "Services": "idea2hypothesis.pipeline.models",
    "Stage": "idea2hypothesis.pipeline.models",
    "GateAnswer": "idea2hypothesis.pipeline.models",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
