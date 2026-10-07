"""Ports (protocols) the engine depends on.

``LLMPort`` and ``LiteraturePort`` live next to their implementations and are re-exported
here so pipeline code has a single import location.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from idea2hypothesis.literature.models import LiteraturePort
from idea2hypothesis.llm.models import LLMPort
from idea2hypothesis.pipeline.events import Event


@runtime_checkable
class EventSink(Protocol):
    """Observer notified after an event has been persisted."""

    async def emit(self, event: Event) -> None: ...


__all__ = ["EventSink", "LLMPort", "LiteraturePort"]
