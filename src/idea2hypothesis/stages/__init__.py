"""The eight stage implementations; each module exposes ``async run(ctx) -> list[str]``."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from idea2hypothesis.pipeline.models import Stage, StageContext
from idea2hypothesis.stages import (
    hypothesis_gen,
    knowledge_extract,
    literature_collect,
    literature_screen,
    problem_decompose,
    search_strategy,
    synthesis,
    topic_init,
)

StageFn = Callable[[StageContext], Awaitable[list[str]]]

STAGE_RUNNERS: dict[Stage, StageFn] = {
    Stage.TOPIC_INIT: topic_init.run,
    Stage.PROBLEM_DECOMPOSE: problem_decompose.run,
    Stage.SEARCH_STRATEGY: search_strategy.run,
    Stage.LITERATURE_COLLECT: literature_collect.run,
    Stage.LITERATURE_SCREEN: literature_screen.run,
    Stage.KNOWLEDGE_EXTRACT: knowledge_extract.run,
    Stage.SYNTHESIS: synthesis.run,
    Stage.HYPOTHESIS_GEN: hypothesis_gen.run,
}

__all__ = ["STAGE_RUNNERS", "StageFn"]
