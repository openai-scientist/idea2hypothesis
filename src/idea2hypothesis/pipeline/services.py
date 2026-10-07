"""Wire :class:`Services` from configuration (any part can be injected for tests)."""

from __future__ import annotations

from idea2hypothesis.config import Config
from idea2hypothesis.literature.models import LiteraturePort
from idea2hypothesis.literature.search import build_literature
from idea2hypothesis.llm.factory import build_llm, build_reviewer
from idea2hypothesis.llm.models import LLMPort
from idea2hypothesis.memory.ideation import IdeationMemory
from idea2hypothesis.memory.retriever import hashing_embed
from idea2hypothesis.pipeline.models import Services
from idea2hypothesis.pipeline.ports import EventSink
from idea2hypothesis.prompts.loader import PromptLoader
from idea2hypothesis.storage.runs import RunStore

MEMORY_DIRNAME = "_memory"


def build_services(
    config: Config,
    *,
    llm: LLMPort | None = None,
    literature: LiteraturePort | None = None,
    reviewer: LLMPort | None = None,
    events: EventSink | None = None,
    store: RunStore | None = None,
) -> Services:
    """Create real clients from ``config``; injected parts replace the configured ones.

    Raises :class:`~idea2hypothesis.llm.models.LLMConfigError` when credentials are missing.
    """
    run_store = store or RunStore(config.runs_root)
    prompts = PromptLoader(config.prompts.domain, config.prompts.override_file or None)
    memory = None
    if config.runtime.ideation_memory:
        memory = IdeationMemory(store_dir=config.runs_root / MEMORY_DIRNAME, embed_fn=hashing_embed)
    return Services(
        config=config,
        llm=llm or build_llm(config.llm),
        literature=literature or build_literature(config.literature, cache_root=config.cache_root),
        prompts=prompts,
        store=run_store,
        events=events,
        reviewer=reviewer if reviewer is not None else build_reviewer(config),
        memory=memory,
    )
