"""Multi-query, multi-provider search with per-source error recording and optional cache."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import httpx

from idea2hypothesis.config import LiteratureConfig
from idea2hypothesis.literature.dedup import deduplicate
from idea2hypothesis.literature.models import Paper, SearchReport, SourceStats
from idea2hypothesis.literature.providers.arxiv import BASE_URL as ARXIV_URL
from idea2hypothesis.literature.providers.arxiv import ArxivProvider
from idea2hypothesis.literature.providers.http import ProviderError
from idea2hypothesis.literature.providers.openalex import BASE_URL as OPENALEX_URL
from idea2hypothesis.literature.providers.openalex import OpenAlexProvider
from idea2hypothesis.literature.providers.semantic_scholar import BASE_URL as S2_URL
from idea2hypothesis.literature.providers.semantic_scholar import SemanticScholarProvider

logger = logging.getLogger(__name__)

SOURCE_INFO: dict[str, tuple[str, str]] = {
    "openalex": ("OpenAlex", OPENALEX_URL),
    "semantic_scholar": ("Semantic Scholar", S2_URL),
    "arxiv": ("arXiv", ARXIV_URL),
}
#: Sources whose records carry no citation count: their 0 means unknown, not uncited.
NO_CITATION_COUNTS = frozenset({"arxiv"})
_MAX_STALE_SEC = 30 * 86400.0  # cached results are only a fallback when a provider fails


class Provider(Protocol):
    name: str

    async def search(self, query: str, *, limit: int, year_min: int = 0) -> list[Paper]: ...


class FileCache:
    """Small JSON cache keyed by (provider, query, limit, year_min); used on provider failure."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _path(self, provider: str, query: str, limit: int, year_min: int) -> Path:
        raw = f"{provider}|{query.strip().lower()}|{limit}|{year_min}"
        return self._root / f"{hashlib.sha256(raw.encode()).hexdigest()[:20]}.json"

    def get(self, provider: str, query: str, limit: int, year_min: int) -> list[Paper] | None:
        path = self._path(provider, query, limit, year_min)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if time.time() - float(data.get("timestamp", 0)) > _MAX_STALE_SEC:
            return None
        return [Paper.from_dict(p) for p in data.get("papers", []) if isinstance(p, dict)]

    def put(
        self, provider: str, query: str, limit: int, year_min: int, papers: list[Paper]
    ) -> None:
        path = self._path(provider, query, limit, year_min)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"timestamp": time.time(), "papers": [p.to_dict() for p in papers]}
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            logger.debug("literature cache write failed", exc_info=True)


class MultiSourceSearch:
    """Implements :class:`LiteraturePort` over a list of providers."""

    def __init__(
        self,
        providers: Sequence[Provider],
        *,
        inter_query_delay: float = 1.0,
        cache: FileCache | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not providers:
            raise ValueError("at least one literature provider is required")
        self._providers = list(providers)
        self._delay = inter_query_delay
        self._cache = cache
        self._sleep = sleep

    async def search(self, queries: list[str], *, limit: int, year_min: int = 0) -> SearchReport:
        report = SearchReport(per_source={p.name: SourceStats() for p in self._providers})
        collected: list[Paper] = []
        for index, query in enumerate(queries):
            if index > 0 and self._delay > 0:
                await self._sleep(self._delay)
            counts = report.per_query.setdefault(query, {})
            for provider in self._providers:
                papers = await self._query_provider(provider, query, limit, year_min, report)
                counts[provider.name] = len(papers)
                collected.extend(papers)
        report.raw_count = len(collected)
        titled = [p for p in collected if p.title.strip()]
        report.dropped_without_title = len(collected) - len(titled)
        unique, removed = deduplicate(titled)
        unique.sort(key=lambda p: (p.citation_count, p.year), reverse=True)
        report.papers = unique
        report.duplicates_removed = removed
        return report

    async def _query_provider(
        self, provider: Provider, query: str, limit: int, year_min: int, report: SearchReport
    ) -> list[Paper]:
        stats = report.per_source[provider.name]
        stats.requests += 1
        try:
            papers = await provider.search(query, limit=limit, year_min=year_min)
        except ProviderError as exc:
            stats.errors.append(f"{query!r}: {exc}")
            cached = self._cache.get(provider.name, query, limit, year_min) if self._cache else None
            if cached:
                stats.served_from_cache += len(cached)
                stats.papers += len(cached)
                return cached
            return []
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            stats.errors.append(f"{query!r}: unexpected {type(exc).__name__}")
            return []
        stats.papers += len(papers)
        if self._cache:
            self._cache.put(provider.name, query, limit, year_min, papers)
        return papers


def build_literature(
    cfg: LiteratureConfig,
    *,
    cache_root: Path | None = None,
    client: httpx.AsyncClient | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> MultiSourceSearch:
    """Create the configured providers; secrets are read from the named environment variables."""
    http = client or httpx.AsyncClient(timeout=cfg.timeout_sec, follow_redirects=True)
    common: dict[str, Any] = {
        "timeout": cfg.timeout_sec,
        "max_retries": cfg.max_retries,
        "sleep": sleep,
    }
    providers: list[Provider] = []
    for name in cfg.sources:
        if name == "openalex":
            providers.append(
                OpenAlexProvider(
                    http,
                    email=cfg.openalex_email,
                    api_key=os.environ.get(cfg.openalex_api_key_env, "")
                    if cfg.openalex_api_key_env
                    else "",
                    **common,
                )
            )
        elif name == "semantic_scholar":
            providers.append(
                SemanticScholarProvider(
                    http,
                    api_key=os.environ.get(cfg.s2_api_key_env, "") if cfg.s2_api_key_env else "",
                    **common,
                )
            )
        elif name == "arxiv":
            providers.append(ArxivProvider(http, **common))
    cache = FileCache(cache_root / "literature") if cfg.cache and cache_root else None
    return MultiSourceSearch(
        providers, inter_query_delay=cfg.inter_query_delay_sec, cache=cache, sleep=sleep
    )
