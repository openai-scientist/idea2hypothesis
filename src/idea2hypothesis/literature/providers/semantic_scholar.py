"""Semantic Scholar paper search with a circuit breaker for the free tier."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from idea2hypothesis.literature.models import Author, Paper, SourceRecord, normalise_doi
from idea2hypothesis.literature.providers.http import (
    CircuitBreaker,
    ProviderError,
    RateSpacer,
    Sleep,
    request_with_retry,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "paperId,title,abstract,year,venue,citationCount,authors,externalIds,url"
MAX_PER_REQUEST = 100


class SemanticScholarProvider:
    name = "semantic_scholar"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        api_key: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
        min_interval: float | None = None,
        sleep: Sleep = asyncio.sleep,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleep = sleep
        interval = min_interval if min_interval is not None else (0.3 if api_key else 1.5)
        self._spacer = RateSpacer(interval, sleep)
        self.breaker = breaker or CircuitBreaker("semantic_scholar")

    async def search(self, query: str, *, limit: int, year_min: int = 0) -> list[Paper]:
        params: dict[str, Any] = {
            "query": query,
            "limit": str(min(limit, MAX_PER_REQUEST)),
            "fields": FIELDS,
        }
        if year_min > 0:
            params["year"] = f"{year_min}-"
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        response = await request_with_retry(
            self._client,
            "GET",
            BASE_URL,
            provider=self.name,
            params=params,
            headers=headers,
            max_retries=self._max_retries,
            timeout=self._timeout,
            breaker=self.breaker,
            spacer=self._spacer,
            sleep=self._sleep,
        )
        try:
            data = response.json().get("data", [])
        except ValueError as exc:
            raise ProviderError("semantic_scholar: invalid JSON response") from exc
        papers: list[Paper] = []
        for item in data if isinstance(data, list) else []:
            try:
                papers.append(parse_paper(item))
            except (KeyError, TypeError, ValueError):
                logger.debug("skipping unparsable Semantic Scholar entry")
        return papers


def parse_paper(item: dict[str, Any]) -> Paper:
    external = item.get("externalIds") or {}
    authors = tuple(
        Author(name=str(a.get("name") or "Unknown"))
        for a in item.get("authors") or []
        if isinstance(a, dict)
    )
    source_id = str(item.get("paperId") or "")
    url = str(item.get("url") or "").strip()
    record = SourceRecord(provider="semantic_scholar", source_id=source_id, url=url)
    return Paper(
        paper_id="",
        title=str(item.get("title") or "").strip(),
        authors=authors,
        year=int(item.get("year") or 0),
        abstract=str(item.get("abstract") or "").strip(),
        venue=str(item.get("venue") or "").strip(),
        citation_count=int(item.get("citationCount") or 0),
        doi=normalise_doi(str(external.get("DOI") or "")),
        arxiv_id=str(external.get("ArXiv") or "").strip(),
        url=url,
        source_records=(record,),
    )
