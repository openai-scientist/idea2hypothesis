"""arXiv search through the public Atom API (no extra dependency)."""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET  # noqa: S405 - responses come from the arXiv API
from datetime import datetime

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

BASE_URL = "https://export.arxiv.org/api/query"
MAX_RESULTS = 300
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_FIELD_PREFIX_RE = re.compile(r"\b(?:ti|au|abs|cat|all|co|jr|rn|id):")


def build_search_query(query: str) -> str:
    """Use field-prefixed queries as given; otherwise AND together ``all:`` terms."""
    if _FIELD_PREFIX_RE.search(query):
        return query
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query)
    return " AND ".join(f"all:{t}" for t in terms) or query


class ArxivProvider:
    name = "arxiv"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        min_interval: float = 3.1,
        sleep: Sleep = asyncio.sleep,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._client = client
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleep = sleep
        self._spacer = RateSpacer(min_interval, sleep)
        self.breaker = breaker or CircuitBreaker("arxiv", cooldown_sec=180.0)

    async def search(self, query: str, *, limit: int, year_min: int = 0) -> list[Paper]:
        params = {
            "search_query": build_search_query(query),
            "start": "0",
            "max_results": str(min(limit, MAX_RESULTS)),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        response = await request_with_retry(
            self._client,
            "GET",
            BASE_URL,
            provider=self.name,
            params=params,
            max_retries=self._max_retries,
            timeout=self._timeout,
            breaker=self.breaker,
            spacer=self._spacer,
            sleep=self._sleep,
        )
        try:
            papers = parse_feed(response.text)
        except ET.ParseError as exc:
            raise ProviderError("arxiv: invalid Atom response") from exc
        return [p for p in papers if not (year_min > 0 and p.year and p.year < year_min)]


def parse_feed(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)  # noqa: S314
    papers: list[Paper] = []
    for entry in root.findall("atom:entry", _NS):
        entry_id = (entry.findtext("atom:id", default="", namespaces=_NS) or "").strip()
        match = re.search(r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$", entry_id)
        arxiv_id = match.group(1) if match else ""
        title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=_NS) or "")
        abstract = re.sub(
            r"\s+", " ", entry.findtext("atom:summary", default="", namespaces=_NS) or ""
        ).strip()
        published = entry.findtext("atom:published", default="", namespaces=_NS) or ""
        year = 0
        if published:
            try:
                year = datetime.fromisoformat(published.replace("Z", "+00:00")).year
            except ValueError:
                year = 0
        authors = tuple(
            Author(name=(a.findtext("atom:name", default="", namespaces=_NS) or "").strip())
            for a in entry.findall("atom:author", _NS)
        )
        category = entry.find("arxiv:primary_category", _NS)
        venue = category.get("term", "") if category is not None else ""
        doi = normalise_doi(entry.findtext("arxiv:doi", default="", namespaces=_NS) or "")
        url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else entry_id
        papers.append(
            Paper(
                paper_id="",
                title=title.strip(),
                authors=authors,
                year=year,
                abstract=abstract,
                venue=venue,
                doi=doi,
                arxiv_id=arxiv_id,
                url=url,
                source_records=(
                    SourceRecord(provider="arxiv", source_id=arxiv_id or entry_id, url=url),
                ),
            )
        )
    return papers
