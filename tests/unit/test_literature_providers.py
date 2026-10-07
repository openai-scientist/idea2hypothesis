from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from idea2hypothesis.config import LiteratureConfig
from idea2hypothesis.literature.models import Paper
from idea2hypothesis.literature.providers.arxiv import ArxivProvider, build_search_query
from idea2hypothesis.literature.providers.http import CircuitBreaker, ProviderError
from idea2hypothesis.literature.providers.openalex import OpenAlexProvider, reconstruct_abstract
from idea2hypothesis.literature.providers.semantic_scholar import SemanticScholarProvider
from idea2hypothesis.literature.search import FileCache, MultiSourceSearch, build_literature

Handler = Callable[[httpx.Request], httpx.Response]


async def no_sleep(_: float) -> None:
    return None


def client_for(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


OPENALEX_PAYLOAD = {
    "results": [
        {
            "id": "https://openalex.org/W123",
            "title": "  Sleep   and Exams ",
            "authorships": [
                {
                    "author": {"display_name": "Ada Lovelace"},
                    "institutions": [{"display_name": "Uni"}],
                }
            ],
            "publication_year": 2021,
            "primary_location": {"source": {"display_name": "Sleep Medicine"}},
            "cited_by_count": 42,
            "doi": "https://doi.org/10.1000/ABC",
            "ids": {
                "openalex": "https://openalex.org/W123",
                "arxiv": "https://arxiv.org/abs/2101.00001",
            },
            "abstract_inverted_index": {"Sleep": [0], "helps": [1], "exams": [2]},
        },
        {"id": "https://openalex.org/W9", "title": "", "authorships": []},
    ]
}

S2_PAYLOAD = {
    "data": [
        {
            "paperId": "abc",
            "title": "Sleep and grades",
            "abstract": "Grades improve.",
            "year": 2020,
            "venue": "NeurIPS Workshop",
            "citationCount": 7,
            "authors": [{"name": "Grace Hopper"}],
            "externalIds": {"DOI": "10.1000/xyz", "ArXiv": "2001.00002"},
            "url": "https://www.semanticscholar.org/paper/abc",
        }
    ]
}

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v2</id>
    <title>Sleep   and
      learning</title>
    <summary>  An abstract
      over lines. </summary>
    <published>2023-01-29T10:00:00Z</published>
    <author><name>Alan Turing</name></author>
    <author><name>Mary Jackson</name></author>
    <arxiv:primary_category term="q-bio.NC"/>
    <arxiv:doi>10.1000/arxiv.1</arxiv:doi>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/1801.00001v1</id>
    <title>Old paper</title>
    <summary>Old.</summary>
    <published>2018-01-01T00:00:00Z</published>
    <author><name>Old Author</name></author>
  </entry>
</feed>"""


async def test_openalex_parsing_and_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=OPENALEX_PAYLOAD)

    provider = OpenAlexProvider(
        client_for(handler),
        email="me@example.org",
        api_key="secret",
        sleep=no_sleep,
        min_interval=0,
    )
    papers = await provider.search("sleep exams", limit=500, year_min=2015)

    params = dict(seen[0].url.params)
    assert params["per_page"] == "50" and params["filter"] == "from_publication_date:2015-01-01"
    assert params["mailto"] == "me@example.org"
    paper = papers[0]
    assert paper.title == "Sleep and Exams"
    assert paper.abstract == "Sleep helps exams"
    assert paper.doi == "10.1000/abc" and paper.arxiv_id == "2101.00001"
    assert paper.venue == "Sleep Medicine" and paper.citation_count == 42
    assert paper.source_records[0].provider == "openalex"
    assert paper.source_records[0].source_id == "https://openalex.org/W123"
    assert paper.source_records[0].retrieved_at


def test_reconstruct_abstract_handles_missing_index() -> None:
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({"b": [1], "a": [0]}) == "a b"


async def test_semantic_scholar_parsing_and_api_key_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=S2_PAYLOAD)

    provider = SemanticScholarProvider(
        client_for(handler), api_key="k", sleep=no_sleep, min_interval=0
    )
    papers = await provider.search("sleep", limit=10)
    assert seen[0].headers["x-api-key"] == "k"
    assert "k" not in str(seen[0].url)
    assert papers[0].doi == "10.1000/xyz" and papers[0].arxiv_id == "2001.00002"
    assert papers[0].source_records[0].source_id == "abc"


async def test_arxiv_atom_parsing_and_year_filter() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=ATOM)

    provider = ArxivProvider(client_for(handler), sleep=no_sleep, min_interval=0)
    papers = await provider.search("sleep learning", limit=5, year_min=2020)
    assert len(papers) == 1
    paper = papers[0]
    assert paper.arxiv_id == "2301.12345" and paper.year == 2023
    assert paper.title == "Sleep and learning" and paper.abstract == "An abstract over lines."
    assert [a.name for a in paper.authors] == ["Alan Turing", "Mary Jackson"]
    assert paper.venue == "q-bio.NC" and paper.doi == "10.1000/arxiv.1"
    assert dict(seen[0].url.params)["search_query"] == "all:sleep AND all:learning"
    assert paper.source_records[0].provider == "arxiv"


def test_arxiv_query_building() -> None:
    assert build_search_query("ti:transformer AND cat:cs.LG") == "ti:transformer AND cat:cs.LG"
    assert build_search_query("dark-matter portal!") == "all:dark-matter AND all:portal"


async def test_429_is_retried_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200, json=OPENALEX_PAYLOAD)

    provider = OpenAlexProvider(client_for(handler), sleep=no_sleep, min_interval=0)
    assert await provider.search("q", limit=5)
    assert attempts["n"] == 2


async def test_server_errors_exhaust_retries_with_a_clean_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    provider = OpenAlexProvider(
        client_for(handler), api_key="topsecret", sleep=no_sleep, max_retries=2, min_interval=0
    )
    with pytest.raises(ProviderError) as info:
        await provider.search("q", limit=5)
    assert "HTTP 503" in str(info.value) and "topsecret" not in str(info.value)


async def test_client_errors_are_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400)

    provider = OpenAlexProvider(client_for(handler), sleep=no_sleep, min_interval=0)
    with pytest.raises(ProviderError):
        await provider.search("q", limit=5)
    assert calls["n"] == 1


async def test_timeouts_and_transport_errors_are_provider_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    provider = ArxivProvider(client_for(handler), sleep=no_sleep, max_retries=2, min_interval=0)
    with pytest.raises(ProviderError, match="timeout"):
        await provider.search("q", limit=5)


async def test_semantic_scholar_circuit_breaker_stops_requests() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429)

    provider = SemanticScholarProvider(
        client_for(handler),
        sleep=no_sleep,
        min_interval=0,
        max_retries=5,
        breaker=CircuitBreaker("s2", threshold=2, cooldown_sec=1000),
    )
    with pytest.raises(ProviderError, match="circuit breaker"):
        await provider.search("q", limit=5)
    first_calls = calls["n"]
    assert first_calls == 2
    with pytest.raises(ProviderError, match="circuit breaker open"):
        await provider.search("q", limit=5)
    assert calls["n"] == first_calls  # no request while the breaker is open


def test_circuit_breaker_recovers_after_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    now = {"t": 0.0}
    monkeypatch.setattr(
        "idea2hypothesis.literature.providers.http.time.monotonic", lambda: now["t"]
    )
    breaker = CircuitBreaker("x", threshold=1, cooldown_sec=10)
    assert breaker.on_failure() is True and not breaker.allow()
    now["t"] = 11.0
    assert breaker.allow() and breaker.state == "half_open"
    breaker.on_success()
    assert breaker.state == "closed"


# -- MultiSourceSearch -----------------------------------------------------------


class StubProvider:
    def __init__(self, name: str, papers: list[Paper] | None = None, error: str = "") -> None:
        self.name = name
        self._papers = papers or []
        self._error = error
        self.calls: list[str] = []

    async def search(self, query: str, *, limit: int, year_min: int = 0) -> list[Paper]:
        self.calls.append(query)
        if self._error:
            raise ProviderError(self._error)
        return self._papers


def paper(title: str, provider: str, doi: str = "", cites: int = 0) -> Paper:
    from idea2hypothesis.literature.models import SourceRecord

    return Paper(
        paper_id="",
        title=title,
        doi=doi,
        citation_count=cites,
        source_records=(
            SourceRecord(provider, f"{provider}-{title[:4]}", "", "2026-01-01T00:00:00+00:00"),
        ),
    )


async def test_per_source_errors_are_recorded_and_other_sources_still_answer() -> None:
    good = StubProvider("openalex", [paper("Sleep and exams", "openalex", "10.1/a")])
    bad = StubProvider("semantic_scholar", error="semantic_scholar: HTTP 429")
    search = MultiSourceSearch([good, bad], inter_query_delay=0, sleep=no_sleep)
    report = await search.search(["q1", "q2"], limit=5)

    assert [p.title for p in report.papers] == ["Sleep and exams"]
    assert report.raw_count == 2 and report.duplicates_removed == 1
    assert report.per_source["openalex"].papers == 2
    assert len(report.per_source["semantic_scholar"].errors) == 2
    assert report.per_query["q1"] == {"openalex": 1, "semantic_scholar": 0}
    assert report.errors and "429" in report.errors[0]


async def test_papers_without_a_title_are_dropped_and_counted() -> None:
    provider = StubProvider("openalex", [paper("", "openalex"), paper("Real title", "openalex")])
    report = await MultiSourceSearch([provider], inter_query_delay=0, sleep=no_sleep).search(
        ["q"], limit=5
    )
    assert [p.title for p in report.papers] == ["Real title"]
    assert report.dropped_without_title == 1


async def test_cache_is_used_only_when_a_provider_fails(tmp_path: Path) -> None:
    cache = FileCache(tmp_path / "cache")
    working = StubProvider("openalex", [paper("Cached result", "openalex", "10.1/c")])
    await MultiSourceSearch([working], inter_query_delay=0, cache=cache, sleep=no_sleep).search(
        ["q"], limit=5
    )

    broken = StubProvider("openalex", error="openalex: HTTP 503")
    report = await MultiSourceSearch(
        [broken], inter_query_delay=0, cache=cache, sleep=no_sleep
    ).search(["q"], limit=5)
    assert [p.title for p in report.papers] == ["Cached result"]
    stats = report.per_source["openalex"]
    assert stats.served_from_cache == 1 and stats.errors  # the failure is still on record

    other = await MultiSourceSearch(
        [broken], inter_query_delay=0, cache=cache, sleep=no_sleep
    ).search(["other"], limit=5)
    assert other.papers == []


def test_build_literature_reads_keys_from_the_named_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("S2_API_KEY", "s2-secret")
    cfg = LiteratureConfig(sources=("semantic_scholar", "arxiv"), cache=True)
    search = build_literature(cfg, cache_root=Path("cache-root"), client=httpx.AsyncClient())
    names: list[Any] = [p.name for p in search._providers]
    assert names == ["semantic_scholar", "arxiv"]
    assert search._providers[0]._api_key == "s2-secret"
