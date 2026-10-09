"""FixtureLiterature: a scripted LiteraturePort. Not real retrieval."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from idea2hypothesis.literature.dedup import deduplicate
from idea2hypothesis.literature.models import (
    Author,
    Paper,
    SearchReport,
    SourceRecord,
    SourceStats,
)

OFFTOPIC_MARK = "[offtopic]"

_ON_TOPIC = [
    ("Sleep duration and academic performance in university students", 2019, "Sleep Medicine"),
    (
        "Sleep deprivation impairs working memory before examinations",
        2018,
        "Journal of Sleep Research",
    ),
    ("Irregular sleep schedules and exam scores: a cohort study", 2021, "Nature Human Behaviour"),
    ("A meta-analysis of sleep and student exam outcomes", 2020, "Psychological Bulletin"),
    ("Napping, memory consolidation and examination results", 2017, "Learning and Memory"),
    ("Sleep quality, stress and grades in college freshmen", 2022, "Frontiers in Psychology"),
    ("Circadian preference and exam performance of students", 2016, "Chronobiology International"),
    ("Actigraphy-measured sleep and standardized test scores", 2023, "SLEEP"),
    ("Sleep extension interventions for student exam preparation", 2020, "Sleep Health"),
]
_OFF_TOPIC = [
    ("Adaptive sleep mode scheduling for wireless sensor networks", 2019, "IEEE Sensors Journal"),
    ("Exam timetabling with integer programming", 2018, "Computers & Operations Research"),
    ("Sleep state replay in neural network training pipelines", 2021, "arXiv cs.LG"),
]


def make_fixture_papers(*, with_abstract: bool = True) -> list[Paper]:
    """Twelve papers (nine on topic, three whose titles carry the off-topic marker)."""
    papers: list[Paper] = []
    rows = [(*row, False) for row in _ON_TOPIC] + [(*row, True) for row in _OFF_TOPIC]
    for i, (title, year, venue, off) in enumerate(rows, 1):
        doi = f"10.1000/fixture.{i}"
        papers.append(
            Paper(
                paper_id="",
                title=f"{title} {OFFTOPIC_MARK}" if off else title,
                authors=(Author(f"Alex Fixture{i}"), Author("Sam Example")),
                year=year,
                abstract=(
                    f"We study {title.lower()} using a controlled design with participants "
                    f"and report effects on outcomes (study {i})."
                    if with_abstract
                    else ""
                ),
                venue=venue,
                citation_count=500 - i * 20,
                doi=doi,
                url=f"https://doi.org/{doi}",
                source_records=(
                    SourceRecord(
                        "openalex",
                        f"W{1000 + i}",
                        f"https://openalex.org/W{1000 + i}",
                        "2026-01-01T00:00:00+00:00",
                    ),
                ),
            )
        )
    return papers


class FixtureLiterature:
    """Returns the same scripted papers for every query (duplicates exercise deduplication)."""

    def __init__(
        self,
        papers: list[Paper] | None = None,
        *,
        source_errors: dict[str, str] | None = None,
        outage: Callable[[list[str]], bool] | None = None,
    ) -> None:
        self.papers = make_fixture_papers() if papers is None else papers
        self.source_errors = source_errors or {}
        self.outage = outage
        self.calls: list[list[str]] = []

    async def search(self, queries: list[str], *, limit: int, year_min: int = 0) -> SearchReport:
        self.calls.append(list(queries))
        report = SearchReport(per_source={"openalex": SourceStats(), "arxiv": SourceStats()})
        if self.outage is not None and self.outage(queries):
            report.per_source["openalex"].errors.append("HTTP 429 (rate limited)")
            return report
        for name, message in self.source_errors.items():
            report.per_source.setdefault(name, SourceStats()).errors.append(message)
        collected: list[Paper] = []
        for query in queries:
            report.per_query[query] = {"openalex": len(self.papers), "arxiv": 0}
            for paper in self.papers:
                collected.append(dataclasses.replace(paper))
                report.per_source["openalex"].papers += 1
            report.per_source["openalex"].requests += 1
        unique, removed = deduplicate(collected)
        report.papers = unique
        report.raw_count = len(collected)
        report.duplicates_removed = removed
        return report
