"""Deduplication with provenance merging and stable identifiers."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable

from idea2hypothesis.literature.models import (
    Paper,
    SourceRecord,
    make_paper_id,
    normalise_arxiv_id,
    normalise_doi,
    normalise_title,
)


def _merge(primary: Paper, other: Paper) -> Paper:
    """Merge two records of the same work: keep the richer one, fill gaps, union provenance."""
    if (other.citation_count, len(other.abstract)) > (
        primary.citation_count,
        len(primary.abstract),
    ):
        primary, other = other, primary
    seen: set[tuple[str, str]] = set()
    records: list[SourceRecord] = []
    for record in (*primary.source_records, *other.source_records):
        key = (record.provider, record.source_id)
        if key not in seen:
            seen.add(key)
            records.append(record)
    return dataclasses.replace(
        primary,
        doi=primary.doi or other.doi,
        arxiv_id=primary.arxiv_id or other.arxiv_id,
        abstract=primary.abstract or other.abstract,
        venue=primary.venue or other.venue,
        url=primary.url or other.url,
        year=primary.year or other.year,
        authors=primary.authors or other.authors,
        source_records=tuple(records),
    )


def deduplicate(papers: Iterable[Paper]) -> tuple[list[Paper], int]:
    """Collapse duplicates by DOI, then arXiv id, then normalised title.

    Returns ``(unique_papers, duplicates_removed)``. Papers get stable ids and unique
    citation keys.
    """
    result: list[Paper] = []
    by_doi: dict[str, int] = {}
    by_arxiv: dict[str, int] = {}
    by_title: dict[str, int] = {}
    removed = 0

    def keys(paper: Paper) -> tuple[str, str, str]:
        return (
            normalise_doi(paper.doi),
            normalise_arxiv_id(paper.arxiv_id),
            normalise_title(paper.title),
        )

    for paper in papers:
        doi, arxiv, title = keys(paper)
        index = None
        for table, key in ((by_doi, doi), (by_arxiv, arxiv), (by_title, title)):
            if key and key in table:
                index = table[key]
                break
        if index is None:
            index = len(result)
            result.append(paper)
        else:
            result[index] = _merge(result[index], paper)
            removed += 1
        merged_doi, merged_arxiv, merged_title = keys(result[index])
        for table, key in (
            (by_doi, merged_doi),
            (by_arxiv, merged_arxiv),
            (by_title, merged_title),
        ):
            if key:
                table[key] = index

    identified = [
        dataclasses.replace(p, paper_id=make_paper_id(p.doi, p.arxiv_id, p.title)) for p in result
    ]
    return assign_unique_cite_keys(identified), removed


def assign_unique_cite_keys(papers: list[Paper]) -> list[Paper]:
    """Give every paper a distinct BibTeX key (``a``, ``b`` ... suffixes on collisions)."""
    used: dict[str, int] = {}
    out: list[Paper] = []
    for paper in papers:
        base = paper.derived_cite_key()
        count = used.get(base, 0)
        used[base] = count + 1
        key = base if count == 0 else f"{base}{_suffix(count)}"
        out.append(dataclasses.replace(paper, bib_key=key))
    return out


def _suffix(index: int) -> str:
    letters = ""
    while index > 0:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("a") + rem) + letters
    return letters
