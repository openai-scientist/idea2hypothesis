"""Literature retrieval, deduplication, BibTeX export and novelty assessment."""

from idea2hypothesis.literature.citations import papers_to_bibtex
from idea2hypothesis.literature.dedup import deduplicate
from idea2hypothesis.literature.models import (
    Author,
    LiteraturePort,
    Paper,
    SearchReport,
    SourceRecord,
    SourceStats,
)
from idea2hypothesis.literature.search import FileCache, MultiSourceSearch, build_literature

__all__ = [
    "Author",
    "FileCache",
    "LiteraturePort",
    "MultiSourceSearch",
    "Paper",
    "SearchReport",
    "SourceRecord",
    "SourceStats",
    "build_literature",
    "deduplicate",
    "papers_to_bibtex",
]
