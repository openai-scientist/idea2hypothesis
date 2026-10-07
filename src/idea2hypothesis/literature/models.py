"""Literature data models: papers, provenance records and search reports."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "into", "over", "upon", "about",
        "through", "using", "based", "towards", "toward", "between", "under", "more", "than",
        "when", "what", "which", "where", "does", "have", "been", "some", "each", "also",
        "much", "very", "learning",
    }
)  # fmt: skip

_ARXIV_CATEGORY_RE = re.compile(
    r"^(?:cs|math|stat|eess|physics|q-bio|q-fin|astro-ph|cond-mat|gr-qc|hep-ex|hep-lat|"
    r"hep-ph|hep-th|nlin|nucl-ex|nucl-th|quant-ph)(?:\.[A-Za-z-]+)?$"
)
_CONFERENCE_KEYWORDS = (
    "conference", "proc", "workshop", "neurips", "icml", "iclr", "aaai", "cvpr", "acl",
    "emnlp", "naacl", "eccv", "iccv", "sigir", "kdd", "www", "ijcai",
)  # fmt: skip


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def normalise_doi(doi: str) -> str:
    value = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value.strip()


def normalise_arxiv_id(arxiv_id: str) -> str:
    value = arxiv_id.strip()
    value = re.sub(r"^(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv:)", "", value, flags=re.IGNORECASE)
    value = re.sub(r"(?:v\d+)?(?:\.pdf)?$", "", value)
    return value


def normalise_title(title: str) -> str:
    folded = unicodedata.normalize("NFKD", title.lower()).encode("ascii", "ignore").decode()
    folded = re.sub(r"[^a-z0-9\s]", "", folded)
    return re.sub(r"\s+", " ", folded).strip()


def make_paper_id(doi: str, arxiv_id: str, title: str) -> str:
    """Stable internal id: derived from DOI, else arXiv id, else normalised title."""
    if doi:
        basis = f"doi:{normalise_doi(doi)}"
    elif arxiv_id:
        basis = f"arxiv:{normalise_arxiv_id(arxiv_id)}"
    else:
        basis = f"title:{normalise_title(title)}"
    return "p-" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class Author:
    name: str
    affiliation: str = ""

    def last_name(self) -> str:
        parts = self.name.strip().split()
        raw = parts[-1] if parts else "unknown"
        folded = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-zA-Z]", "", folded).lower() or "unknown"


@dataclass(frozen=True)
class SourceRecord:
    """Where and when a paper record was retrieved."""

    provider: str
    source_id: str
    url: str = ""
    retrieved_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "source_id": self.source_id,
            "url": self.url,
            "retrieved_at": self.retrieved_at,
        }


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    authors: tuple[Author, ...] = ()
    year: int = 0
    abstract: str = ""
    venue: str = ""
    citation_count: int = 0
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    source_records: tuple[SourceRecord, ...] = ()
    bib_key: str = ""

    @property
    def providers(self) -> tuple[str, ...]:
        seen: list[str] = []
        for record in self.source_records:
            if record.provider not in seen:
                seen.append(record.provider)
        return tuple(seen)

    @property
    def cite_key(self) -> str:
        """Citation key; ``lastname<year><keyword>`` unless a unique key was assigned."""
        return self.bib_key or self.derived_cite_key()

    def derived_cite_key(self) -> str:
        last = self.authors[0].last_name() if self.authors else "anon"
        year = str(self.year) if self.year else "0000"
        keyword = ""
        for word in self.title.split():
            cleaned = re.sub(r"[^a-zA-Z]", "", word).lower()
            if len(cleaned) > 3 and cleaned not in _STOPWORDS:
                keyword = cleaned
                break
        return f"{last}{year}{keyword}"

    def to_bibtex(self) -> str:
        venue = self.venue or ""
        is_category = bool(_ARXIV_CATEGORY_RE.match(venue))
        authors = " and ".join(a.name for a in self.authors) or "Unknown"
        if venue and not is_category and any(k in venue.lower() for k in _CONFERENCE_KEYWORDS):
            entry_type, venue_line = "inproceedings", f"  booktitle = {{{venue}}},"
        elif self.arxiv_id and (not venue or is_category):
            entry_type, venue_line = (
                "article",
                f"  journal = {{arXiv preprint arXiv:{self.arxiv_id}}},",
            )
        else:
            entry_type = "article"
            venue_line = f"  journal = {{{venue}}}," if venue else ""
        lines = [f"@{entry_type}{{{self.cite_key},", f"  title = {{{self.title}}},"]
        lines.append(f"  author = {{{authors}}},")
        lines.append(f"  year = {{{self.year or 'Unknown'}}},")
        if venue_line:
            lines.append(venue_line)
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.arxiv_id:
            lines += [f"  eprint = {{{self.arxiv_id}}},", "  archiveprefix = {arXiv},"]
        if self.url:
            lines.append(f"  url = {{{self.url}}},")
        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": [{"name": a.name, "affiliation": a.affiliation} for a in self.authors],
            "year": self.year,
            "abstract": self.abstract,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "cite_key": self.cite_key,
            "source_records": [r.to_dict() for r in self.source_records],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Paper:
        authors = tuple(
            Author(str(a.get("name", "")), str(a.get("affiliation", "")))
            for a in data.get("authors", [])
            if isinstance(a, dict)
        )
        records = tuple(
            SourceRecord(
                provider=str(r.get("provider", "")),
                source_id=str(r.get("source_id", "")),
                url=str(r.get("url", "")),
                retrieved_at=str(r.get("retrieved_at", "")),
            )
            for r in data.get("source_records", [])
            if isinstance(r, dict)
        )
        return cls(
            paper_id=str(data.get("paper_id", "")),
            title=str(data.get("title", "")),
            authors=authors,
            year=int(data.get("year") or 0),
            abstract=str(data.get("abstract", "")),
            venue=str(data.get("venue", "")),
            citation_count=int(data.get("citation_count") or 0),
            doi=str(data.get("doi", "")),
            arxiv_id=str(data.get("arxiv_id", "")),
            url=str(data.get("url", "")),
            source_records=records,
            bib_key=str(data.get("cite_key", "")),
        )


@dataclass
class SourceStats:
    """Per-provider outcome of one search report."""

    requests: int = 0
    papers: int = 0
    errors: list[str] = field(default_factory=list)
    served_from_cache: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "papers": self.papers,
            "errors": list(self.errors),
            "served_from_cache": self.served_from_cache,
        }


@dataclass
class SearchReport:
    """Deduplicated result of a multi-query, multi-provider search."""

    papers: list[Paper] = field(default_factory=list)
    raw_count: int = 0
    duplicates_removed: int = 0
    per_source: dict[str, SourceStats] = field(default_factory=dict)
    per_query: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped_without_title: int = 0

    @property
    def errors(self) -> list[str]:
        return [f"{name}: {err}" for name, s in self.per_source.items() for err in s.errors]


@runtime_checkable
class LiteraturePort(Protocol):
    async def search(
        self, queries: list[str], *, limit: int, year_min: int = 0
    ) -> SearchReport: ...
