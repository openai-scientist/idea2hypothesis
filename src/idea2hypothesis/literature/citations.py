"""BibTeX export."""

from __future__ import annotations

from collections.abc import Iterable

from idea2hypothesis.literature.models import Paper


def papers_to_bibtex(papers: Iterable[Paper]) -> str:
    """Render one BibTeX file; an empty input yields an empty string."""
    entries = [p.to_bibtex() for p in papers]
    return "\n\n".join(entries) + "\n" if entries else ""
