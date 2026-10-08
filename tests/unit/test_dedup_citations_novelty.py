from __future__ import annotations

from idea2hypothesis.literature.citations import papers_to_bibtex
from idea2hypothesis.literature.dedup import deduplicate
from idea2hypothesis.literature.models import (
    Author,
    Paper,
    SourceRecord,
    make_paper_id,
    normalise_arxiv_id,
    normalise_doi,
    normalise_title,
)
from idea2hypothesis.literature.novelty import DISCLAIMER, check_novelty, extract_keywords
from tests.fixtures import FixtureLiterature


def mk(title: str, provider: str, sid: str, **kw: object) -> Paper:
    return Paper(
        paper_id="",
        title=title,
        source_records=(
            SourceRecord(provider, sid, f"https://{provider}/{sid}", "2026-01-01T00:00:00Z"),
        ),
        **kw,  # type: ignore[arg-type]
    )


def test_each_merged_record_keeps_what_its_source_said() -> None:
    papers = [
        mk("Sparse attention", "arxiv", "2301.1", arxiv_id="2301.1", citation_count=0),
        mk("Sparse attention", "openalex", "W9", doi="10.1/s", citation_count=40),
        mk("Sparse attention", "openalex", "W9", doi="10.1/s", citation_count=40),  # same record
    ]
    (paper,), removed = deduplicate(papers)
    assert removed == 2
    kept, other = paper.source_records  # the record whose metadata was kept comes first
    assert (kept.provider, kept.citations, kept.has_doi) == ("openalex", 40, True)
    assert (other.provider, other.citations, other.has_doi) == ("arxiv", 0, False)
    again = Paper.from_dict(paper.to_dict())
    assert again.source_records == paper.source_records


def test_normalisers() -> None:
    assert normalise_doi("https://doi.org/10.1000/ABC") == "10.1000/abc"
    assert normalise_doi("doi:10.1/x ") == "10.1/x"
    assert normalise_arxiv_id("arXiv:2301.00001v3") == "2301.00001"
    assert normalise_arxiv_id("https://arxiv.org/abs/2301.00001v2") == "2301.00001"
    assert normalise_title("A  Study: of ÉCOLE!") == "a study of ecole"


def test_dedup_by_doi_arxiv_and_title_merges_provenance() -> None:
    papers = [
        mk("Sleep and exams", "openalex", "W1", doi="10.1/A", citation_count=10),
        mk(
            "Sleep & Exams!",
            "semantic_scholar",
            "S1",
            doi="https://doi.org/10.1/a",
            abstract="Long abstract",
        ),
        mk("Other title entirely", "arxiv", "2301.00001", arxiv_id="2301.00001v1"),
        mk("Different words", "openalex", "W2", arxiv_id="arXiv:2301.00001"),
        mk("Same Title", "openalex", "W3"),
        mk("same title", "arxiv", "x"),
        mk("Unique one", "arxiv", "u"),
    ]
    unique, removed = deduplicate(papers)

    assert removed == 3 and len(unique) == 4
    first = next(p for p in unique if p.doi)
    assert first.abstract == "Long abstract"  # gap filled from the other record
    assert {r.provider for r in first.source_records} == {"openalex", "semantic_scholar"}
    assert all(p.source_records for p in unique)


def test_ids_are_stable_and_derived_from_identity() -> None:
    assert make_paper_id("10.1/A", "", "t") == make_paper_id("https://doi.org/10.1/a", "x", "other")
    assert make_paper_id("", "2301.00001v2", "t") == make_paper_id("", "2301.00001", "u")
    assert make_paper_id("", "", "A Title") == make_paper_id("", "", "a  title!")
    assert make_paper_id("10.1/a", "", "t").startswith("p-")
    one, _ = deduplicate([mk("Same", "a", "1", doi="10.1/z")])
    two, _ = deduplicate([mk("Same", "b", "2", doi="10.1/z"), mk("Zed", "b", "3")])
    assert one[0].paper_id == two[0].paper_id


def test_cite_keys_are_unique() -> None:
    author = (Author("Jane Smith"),)
    papers = [
        mk(
            f"Transformer variant {i} notes",
            "oa",
            str(i),
            authors=author,
            year=2024,
            doi=f"10.1/{i}",
        )
        for i in range(3)
    ]
    unique, _ = deduplicate(papers)
    keys = [p.cite_key for p in unique]
    assert len(set(keys)) == 3 and keys[0] == "smith2024transformer"


def test_bibtex_entries() -> None:
    article = Paper(
        "p",
        "Great Results",
        (Author("Ana Perez"), Author("Bo Li")),
        2020,
        venue="Nature",
        doi="10.1/n",
    )
    conference = Paper("q", "Fast Models", (Author("Cy Wu"),), 2021, venue="ICML Conference")
    preprint = Paper(
        "r", "Open Preprint", (Author("Di Xu"),), 2022, venue="cs.LG", arxiv_id="2201.00001"
    )
    text = papers_to_bibtex([article, conference, preprint])
    assert "@article{perez2020great," in text and "author = {Ana Perez and Bo Li}" in text
    assert "@inproceedings{wu2021fast," in text and "booktitle = {ICML Conference}" in text
    assert "arXiv preprint arXiv:2201.00001" in text and "archiveprefix = {arXiv}" in text
    assert papers_to_bibtex([]) == ""


def test_paper_roundtrip_through_dict_keeps_provenance_and_key() -> None:
    original = deduplicate([mk("Roundtrip paper", "openalex", "W7", doi="10.1/r", year=2020)])[0][0]
    restored = Paper.from_dict(original.to_dict())
    assert restored == original and restored.cite_key == original.cite_key


async def test_novelty_report_is_labelled_and_per_hypothesis() -> None:
    hypotheses = [
        {"id": "H1", "statement": "Sleep duration changes exam score in university students"},
        {"id": "H2", "statement": "Quantum annealing improves protein folding bandwidth"},
    ]
    report = await check_novelty("sleep and exams", hypotheses, literature=FixtureLiterature())
    assert report["kind"] == "novelty_assessment" and report["disclaimer"] == DISCLAIMER
    assert report["total_papers_retrieved"] > 0 and report["search_coverage"] == "full"
    closest = {r["hypothesis_id"]: r["closest_paper"] for r in report["per_hypothesis"]}
    assert closest["H1"]["similarity"] > closest["H2"]["similarity"]
    assert 0.0 <= report["novelty_score"] <= 1.0


async def test_novelty_without_any_coverage_is_flagged_not_perfect() -> None:
    class Failing:
        async def search(self, queries, *, limit, year_min=0):  # type: ignore[no-untyped-def]
            raise RuntimeError("network down")

    report = await check_novelty(
        "topic words", [{"id": "H1", "statement": "x y z claim"}], literature=Failing()
    )
    assert report["assessment"] == "insufficient_data"
    assert report["recommendation"] == "proceed_with_caution"
    assert report["search_errors"]


def test_extract_keywords_drops_stop_words_and_short_tokens() -> None:
    assert extract_keywords("The effect of sleep on an exam, ok") == ["effect", "sleep", "exam"]
