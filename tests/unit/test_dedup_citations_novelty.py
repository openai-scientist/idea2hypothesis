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
from idea2hypothesis.literature.novelty import (
    DISCLAIMER,
    PAPERS_PER_HYPOTHESIS,
    check_novelty,
    extract_keywords,
    overlap,
)
from tests.fixtures import FixtureLiterature, FixtureNoveltyJudge

QUERIES = ["sleep duration exam performance", "quantum annealing protein folding"]


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


async def test_novelty_report_carries_the_judge_verdict_per_hypothesis() -> None:
    hypotheses = [
        {"id": "H1", "statement": "Sleep duration changes exam score in university students"},
        {"id": "H2", "statement": "Quantum annealing improves protein folding bandwidth"},
    ]
    judge = FixtureNoveltyJudge({"H1": "tested"})
    report = await check_novelty(
        "sleep and exams", hypotheses, queries=QUERIES, literature=FixtureLiterature(),
        judge=judge,
    )  # fmt: skip
    assert report["kind"] == "novelty_assessment" and report["disclaimer"] == DISCLAIMER
    assert report["total_papers_retrieved"] > 0 and report["search_coverage"] == "full"
    rows = {r["hypothesis_id"]: r for r in report["per_hypothesis"]}
    assert rows["H1"]["verdict"] == "tested" and rows["H2"]["verdict"] == "new"
    assert rows["H1"]["reason"] == "tested by fixture"
    # the judge reads at most five papers per hypothesis, the closest first
    (items,) = judge.calls
    assert all(len(i["papers"]) <= PAPERS_PER_HYPOTHESIS for i in items)
    assert "sleep" in items[0]["papers"][0]["title"].lower()
    assert rows["H1"]["closest_paper"]["paper_id"] == items[0]["papers"][0]["paper_id"]
    assert report["similar_papers_found"] == 1
    assert report["similar_papers"][0]["hypothesis_id"] == "H1"
    assert report["novelty_score"] == 0.5 and report["assessment"] == "moderate"
    assert report["recommendation"] == "differentiate_or_reconsider"


async def test_related_work_is_not_read_as_a_prior_test() -> None:
    hypotheses = [
        {"id": f"H{i}", "statement": "Sleep duration and exam scores of students"}
        for i in range(1, 4)
    ]
    report = await check_novelty(
        "sleep and exams", hypotheses, queries=QUERIES, literature=FixtureLiterature(),
        judge=FixtureNoveltyJudge({"H1": "related", "H2": "related"}),
    )  # fmt: skip
    assert report["similar_papers_found"] == 0  # related papers are listed, not counted as tests
    assert {p["verdict"] for p in report["similar_papers"]} == {"related"}
    assert report["novelty_score"] == round(2 / 3, 3) and report["recommendation"] == "proceed"


def test_overlap_is_the_share_of_the_hypothesis_words_a_paper_holds() -> None:
    keywords = extract_keywords("Sleep duration and exam scores")
    long_abstract = "We measured sleep duration and exam score. " + "Unrelated filler words. " * 80
    assert overlap(keywords, "Sleep and exams", long_abstract) == 1.0  # length does not dilute it
    assert overlap(keywords, "Glacier albedo", "monsoon onset over deltas") == 0.0


async def test_novelty_without_any_coverage_is_flagged_not_perfect() -> None:
    class Failing:
        async def search(self, queries, *, limit, year_min=0):  # type: ignore[no-untyped-def]
            raise RuntimeError("network down")

    judge = FixtureNoveltyJudge()
    report = await check_novelty(
        "topic words",
        [{"id": "H1", "statement": "x y z claim"}],
        queries=["x y z"],
        literature=Failing(),
        judge=judge,
    )
    assert report["assessment"] == "insufficient_data" and report["novelty_score"] is None
    assert report["recommendation"] == "proceed_with_caution"
    assert report["search_errors"] and judge.calls == []  # nothing to read, nothing judged


async def test_a_failed_judgement_gives_no_verdict() -> None:
    report = await check_novelty(
        "sleep and exams",
        [{"id": "H1", "statement": "Sleep duration changes exam score"}],
        queries=QUERIES,
        literature=FixtureLiterature(),
        judge=FixtureNoveltyJudge(fail=True),
    )
    assert report["per_hypothesis"][0]["verdict"] is None
    assert report["assessment"] == "insufficient_data" and report["novelty_score"] is None


async def test_novelty_checked_only_against_the_run_papers_is_not_a_plain_proceed() -> None:
    seen = [{"paper_id": "p-1", "title": "Sleep and exams", "abstract": "sleep duration exam"}]
    report = await check_novelty(
        "sleep and exams",
        [{"id": "H1", "statement": "Quantum annealing improves protein folding bandwidth"}],
        queries=["quantum annealing protein folding"],
        literature=FixtureLiterature([], source_errors={"openalex": "HTTP 429 (rate limited)"}),
        judge=FixtureNoveltyJudge(),
        papers_already_seen=seen,
    )
    assert report["total_papers_retrieved"] == 0
    assert report["search_coverage"] == "run_corpus_only"
    assert report["assessment"] == "high"
    assert report["recommendation"] == "proceed_with_caution"
    assert any("429" in e for e in report["search_errors"])


async def test_novelty_searches_the_given_queries_and_reads_the_closest_paper() -> None:
    papers = [
        mk(f"Glacier albedo record {i}", "openalex", f"W{i}", citation_count=900 - i,
           abstract="Glacier albedo and monsoon onset over coastal deltas.")
        for i in range(40)
    ]  # fmt: skip
    papers.append(
        mk(
            "Quantum annealing improves protein folding bandwidth",
            "arxiv",
            "2401.1",
            arxiv_id="2401.1",
            citation_count=0,
            abstract="We test whether quantum annealing improves protein folding bandwidth.",
        )
    )
    literature = FixtureLiterature(papers)
    judge = FixtureNoveltyJudge({"H1": "tested"})
    report = await check_novelty(
        "folding",
        [{"id": "H1", "statement": "Quantum annealing improves protein folding bandwidth"}],
        queries=["quantum annealing protein folding"],
        literature=literature,
        judge=judge,
    )
    assert literature.calls == [["quantum annealing protein folding"]]
    assert report["search_queries"] == ["quantum annealing protein folding"]
    assert report["papers_compared"] == 41
    # the closest paper is the least cited of 41: it is still read, and read first
    assert judge.calls[0][0]["papers"][0]["title"].startswith("Quantum annealing")
    closest = report["per_hypothesis"][0]["closest_paper"]
    assert closest["title"] == "Quantum annealing improves protein folding bandwidth"
    assert report["similar_papers_found"] == 1


def test_extract_keywords_drops_stop_words_and_short_tokens() -> None:
    assert extract_keywords("The effect of sleep on an exam, ok") == ["effect", "sleep", "exam"]
