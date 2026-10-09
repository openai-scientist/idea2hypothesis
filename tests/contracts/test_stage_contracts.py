"""Per-stage schema and reference-integrity checks on the artifacts of a real (fixture) run."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from idea2hypothesis.pipeline.contracts import CONTRACTS, missing_inputs, validate_stage
from idea2hypothesis.pipeline.models import Stage
from idea2hypothesis.pipeline.runner import run_pipeline
from idea2hypothesis.storage.artifacts import ArtifactStore
from tests.conftest import make_services, request


@pytest.fixture(scope="module")
async def finished_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("contract-run")
    services = make_services(tmp, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status.value == "completed", result.error
    return services.store.run_dir(result.run_id)


@pytest.fixture
def art(finished_run: Path, tmp_path: Path) -> ArtifactStore:
    """A private copy so tests can corrupt artifacts freely."""
    copy = tmp_path / "copy"
    shutil.copytree(finished_run, copy)
    return ArtifactStore(copy)


def edit_json(art: ArtifactStore, stage: int, name: str, fn: Callable[[Any], None]) -> None:
    data = art.read_json(stage, name)
    fn(data)
    art.write_json(stage, name, data)


def edit_jsonl(art: ArtifactStore, stage: int, name: str, fn: Callable[[list[dict]], None]) -> None:
    rows = art.read_jsonl(stage, name)
    fn(rows)
    art.write_jsonl(stage, name, rows)


@pytest.mark.parametrize("stage", list(Stage))
def test_every_stage_of_a_good_run_satisfies_its_contract(art: ArtifactStore, stage: Stage) -> None:
    findings = validate_stage(stage, art)
    assert findings.ok, findings.errors
    assert missing_inputs(stage, art) == []
    assert CONTRACTS[stage].dod and CONTRACTS[stage].error_code


def test_json_artifacts_carry_a_schema_version(art: ArtifactStore) -> None:
    for stage, name in [
        (1, "goal.json"), (2, "problem_tree.json"), (2, "topic_evaluation.json"),
        (3, "queries.json"), (3, "sources.json"), (4, "search_meta.json"),
        (5, "review.json"), (5, "screen_meta.json"), (6, "knowledge_meta.json"),
        (7, "synthesis.json"), (8, "hypotheses.json"), (9, "argument_map.json"),
        (9, "semantic_graph.json"), (9, "research_canvas.json"),
    ]:  # fmt: skip
        # syntheses from schema 2 on give each tension an id and two sides of cards
        expected = 2 if name == "synthesis.json" else 1
        assert art.read_json(stage, name)["schema_version"] == expected, name
    for stage in range(1, 10):
        assert art.read_manifest(stage)["schema_version"] == 1


def test_stage1_requires_the_goal_fields(art: ArtifactStore) -> None:
    edit_json(art, 1, "goal.json", lambda d: d.update(objective=""))
    assert any("objective" in e for e in validate_stage(Stage.TOPIC_INIT, art).errors)


def test_stage2_requires_three_linked_sub_questions_and_valid_scores(art: ArtifactStore) -> None:
    edit_json(art, 2, "problem_tree.json", lambda d: d.update(sub_questions=d["sub_questions"][:2]))
    assert any("at least 3" in e for e in validate_stage(Stage.PROBLEM_DECOMPOSE, art).errors)


def test_stage2_rejects_out_of_range_topic_scores(art: ArtifactStore) -> None:
    edit_json(art, 2, "topic_evaluation.json", lambda d: d.update(novelty=11))
    assert any("novelty" in e for e in validate_stage(Stage.PROBLEM_DECOMPOSE, art).errors)


def test_stage3_queries_must_link_to_existing_sub_questions(art: ArtifactStore) -> None:
    edit_json(art, 3, "queries.json", lambda d: d["queries"][0].update(sub_question_ids=["SQ99"]))
    assert any(
        "unknown sub-questions" in e for e in validate_stage(Stage.SEARCH_STRATEGY, art).errors
    )


def test_stage3_needs_two_strategies(art: ArtifactStore) -> None:
    import yaml

    plan = yaml.safe_load(art.read_text(3, "search_plan.yaml"))
    plan["search_strategies"] = plan["search_strategies"][:1]
    art.write_text(3, "search_plan.yaml", yaml.safe_dump(plan))
    assert any("at least 2" in e for e in validate_stage(Stage.SEARCH_STRATEGY, art).errors)


def test_stage4_candidates_need_provenance(art: ArtifactStore) -> None:
    edit_jsonl(art, 4, "candidates.jsonl", lambda rows: rows[0].update(source_records=[]))
    assert any("provenance" in e for e in validate_stage(Stage.LITERATURE_COLLECT, art).errors)


def test_stage4_rejects_duplicates_and_bib_mismatch(art: ArtifactStore) -> None:
    edit_jsonl(
        art, 4, "candidates.jsonl", lambda rows: rows.append(dict(rows[0], paper_id="p-dup"))
    )
    errors = validate_stage(Stage.LITERATURE_COLLECT, art).errors
    assert any("duplicate" in e for e in errors) and any("references.bib" in e for e in errors)


def test_stage4_candidates_are_unique_real_records(art: ArtifactStore) -> None:
    rows = art.read_jsonl(4, "candidates.jsonl")
    assert len({r["paper_id"] for r in rows}) == len(rows) == 12
    assert all(r["source_records"][0]["provider"] and r["cite_key"] for r in rows)
    bib = art.read_text(4, "references.bib")
    assert all(f"{{{r['cite_key']}," in bib for r in rows)
    meta = art.read_json(4, "search_meta.json")
    assert meta["raw"] > meta["unique"] and meta["duplicates"] == meta["raw"] - meta["unique"]


def test_stage5_shortlist_must_come_from_candidates_with_scores(art: ArtifactStore) -> None:
    edit_jsonl(art, 5, "shortlist.jsonl", lambda rows: rows[0].update(paper_id="p-ghost"))
    errors = validate_stage(Stage.LITERATURE_SCREEN, art).errors
    assert any("not a collected candidate" in e for e in errors)


def test_stage5_scores_must_be_in_unit_interval(art: ArtifactStore) -> None:
    edit_jsonl(art, 5, "shortlist.jsonl", lambda rows: rows[0].update(relevance_score=7))
    assert any("relevance_score" in e for e in validate_stage(Stage.LITERATURE_SCREEN, art).errors)


def test_stage5_every_candidate_needs_a_decision_and_reason(art: ArtifactStore) -> None:
    edit_json(art, 5, "review.json", lambda d: d["decisions"].pop())
    assert any("every candidate" in e for e in validate_stage(Stage.LITERATURE_SCREEN, art).errors)


def test_stage5_decisions_have_reasons_and_unscored_papers_have_no_scores(
    art: ArtifactStore,
) -> None:
    def blank_reason(d: dict) -> None:
        d["decisions"][0]["reason"] = ""

    edit_json(art, 5, "review.json", blank_reason)
    assert any("no reason" in e for e in validate_stage(Stage.LITERATURE_SCREEN, art).errors)

    def fake_score(d: dict) -> None:
        d["decisions"][0].update(reason="r", decision="unscored", relevance_score=0.5)

    edit_json(art, 5, "review.json", fake_score)
    errors = validate_stage(Stage.LITERATURE_SCREEN, art).errors
    assert any("must not carry scores" in e for e in errors)


def test_stage5_empty_shortlist_is_invalid(art: ArtifactStore) -> None:
    art.write_jsonl(5, "shortlist.jsonl", [])
    assert any(
        "shortlist is empty" in e for e in validate_stage(Stage.LITERATURE_SCREEN, art).errors
    )


def test_stage6_cards_reference_shortlisted_papers_and_use_null_for_unknowns(
    art: ArtifactStore,
) -> None:
    cards = sorted((art.stage_dir(6) / "cards").glob("*.json"))
    assert len(cards) == 9
    first = json.loads(cards[0].read_text(encoding="utf-8"))
    assert (
        first["evidence_scope"] == "abstract" and first["data"] is None and first["metrics"] is None
    )
    assert first["card_id"] == f"card-{first['paper_id']}"

    first["paper_id"] = "p-ghost"
    cards[0].write_text(json.dumps(first), encoding="utf-8")
    assert any(
        "not in the shortlist" in e for e in validate_stage(Stage.KNOWLEDGE_EXTRACT, art).errors
    )


def test_stage6_rejects_invented_evidence_scope_and_template_values(art: ArtifactStore) -> None:
    path = sorted((art.stage_dir(6) / "cards").glob("*.json"))[0]
    card = json.loads(path.read_text(encoding="utf-8"))
    card["evidence_scope"] = "read in full"
    card["problem"] = ["not text"]
    path.write_text(json.dumps(card), encoding="utf-8")
    errors = validate_stage(Stage.KNOWLEDGE_EXTRACT, art).errors
    assert any("evidence_scope" in e for e in errors) and any("problem" in e for e in errors)


def test_stage7_gaps_link_to_sub_questions_and_cards(art: ArtifactStore) -> None:
    synthesis = art.read_json(7, "synthesis.json")
    assert len(synthesis["gaps"]) >= 2
    edit_json(art, 7, "synthesis.json", lambda d: d["gaps"][0].update(card_ids=["card-ghost"]))
    assert any("unknown cards" in e for e in validate_stage(Stage.SYNTHESIS, art).errors)


def test_stage7_needs_two_gaps_and_sub_question_links(art: ArtifactStore) -> None:
    edit_json(art, 7, "synthesis.json", lambda d: d.update(gaps=d["gaps"][:1]))
    assert any("at least 2" in e for e in validate_stage(Stage.SYNTHESIS, art).errors)


def test_stage7_unsupported_numbers_are_flagged_as_warnings(art: ArtifactStore) -> None:
    edit_json(
        art, 7, "synthesis.json", lambda d: d.update(overview="Sleep raises scores by 47.3%.")
    )
    findings = validate_stage(Stage.SYNTHESIS, art)
    assert findings.ok and any("47.3" in w for w in findings.warnings)


def test_stage7_stamps_and_ids_are_not_read_as_numbers(art: ArtifactStore) -> None:
    edit_json(
        art,
        7,
        "synthesis.json",
        lambda d: d.update(
            generated_at="2026-10-09T18:56:21.222+00:00",
            topic="Sleep in 2026 cohorts",
            overview="Sleep raises scores by 47.3%.",
        ),
    )
    warnings = validate_stage(Stage.SYNTHESIS, art).warnings
    flagged = [w for w in warnings if "does not appear in any card" in w]
    assert len(flagged) == 1 and "47.3" in flagged[0]


def test_stage8_references_resolve_to_gaps_and_evidence(art: ArtifactStore) -> None:
    edit_json(art, 8, "hypotheses.json", lambda d: d["hypotheses"][0].update(gap_id="G99"))
    assert any("gap_id" in e for e in validate_stage(Stage.HYPOTHESIS_GEN, art).errors)
    edit_json(
        art,
        8,
        "hypotheses.json",
        lambda d: d["hypotheses"][0].update(gap_id="G1", evidence_refs=["x"]),
    )
    assert any("do not resolve" in e for e in validate_stage(Stage.HYPOTHESIS_GEN, art).errors)


def test_stage8_requires_falsification_criteria_and_a_perspectives_directory(
    art: ArtifactStore,
) -> None:
    edit_json(
        art, 8, "hypotheses.json", lambda d: d["hypotheses"][0].update(falsification_criteria="")
    )
    assert any(
        "falsification_criteria" in e for e in validate_stage(Stage.HYPOTHESIS_GEN, art).errors
    )
    shutil.rmtree(art.stage_dir(8) / "perspectives")
    assert any("perspectives" in e for e in validate_stage(Stage.HYPOTHESIS_GEN, art).errors)


def test_novelty_report_is_labelled_as_an_assessment(art: ArtifactStore) -> None:
    report = art.read_json(8, "novelty_report.json")
    assert report["kind"] == "novelty_assessment"
    assert report["disclaimer"] == "heuristic assessment, not proof of novelty"
    edit_json(art, 8, "novelty_report.json", lambda d: d.pop("kind"))
    assert any("novelty" in e for e in validate_stage(Stage.HYPOTHESIS_GEN, art).errors)


def test_missing_output_is_reported_by_name(art: ArtifactStore) -> None:
    art.path(7, "synthesis.md").unlink()
    assert any("synthesis.md is missing" in e for e in validate_stage(Stage.SYNTHESIS, art).errors)


def test_missing_inputs_names_the_upstream_file(art: ArtifactStore) -> None:
    art.path(4, "candidates.jsonl").unlink()
    assert missing_inputs(Stage.LITERATURE_SCREEN, art) == ["stage-04/candidates.jsonl"]


def test_stage9_judges_every_clustered_card_and_grounds_every_hypothesis(
    art: ArtifactStore,
) -> None:
    edit_json(art, 9, "argument_map.json", lambda d: d["evidence_links"].pop())
    assert any("not judged" in e for e in validate_stage(Stage.ARGUMENT_MAP, art).errors)
    edit_json(art, 9, "argument_map.json", lambda d: d.update(rationales=d["rationales"][1:]))
    errors = validate_stage(Stage.ARGUMENT_MAP, art).errors
    assert any("no claim as rationale" in e for e in errors)


def test_stage9_graph_keeps_the_relation_direction(art: ArtifactStore) -> None:
    def flip(d: dict) -> None:
        r = next(r for r in d["relations"] if r["relation"] == "addresses")
        r["from"], r["to"] = r["to"], r["from"]

    edit_json(art, 9, "semantic_graph.json", flip)
    errors = validate_stage(Stage.ARGUMENT_MAP, art).errors
    assert any("must read from hypothesis to gap" in e for e in errors)


def test_stage9_graph_says_which_relations_are_judged(art: ArtifactStore) -> None:
    graph = art.read_json(9, "semantic_graph.json")
    judged = art.read_json(9, "argument_map.json")
    status = {r["relation"]: r["status"] for r in graph["relations"]}
    assert status["supports"] == status["provides_rationale_for"] == "unreviewed"
    assert status["decomposes_into"] == status["addresses"] == status["motivates"] == "stated"
    unrelated = [x for x in judged["evidence_links"] if x["relation"] == "unrelated"]
    linked = [r for r in graph["relations"] if r["relation"] in ("supports", "contradicts")]
    assert unrelated and len(linked) == len(judged["evidence_links"]) - len(unrelated)
    hypotheses = art.read_json(8, "hypotheses.json")["hypotheses"]
    assert {e["id"] for e in graph["entities"] if e["type"] == "hypothesis"} == {
        f"H:{h['id']}" for h in hypotheses
    }


def test_stage9_canvas_holds_nine_pieces_and_findings_wait(art: ArtifactStore) -> None:
    pieces = art.read_json(9, "research_canvas.json")["pieces"]
    assert len(pieces) == 9 and pieces[0]["id"] == "puzzle"
    assert next(p for p in pieces if p["id"] == "findings")["status"] == "pending"
    edit_json(art, 9, "research_canvas.json", lambda d: d["pieces"].pop())
    assert any("nine pieces" in e for e in validate_stage(Stage.ARGUMENT_MAP, art).errors)


def test_stage6_cards_quote_their_abstract_word_for_word(art: ArtifactStore) -> None:
    path = sorted((art.stage_dir(6) / "cards").glob("*.json"))[0]
    card = json.loads(path.read_text(encoding="utf-8"))
    assert card["schema_version"] == 2 and card["quotes"]["findings"]
    assert validate_stage(Stage.KNOWLEDGE_EXTRACT, art).ok

    card["quotes"]["findings"] = ["report large gains on every outcome measured"]
    path.write_text(json.dumps(card), encoding="utf-8")
    errors = validate_stage(Stage.KNOWLEDGE_EXTRACT, art).errors
    assert any("findings quote" in e and "word for word" in e for e in errors)

    del card["quotes"]
    path.write_text(json.dumps(card), encoding="utf-8")
    assert any("'quotes'" in e for e in validate_stage(Stage.KNOWLEDGE_EXTRACT, art).errors)


def test_stage6_cards_written_before_quotes_still_validate(art: ArtifactStore) -> None:
    path = sorted((art.stage_dir(6) / "cards").glob("*.json"))[0]
    card = json.loads(path.read_text(encoding="utf-8"))
    card["schema_version"] = 1
    del card["quotes"]
    path.write_text(json.dumps(card), encoding="utf-8")
    assert validate_stage(Stage.KNOWLEDGE_EXTRACT, art).ok
