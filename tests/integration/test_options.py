"""Optional behaviour: debate, reviewer, domains, memory, hardware, novelty, data edge cases."""

from __future__ import annotations

import json
import re
from pathlib import Path

from idea2hypothesis.memory.ideation import IdeationMemory
from idea2hypothesis.memory.retriever import hashing_embed
from idea2hypothesis.pipeline.models import RunStatus
from idea2hypothesis.pipeline.runner import run_pipeline
from idea2hypothesis.prompts.loader import PromptLoader
from tests.conftest import make_config, make_services, request
from tests.fixtures import FixtureLiterature, FixtureLLM, PromptInfo, make_fixture_papers


async def test_debate_rounds_with_an_independent_reviewer(tmp_path: Path) -> None:
    llm, reviewer = FixtureLLM(), FixtureLLM()
    cfg = make_config(tmp_path, review={"mode": "auto"}, llm={"debate_rounds": 1})
    services = make_services(tmp_path, llm=llm, config=cfg)
    services.reviewer = reviewer
    result = await run_pipeline(request(), services)

    assert result.status is RunStatus.COMPLETED, result.error
    assert llm.count("debate_critique") == 3  # one per role
    # only the perspectives that were challenged answer (the fixture challenges two of them)
    assert llm.count("debate_answer") == 2
    assert reviewer.count("debate_judge") == 1 and llm.count("debate_judge") == 0
    art = services.store.artifacts(result.run_id)
    files = set(art.list_files(8))
    assert {"perspectives/innovator.r1.json", "perspectives/debate_record.json"} <= files
    record = art.read_json(8, "perspectives/debate_record.json")
    assert record["independent_judge"] is True and record["rounds"] == 1
    final_prompt = [c for c in llm.calls if c.key == "hypothesis_gen"][0].user
    assert "Independent reviewer assessment" in final_prompt
    assert art.read_json(8, "hypotheses.json")["debate_rounds"] == 1


async def test_the_debate_sees_what_each_card_reports(tmp_path: Path) -> None:
    # Without the cards' findings no critic could tell a claim a card already reports.
    llm = FixtureLLM()
    cfg = make_config(tmp_path, review={"mode": "auto"}, llm={"debate_rounds": 1})
    services = make_services(tmp_path, llm=llm, config=cfg)
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED, result.error
    art = services.store.artifacts(result.run_id)
    card = next(iter(sorted((art.stage_dir(6) / "cards").glob("*.json"))))
    findings = json.loads(card.read_text())["findings"][:60]
    for key in ("perspective", "debate_critique", "debate_answer", "debate_review",
                "hypothesis_gen"):  # fmt: skip
        prompts = [c.user for c in llm.calls if c.key == key]
        assert prompts and all(findings in p for p in prompts), key


async def test_debate_without_a_reviewer_warns_that_the_judge_is_not_independent(
    tmp_path: Path,
) -> None:
    cfg = make_config(tmp_path, review={"mode": "auto"}, llm={"debate_rounds": 1})
    services = make_services(tmp_path, config=cfg)
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    done = [
        e
        for e in services.store.read_events(result.run_id)
        if e.stage == 8 and e.type == "stage.completed"
    ]
    assert any("not independent" in w for w in done[0].data["warnings"])


async def test_a_withdrawn_hypothesis_leaves_the_debate_and_the_final_set(tmp_path: Path) -> None:
    def withdraw_all(info: object) -> dict:
        user = info.user  # type: ignore[attr-defined]
        own = user.split("numbered:\n", 1)[1].split("\n\nChallenges to your", 1)[0]
        hyps = [json.loads(p) for p in re.split(r"^\d+\. ", own, flags=re.MULTILINE)[1:]]
        asked = re.findall(r"^(\d+)\. \(to your hypothesis (\d+)", user, re.MULTILINE)
        answers = [{"challenge": int(n), "action": "withdraw", "text": "it cannot fail"}
                   for n, _ in asked]  # fmt: skip
        return {
            "answers": answers,
            "hypotheses": hyps,
            "withdrawn": sorted({int(k) for _, k in asked}),
        }

    llm = FixtureLLM(overrides={"debate_answer": withdraw_all})
    cfg = make_config(tmp_path, review={"mode": "auto"}, llm={"debate_rounds": 1})
    services = make_services(tmp_path, llm=llm, config=cfg)
    result = await run_pipeline(request(), services)

    assert result.status is RunStatus.COMPLETED, result.error
    art = services.store.artifacts(result.run_id)
    record = art.read_json(8, "perspectives/debate_record.json")
    assert record["withdrawn"] == {"innovator": [1], "pragmatist": [1]}
    assert record["answers"]["innovator"] == {"revise": 0, "defend": 0, "withdraw": 2}
    gone = [art.read_json(8, f"perspectives/{r}.json")["hypotheses"][0]["statement"]
            for r in ("innovator", "pragmatist")]  # fmt: skip
    kept = art.read_json(8, "perspectives/innovator.json")["hypotheses"][1]["statement"]
    for key in ("debate_judge", "hypothesis_gen"):
        prompt = [c for c in llm.calls if c.key == key][0].user
        assert not any(g in prompt for g in gone) and kept in prompt, key


async def test_tensions_left_open_are_reported_and_one_must_be_settled(tmp_path: Path) -> None:
    def two_tensions(info: object) -> dict:
        syn = FixtureLLM()._default_synthesis(info)  # type: ignore[arg-type]
        x1 = syn["tensions"][0]
        syn["tensions"].append({**x1, "id": "X2", "text": "self-report versus actigraphy"})
        return syn

    def unsettled(info: object) -> dict:
        data = FixtureLLM()._default_hypothesis_gen(info)  # type: ignore[arg-type]
        for h in data["hypotheses"]:
            h["tension_ids"] = []
        return data

    llm = FixtureLLM(overrides={"synthesis": two_tensions, "hypothesis_gen": [unsettled]})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)

    assert result.status is RunStatus.COMPLETED, result.error
    art = services.store.artifacts(result.run_id)
    doc = art.read_json(8, "hypotheses.json")
    assert doc["hypotheses"][0]["tension_ids"] == ["X1"] and doc["open_tensions"] == ["X2"]
    assert "## Open tensions" in art.read_text(8, "hypotheses.md")
    calls = [c for c in art.read_llm_calls(8, 1) if c["label"] == "hypothesis_gen"]
    assert [c["outcome"] for c in calls] == ["rejected", "accepted"]
    assert any("must settle one of them" in p for p in calls[0]["problems"])


async def test_without_debate_there_are_no_rebuttals(tmp_path: Path) -> None:
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    await run_pipeline(request(), services)
    assert llm.count("debate_critique") == 0 and llm.count("debate_answer") == 0
    assert llm.count("debate_judge") == 0
    assert llm.count("perspective") == 3


async def test_a_failed_perspective_is_dropped_but_the_stage_continues(tmp_path: Path) -> None:
    llm = FixtureLLM(overrides={"perspective": ["bad", "bad", "bad"]})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    art = services.store.artifacts(result.run_id)
    perspectives = [
        f for f in art.list_files(8) if f.startswith("perspectives/") and f.endswith(".json")
    ]
    assert 1 <= len(perspectives) < 3


async def test_no_usable_perspective_fails_instead_of_inventing_hypotheses(tmp_path: Path) -> None:
    llm = FixtureLLM(overrides={"perspective": ["bad"] * 20})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.FAILED
    assert result.error is not None and result.error.code == "NO_PERSPECTIVES"
    assert not services.store.artifacts(result.run_id).exists(8, "hypotheses.json")


async def test_duplicate_novelty_text_is_sent_back_for_repair(tmp_path: Path) -> None:
    def duplicated(info) -> dict:
        data = FixtureLLM()._default_hypothesis_gen(info)
        for h in data["hypotheses"]:
            h["novelty"] = "Isolates causal mechanisms under controlled conditions."
        return data

    llm = FixtureLLM(overrides={"hypothesis_gen": [duplicated]})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    assert llm.count("hypothesis_gen") == 2


async def test_hep_domain_uses_domain_prompts_and_roles(tmp_path: Path) -> None:
    llm = FixtureLLM()
    cfg = make_config(tmp_path, review={"mode": "auto"}, prompts={"domain": "hep"})
    services = make_services(tmp_path, llm=llm, config=cfg)
    result = await run_pipeline(
        request(topic="Scalar mediator dark matter simplified model"), services
    )
    assert result.status is RunStatus.COMPLETED, result.error
    assert any("HEP-ph" in c.system for c in llm.calls if c.key == "literature_screen")
    assert any("particle physicist" in c.system for c in llm.calls if c.key == "perspective")
    snapshot = services.store.read_snapshot(result.run_id, "prompts")
    assert snapshot["domain"] == "hep"


async def test_snapshots_record_names_not_secrets(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("I2H_LLM_API_KEY", "sk-should-never-be-stored")
    services = make_services(tmp_path, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    run_dir = services.store.run_dir(result.run_id)
    config_text = (run_dir / "config.snapshot.json").read_text(encoding="utf-8")
    assert "sk-should-never-be-stored" not in config_text
    assert json.loads(config_text)["llm"]["api_key_env"] == "I2H_LLM_API_KEY"
    prompts = json.loads((run_dir / "prompts.snapshot.json").read_text(encoding="utf-8"))
    assert prompts["content_sha256"] == PromptLoader("ml").snapshot()["content_sha256"]


async def test_hardware_profile_only_when_the_advisory_is_enabled(tmp_path: Path) -> None:
    on = make_services(tmp_path / "on", review={"mode": "auto"})
    result = await run_pipeline(request(), on)
    art = on.store.artifacts(result.run_id)
    assert art.read_json(1, "hardware_profile.json")["gpu_type"] == "cpu"

    off = make_services(
        tmp_path / "off", review={"mode": "auto"}, research={"hardware_advisory": False}
    )
    result = await run_pipeline(request(), off)
    assert not off.store.artifacts(result.run_id).exists(1, "hardware_profile.json")


async def test_novelty_report_can_be_disabled(tmp_path: Path) -> None:
    services = make_services(tmp_path, review={"mode": "auto"}, research={"novelty_check": False})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    assert not services.store.artifacts(result.run_id).exists(8, "novelty_report.json")


async def test_source_errors_are_recorded_in_search_meta_and_warnings(tmp_path: Path) -> None:
    literature = FixtureLiterature(source_errors={"semantic_scholar": "HTTP 429 (rate limited)"})
    services = make_services(tmp_path, literature=literature, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    meta = services.store.artifacts(result.run_id).read_json(4, "search_meta.json")
    assert any("semantic_scholar" in e and "429" in e for e in meta["errors"])
    assert meta["per_source"]["semantic_scholar"]["errors"]
    completed = [
        e
        for e in services.store.read_events(result.run_id)
        if e.stage == 4 and e.type == "stage.completed"
    ]
    assert any("429" in w for w in completed[0].data["warnings"])
    # every query fails the same way: one line for the source, not one per query
    assert len(meta["per_source"]["semantic_scholar"]["errors"]) > 1
    assert sum("semantic_scholar:" in w for w in completed[0].data["warnings"]) == 1


async def test_papers_without_abstracts_are_set_aside_before_screening(tmp_path: Path) -> None:
    papers = make_fixture_papers()
    import dataclasses

    papers[0] = dataclasses.replace(papers[0], abstract="")
    services = make_services(
        tmp_path, literature=FixtureLiterature(papers), review={"mode": "auto"}
    )
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    art = services.store.artifacts(result.run_id)
    review = art.read_json(5, "review.json")
    blank = next(d for d in review["decisions"] if d["title"] == papers[0].title)
    assert blank["decision"] == "prefiltered" and blank["relevance_score"] is None
    assert "no abstract" in blank["reason"]
    assert art.read_json(5, "screen_meta.json")["no_abstract"] == 1
    meta = art.read_json(6, "knowledge_meta.json")
    assert meta["cards"] == 8 and meta["skipped"] == []
    assert len(list((art.stage_dir(6) / "cards").glob("*.json"))) == 8


async def test_a_failed_novelty_search_is_reported_not_read_as_novel(tmp_path: Path) -> None:
    # Stage 4 searches one query at a time; the novelty check searches several at once.
    literature = FixtureLiterature(outage=lambda queries: len(queries) > 1)
    services = make_services(tmp_path, literature=literature, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    report = services.store.artifacts(result.run_id).read_json(8, "novelty_report.json")
    assert report["total_papers_retrieved"] == 0
    assert report["search_coverage"] == "run_corpus_only"
    assert report["recommendation"] in ("proceed_with_caution", "differentiate",
                                        "differentiate_or_reconsider")  # fmt: skip
    completed = [
        e
        for e in services.store.read_events(result.run_id)
        if e.stage == 8 and e.type == "stage.completed"
    ]
    assert any(
        "compared only with the run's own papers" in w and "search errors" in w
        for w in completed[0].data["warnings"]
    )


async def test_the_novelty_search_sends_one_short_query_per_hypothesis(tmp_path: Path) -> None:
    literature = FixtureLiterature()
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, literature=literature, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    art = services.store.artifacts(result.run_id)
    hypotheses = art.read_json(8, "hypotheses.json")["hypotheses"]
    report = art.read_json(8, "novelty_report.json")
    assert llm.count("novelty_queries") == 1
    assert report["search_queries"] == ["sleep duration exam performance"] * len(hypotheses)
    assert literature.calls[-1] == report["search_queries"]
    assert report["search_coverage"] == "full"


async def test_novelty_queries_that_never_come_back_valid_skip_the_search_only(
    tmp_path: Path,
) -> None:
    sentence = {"queries": [{"hypothesis_id": "H1", "query": "a whole sentence " * 4}]}
    llm = FixtureLLM(overrides={"novelty_queries": sentence})
    literature = FixtureLiterature()
    services = make_services(tmp_path, llm=llm, literature=literature, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    report = services.store.artifacts(result.run_id).read_json(8, "novelty_report.json")
    assert report["search_queries"] == [] and report["total_papers_retrieved"] == 0
    assert report["search_coverage"] == "run_corpus_only"
    assert report["search_errors"][0].startswith("no search: the search queries could not")
    completed = [
        e
        for e in services.store.read_events(result.run_id)
        if e.stage == 8 and e.type == "stage.completed"
    ]
    warnings = completed[0].data["warnings"]
    assert any("compared only with the run's own papers" in w for w in warnings)


async def test_a_hypothesis_a_paper_already_tests_is_flagged(tmp_path: Path) -> None:
    def tested_first(info: PromptInfo) -> dict:
        items = info.section_json("Hypotheses and the papers to read:\n")
        return {
            "judgements": [
                {
                    "hypothesis_id": item["id"],
                    "verdict": "tested" if i == 0 else "new",
                    "paper_ids": [item["papers"][0]["paper_id"]] if i == 0 else [],
                    "reason": "Its abstract reports the same test." if i == 0 else "Other topics.",
                }
                for i, item in enumerate(items)
            ]
        }

    llm = FixtureLLM(overrides={"novelty_judge": tested_first})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    art = services.store.artifacts(result.run_id)
    report = art.read_json(8, "novelty_report.json")
    first = report["per_hypothesis"][0]
    assert first["verdict"] == "tested" and first["reason"] == "Its abstract reports the same test."
    assert report["similar_papers_found"] == 1
    assert report["recommendation"] in ("differentiate", "differentiate_or_reconsider")
    completed = [
        e
        for e in services.store.read_events(result.run_id)
        if e.stage == 8 and e.type == "stage.completed"
    ]
    assert any(
        f"may already test {first['hypothesis_id']}" in w for w in completed[0].data["warnings"]
    )


async def test_papers_that_cannot_be_judged_give_no_verdict_and_the_run_goes_on(
    tmp_path: Path,
) -> None:
    llm = FixtureLLM(overrides={"novelty_judge": {"judgements": []}})
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    report = services.store.artifacts(result.run_id).read_json(8, "novelty_report.json")
    assert report["assessment"] == "insufficient_data" and report["novelty_score"] is None
    assert report["judge_errors"][0].startswith("no judgement: the papers could not be judged")
    assert all(r["verdict"] is None for r in report["per_hypothesis"])
    completed = [
        e
        for e in services.store.read_events(result.run_id)
        if e.stage == 8 and e.type == "stage.completed"
    ]
    assert any(w.startswith("novelty was not judged") for w in completed[0].data["warnings"])


async def test_the_argument_map_sees_what_each_card_studies(tmp_path: Path) -> None:
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    prompt = next(c for c in llm.calls if c.key == "argument_map")
    claims = prompt.section_json("Claims and their cards:\n", "Hypotheses:")
    card = claims[0]["cards"][0]
    assert {"title", "method", "findings", "data"} <= card.keys()
    assert card["title"] and card["method"]


async def test_when_no_paper_matches_the_topic_keywords_the_reviewer_judges_all(
    tmp_path: Path,
) -> None:
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    result = await run_pipeline(
        request(topic="Zebra migration corridors", domains=("ecology",)), services
    )
    art = services.store.artifacts(result.run_id)
    meta = art.read_json(5, "screen_meta.json")
    assert meta["prefilter_bypassed"] is True and meta["prefiltered"] == 0
    assert llm.count("literature_screen") >= 1
    assert result.status is RunStatus.COMPLETED


async def test_prefiltered_papers_are_listed_with_a_reason_and_no_scores(tmp_path: Path) -> None:
    papers = make_fixture_papers()
    import dataclasses

    papers.append(
        dataclasses.replace(
            papers[0],
            title="Quark gluon plasma dynamics",
            abstract="Heavy ion collisions.",
            doi="10.1000/quark",
            url="",
            source_records=papers[0].source_records,
        )
    )
    services = make_services(
        tmp_path, literature=FixtureLiterature(papers), review={"mode": "auto"}
    )
    result = await run_pipeline(request(), services)
    review = services.store.artifacts(result.run_id).read_json(5, "review.json")
    quark = next(d for d in review["decisions"] if "Quark" in d["title"])
    assert quark["decision"] == "prefiltered" and quark["relevance_score"] is None
    assert quark["reason"] and review["summary"]["prefiltered"] == 1


async def test_ideation_memory_is_recorded_and_recalled(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, review={"mode": "auto"}, runtime={"ideation_memory": True})
    memory = IdeationMemory(store_dir=tmp_path / "mem", embed_fn=hashing_embed)
    memory.record_topic_outcome(
        "Old failed idea about tides", "failure", 2.0, reason="NO_LITERATURE"
    )
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, config=cfg)
    services.memory = memory

    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.COMPLETED
    first_prompt = [c for c in llm.calls if c.key == "topic_init"][0].user
    assert "Old failed idea about tides" in first_prompt  # anti-pattern reaches the prompt
    contents = [e.content for e in memory.store.get_all("ideation")]
    assert any(c.startswith("Topic: Effect of sleep") and "success" in c for c in contents)
    assert any(c.startswith("Hypothesis:") for c in contents)
    assert (tmp_path / "mem" / "ideation.jsonl").is_file()  # persisted


async def test_failed_topics_become_anti_patterns(tmp_path: Path) -> None:
    cfg = make_config(tmp_path, review={"mode": "auto"}, runtime={"ideation_memory": True})
    memory = IdeationMemory(store_dir=tmp_path / "mem", embed_fn=hashing_embed)
    services = make_services(tmp_path, config=cfg, literature=FixtureLiterature(papers=[]))
    services.memory = memory
    result = await run_pipeline(request(), services)
    assert result.status is RunStatus.FAILED
    patterns = memory.get_anti_patterns()
    assert len(patterns) == 1 and "NO_LITERATURE" in patterns[0]


async def test_memory_stays_untouched_when_disabled(tmp_path: Path) -> None:
    memory = IdeationMemory(store_dir=tmp_path / "mem")
    services = make_services(tmp_path, review={"mode": "auto"})
    services.memory = memory
    await run_pipeline(request(), services)
    assert memory.store.count() == 0


async def test_constraints_reach_the_prompts(tmp_path: Path) -> None:
    llm = FixtureLLM()
    services = make_services(tmp_path, llm=llm, review={"mode": "auto"})
    await run_pipeline(request(constraints=("no new data collection", "single GPU")), services)
    assert (
        "no new data collection; single GPU"
        in [c for c in llm.calls if c.key == "topic_init"][0].user
    )
    assert "single GPU" in [c for c in llm.calls if c.key == "hypothesis_gen"][0].user
