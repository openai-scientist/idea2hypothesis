from __future__ import annotations

import re
from pathlib import Path

import pytest

from idea2hypothesis.prompts import DOMAINS, PromptError, PromptLoader

STAGE_VARIABLES = {
    "topic_init": dict(topic="t", domains="d", constraints="c", memory_context="", feedback=""),
    "topic_evaluation": dict(topic="t", goal_json="{}"),
    "problem_decompose": dict(topic="t", goal_json="{}", feedback=""),
    "search_strategy": dict(topic="t", problem_tree_json="{}", feedback="", year_hint=""),
    "literature_screen": dict(
        topic="t", domains="d", min_relevance=0.7, min_quality=0.5, candidates_json="[]"
    ),
    "knowledge_extract": dict(topic="t", paper_json="{}"),
    "synthesis": dict(topic="t", problem_tree_json="{}", cards_json="[]"),
    "hypothesis_gen": dict(
        topic="t",
        constraints="c",
        memory_context="",
        valid_refs="r",
        valid_gaps="g",
        synthesis_json="{}",
        perspectives="p",
        judge_assessment="",
        feedback="",
        min_hypotheses="3",
        max_hypotheses="6",
    ),
    "debate_critique": dict(
        role="r", own_position="x", others="y", valid_refs="r", valid_gaps="g", synthesis_json="{}"
    ),
    "debate_answer": dict(
        role="r",
        own_position="x",
        challenges="y",
        valid_refs="r",
        valid_gaps="g",
        synthesis_json="{}",
    ),
    "debate_review": dict(role="r", answered="a", added="n", valid_refs="r", synthesis_json="{}"),
    "debate_judge": dict(perspectives="p"),
}
ROLE_VARIABLES = dict(
    topic="t", constraints="c", feedback="", synthesis_json="{}", valid_refs="r", valid_gaps="g"
)


def test_assets_load_from_the_installed_package_not_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    loader = PromptLoader("ml")
    assert set(STAGE_VARIABLES) <= set(loader.keys())
    assert loader.role_names() == ["innovator", "pragmatist", "contrarian"]


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_prompt_renders_in_every_domain(domain: str) -> None:
    loader = PromptLoader(domain)
    for key, variables in STAGE_VARIABLES.items():
        rendered = loader.render(key, **variables)
        assert rendered.user and rendered.system
        assert "{{" not in rendered.user + rendered.system, key
    for role in loader.role_names():
        rendered = loader.render_role(role, **ROLE_VARIABLES)
        assert rendered.json_mode and "{{" not in rendered.user


def test_required_variables_match_what_the_stages_pass() -> None:
    loader = PromptLoader("ml")
    for key, variables in STAGE_VARIABLES.items():
        assert loader.required_variables(key) == set(variables), key


def test_every_stage_prompt_requests_a_single_json_object() -> None:
    loader = PromptLoader("ml")
    for key in (
        "topic_init",
        "problem_decompose",
        "search_strategy",
        "literature_screen",
        "knowledge_extract",
        "synthesis",
        "hypothesis_gen",
        "topic_evaluation",
    ):
        rendered = loader.render(key, **STAGE_VARIABLES[key])
        assert rendered.json_mode, key
        assert "ONE JSON object" in rendered.user, key


def test_missing_variable_is_an_error_naming_it() -> None:
    loader = PromptLoader("ml")
    with pytest.raises(PromptError, match="missing template variables.*topic"):
        loader.render("topic_evaluation", goal_json="{}")


def test_unknown_key_and_domain() -> None:
    with pytest.raises(PromptError, match="unknown prompt key"):
        PromptLoader("ml").render("experiment_design")
    with pytest.raises(PromptError, match="unknown prompt domain"):
        PromptLoader("chemistry")


def test_reliability_rules_are_in_the_prompts() -> None:
    loader = PromptLoader("ml")
    topic = loader.render("topic_init", **STAGE_VARIABLES["topic_init"])
    assert "researchable" in topic.user and "Wi-Fi" in topic.system  # out-of-scope guard
    screen = loader.render("literature_screen", **STAGE_VARIABLES["literature_screen"])
    assert "false_friend" in screen.user
    assert "staleness" in screen.system  # polysemy example
    hyp = loader.render("hypothesis_gen", **STAGE_VARIABLES["hypothesis_gen"])
    for needle in ("FALSIFIABLE", "EVIDENCE-LED DIRECTION", "DISTINCT MECHANISMS", "TRACEABLE"):
        assert needle in hyp.user
    assert "falsification_criteria" in hyp.user and "estimand" in hyp.user
    # The set's direction and surprise follow the evidence; nothing forces either.
    assert "DIRECTIONALLY DIVERSE" not in hyp.user and "SURPRISING" not in hyp.system
    card = loader.render("knowledge_extract", **STAGE_VARIABLES["knowledge_extract"])
    assert "quotes" in card.user and "clearly implies" not in card.user


def test_domain_overrides_replace_only_what_they_define() -> None:
    ml = PromptLoader("ml").render("literature_screen", **STAGE_VARIABLES["literature_screen"])
    hep = PromptLoader("hep").render("literature_screen", **STAGE_VARIABLES["literature_screen"])
    assert "HEP-ph" in hep.system and "HEP-ph" not in ml.system
    assert "HEP-ph guidance" in hep.user and "SCREENING RULES" in hep.user  # schema inherited
    bio = PromptLoader("biology")
    assert "COBRApy" in bio.render("hypothesis_gen", **STAGE_VARIABLES["hypothesis_gen"]).system
    assert bio.render("synthesis", **STAGE_VARIABLES["synthesis"]).user == (
        PromptLoader("ml").render("synthesis", **STAGE_VARIABLES["synthesis"]).user
    )
    assert PromptLoader("hep").role_names() == ["theorist", "phenomenologist", "experimentalist"]
    assert PromptLoader("biology").role_names() == [
        "model_builder",
        "fba_analyst",
        "experimentalist",
    ]


def test_user_override_file(tmp_path: Path) -> None:
    override = tmp_path / "over.yaml"
    override.write_text(
        "stages:\n  topic_evaluation:\n    system: custom system\n"
        "blocks:\n  json_rules: CUSTOM RULES\n",
        encoding="utf-8",
    )
    loader = PromptLoader("ml", override)
    rendered = loader.render("topic_evaluation", **STAGE_VARIABLES["topic_evaluation"])
    assert rendered.system == "custom system" and "CUSTOM RULES" in rendered.user
    assert loader.snapshot()["override_file"] == str(override)


def test_bad_override_files_are_rejected(tmp_path: Path) -> None:
    bad_key = tmp_path / "a.yaml"
    bad_key.write_text("stages:\n  nonexistent:\n    system: x\n", encoding="utf-8")
    with pytest.raises(PromptError, match="unknown prompt key"):
        PromptLoader("ml", bad_key)
    bad_field = tmp_path / "b.yaml"
    bad_field.write_text("stages:\n  topic_init:\n    sistem: x\n", encoding="utf-8")
    with pytest.raises(PromptError, match="unknown fields"):
        PromptLoader("ml", bad_field)
    with pytest.raises(PromptError, match="cannot read"):
        PromptLoader("ml", tmp_path / "missing.yaml")


def test_snapshot_is_stable_and_changes_with_content(tmp_path: Path) -> None:
    first = PromptLoader("ml").snapshot()
    assert first == PromptLoader("ml").snapshot()
    assert re.fullmatch(r"[0-9a-f]{64}", first["content_sha256"])
    assert PromptLoader("hep").snapshot()["content_sha256"] != first["content_sha256"]
    assert "topic_init" in first["prompts"] and "json_rules" in first["blocks"]


def test_unknown_block_is_reported(tmp_path: Path) -> None:
    override = tmp_path / "o.yaml"
    override.write_text(
        "stages:\n  topic_evaluation:\n    user: 'Hi {{> nowhere}} {{topic}}'\n", encoding="utf-8"
    )
    loader = PromptLoader("ml", override)
    with pytest.raises(PromptError, match="unknown prompt block"):
        loader.render("topic_evaluation", topic="t", goal_json="{}")


def test_extracting_and_judging_prompts_run_at_temperature_zero() -> None:
    loader = PromptLoader("ml")
    for key in (
        "topic_init", "topic_evaluation", "problem_decompose", "search_strategy",
        "literature_screen", "knowledge_extract", "synthesis", "argument_map", "debate_judge",
    ):  # fmt: skip
        variables = STAGE_VARIABLES.get(key) or dict(
            topic="t", claims_json="[]", hypotheses_json="[]"
        )
        assert loader.render(key, **variables).temperature == 0, key
    # Proposing ideas keeps the configured temperature.
    assert loader.render("hypothesis_gen", **STAGE_VARIABLES["hypothesis_gen"]).temperature is None
    role = loader.role_names()[0]
    assert loader.render_role(role, **ROLE_VARIABLES).temperature is None


def test_a_prompt_temperature_outside_zero_to_one_is_rejected(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text("stages:\n  synthesis:\n    temperature: 1.5\n", encoding="utf-8")
    with pytest.raises(PromptError, match="temperature"):
        PromptLoader("ml", override)
    override.write_text("stages:\n  synthesis:\n    temperature: 0.2\n", encoding="utf-8")
    loader = PromptLoader("ml", override)
    assert loader.render("synthesis", **STAGE_VARIABLES["synthesis"]).temperature == 0.2
