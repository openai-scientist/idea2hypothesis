"""Stage 2 - PROBLEM_DECOMPOSE: prioritised sub-questions and topic evaluation."""

from __future__ import annotations

from typing import Any

from idea2hypothesis.pipeline.contracts import check_problem_tree, check_topic_evaluation
from idea2hypothesis.pipeline.models import StageContext
from idea2hypothesis.stages.base import StageFailure, compact_json, request_json
from idea2hypothesis.stages.topic_init import goal_for_prompt
from idea2hypothesis.storage.runs import utc_now

STAGE = 2


def render_tree_markdown(tree: dict[str, Any]) -> str:
    lines = [
        "# Problem decomposition",
        "",
        f"**Topic:** {tree['topic']}",
        "",
        "## Sub-questions",
        "",
    ]
    for q in tree["sub_questions"]:
        lines.append(f"{q['priority']}. **{q['id']}** - {q['text']}")
        lines.append(f"   - Serves: {q['goal_link']}")
        if q.get("tests"):
            lines.append(f"   - Evidence needed: {q['tests']}")
    risks = tree.get("risks") or []
    if risks:
        lines += ["", "## Risks", ""]
        lines += [
            f"- **{r.get('id', '')}** ({r.get('level', 'n/a')}, {r.get('sub_question_id', '-')}) "
            f"{r.get('text', '')}"
            for r in risks
            if isinstance(r, dict)
        ]
    return "\n".join(lines) + "\n"


async def run(ctx: StageContext) -> list[str]:
    goal = ctx.artifacts.read_json(1, "goal.json")
    goal_json = compact_json(goal_for_prompt(goal))

    prompt = ctx.prompts.render(
        "problem_decompose", topic=ctx.topic, goal_json=goal_json, feedback=ctx.feedback
    )
    tree, warnings = await request_json(
        ctx, prompt, label="problem_decompose", validate=check_problem_tree
    )
    questions = sorted(tree["sub_questions"], key=lambda q: q["priority"])
    tree["sub_questions"] = questions
    tree["priority_ranking"] = [q["id"] for q in questions]
    tree = {"schema_version": 1, "topic": ctx.topic, **tree, "generated_at": utc_now()}

    eval_prompt = ctx.prompts.render("topic_evaluation", topic=ctx.topic, goal_json=goal_json)
    evaluation, eval_warnings = await request_json(
        ctx,
        eval_prompt,
        label="topic_evaluation",
        validate=lambda d: check_topic_evaluation(d, require_reasons=True),
    )
    dims = ("novelty", "specificity", "feasibility")
    scores = [float(evaluation[k]) for k in dims]
    overall = round(sum(scores) / 3.0, 1)
    evaluation = {
        "schema_version": 1,
        "novelty": evaluation["novelty"],
        "specificity": evaluation["specificity"],
        "feasibility": evaluation["feasibility"],
        "overall": overall,
        "threshold": ctx.config.research.min_topic_score,
        "reasons": {k: str(evaluation["reasons"][k]).strip() for k in dims},
        "suggestion": str(evaluation["suggestion"]).strip(),
        # No paper has been retrieved yet: the scores are the model's prior, not a finding.
        "basis": "model judgement before any literature search",
    }

    ctx.artifacts.write_json(STAGE, "problem_tree.json", tree)
    ctx.artifacts.write_text(STAGE, "problem_tree.md", render_tree_markdown(tree))
    ctx.artifacts.write_json(STAGE, "topic_evaluation.json", evaluation)

    if overall < ctx.config.research.min_topic_score:
        raise StageFailure(
            "TOPIC_BELOW_THRESHOLD",
            f"topic scored {overall}/10, below the required {ctx.config.research.min_topic_score}. "
            f"{evaluation['suggestion']}".strip(),
            (*warnings, *eval_warnings),
        )
    return [*warnings, *eval_warnings]
