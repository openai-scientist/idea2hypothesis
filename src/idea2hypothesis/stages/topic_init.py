"""Stage 1 - TOPIC_INIT: researchability guard and structured research goal."""

from __future__ import annotations

import asyncio
from typing import Any

from idea2hypothesis.pipeline.contracts import check_goal
from idea2hypothesis.pipeline.models import StageContext
from idea2hypothesis.resources.hardware import detect_hardware
from idea2hypothesis.stages.base import StageFailure, bullet_list, request_json
from idea2hypothesis.storage.runs import utc_now

STAGE = 1
GOAL_KEYS = (
    "working_title",
    "problem",
    "objective",
    "novel_angle",
    "scope",
    "smart_goal",
    "constraints",
    "success_criteria",
    "benchmark",
)


def goal_for_prompt(goal: dict[str, Any]) -> dict[str, Any]:
    """The part of ``goal.json`` shown to later prompts."""
    return {k: goal[k] for k in GOAL_KEYS if goal.get(k)}


def render_goal_markdown(goal: dict[str, Any]) -> str:
    lines = [
        f"# {goal.get('working_title') or goal['topic']}",
        "",
        f"**Topic:** {goal['topic']}",
        "",
    ]
    if not goal["researchable"]:
        lines += ["## Not researchable", "", str(goal.get("rejection_reason", "")), ""]
        return "\n".join(lines)
    sections = (
        ("Problem", goal.get("problem")),
        ("Objective", goal.get("objective")),
        ("Novel angle", goal.get("novel_angle")),
        ("Scope", goal.get("scope")),
    )
    for title, body in sections:
        lines += [f"## {title}", "", str(body or ""), ""]
    smart = goal.get("smart_goal")
    if isinstance(smart, dict):
        lines += ["## SMART goal", ""]
        lines += [f"- **{k.replace('_', ' ').title()}:** {v}" for k, v in smart.items()]
        lines.append("")
    for title, key in (("Constraints", "constraints"), ("Success criteria", "success_criteria")):
        values = goal.get(key) or []
        if values:
            lines += [f"## {title}", ""] + [f"- {v}" for v in values] + [""]
    bench = goal.get("benchmark")
    if isinstance(bench, dict) and bench.get("name"):
        metrics = ", ".join(bench.get("typical_metrics") or [])
        lines += [
            "## Benchmark",
            "",
            f"- **Name:** {bench['name']}",
            f"- **Source:** {bench.get('source', '')}",
            f"- **Typical metrics:** {metrics}",
            "",
        ]
    return "\n".join(lines)


def memory_context(ctx: StageContext) -> str:
    """Recalled past directions and anti-patterns (empty when memory is disabled)."""
    if ctx.memory is None:
        return ""
    parts = []
    recalled = ctx.memory.recall_similar_topics(ctx.topic)
    if recalled:
        parts.append(recalled)
    anti = ctx.memory.get_anti_patterns()[:5]
    if anti:
        parts.append(
            "Directions that failed before (avoid repeating them):\n- " + "\n- ".join(anti)
        )
    return "\n".join(parts)


async def run(ctx: StageContext) -> list[str]:
    prompt = ctx.prompts.render(
        "topic_init",
        topic=ctx.topic,
        domains=bullet_list(ctx.domains, "general"),
        constraints=bullet_list(ctx.constraints),
        memory_context=memory_context(ctx),
        feedback=ctx.feedback,
    )
    data, warnings = await request_json(ctx, prompt, label="topic_init", validate=check_goal)
    goal: dict[str, Any] = {
        "schema_version": 1,
        "topic": ctx.topic,
        "domains": list(ctx.domains),
        "research_constraints": list(ctx.constraints),
        **data,
        "generated_at": utc_now(),
    }
    ctx.artifacts.write_json(STAGE, "goal.json", goal)
    ctx.artifacts.write_text(STAGE, "goal.md", render_goal_markdown(goal))

    if ctx.config.research.hardware_advisory:
        detect = ctx.hardware or detect_hardware
        profile = await asyncio.to_thread(detect)
        ctx.artifacts.write_json(STAGE, "hardware_profile.json", profile.to_dict())
        if profile.warning:
            warnings = (*warnings, f"hardware advisory: {profile.warning}")

    if not goal["researchable"]:
        raise StageFailure("TOPIC_NOT_RESEARCHABLE", str(goal["rejection_reason"]), warnings)
    return list(warnings)
