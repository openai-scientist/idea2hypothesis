"""Stage review routes: run one stage, read and edit its artifacts, health and decisions.

These routes drive the same engine as ``/runs`` (through :class:`RunService`). Reading returns
the human-facing Markdown by default (``?format=json`` returns the JSON artifact); edits
replace the JSON artifact (the Markdown is re-rendered), are validated against the stage
contract and invalidate every later stage.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any

import yaml
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, PlainTextResponse

from idea2hypothesis.api.routes.engine_runs import get_service
from idea2hypothesis.api.schemas import (
    Phase1StartRequest,
    Stage1RunRequest,
    TextContentUpdate,
)
from idea2hypothesis.api.service import STAGE_NAMES, RunService, ServiceError
from idea2hypothesis.pipeline.models import Stage
from idea2hypothesis.stages.hypothesis_gen import render_hypotheses_markdown
from idea2hypothesis.stages.problem_decompose import render_tree_markdown
from idea2hypothesis.stages.synthesis import render_synthesis_markdown
from idea2hypothesis.stages.topic_init import render_goal_markdown
from idea2hypothesis.storage.artifacts import ArtifactStore

router = APIRouter()

PHASE1 = "0. Phase 1: Full Orchestration & Runs Management"
TAGS = {
    1: "Stage 1: Topic Init",
    2: "Stage 2: Problem Decompose",
    3: "Stage 3: Search Strategy",
    4: "Stage 4: Literature Collect",
    5: "Stage 5: Literature Screen (Gate)",
    6: "Stage 6: Knowledge Extract",
    7: "Stage 7: Synthesis",
    8: "Stage 8: Hypothesis Gen",
}
Run = Callable[..., Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _art(service: RunService, run_id: str) -> ArtifactStore:
    return service.store.artifacts(service.stage_run_id(run_id))


def _text(service: RunService, run_id: str, stage: int, name: str) -> str:
    art = _art(service, run_id)
    if not art.exists(stage, name):
        raise ServiceError(404, f"{name} does not exist yet in stage {stage}")
    return art.read_text(stage, name)


def _json(service: RunService, run_id: str, stage: int, name: str) -> Any:
    try:
        return json.loads(_text(service, run_id, stage, name))
    except ValueError as exc:
        raise ServiceError(500, f"{name} is not valid JSON: {exc}") from exc


def _jsonl(service: RunService, run_id: str, stage: int, name: str) -> list[dict[str, Any]]:
    rows = []
    for line in _text(service, run_id, stage, name).splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _dump(doc: Any) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def _document_or_text(
    service: RunService, run_id: str, stage: int, md: str, js: str, fmt: str
) -> Any:
    if fmt == "json":
        return _json(service, run_id, stage, js)
    return PlainTextResponse(_text(service, run_id, stage, md))


async def _put_document(
    service: RunService,
    run_id: str,
    stage: int,
    js: str,
    md: str,
    render: Callable[[dict[str, Any]], str],
    body: TextContentUpdate,
) -> dict[str, Any]:
    """Merge the supplied JSON object over the stored document, re-render, validate, store."""
    current = _json(service, run_id, stage, js)
    try:
        supplied = json.loads(body.content)
    except ValueError as exc:
        raise ServiceError(
            422, f"content must be the JSON document of {js} (or a part of it): {exc}"
        ) from exc
    if not isinstance(supplied, dict):
        raise ServiceError(422, f"content must be a JSON object ({js})")
    document = {**current, **supplied}
    try:
        markdown = render(document)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ServiceError(422, f"{js} is missing or has an invalid field: {exc!r}") from exc
    writes = {js: _dump(document), md: markdown}
    return await service.edit_artifacts(run_id, Stage(stage), writes, note="PUT " + js)


def _plan_rows(plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for strategy in plan.get("search_strategies") or []:
        for text in strategy.get("queries") or []:
            text = str(text).strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                rows.append(
                    {
                        "id": f"q{len(rows) + 1}",
                        "text": text,
                        "strategy": strategy.get("name"),
                        "sub_question_ids": list(strategy.get("sub_question_ids") or []),
                    }
                )
    return rows


def _year_min(plan: dict[str, Any], previous: Any) -> int:
    filters = plan.get("filters") if isinstance(plan.get("filters"), dict) else {}
    value = filters.get("min_year")
    return value if isinstance(value, int) and value > 0 else int(previous or 0)


# ---------------------------------------------------------------------------
# 0. Phase 1: full orchestration and run management
# ---------------------------------------------------------------------------


@router.post("/api/phase1/start", tags=[PHASE1], summary="Start all 8 stages in the background")
async def start_phase1(
    req: Phase1StartRequest, service: RunService = Depends(get_service)
) -> dict[str, Any]:
    """Starts a full run. ``auto_approve=true`` uses review mode ``auto``; ``false`` uses
    ``copilot`` (the run stops at the screening gate). ``quality_threshold`` is accepted for
    compatibility and has no effect."""
    run_id = await service.start_phase1(
        req.topic,
        req.domains,
        auto_approve=req.auto_approve,
        provider=req.llm_provider,
        model=req.model,
    )
    return {
        "message": "Phase 1 started",
        "run_id": run_id,
        "status": "running",
        "output_dir": str(service.store.run_dir(run_id).resolve()),
    }


@router.get("/api/phase1/status", tags=[PHASE1], summary="Progress of the most recent run")
async def get_phase1_status(service: RunService = Depends(get_service)) -> dict[str, Any]:
    return service.overview(service.latest_run())


@router.post("/api/phase1/stop", tags=[PHASE1], summary="Cancel the active run")
async def stop_phase1(service: RunService = Depends(get_service)) -> dict[str, str]:
    active = [r for r in service.store.list_run_ids() if service.control.is_active(r)]
    if not active:
        raise ServiceError(404, "No pipeline is running")
    await service.cancel(active[-1])
    return {"status": "stopped"}


@router.get("/api/phase1/runs", tags=[PHASE1], summary="List runs")
async def list_runs(service: RunService = Depends(get_service)) -> list[dict[str, Any]]:
    return service.list_runs()


@router.get("/api/phase1/runs/{run_id}/summary", tags=[PHASE1], summary="Phase 1 result bundle")
async def get_run_summary(
    run_id: str, service: RunService = Depends(get_service)
) -> dict[str, Any]:
    art = _art(service, run_id)

    def text(stage: int, name: str) -> str:
        return art.read_text(stage, name) if art.exists(stage, name) else ""

    def doc(stage: int, name: str) -> Any:
        return art.read_json(stage, name) if art.exists(stage, name) else None

    queries = doc(3, "queries.json")
    candidates = art.read_jsonl(4, "candidates.jsonl") if art.exists(4, "candidates.jsonl") else []
    return {
        "run_id": run_id,
        "stage_1_goal": text(1, "goal.md"),
        "stage_2_problem_tree": text(2, "problem_tree.md"),
        "stage_2_evaluation": doc(2, "topic_evaluation.json"),
        "stage_3_search_plan": text(3, "search_plan.yaml"),
        "stage_3_queries": [q["text"] for q in queries["queries"]] if queries else [],
        "stage_4_references_count": len(candidates),
        "stage_5_shortlist_count": len(art.read_jsonl(5, "shortlist.jsonl"))
        if art.exists(5, "shortlist.jsonl")
        else 0,
        "stage_6_knowledge_cards_count": len(
            [n for n in art.list_files(6) if n.startswith("cards/") and n.endswith(".json")]
        ),
        "stage_7_synthesis": text(7, "synthesis.md"),
        "stage_8_hypotheses": text(8, "hypotheses.md"),
        "stage_8_novelty_report": doc(8, "novelty_report.json"),
    }


@router.get("/api/phase1/runs/{run_id}/checkpoint", tags=[PHASE1], summary="Run checkpoint")
async def get_checkpoint(run_id: str, service: RunService = Depends(get_service)) -> dict[str, Any]:
    return service.store.read_checkpoint(service.stage_run_id(run_id))


@router.get(
    "/api/phase1/runs/{run_id}/health-overview", tags=[PHASE1], summary="Health of stages 1-8"
)
async def get_health_overview(
    run_id: str, service: RunService = Depends(get_service)
) -> list[dict[str, Any]]:
    return service.health_overview(service.stage_run_id(run_id))


@router.delete("/api/phase1/runs/{run_id}", tags=[PHASE1], summary="Delete a run")
async def delete_run(run_id: str, service: RunService = Depends(get_service)) -> dict[str, str]:
    await service.delete_run(service.stage_run_id(run_id))
    return {"status": "deleted", "run_id": run_id}


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


@router.post("/api/stage1/run", tags=[TAGS[1]], summary="Run stage 1 (creates the run)")
async def run_stage_1(
    req: Stage1RunRequest, service: RunService = Depends(get_service)
) -> dict[str, Any]:
    run_id, services = await service.create_stage_run(
        req.topic, req.run_id, req.llm_provider, req.model
    )
    return await service.run_stage(run_id, Stage.TOPIC_INIT, services=services)


@router.get("/api/stage1/{run_id}/goal", tags=[TAGS[1]], summary="Research goal (Markdown)")
async def get_stage1_goal(
    run_id: str,
    format: str = Query("md", pattern="^(md|json)$"),
    service: RunService = Depends(get_service),
) -> Any:
    return _document_or_text(service, run_id, 1, "goal.md", "goal.json", format)


@router.put("/api/stage1/{run_id}/goal", tags=[TAGS[1]], summary="[HITL] Edit the goal (JSON)")
async def update_stage1_goal(
    run_id: str, body: TextContentUpdate, service: RunService = Depends(get_service)
) -> dict[str, Any]:
    service.stage_run_id(run_id)
    return await _put_document(
        service, run_id, 1, "goal.json", "goal.md", render_goal_markdown, body
    )


@router.get("/api/stage1/{run_id}/hardware", tags=[TAGS[1]], summary="Detected hardware")
async def get_stage1_hardware(run_id: str, service: RunService = Depends(get_service)) -> Any:
    return _json(service, run_id, 1, "hardware_profile.json")


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------


@router.get("/api/stage2/{run_id}/problem-tree", tags=[TAGS[2]], summary="Problem tree (Markdown)")
async def get_stage2_problem_tree(
    run_id: str,
    format: str = Query("md", pattern="^(md|json)$"),
    service: RunService = Depends(get_service),
) -> Any:
    return _document_or_text(service, run_id, 2, "problem_tree.md", "problem_tree.json", format)


@router.put(
    "/api/stage2/{run_id}/problem-tree",
    tags=[TAGS[2]],
    summary="[HITL] Edit the problem tree (JSON)",
)
async def update_stage2_problem_tree(
    run_id: str, body: TextContentUpdate, service: RunService = Depends(get_service)
) -> dict[str, Any]:
    service.stage_run_id(run_id)
    return await _put_document(
        service, run_id, 2, "problem_tree.json", "problem_tree.md", render_tree_markdown, body
    )


@router.get("/api/stage2/{run_id}/evaluation", tags=[TAGS[2]], summary="Topic evaluation")
async def get_stage2_evaluation(run_id: str, service: RunService = Depends(get_service)) -> Any:
    return _json(service, run_id, 2, "topic_evaluation.json")


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------


@router.get("/api/stage3/{run_id}/plan", tags=[TAGS[3]], summary="Search plan (YAML)")
async def get_stage3_search_plan(run_id: str, service: RunService = Depends(get_service)) -> Any:
    return PlainTextResponse(_text(service, run_id, 3, "search_plan.yaml"))


@router.put(
    "/api/stage3/{run_id}/plan", tags=[TAGS[3]], summary="[HITL] Edit the search plan (YAML)"
)
async def update_stage3_search_plan(
    run_id: str, body: TextContentUpdate, service: RunService = Depends(get_service)
) -> dict[str, Any]:
    service.stage_run_id(run_id)
    try:
        plan = yaml.safe_load(body.content)
    except yaml.YAMLError as exc:
        raise ServiceError(422, f"content is not valid YAML: {exc}") from exc
    if not isinstance(plan, dict):
        raise ServiceError(422, "content must be a YAML mapping with search_strategies")
    previous = _json(service, run_id, 3, "queries.json")
    queries = {
        "schema_version": 1,
        "year_min": _year_min(plan, previous.get("year_min")),
        "queries": _plan_rows(plan),
    }
    writes = {"search_plan.yaml": body.content, "queries.json": _dump(queries)}
    return await service.edit_artifacts(run_id, Stage.SEARCH_STRATEGY, writes, note="PUT plan")


@router.get("/api/stage3/{run_id}/queries", tags=[TAGS[3]], summary="Search queries")
async def get_stage3_queries(
    run_id: str,
    format: str = Query("list", pattern="^(list|json)$"),
    service: RunService = Depends(get_service),
) -> Any:
    doc = _json(service, run_id, 3, "queries.json")
    return doc if format == "json" else [q["text"] for q in doc["queries"]]


@router.put("/api/stage3/{run_id}/queries", tags=[TAGS[3]], summary="[HITL] Replace the queries")
async def update_stage3_queries(
    run_id: str, queries: list[str], service: RunService = Depends(get_service)
) -> dict[str, Any]:
    """New queries get strategy ``manual`` and are linked to every sub-question."""
    service.stage_run_id(run_id)
    doc = _json(service, run_id, 3, "queries.json")
    plan = yaml.safe_load(_text(service, run_id, 3, "search_plan.yaml"))
    tree = _json(service, run_id, 2, "problem_tree.json")
    all_sq = [q["id"] for q in tree["sub_questions"]]
    known = {r["text"].lower(): r for r in doc["queries"]}
    rows: list[dict[str, Any]] = []
    for text in (q.strip() for q in queries):
        if not text or text.lower() in {r["text"].lower() for r in rows}:
            continue
        old = known.get(text.lower())
        rows.append(
            {
                "id": f"q{len(rows) + 1}",
                "text": text,
                "strategy": old["strategy"] if old else "manual",
                "sub_question_ids": list(old["sub_question_ids"]) if old else list(all_sq),
            }
        )
    old_strategies = {s["name"]: s for s in plan.get("search_strategies", [])}
    strategies: list[dict[str, Any]] = []
    for row in rows:
        entry = next((s for s in strategies if s["name"] == row["strategy"]), None)
        if entry is None:
            base = old_strategies.get(row["strategy"], {})
            entry = {
                "name": row["strategy"],
                "rationale": base.get("rationale", "queries added by a reviewer"),
                "sub_question_ids": list(base.get("sub_question_ids") or row["sub_question_ids"]),
                "queries": [],
            }
            strategies.append(entry)
        entry["queries"].append(row["text"])
    new_plan = {**plan, "search_strategies": strategies}
    writes = {
        "queries.json": _dump({**doc, "queries": rows}),
        "search_plan.yaml": yaml.safe_dump(new_plan, sort_keys=False, allow_unicode=True),
    }
    return await service.edit_artifacts(run_id, Stage.SEARCH_STRATEGY, writes, note="PUT queries")


@router.get("/api/stage3/{run_id}/sources", tags=[TAGS[3]], summary="Enabled sources")
async def get_stage3_sources(run_id: str, service: RunService = Depends(get_service)) -> Any:
    return _json(service, run_id, 3, "sources.json")


# ---------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------


@router.get("/api/stage4/{run_id}/candidates", tags=[TAGS[4]], summary="Collected candidates")
async def get_stage4_candidates(
    run_id: str, limit: int = Query(50, ge=1, le=500), service: RunService = Depends(get_service)
) -> list[dict[str, Any]]:
    return _jsonl(service, run_id, 4, "candidates.jsonl")[:limit]


@router.get(
    "/api/stage4/{run_id}/download-bibtex", tags=[TAGS[4]], summary="Download references.bib"
)
async def download_stage4_bibtex(
    run_id: str, service: RunService = Depends(get_service)
) -> FileResponse:
    art = _art(service, run_id)
    if not art.exists(4, "references.bib"):
        raise ServiceError(404, "references.bib does not exist yet")
    return FileResponse(
        path=art.path(4, "references.bib"),
        filename=f"{run_id}_references.bib",
        media_type="text/plain",
    )


@router.get(
    "/api/stage4/{run_id}/references-text", tags=[TAGS[4]], summary="references.bib as text"
)
async def get_stage4_references_text(
    run_id: str, service: RunService = Depends(get_service)
) -> Any:
    return PlainTextResponse(_text(service, run_id, 4, "references.bib"))


@router.get("/api/stage4/{run_id}/stats", tags=[TAGS[4]], summary="Collection statistics")
async def get_stage4_stats(run_id: str, service: RunService = Depends(get_service)) -> Any:
    return _json(service, run_id, 4, "search_meta.json")


# ---------------------------------------------------------------------------
# Stage 5
# ---------------------------------------------------------------------------


@router.get("/api/stage5/{run_id}/shortlist", tags=[TAGS[5]], summary="Shortlist")
async def get_stage5_shortlist(
    run_id: str, service: RunService = Depends(get_service)
) -> list[dict[str, Any]]:
    return _jsonl(service, run_id, 5, "shortlist.jsonl")


@router.put(
    "/api/stage5/{run_id}/shortlist", tags=[TAGS[5]], summary="[HITL] Replace the shortlist"
)
async def update_stage5_shortlist(
    run_id: str, papers: list[dict[str, Any]], service: RunService = Depends(get_service)
) -> dict[str, Any]:
    """Rows are full shortlist rows (as returned by GET). Removed papers become
    ``dropped_by_reviewer``; a paper can only be added with its own scores and reason."""
    service.stage_run_id(run_id)
    candidates = {c["paper_id"]: c for c in _jsonl(service, run_id, 4, "candidates.jsonl")}
    review = _json(service, run_id, 5, "review.json")
    meta = _json(service, run_id, 5, "screen_meta.json")
    previous = {r["paper_id"]: r for r in _jsonl(service, run_id, 5, "shortlist.jsonl")}
    rows = []
    for paper in papers:
        pid = paper.get("paper_id")
        if pid not in candidates:
            raise ServiceError(422, f"paper {pid!r} is not a collected candidate")
        rows.append({**candidates[pid], **previous.get(pid, {}), **paper})
    kept = {r["paper_id"] for r in rows}
    for decision in review["decisions"]:
        pid = decision["paper_id"]
        row = next((r for r in rows if r["paper_id"] == pid), None)
        if row is not None:
            decision.update(
                decision="kept",
                relevance_score=row.get("relevance_score"),
                quality_score=row.get("quality_score"),
                reason=row.get("keep_reason") or decision["reason"],
            )
        elif decision["decision"] == "kept":
            decision.update(
                decision="dropped_by_reviewer",
                reason=f"{decision['reason']} | removed by reviewer edit",
            )
    summary = review["summary"]
    summary["kept"] = len(kept)
    summary["rejected"] = sum(1 for d in review["decisions"] if d["decision"] == "rejected")
    review["human_review"] = {"decision": "edited", "kept": sorted(kept)}
    meta["kept"] = len(kept)
    writes = {
        "shortlist.jsonl": "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        "review.json": _dump(review),
        "screen_meta.json": _dump(meta),
    }
    return await service.edit_artifacts(
        run_id, Stage.LITERATURE_SCREEN, writes, note="PUT shortlist"
    )


@router.post(
    "/api/stage5/{run_id}/approve", tags=[TAGS[5]], summary="[HITL GATE] Approve the screening gate"
)
async def approve_stage5_gate(
    run_id: str,
    reason: str = Query("Approved by researcher"),
    service: RunService = Depends(get_service),
) -> dict[str, str]:
    await service.approve_open_gate(service.stage_run_id(run_id), "screen", reason)
    return {"status": "gate_approved", "run_id": run_id}


# ---------------------------------------------------------------------------
# Stage 6
# ---------------------------------------------------------------------------


def _card_names(service: RunService, run_id: str) -> list[str]:
    art = _art(service, run_id)
    return sorted(
        PurePosixPath(n).name
        for n in art.list_files(6)
        if n.startswith("cards/") and n.endswith(".json")
    )


@router.get("/api/stage6/{run_id}/cards", tags=[TAGS[6]], summary="Card file names")
async def list_stage6_cards(run_id: str, service: RunService = Depends(get_service)) -> list[str]:
    return _card_names(service, run_id)


@router.get("/api/stage6/{run_id}/cards-merged", tags=[TAGS[6]], summary="All cards")
async def get_stage6_cards_merged(
    run_id: str, service: RunService = Depends(get_service)
) -> list[dict[str, Any]]:
    return [_json(service, run_id, 6, f"cards/{name}") for name in _card_names(service, run_id)]


@router.get("/api/stage6/{run_id}/cards/{card_name}", tags=[TAGS[6]], summary="One card")
async def get_stage6_card(
    run_id: str, card_name: str, service: RunService = Depends(get_service)
) -> Any:
    if card_name not in _card_names(service, run_id):
        raise ServiceError(404, f"card {card_name} not found")
    return _json(service, run_id, 6, f"cards/{card_name}")


# ---------------------------------------------------------------------------
# Stage 7
# ---------------------------------------------------------------------------


@router.get("/api/stage7/{run_id}/synthesis", tags=[TAGS[7]], summary="Synthesis (Markdown)")
async def get_stage7_synthesis(
    run_id: str,
    format: str = Query("md", pattern="^(md|json)$"),
    service: RunService = Depends(get_service),
) -> Any:
    return _document_or_text(service, run_id, 7, "synthesis.md", "synthesis.json", format)


@router.put(
    "/api/stage7/{run_id}/synthesis", tags=[TAGS[7]], summary="[HITL] Edit the synthesis (JSON)"
)
async def update_stage7_synthesis(
    run_id: str, body: TextContentUpdate, service: RunService = Depends(get_service)
) -> dict[str, Any]:
    service.stage_run_id(run_id)
    return await _put_document(
        service, run_id, 7, "synthesis.json", "synthesis.md", render_synthesis_markdown, body
    )


# ---------------------------------------------------------------------------
# Stage 8
# ---------------------------------------------------------------------------


@router.get("/api/stage8/{run_id}/hypotheses", tags=[TAGS[8]], summary="Hypotheses (Markdown)")
async def get_stage8_hypotheses(
    run_id: str,
    format: str = Query("md", pattern="^(md|json)$"),
    service: RunService = Depends(get_service),
) -> Any:
    return _document_or_text(service, run_id, 8, "hypotheses.md", "hypotheses.json", format)


@router.put(
    "/api/stage8/{run_id}/hypotheses", tags=[TAGS[8]], summary="[HITL] Edit the hypotheses (JSON)"
)
async def update_stage8_hypotheses(
    run_id: str, body: TextContentUpdate, service: RunService = Depends(get_service)
) -> dict[str, Any]:
    service.stage_run_id(run_id)
    return await _put_document(
        service, run_id, 8, "hypotheses.json", "hypotheses.md", render_hypotheses_markdown, body
    )


@router.get(
    "/api/stage8/{run_id}/novelty", tags=[TAGS[8]], summary="Novelty assessment (heuristic)"
)
async def get_stage8_novelty(run_id: str, service: RunService = Depends(get_service)) -> Any:
    return _json(service, run_id, 8, "novelty_report.json")


def _perspective_files(service: RunService, run_id: str) -> list[str]:
    art = _art(service, run_id)
    return sorted(
        n.removeprefix("perspectives/") for n in art.list_files(8) if n.startswith("perspectives/")
    )


@router.get("/api/stage8/{run_id}/perspectives", tags=[TAGS[8]], summary="Perspective files")
async def list_stage8_perspectives(
    run_id: str, service: RunService = Depends(get_service)
) -> list[str]:
    return _perspective_files(service, run_id)


@router.get(
    "/api/stage8/{run_id}/perspectives/{filename}", tags=[TAGS[8]], summary="One perspective file"
)
async def get_stage8_perspective(
    run_id: str, filename: str, service: RunService = Depends(get_service)
) -> Any:
    if filename not in _perspective_files(service, run_id):
        raise ServiceError(404, f"perspective file {filename} not found")
    return PlainTextResponse(_text(service, run_id, 8, f"perspectives/{filename}"))


# ---------------------------------------------------------------------------
# Uniform per-stage routes: run (stages 2-8), decision and health (stages 1-8)
# ---------------------------------------------------------------------------


def _run_route(stage: int) -> Run:
    if stage == 5:

        async def run_stage_5(
            run_id: str,
            auto_approve: bool = Query(True),
            service: RunService = Depends(get_service),
        ) -> dict[str, Any]:
            """Gates follow the run's review mode; ``auto_approve=false`` switches an
            ``auto``/``light`` run to ``copilot`` so the screening gate opens on resume."""
            return await service.run_stage(
                service.stage_run_id(run_id), Stage(5), auto_approve=auto_approve
            )

        return run_stage_5

    async def run_stage(run_id: str, service: RunService = Depends(get_service)) -> dict[str, Any]:
        return await service.run_stage(service.stage_run_id(run_id), Stage(stage))

    return run_stage


def _decision_route(stage: int) -> Run:
    async def decision(run_id: str, service: RunService = Depends(get_service)) -> dict[str, Any]:
        return service.decision(service.stage_run_id(run_id), stage)

    return decision


def _health_route(stage: int) -> Run:
    async def health(run_id: str, service: RunService = Depends(get_service)) -> dict[str, Any]:
        return service.health(service.stage_run_id(run_id), stage)

    return health


for _n in range(1, 9):
    _name = STAGE_NAMES[_n].replace("_", " ").title()
    if _n > 1:
        router.add_api_route(
            f"/api/stage{_n}/{{run_id}}/run",
            _run_route(_n),
            methods=["POST"],
            tags=[TAGS[_n]],
            summary=f"Run only stage {_n} ({_name})",
            name=f"run_stage_{_n}",
        )
    router.add_api_route(
        f"/api/stage{_n}/{{run_id}}/decision",
        _decision_route(_n),
        methods=["GET"],
        tags=[TAGS[_n]],
        summary=f"Outcome of stage {_n}",
        name=f"get_stage{_n}_decision",
    )
    router.add_api_route(
        f"/api/stage{_n}/{{run_id}}/health",
        _health_route(_n),
        methods=["GET"],
        tags=[TAGS[_n]],
        summary=f"Timing and artifacts of stage {_n}",
        name=f"get_stage{_n}_health",
    )
