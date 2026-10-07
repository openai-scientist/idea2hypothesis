"""Build ``data.js`` for the developer dashboard from a ``runs/<run-id>`` directory.

Developer tool, not part of the ``idea2hypothesis`` package. It only reads the files the
engine writes (``run.json``, ``checkpoint.json``, ``events.jsonl`` and ``stage-NN/`` artifacts)
and never invents values: a stage that has not produced an artifact is reported as missing.

Usage::

    python tools/dashboard/build_data.py [--runs-dir runs] [--run-id RUN_ID] [--out data.js]

``I2H_RUNS_DIR`` replaces the default runs directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "data.js"
STAGE_NAMES = {
    1: "Topic Initialization",
    2: "Problem Decomposition",
    3: "Search Strategy",
    4: "Literature Collection",
    5: "Literature Screening",
    6: "Knowledge Extraction",
    7: "Synthesis and Gaps",
    8: "Hypothesis Generation",
}


def default_runs_dir() -> Path:
    return Path(os.environ.get("I2H_RUNS_DIR") or "runs").resolve()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def list_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Run summaries, newest first. Directories without ``run.json`` are ignored."""
    runs: list[dict[str, Any]] = []
    if not runs_dir.is_dir():
        return runs
    for entry in runs_dir.iterdir():
        record = _json(entry / "run.json") if entry.is_dir() else None
        if isinstance(record, dict):
            runs.append(
                {
                    "run_id": record.get("run_id", entry.name),
                    "topic": record.get("topic", ""),
                    "status": record.get("status", "unknown"),
                    "created_at": record.get("created_at", ""),
                    "mtime": (entry / "run.json").stat().st_mtime,
                }
            )
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _durations(events: list[dict[str, Any]]) -> dict[int, float]:
    """Seconds between ``stage.started`` and the matching ``stage.completed``/``failed``."""
    started: dict[int, datetime] = {}
    out: dict[int, float] = {}
    for event in events:
        stage = event.get("stage")
        ts = _parse_ts(str(event.get("timestamp", "")))
        if stage is None or ts is None:
            continue
        if event.get("type") == "stage.started":
            started[stage] = ts
        elif event.get("type") in ("stage.completed", "stage.failed") and stage in started:
            out[stage] = round((ts - started[stage]).total_seconds(), 1)
    return out


def _stage_payload(stage: int, sdir: Path) -> dict[str, Any]:
    """Artifacts of one stage, keyed by what the dashboard renders."""
    if stage == 1:
        return {
            "goal": _json(sdir / "goal.json"),
            "goal_md": _text(sdir / "goal.md"),
            "hardware": _json(sdir / "hardware_profile.json"),
        }
    if stage == 2:
        return {
            "problem_tree": _json(sdir / "problem_tree.json"),
            "problem_tree_md": _text(sdir / "problem_tree.md"),
            "topic_evaluation": _json(sdir / "topic_evaluation.json"),
        }
    if stage == 3:
        return {
            "queries": _json(sdir / "queries.json"),
            "sources": _json(sdir / "sources.json"),
            "search_plan_yaml": _text(sdir / "search_plan.yaml"),
        }
    if stage == 4:
        return {"search_meta": _json(sdir / "search_meta.json")}
    if stage == 5:
        review = _json(sdir / "review.json") or {}
        return {
            "shortlist": _jsonl(sdir / "shortlist.jsonl"),
            "screen_meta": _json(sdir / "screen_meta.json"),
            "review_summary": review.get("summary"),
            "decisions": review.get("decisions", []),
        }
    if stage == 6:
        cards = [_json(p) for p in sorted((sdir / "cards").glob("*.json"))]
        return {
            "cards": [c for c in cards if isinstance(c, dict)],
            "knowledge_meta": _json(sdir / "knowledge_meta.json"),
        }
    if stage == 7:
        return {
            "synthesis": _json(sdir / "synthesis.json"),
            "synthesis_md": _text(sdir / "synthesis.md"),
        }
    perspectives: dict[str, Any] = {}
    for path in sorted((sdir / "perspectives").glob("*.json")):
        perspectives[path.stem] = _json(path)
    return {
        "hypotheses": _json(sdir / "hypotheses.json"),
        "hypotheses_md": _text(sdir / "hypotheses.md"),
        "novelty_report": _json(sdir / "novelty_report.json"),
        "perspectives": perspectives,
    }


def _literature(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    years: dict[str, int] = {}
    providers: dict[str, int] = {}
    for row in candidates:
        year = str(row.get("year") or "unknown")
        years[year] = years.get(year, 0) + 1
        for provider in {r.get("provider", "unknown") for r in row.get("source_records", [])}:
            providers[provider] = providers.get(provider, 0) + 1
    rows = [
        {
            "paper_id": r.get("paper_id"),
            "title": r.get("title"),
            "year": r.get("year") or None,
            "venue": r.get("venue") or "",
            "citation_count": r.get("citation_count") or 0,
            "providers": sorted({s.get("provider", "") for s in r.get("source_records", [])}),
            "doi": r.get("doi", ""),
            "arxiv_id": r.get("arxiv_id", ""),
            "url": r.get("url", ""),
            "abstract": r.get("abstract", ""),
        }
        for r in candidates
    ]
    return {
        "total_candidates": len(rows),
        "year_distribution": dict(sorted(years.items())),
        "source_distribution": providers,
        "candidates": rows,
    }


def build_data(runs_dir: Path, run_id: str | None = None) -> dict[str, Any] | None:
    """Return the dashboard data of ``run_id`` (default: the newest run) or ``None``."""
    runs = list_runs(runs_dir)
    if run_id is None:
        if not runs:
            return None
        run_id = runs[0]["run_id"]
    run_dir = runs_dir / run_id
    record = _json(run_dir / "run.json")
    if not isinstance(record, dict):
        return None
    checkpoint = _json(run_dir / "checkpoint.json") or {}
    events = _jsonl(run_dir / "events.jsonl")
    durations = _durations(events)

    stages: dict[str, Any] = {}
    for n in range(1, 9):
        key = f"stage-{n:02d}"
        entry = (checkpoint.get("stages") or {}).get(str(n), {})
        sdir = run_dir / key
        stages[key] = {
            "stage_number": n,
            "stage_key": key,
            "name": STAGE_NAMES[n],
            "status": entry.get("status", "pending"),
            "attempt": entry.get("attempt"),
            "completed_at": entry.get("completed_at"),
            "duration_sec": durations.get(n),
            "usage": entry.get("usage"),
            "error": entry.get("error"),
            "artifacts": entry.get("artifacts", []),
            **(_stage_payload(n, sdir) if sdir.is_dir() else {}),
        }

    candidates = _jsonl(run_dir / "stage-04" / "candidates.jsonl")
    return {
        "run_id": run_id,
        "topic": record.get("topic", ""),
        "domains": record.get("domains", []),
        "review_mode": record.get("review_mode"),
        "status": record.get("status"),
        "attempt": record.get("attempt"),
        "created_at": record.get("created_at"),
        "pause_reason": record.get("pause_reason"),
        "error": record.get("error"),
        "gate": record.get("gate"),
        "usage": record.get("usage"),
        "stages": stages,
        "literature": _literature(candidates),
        "events": events[-200:],
        "runs": [{k: v for k, v in r.items() if k != "mtime"} for r in runs],
    }


def render_js(data: dict[str, Any] | None) -> str:
    body = json.dumps(data, indent=2, ensure_ascii=False).replace("</", "<\\/")
    return f"// Generated by tools/dashboard/build_data.py\nwindow.I2H_DATA = {body};\n"


def write_data_js(runs_dir: Path, run_id: str | None = None, out: Path = DEFAULT_OUT) -> Path:
    out.write_text(render_js(build_data(runs_dir, run_id)), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate data.js for the dashboard.")
    parser.add_argument("--runs-dir", type=Path, default=default_runs_dir())
    parser.add_argument("--run-id", default=None, help="default: newest run")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    data = build_data(args.runs_dir, args.run_id)
    if data is None:
        print(f"no run found in {args.runs_dir}", file=sys.stderr)
    args.out.write_text(render_js(data), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")
    return 0 if data is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
