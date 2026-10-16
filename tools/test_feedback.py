"""Offline test selection shared by edit commands, pre-push and CI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANES = ("unit", "pipeline", "api")
API_UNIT = ("test_api_idle.py", "test_stage_helpers.py")
UNIT = (
    "test_card_salvage.py", "test_compact_json.py", "test_config.py",
    "test_dedup_citations_novelty.py", "test_filecache.py", "test_literature_providers.py",
    "test_llm.py",
    "test_llm_factory_key.py", "test_memory.py", "test_parsing.py", "test_prompts.py",
    "test_storage.py", "test_test_feedback.py",
)
PIPELINE = (
    "test_full_run.py", "test_gates_and_resume.py", "test_options.py", "test_progress.py",
    "test_reuse.py", "test_review_phase.py", "test_stops_and_failures.py",
)
API = (
    "test_api_bedrock.py", "test_api_engine_runs.py", "test_api_platform_events.py",
    "test_api_security.py", "test_api_stages.py", "test_api_webhooks.py",
)
PLATFORM = (
    "test_activity_events.py", "test_contract_api.py", "test_contract_gates.py",
    "test_contract_llm.py", "test_contract_models.py", "test_contract_runs.py",
    "test_contract_suite.py", "test_evidence_and_reuse.py", "test_question_gates.py",
)
TEST_LANES = {
    **{f"tests/unit/{name}": "api" for name in API_UNIT},
    **{f"tests/unit/{name}": "unit" for name in UNIT},
    **{f"tests/integration/{name}": "pipeline" for name in PIPELINE},
    "tests/contracts/test_stage_contracts.py": "pipeline",
    **{f"tests/contracts/{name}": "api" for name in API},
    **{f"tests/contract/{name}": "api" for name in PLATFORM},
}
NARRATIVE = {"README.md", "AGENTS.md", "docs/test-audit.md", "docs/test-audit-ledger.csv"}
GENERATED = {
    ".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".cache", "dist", "build", ".worktrees",
}


def full(reason: str) -> dict:
    return {"full": True, "lanes": list(LANES), "tests": ["tests"], "reason": reason}


def lane_selection(lanes: Iterable[str], reason: str) -> dict:
    owners = set(lanes)
    chosen = [lane for lane in LANES if lane in owners]
    return {
        "full": False, "lanes": chosen,
        "tests": [path for path, lane in TEST_LANES.items() if lane in chosen],
        "reason": reason,
    }


def select_paths(paths: Iterable[str], root: Path = ROOT) -> dict:
    lanes = set()
    for raw in paths:
        path = Path(raw.replace("\\", "/"))
        if path.is_absolute():
            try:
                path = path.relative_to(root)
            except ValueError:
                return full("path outside repository")
        name = path.as_posix()
        if ".." in path.parts:
            return full("path outside repository")
        if name in NARRATIVE:
            continue
        if not (root / path).is_file():
            return full("deleted, renamed or missing path")
        if name in TEST_LANES:
            lanes.add(TEST_LANES[name])
        elif name.startswith("src/idea2hypothesis/api/") and path.suffix == ".py":
            lanes.add("api")
        else:
            # Core imports fan out through the runner and API; assets and helpers are shared.
            return full("shared core, runtime input or unmapped path")
    return lane_selection(lanes, "affected owners" if lanes else "narrative only or no changes")


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True,
    ).stdout


def changed_paths(base: str, root: Path = ROOT, *, committed_only: bool = False) -> set[str]:
    merge_base = git(root, "merge-base", base, "HEAD").decode().strip()
    if not merge_base:
        raise ValueError("empty merge base")
    reports = [git(root, "diff", "--name-only", "-z", "--no-renames", merge_base, "HEAD")]
    if not committed_only:
        reports.extend([
            git(root, "diff", "--name-only", "-z", "--no-renames", "--cached"),
            git(root, "diff", "--name-only", "-z", "--no-renames"),
        ])
    paths = {p.decode("utf-8") for report in reports for p in report.split(b"\0") if p}
    if not committed_only:
        untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z")
        # Include ignored code/assets in relevant roots without scanning runs or .env.
        ignored = git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--",
                      "src", "tests", "tools", "configs", ".github")
        for raw in (untracked + ignored).split(b"\0"):
            if not raw:
                continue
            name = raw.decode("utf-8")
            parts = Path(name).parts
            if (parts[0] not in GENERATED and "__pycache__" not in parts
                    and name != "tools/dashboard/data.js"):
                paths.add(name)
    return paths


def affected(base: str | None, root: Path = ROOT, *, committed_only: bool = False) -> dict:
    if not base:
        return full("no diff base supplied")
    try:
        return select_paths(changed_paths(base, root, committed_only=committed_only), root)
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError):
        return full("diff/base unavailable; conservative full fallback")


def gate_passes(needs: dict) -> bool:
    return (
        {"changes", "test", "names"} <= needs.keys()
        and needs["changes"].get("result") == "success"
        and all(job.get("result") in {"success", "skipped"} for job in needs.values())
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--files", nargs="+", help="explicit scoped edit (not a pre-push check)")
    scope.add_argument("--lane", choices=LANES, help="explicit scoped lane")
    scope.add_argument("--full", action="store_true")
    scope.add_argument("--gate", help="CI needs JSON; fail closed on failure or cancellation")
    parser.add_argument("--base", help="target ref/SHA; compare HEAD against its merge base")
    parser.add_argument("--committed-only", action="store_true", help="CI: omit worktree changes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args(argv)
    if args.gate is not None:
        try:
            return 0 if gate_passes(json.loads(args.gate)) else 1
        except (ValueError, TypeError, AttributeError):
            return 1
    if args.full:
        selection = full("forced full suite")
    elif args.lane:
        selection = lane_selection([args.lane], "SCOPED edit lane; affected check still required")
    elif args.files:
        selection = select_paths(args.files)
        # Exact test edits can use their smallest existing behavioral boundary.
        if all(path in TEST_LANES and (ROOT / path).is_file() for path in args.files):
            selection["tests"] = list(dict.fromkeys(args.files))
        selection["reason"] = "SCOPED edit selection; affected check still required"
    else:
        selection = affected(args.base, committed_only=args.committed_only)
    print(json.dumps(selection), flush=True)
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"run={str(bool(selection['tests'])).lower()}\n")
            output.write(f"full={str(selection['full']).lower()}\n")
    if args.dry_run or not selection["tests"]:
        return 0
    command = [sys.executable, "-m", "pytest", "-q", *selection["tests"]]
    if args.fail_fast:
        command.append("-x")
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
