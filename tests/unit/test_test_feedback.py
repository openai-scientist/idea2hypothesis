"""Selection must never silently omit a changed owner or a failed CI dependency."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from tools import test_feedback as feedback


@pytest.mark.parametrize(
    ("path", "lanes"),
    [
        ("tests/unit/test_llm.py", ["unit"]),
        ("tests/integration/test_full_run.py", ["pipeline"]),
        ("tests/contracts/test_stage_contracts.py", ["pipeline"]),
        ("tests/unit/test_stage_helpers.py", ["api"]),
        ("src/idea2hypothesis/api/platform_events.py", ["api"]),
        ("src/idea2hypothesis/api/contract/routes.py", ["api"]),
        ("src/idea2hypothesis/pipeline/runner.py", list(feedback.LANES)),
        ("src/idea2hypothesis/llm/parsing.py", list(feedback.LANES)),
        ("src/idea2hypothesis/prompts/stages.yaml", list(feedback.LANES)),
        ("tests/conftest.py", list(feedback.LANES)),
        ("tests/fixtures/api_helpers.py", list(feedback.LANES)),
        ("tests/contract/fake_platform.py", list(feedback.LANES)),
        ("tests/contract/vendor/README.md", list(feedback.LANES)),
        ("pyproject.toml", list(feedback.LANES)),
        (".github/workflows/ci.yml", list(feedback.LANES)),
        ("unknown/new.py", list(feedback.LANES)),
        ("README.md", []),
        ("AGENTS.md", []),
    ],
)
def test_owner_and_shared_boundaries(path, lanes):
    selected = feedback.select_paths([path])
    assert selected["lanes"] == lanes
    if lanes == ["api"]:
        assert "tests/unit/test_stage_helpers.py" in selected["tests"]
        assert "tests/unit/test_api_idle.py" in selected["tests"]
        assert "tests/contract/test_activity_events.py" in selected["tests"]


def test_union_and_docs_do_not_hide_code():
    selection = feedback.select_paths([
        "README.md", "tests/unit/test_llm.py", "tests/integration/test_full_run.py",
    ])
    assert selection["lanes"] == ["unit", "pipeline"]
    assert feedback.select_paths(["README.md", "new.py"])["full"]
    assert feedback.select_paths(["../outside.py"])["full"]


def test_every_current_test_has_an_owner():
    actual = {p.relative_to(feedback.ROOT).as_posix()
              for p in (feedback.ROOT / "tests").rglob("test_*.py")}
    assert actual == set(feedback.TEST_LANES)


def git(root, *args):
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
         "-c", "user.name=Test Fixture", "-c", "user.email=fixture@example.test", *args],
        cwd=root, check=True, capture_output=True,
    ).stdout.decode().strip()


@pytest.fixture
def repo(tmp_path):
    # Commits are disposable fixture data, never history in a user's workspace.
    assert not tmp_path.resolve().is_relative_to(feedback.ROOT)
    git(tmp_path, "init")
    for name in ("README.md", "tests/unit/test_llm.py", "tests/integration/test_full_run.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-m", "fixture base")
    return tmp_path, git(tmp_path, "rev-parse", "HEAD")


def test_real_git_includes_staged_unstaged_untracked_and_ignored_code(repo):
    root, base = repo
    (root / "tests/unit/test_llm.py").write_text("staged\n", encoding="utf-8")
    git(root, "add", "tests/unit/test_llm.py")
    (root / "tests/integration/test_full_run.py").write_text("unstaged\n", encoding="utf-8")
    (root / "uv.lock").write_text("untracked\n", encoding="utf-8")
    (root / ".gitignore").write_text("src/\n__pycache__/\n", encoding="utf-8")
    ignored = root / "src/new.py"
    ignored.parent.mkdir()
    ignored.write_text("ignored code\n", encoding="utf-8")
    cache = root / "tests/__pycache__/cache.py"
    cache.parent.mkdir()
    cache.write_text("generated\n", encoding="utf-8")
    changed = feedback.changed_paths(base, root)
    assert {"tests/unit/test_llm.py", "tests/integration/test_full_run.py", "uv.lock",
            "src/new.py"} <= changed
    assert "tests/__pycache__/cache.py" not in changed
    assert feedback.affected(base, root)["full"]
    assert feedback.affected(base, root, committed_only=True)["tests"] == []


@pytest.mark.parametrize("operation", ["rename", "delete"])
def test_real_git_rename_and_delete_fall_back_to_full(repo, operation):
    root, base = repo
    original = root / "tests/unit/test_llm.py"
    if operation == "rename":
        original.rename(root / "tests/unit/test_renamed.py")
    else:
        original.unlink()
    git(root, "add", "-A")
    changed = feedback.changed_paths(base, root)
    assert "tests/unit/test_llm.py" in changed
    assert feedback.affected(base, root)["full"]


def test_real_git_merge_base_includes_every_push(repo):
    root, base = repo
    (root / "tests/unit/test_llm.py").write_text("first push\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "fixture first push")
    (root / "README.md").write_text("second push\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "fixture second push")
    assert feedback.affected(base, root, committed_only=True)["lanes"] == ["unit"]


def test_real_git_staged_change_is_not_hidden_by_unstaged_restoration(repo):
    root, base = repo
    path = root / "tests/unit/test_llm.py"
    path.write_text("staged change\n", encoding="utf-8")
    git(root, "add", "tests/unit/test_llm.py")
    path.write_text("original\n", encoding="utf-8")
    assert feedback.affected(base, root)["lanes"] == ["unit"]


def test_missing_base_and_diff_failure_select_full(repo, monkeypatch):
    root, _ = repo
    assert feedback.affected(None, root)["full"]
    assert feedback.affected("missing-ref", root)["full"]

    def fail(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(feedback, "git", fail)
    assert feedback.affected("HEAD", root)["full"]


@pytest.mark.parametrize("job", ["changes", "test", "names"])
@pytest.mark.parametrize("result", ["failure", "cancelled"])
def test_gate_rejects_failed_or_cancelled_dependencies(job, result):
    needs = {name: {"result": "success"} for name in ("changes", "test", "names")}
    needs[job]["result"] = result
    assert not feedback.gate_passes(needs)


def test_gate_allows_skipped_tests_but_requires_successful_selection():
    needs = {"changes": {"result": "success"}, "test": {"result": "skipped"},
             "names": {"result": "success"}}
    assert feedback.gate_passes(needs)
    needs["changes"]["result"] = "skipped"
    assert not feedback.gate_passes(needs)
    assert not feedback.gate_passes({})
    assert feedback.main(["--gate", "null"]) == 1


def test_cli_scoped_source_and_ci_share_selection(tmp_path):
    output = tmp_path / "github-output"
    command = [sys.executable, str(feedback.ROOT / "tools/test_feedback.py"),
               "--files", "src/idea2hypothesis/api/platform_events.py", "--dry-run"]
    result = subprocess.run(command + ["--github-output", str(output)],
                            check=True, capture_output=True, text=True)
    selected = json.loads(result.stdout)
    assert selected["lanes"] == ["api"]
    assert "tests/unit/test_stage_helpers.py" in selected["tests"]
    assert output.read_text(encoding="utf-8") == "run=true\nfull=false\n"


def test_cli_exact_test_executes_the_small_boundary(monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(feedback.subprocess, "run", run)
    assert feedback.main(["--files", "tests/unit/test_parsing.py"]) == 7
    assert commands == [[sys.executable, "-m", "pytest", "-q", "tests/unit/test_parsing.py"]]
