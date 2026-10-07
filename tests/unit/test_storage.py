from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from idea2hypothesis.storage.artifacts import (
    ArtifactStore,
    write_bytes_atomic,
    write_json_atomic,
)
from idea2hypothesis.storage.runs import RunExistsError, RunNotFoundError, RunStore


def make_run(store: RunStore, run_id: str = "run-1") -> None:
    store.create(
        run_id,
        {"status": "running", "attempt": 1},
        config_snapshot={"llm": {"api_key_env": "X"}},
        prompts_snapshot={"content_sha256": "abc"},
    )


def test_atomic_write_replaces_without_leaving_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "data.json"
    write_json_atomic(target, {"a": 1})
    write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
    assert [p.name for p in target.parent.iterdir()] == ["data.json"]


def test_failed_atomic_write_keeps_the_previous_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "data.txt"
    write_bytes_atomic(target, b"old")

    def boom(*_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        write_bytes_atomic(target, b"new")
    monkeypatch.undo()
    assert target.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["data.txt"]


def test_artifact_paths_cannot_escape_the_stage_directory(tmp_path: Path) -> None:
    art = ArtifactStore(tmp_path / "run")
    with pytest.raises(ValueError, match="escapes"):
        art.path(1, "../../secrets.txt")
    art.write_text(1, "ok/inner.txt", "x")
    assert art.read_text(1, "ok/inner.txt") == "x"


def test_json_and_jsonl_roundtrip_with_unicode(tmp_path: Path) -> None:
    art = ArtifactStore(tmp_path / "run")
    art.write_json(2, "a.json", {"t": "Đề tài"})
    art.write_jsonl(2, "b.jsonl", [{"i": 1}, {"i": 2}])
    assert art.read_json(2, "a.json") == {"t": "Đề tài"}
    assert art.read_jsonl(2, "b.jsonl") == [{"i": 1}, {"i": 2}]
    assert art.list_files(2) == ["a.json", "b.jsonl"]


def test_manifest_detects_tampering_and_foreign_runs(tmp_path: Path) -> None:
    art = ArtifactStore(tmp_path / "run")
    art.write_text(3, "x.txt", "hello")
    manifest = art.write_manifest(3, run_id="run-1", attempt=2)
    assert manifest["schema_version"] == 1 and manifest["attempt"] == 2
    assert [f["path"] for f in manifest["files"]] == ["x.txt"]
    assert art.verify_manifest(3, run_id="run-1")
    assert not art.verify_manifest(3, run_id="other-run")
    art.write_text(3, "x.txt", "changed")
    assert not art.verify_manifest(3, run_id="run-1")


def test_invalidate_from_moves_stages_into_the_attempt_archive(tmp_path: Path) -> None:
    art = ArtifactStore(tmp_path / "run")
    for stage in (2, 3, 4, 5):
        art.write_text(stage, "f.txt", f"stage {stage}")
    moved = art.invalidate_from(3, attempt=1)
    assert moved == [3, 4, 5]
    assert art.exists(2, "f.txt") and not art.stage_dir(3).exists()
    archived = tmp_path / "run" / "attempts" / "1" / "stage-04" / "f.txt"
    assert archived.read_text(encoding="utf-8") == "stage 4"


def test_run_lifecycle_and_snapshots(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    make_run(store)
    assert store.exists("run-1") and store.list_run_ids() == ["run-1"]
    run_dir = store.run_dir("run-1")
    names = {p.name for p in run_dir.iterdir()}
    assert {"run.json", "config.snapshot.json", "prompts.snapshot.json"} <= names
    record = store.update_run("run-1", status="paused", pause_reason="user")
    assert record["status"] == "paused" and store.read_run("run-1")["pause_reason"] == "user"
    assert store.read_snapshot("run-1", "config")["llm"]["api_key_env"] == "X"
    with pytest.raises(RunExistsError):
        make_run(store)
    with pytest.raises(RunNotFoundError):
        store.read_run("nope")
    for bad in ("../x", "a/b", "", "_index"):
        with pytest.raises(ValueError):
            store.run_dir(bad)


def test_checkpoint_roundtrip_has_defaults(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    make_run(store)
    checkpoint = store.read_checkpoint("run-1")
    assert checkpoint["stages"] == {} and checkpoint["run_id"] == "run-1"
    checkpoint["stages"]["1"] = {"status": "completed"}
    store.write_checkpoint("run-1", checkpoint)
    assert store.read_checkpoint("run-1")["stages"]["1"]["status"] == "completed"


def test_event_sequence_is_contiguous_and_survives_restart(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    make_run(store)
    for i in range(3):
        store.append_event("run-1", "stage.started", stage=i, data={"i": i})
    again = RunStore(tmp_path / "runs")  # simulated process restart
    event = again.append_event("run-1", "stage.completed", stage=2)
    assert event.seq == 4
    assert [e.seq for e in again.read_events("run-1")] == [1, 2, 3, 4]
    assert [e.seq for e in again.read_events("run-1", after_seq=2)] == [3, 4]
    assert [e.seq for e in again.read_events("run-1", after_seq=0, limit=2)] == [1, 2]


def test_a_torn_final_event_line_is_ignored_and_healed(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    make_run(store)
    store.append_event("run-1", "run.started")
    path = store.run_dir("run-1") / "events.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"run_id": "run-1", "seq": 2, "ty')  # crash mid-write
    fresh = RunStore(tmp_path / "runs")
    assert [e.seq for e in fresh.read_events("run-1")] == [1]
    event = fresh.append_event("run-1", "run.paused")
    assert event.seq == 2
    assert [e.seq for e in fresh.read_events("run-1")] == [1, 2]


def test_concurrent_appends_get_unique_sequence_numbers(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    make_run(store)

    def worker() -> None:
        for _ in range(10):
            store.append_event("run-1", "stage.started")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [e.seq for e in store.read_events("run-1")] == list(range(1, 41))


def test_platform_index_uses_safe_names(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    make_run(store)
    weird = "plat/../id with spaces"
    assert store.claim_platform_id(weird, "run-1") == "run-1"
    assert store.claim_platform_id(weird, "run-2") == "run-1"
    assert store.find_by_platform(weird) == "run-1"
    assert store.find_by_platform("unknown") is None
    index_files = list((tmp_path / "runs" / "_index" / "platform").iterdir())
    assert len(index_files) == 1 and "/" not in index_files[0].name
    assert store.list_run_ids() == ["run-1"]  # the index directory is not a run
