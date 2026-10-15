"""Incremental read cache for the append-only event logs."""

from __future__ import annotations

import json
from pathlib import Path

from idea2hypothesis.api.platform_events import PlatformEventLog
from idea2hypothesis.storage.runs import JsonlCache, RunStore


def _env(kind: str = "x") -> dict:
    return {"type": kind, "payload": {"n": [1]}}


def _full_read(store: RunStore, run_id: str):
    return RunStore(store.root).read_events(run_id)


def test_append_parses_only_the_tail(tmp_path: Path) -> None:
    parsed: list[str] = []

    def parse(line: str) -> dict:
        parsed.append(line)
        return json.loads(line)

    path = tmp_path / "log.jsonl"
    cache = JsonlCache(parse)
    path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    assert cache.read(path) == [{"a": 1}, {"a": 2}]
    assert cache.read(path) == [{"a": 1}, {"a": 2}]
    assert len(parsed) == 2  # unchanged file: nothing re-parsed
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"a": 3}\n')
    assert cache.read(path) == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert len(parsed) == 3  # only the new line


def test_run_store_incremental_equals_full_read(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    (tmp_path / "r1").mkdir(exist_ok=True)
    for i in range(5):
        store.append_event("r1", "t", data={"i": i})
        assert store.read_events("r1") == _full_read(store, "r1")
    assert [e.seq for e in store.read_events("r1", after_seq=3)] == [4, 5]
    assert [e.seq for e in store.read_events("r1", limit=2)] == [1, 2]


def test_torn_tail_is_ignored_then_picked_up_when_completed(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    cache = JsonlCache(json.loads)
    path.write_bytes(b'{"a": 1}\n{"a"')
    assert cache.read(path) == [{"a": 1}]
    with path.open("ab") as fh:
        fh.write(b": 2}\n")
    assert cache.read(path) == [{"a": 1}, {"a": 2}]


def test_valid_unterminated_final_line_is_returned_not_duplicated(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    cache = JsonlCache(json.loads)
    path.write_bytes(b'{"a": 1}\n{"a": 2}')
    assert cache.read(path) == [{"a": 1}, {"a": 2}]
    with path.open("ab") as fh:
        fh.write(b'\n{"a": 3}\n')
    assert cache.read(path) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_stop_on_error_ends_parsing_for_good(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    cache = JsonlCache(json.loads, stop_on_error=True)
    path.write_bytes(b'{"a": 1}\nnot json\n{"a": 3}\n')
    assert cache.read(path) == [{"a": 1}]
    with path.open("ab") as fh:
        fh.write(b'{"a": 4}\n')
    assert cache.read(path) == [{"a": 1}]


def test_stop_on_error_torn_tail_is_retried_when_completed(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    cache = JsonlCache(json.loads, stop_on_error=True)
    path.write_bytes(b'{"a": 1}\n{"a"')
    assert cache.read(path) == [{"a": 1}]
    with path.open("ab") as fh:
        fh.write(b': 2}\n{"a": 3}\n')
    assert cache.read(path) == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_run_store_read_events_returns_copies(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    (tmp_path / "r1").mkdir()
    store.append_event("r1", "t", data={"k": [1]})
    store.read_events("r1")[0].data["k"].append(2)
    assert store.read_events("r1")[0].data == {"k": [1]}


def test_skip_mode_skips_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"a": 1}\nnot json\n\n{"a": 3}\n')
    assert JsonlCache(json.loads).read(path) == [{"a": 1}, {"a": 3}]


def test_missing_shrunk_and_replaced_files_are_re_read(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    cache = JsonlCache(json.loads)
    assert cache.read(path) == []
    path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    assert len(cache.read(path)) == 2
    path.write_text('{"b": 9}\n', encoding="utf-8")  # shrink
    assert cache.read(path) == [{"b": 9}]
    path.unlink()
    assert cache.read(path) == []


def test_external_append_by_second_store_is_seen(tmp_path: Path) -> None:
    first = RunStore(tmp_path)
    (tmp_path / "r1").mkdir()
    first.append_event("r1", "t")
    assert len(first.read_events("r1")) == 1
    second = RunStore(tmp_path)
    second.append_event("r1", "t")
    second.append_event("r1", "t")
    assert [e.seq for e in first.read_events("r1")] == [1, 2, 3]


def test_platform_log_heal_resets_the_cache(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    (tmp_path / "r1").mkdir()
    log = PlatformEventLog(store)
    log.append("r1", 1, [_env("a"), _env("b")])
    assert [r["source_seq"] for r in log.read("r1")] == [1, 2]
    # a later batch of two envelopes only half written: the next process heals it away
    path = log.path("r1")
    row = {"source_seq": 3, **_env("c"), "_core_seq": 2, "_batch": [0, 2]}
    with path.open("ab") as fh:
        fh.write((json.dumps(row) + "\n").encode())
    reopened = PlatformEventLog(RunStore(tmp_path))
    assert [r["source_seq"] for r in reopened.read("r1")] == [1, 2]
    assert [json.loads(x)["source_seq"] for x in path.read_text().splitlines()] == [1, 2]
    reopened.append("r1", 2, [_env("c")])
    assert [r["source_seq"] for r in reopened.read("r1")] == [1, 2, 3]


def test_platform_read_returns_copies(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    (tmp_path / "r1").mkdir()
    log = PlatformEventLog(store)
    log.append("r1", 1, [_env("a")])
    log.read("r1")[0]["payload"]["n"].append(2)
    assert log.read("r1")[0]["payload"]["n"] == [1]
