"""Stat-validated write-through file cache."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from idea2hypothesis.storage.artifacts import write_bytes_atomic
from idea2hypothesis.storage.filecache import FileCache, signature


def _count_opens(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    opened: list[str] = []
    real = Path.read_bytes

    def spy(self: Path) -> bytes:
        opened.append(str(self))
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", spy)
    return opened


def test_second_read_does_not_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(b"one")
    cache = FileCache()
    opened = _count_opens(monkeypatch)
    assert cache.read_bytes(path) == b"one"
    assert cache.read_bytes(path) == b"one"
    assert len(opened) == 1


def test_write_through_then_external_change_is_seen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.txt"
    cache = FileCache()
    path.write_bytes(b"first")
    cache.wrote(path, b"first")
    opened = _count_opens(monkeypatch)
    assert cache.read_bytes(path) == b"first"
    assert opened == []
    path.write_bytes(b"other-content")  # plain open(), as a tamper test or another process does
    assert cache.read_bytes(path) == b"other-content"
    assert len(opened) == 1


def test_same_size_external_change_is_seen(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    cache = FileCache()
    path.write_bytes(b"aaaa")
    cache.wrote(path, b"aaaa")
    path.write_bytes(b"bbbb")
    os.utime(path, ns=(1, 2))
    assert cache.read_bytes(path) == b"bbbb"


def test_append_extends_entry_and_unknown_change_drops_it(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    cache = FileCache()
    path.write_bytes(b"a\n")
    cache.wrote(path, b"a\n")
    before = signature(path)
    with path.open("ab") as handle:
        handle.write(b"b\n")
    cache.appended(path, before, b"b\n")
    assert cache.last_byte(path) == b"\n"
    assert cache.read_bytes(path) == b"a\nb\n"

    with path.open("ab") as handle:
        handle.write(b"x\n")  # someone else appends
    stale = signature(path)
    with path.open("ab") as handle:
        handle.write(b"c\n")
    cache.appended(path, stale, b"c\n")  # entry no longer matches: dropped, not extended
    assert cache.read_bytes(path) == b"a\nb\nx\nc\n"


def test_missing_file_raises_like_open(tmp_path: Path) -> None:
    cache = FileCache()
    with pytest.raises(FileNotFoundError):
        cache.read_bytes(tmp_path / "none")
    with pytest.raises(FileNotFoundError):
        cache.read_text(tmp_path / "none")


def test_forget_and_size_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = FileCache(max_total=10, max_file=8)
    big = tmp_path / "big"
    big.write_bytes(b"x" * 9)
    opened = _count_opens(monkeypatch)
    cache.read_bytes(big)
    cache.read_bytes(big)
    assert len(opened) == 2  # larger than max_file: never cached

    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"1234567")
    b.write_bytes(b"1234567")
    opened.clear()
    cache.read_bytes(a)
    cache.read_bytes(b)  # evicts a (total bound)
    cache.read_bytes(a)
    assert len(opened) == 3
    cache.forget(a)
    cache.read_bytes(a)
    assert len(opened) == 4


def test_atomic_write_populates_global_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from idea2hypothesis.storage import filecache

    path = tmp_path / "x.json"
    write_bytes_atomic(path, b'{"a": 1}\n')
    opened = _count_opens(monkeypatch)
    assert filecache.read_json(path) == {"a": 1}
    assert opened == []
