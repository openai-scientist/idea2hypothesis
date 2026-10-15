"""Filesystem run store: run record, checkpoint, event log and platform index."""

from __future__ import annotations

import copy
import json
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any, Generic, TypeVar

from idea2hypothesis.pipeline.events import Event
from idea2hypothesis.storage.artifacts import ArtifactStore, write_json_atomic

SCHEMA_VERSION = 1
INDEX_DIRNAME = "_index"


class RunExistsError(Exception):
    """A run with the requested id already exists."""


class RunNotFoundError(KeyError):
    """No such run."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def safe_index_name(identifier: str) -> str:
    """Filesystem-safe, collision-free file stem for an external identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", identifier)[:80]
    if cleaned != identifier or not cleaned:
        cleaned = f"{cleaned}-{sha1(identifier.encode('utf-8')).hexdigest()[:10]}"
    return cleaned


_T = TypeVar("_T")


@dataclass
class _Tail(Generic[_T]):
    sig: tuple[int, int, int]
    offset: int = 0  # end of the last complete line consumed
    records: list[_T] = field(default_factory=list)  # parsed from complete lines
    partial: list[_T] = field(default_factory=list)  # parsed from an unterminated final line
    stopped: bool = False  # a bad line ended parsing (``stop_on_error`` only)


class JsonlCache(Generic[_T]):
    """Stat-validated incremental reader for append-only JSONL files.

    A read stats the file. An unchanged file returns the cached records, a file that only grew
    (same inode, larger size) parses just the new bytes, and any other change re-reads it all.
    ``parse`` raises ``ValueError``/``KeyError`` for a bad line: it is skipped, or with
    ``stop_on_error`` it ends parsing. An unterminated final line is parsed for the result but
    not consumed, so a torn write is retried once it is completed. Records are shared with the
    cache: callers must treat them as read-only.
    """

    def __init__(self, parse: Callable[[str], _T], *, stop_on_error: bool = False) -> None:
        self._parse = parse
        self._stop_on_error = stop_on_error
        self._lock = threading.Lock()
        self._entries: dict[Path, _Tail[_T]] = {}

    def drop(self, path: Path) -> None:
        with self._lock:
            self._entries.pop(path, None)

    def read(self, path: Path) -> list[_T]:
        with self._lock:
            try:
                st = os.stat(path)
            except OSError:
                self._entries.pop(path, None)
                return []
            sig = (st.st_size, st.st_mtime_ns, st.st_ino)
            entry = self._entries.get(path)
            if entry is not None and entry.sig == sig:
                return entry.records + entry.partial
            if entry is None or sig[2] != entry.sig[2] or sig[0] <= entry.sig[0]:
                entry = _Tail(sig)
            entry.sig = sig
            try:
                with path.open("rb") as handle:
                    handle.seek(entry.offset)
                    data = handle.read()
            except OSError:
                self._entries.pop(path, None)
                return []
            cut = data.rfind(b"\n") + 1
            if not entry.stopped:
                complete = data[:cut].decode("utf-8")
                entry.records.extend(self._parse_lines(entry, complete.splitlines()))
                entry.partial = []
                if not entry.stopped:
                    try:
                        rest = data[cut:].decode("utf-8")
                    except UnicodeDecodeError:
                        rest = ""
                    entry.partial = self._parse_lines(entry, rest.splitlines(), final=False)
            entry.offset += cut
            self._entries[path] = entry
            return entry.records + entry.partial

    def _parse_lines(self, entry: _Tail[_T], lines: list[str], *, final: bool = True) -> list[_T]:
        # ``final=False`` is an unterminated tail: a bad line there may just be torn, so it
        # must not end parsing for good.
        out: list[_T] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                out.append(self._parse(line))
            except (ValueError, KeyError):
                if self._stop_on_error:
                    entry.stopped = entry.stopped or final
                    break
        return out


class RunStore:
    """All state of a run lives under ``<root>/<run_id>/`` as plain files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._guard = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}
        self._next_seq: dict[str, int] = {}
        self._events = JsonlCache(lambda line: Event.from_dict(json.loads(line)))

    # -- paths and locks --------------------------------------------------

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", run_id) or run_id == INDEX_DIRNAME:
            raise ValueError(f"invalid run id: {run_id!r}")
        return self.root / run_id

    def lock(self, run_id: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(run_id, threading.RLock())

    def artifacts(self, run_id: str) -> ArtifactStore:
        return ArtifactStore(self.run_dir(run_id))

    def exists(self, run_id: str) -> bool:
        try:
            return (self.run_dir(run_id) / "run.json").is_file()
        except ValueError:
            return False

    def list_run_ids(self) -> list[str]:
        return sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and p.name != INDEX_DIRNAME and (p / "run.json").is_file()
        )

    # -- run record -------------------------------------------------------

    def create(
        self,
        run_id: str,
        record: dict[str, Any],
        *,
        config_snapshot: dict[str, Any],
        prompts_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        with self.lock(run_id):
            if self.exists(run_id):
                raise RunExistsError(run_id)
            directory = self.run_dir(run_id)
            directory.mkdir(parents=True, exist_ok=True)
            now = utc_now()
            full = {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "created_at": now,
                "updated_at": now,
                **record,
            }
            write_json_atomic(directory / "config.snapshot.json", config_snapshot)
            write_json_atomic(directory / "prompts.snapshot.json", prompts_snapshot)
            write_json_atomic(directory / "run.json", full)
            return full

    def read_run(self, run_id: str) -> dict[str, Any]:
        try:
            raw = (self.run_dir(run_id) / "run.json").read_text(encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise RunNotFoundError(run_id) from exc
        return json.loads(raw)

    def update_run(self, run_id: str, **fields: Any) -> dict[str, Any]:
        with self.lock(run_id):
            record = self.read_run(run_id)
            record.update(fields)
            record["updated_at"] = utc_now()
            write_json_atomic(self.run_dir(run_id) / "run.json", record)
            return record

    def read_snapshot(self, run_id: str, name: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / f"{name}.snapshot.json"
        return json.loads(path.read_text(encoding="utf-8"))

    # -- checkpoint -------------------------------------------------------

    def read_checkpoint(self, run_id: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / "checkpoint.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data.setdefault("schema_version", SCHEMA_VERSION)
        data.setdefault("run_id", run_id)
        data.setdefault("stages", {})
        return data

    def write_checkpoint(self, run_id: str, checkpoint: dict[str, Any]) -> None:
        with self.lock(run_id):
            checkpoint["updated_at"] = utc_now()
            write_json_atomic(self.run_dir(run_id) / "checkpoint.json", checkpoint)

    # -- events -----------------------------------------------------------

    def _events_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "events.jsonl"

    def _scan_events(self, run_id: str) -> list[Event]:
        return self._events.read(self._events_path(run_id))

    def append_event(
        self,
        run_id: str,
        type_: str,
        *,
        stage: int | None = None,
        attempt: int = 1,
        data: dict[str, Any] | None = None,
    ) -> Event:
        """Append one event with the next sequence number; the write is fsynced."""
        with self.lock(run_id):
            if run_id not in self._next_seq:
                existing = self._scan_events(run_id)
                self._next_seq[run_id] = (existing[-1].seq if existing else 0) + 1
            seq = self._next_seq[run_id]
            event = Event(
                run_id=run_id,
                seq=seq,
                type=type_,
                timestamp=utc_now(),
                attempt=attempt,
                stage=stage,
                data=data or {},
            )
            path = self._events_path(run_id)
            line = json.dumps(event.to_dict(), ensure_ascii=False) + "\n"
            prefix = b""
            if path.exists() and path.stat().st_size > 0:
                with path.open("rb") as probe:
                    probe.seek(-1, os.SEEK_END)
                    if probe.read(1) != b"\n":  # heal a torn final line
                        prefix = b"\n"
            with path.open("ab") as handle:
                if prefix:
                    handle.write(prefix)
                handle.write(line.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            self._next_seq[run_id] = seq + 1
            return event

    def read_events(self, run_id: str, after_seq: int = 0, limit: int | None = None) -> list[Event]:
        events = [e for e in self._scan_events(run_id) if e.seq > after_seq]
        events = events[:limit] if limit is not None else events
        return [replace(e, data=copy.deepcopy(e.data)) for e in events]

    # -- platform index ---------------------------------------------------

    def _platform_path(self, platform_run_id: str) -> Path:
        return self.root / INDEX_DIRNAME / "platform" / f"{safe_index_name(platform_run_id)}.json"

    def _read_platform(self, platform_run_id: str) -> str | None:
        try:
            data = json.loads(self._platform_path(platform_run_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        run_id = data.get("run_id")
        return run_id if isinstance(run_id, str) else None

    def claim_platform_id(self, platform_run_id: str, run_id: str) -> str:
        """Atomically map ``platform_run_id`` to ``run_id``; returns the winning run id.

        The claim is recorded before the run directory exists, so a caller that loses the race
        (or recovers from a crash) receives the id it must create or reuse.
        """
        with self._guard:
            existing = self._read_platform(platform_run_id)
            if existing:
                return existing
            write_json_atomic(
                self._platform_path(platform_run_id),
                {"platform_run_id": platform_run_id, "run_id": run_id, "created_at": utc_now()},
            )
            return run_id

    def find_by_platform(self, platform_run_id: str) -> str | None:
        """Run id previously mapped to ``platform_run_id`` if that run exists."""
        run_id = self._read_platform(platform_run_id)
        return run_id if run_id and self.exists(run_id) else None
