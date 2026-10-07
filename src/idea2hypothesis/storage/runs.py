"""Filesystem run store: run record, checkpoint, event log and platform index."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any

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


class RunStore:
    """All state of a run lives under ``<root>/<run_id>/`` as plain files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._guard = threading.Lock()
        self._locks: dict[str, threading.RLock] = {}
        self._next_seq: dict[str, int] = {}

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
        path = self._events_path(run_id)
        events: list[Event] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return events
        for line in lines:
            if not line.strip():
                continue
            try:
                events.append(Event.from_dict(json.loads(line)))
            except (ValueError, KeyError):
                continue  # a torn final line after a crash is ignored
        return events

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
        return events[:limit] if limit is not None else events

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
