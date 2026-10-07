"""Per-run stage artifacts: atomic writes, manifests and attempt versioning."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
STAGE_COUNT = 8


def stage_dirname(stage: int) -> str:
    return f"stage-{int(stage):02d}"


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write via a temporary file in the same directory, then ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(6):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:  # transient on Windows while a reader holds the file
                if attempt == 5:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def write_json_atomic(path: Path, obj: Any) -> None:
    write_bytes_atomic(path, (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


class ArtifactStore:
    """Reads and writes the ``stage-NN`` directories of one run."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)

    # -- paths ------------------------------------------------------------

    def stage_dir(self, stage: int) -> Path:
        return self.run_dir / stage_dirname(stage)

    def path(self, stage: int, relpath: str) -> Path:
        base = self.stage_dir(stage).resolve()
        target = (base / relpath).resolve()
        if base != target and base not in target.parents:
            raise ValueError(f"artifact path escapes the stage directory: {relpath!r}")
        return target

    def exists(self, stage: int, relpath: str) -> bool:
        return self.path(stage, relpath).exists()

    # -- writes -----------------------------------------------------------

    def write_text(self, stage: int, relpath: str, text: str) -> None:
        write_bytes_atomic(self.path(stage, relpath), text.encode("utf-8"))

    def write_json(self, stage: int, relpath: str, obj: Any) -> None:
        write_json_atomic(self.path(stage, relpath), obj)

    def write_jsonl(self, stage: int, relpath: str, rows: Iterable[dict[str, Any]]) -> None:
        body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        write_bytes_atomic(self.path(stage, relpath), body.encode("utf-8"))

    # -- reads ------------------------------------------------------------

    def read_text(self, stage: int, relpath: str) -> str:
        return self.path(stage, relpath).read_text(encoding="utf-8")

    def read_json(self, stage: int, relpath: str) -> Any:
        return json.loads(self.read_text(stage, relpath))

    def read_jsonl(self, stage: int, relpath: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in self.read_text(stage, relpath).splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def list_files(self, stage: int) -> list[str]:
        base = self.stage_dir(stage)
        if not base.is_dir():
            return []
        return sorted(
            p.relative_to(base).as_posix()
            for p in base.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        )

    # -- manifests --------------------------------------------------------

    def write_manifest(
        self, stage: int, *, run_id: str, attempt: int, files: Sequence[str] | None = None
    ) -> dict[str, Any]:
        names = list(files) if files is not None else self.list_files(stage)
        entries = []
        for name in names:
            data = self.path(stage, name).read_bytes()
            entries.append(
                {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "stage": stage,
            "attempt": attempt,
            "files": entries,
        }
        self.write_json(stage, "manifest.json", manifest)
        return manifest

    def read_manifest(self, stage: int) -> dict[str, Any] | None:
        try:
            manifest = self.read_json(stage, "manifest.json")
        except (OSError, ValueError):
            return None
        return manifest if isinstance(manifest, dict) else None

    def verify_manifest(self, stage: int, *, run_id: str) -> bool:
        """True if the manifest belongs to this run and every listed file is intact."""
        manifest = self.read_manifest(stage)
        if not manifest or manifest.get("run_id") != run_id or manifest.get("stage") != stage:
            return False
        for entry in manifest.get("files", []):
            try:
                data = self.path(stage, entry["path"]).read_bytes()
            except (OSError, KeyError, ValueError):
                return False
            if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
                return False
        return True

    # -- versioning -------------------------------------------------------

    def reset_stage(self, stage: int) -> None:
        shutil.rmtree(self.stage_dir(stage), ignore_errors=True)
        self.stage_dir(stage).mkdir(parents=True, exist_ok=True)

    def invalidate_from(self, stage: int, attempt: int) -> list[int]:
        """Move ``stage-NN`` (NN >= ``stage``) into ``attempts/<attempt>/`` before a rerun."""
        moved: list[int] = []
        archive = self.run_dir / "attempts" / str(attempt)
        for n in range(stage, STAGE_COUNT + 1):
            source = self.stage_dir(n)
            if not source.exists():
                continue
            archive.mkdir(parents=True, exist_ok=True)
            target = archive / stage_dirname(n)
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(source), str(target))
            moved.append(n)
        return moved
