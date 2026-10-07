"""JSONL-backed ideation memory storage."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VALID_CATEGORIES = ("ideation",)


@dataclass
class MemoryEntry:
    id: str
    category: str
    content: str
    metadata: dict[str, Any]
    embedding: list[float]
    confidence: float
    created_at: str
    last_accessed: str
    access_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        return cls(
            id=str(data.get("id", "")),
            category=str(data.get("category", "")),
            content=str(data.get("content", "")),
            metadata=data.get("metadata") or {},
            embedding=data.get("embedding") or [],
            confidence=float(data.get("confidence", 0.5)),
            created_at=str(data.get("created_at", "")),
            last_accessed=str(data.get("last_accessed", "")),
            access_count=int(data.get("access_count", 0)),
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class MemoryStore:
    """Entries grouped by category, persisted as ``<category>.jsonl`` files."""

    def __init__(
        self,
        store_dir: str | Path,
        max_entries_per_category: int = 500,
        confidence_threshold: float = 0.3,
    ) -> None:
        self._store_dir = Path(store_dir)
        self._max_per_category = max_entries_per_category
        self._confidence_threshold = confidence_threshold
        self._entries: dict[str, list[MemoryEntry]] = {cat: [] for cat in VALID_CATEGORIES}
        self._dirty = False

    @property
    def store_dir(self) -> Path:
        return self._store_dir

    def add(
        self,
        category: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        confidence: float = 0.5,
    ) -> str:
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Invalid category '{category}'. Must be one of {VALID_CATEGORIES}")
        now = _now()
        entry = MemoryEntry(
            id=uuid.uuid4().hex[:12],
            category=category,
            content=content,
            metadata=metadata or {},
            embedding=embedding or [],
            confidence=confidence,
            created_at=now,
            last_accessed=now,
            access_count=0,
        )
        self._entries[category].append(entry)
        self._dirty = True
        entries = self._entries[category]
        if len(entries) > self._max_per_category:
            entries.sort(key=lambda e: e.confidence, reverse=True)
            self._entries[category] = entries[: self._max_per_category]
        return entry.id

    def get(self, entry_id: str) -> MemoryEntry | None:
        for entries in self._entries.values():
            for entry in entries:
                if entry.id == entry_id:
                    return entry
        return None

    def get_all(self, category: str | None = None) -> list[MemoryEntry]:
        if category:
            return list(self._entries.get(category, []))
        return [e for entries in self._entries.values() for e in entries]

    def update_confidence(self, entry_id: str, delta: float) -> bool:
        entry = self.get(entry_id)
        if entry is None:
            return False
        entry.confidence = max(0.0, min(1.0, entry.confidence + delta))
        self._dirty = True
        return True

    def mark_accessed(self, entry_id: str) -> bool:
        entry = self.get(entry_id)
        if entry is None:
            return False
        entry.last_accessed = _now()
        entry.access_count += 1
        self._dirty = True
        return True

    def prune(self, confidence_threshold: float | None = None, max_age_days: float = 365.0) -> int:
        threshold = (
            confidence_threshold if confidence_threshold is not None else self._confidence_threshold
        )
        now = datetime.now(UTC)
        removed = 0
        for category in VALID_CATEGORIES:
            kept: list[MemoryEntry] = []
            for entry in self._entries[category]:
                try:
                    created = datetime.fromisoformat(entry.created_at)
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=UTC)
                    age_days = (now - created).total_seconds() / 86400.0
                except (ValueError, TypeError):
                    age_days = 0.0
                if entry.confidence >= threshold and age_days <= max_age_days:
                    kept.append(entry)
            removed += len(self._entries[category]) - len(kept)
            self._entries[category] = kept
        if removed:
            self._dirty = True
        return removed

    def save(self) -> None:
        self._store_dir.mkdir(parents=True, exist_ok=True)
        for category in VALID_CATEGORIES:
            path = self._store_dir / f"{category}.jsonl"
            tmp = path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as handle:
                for entry in self._entries[category]:
                    handle.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            tmp.replace(path)
        self._dirty = False

    def load(self) -> int:
        total = 0
        for category in VALID_CATEGORIES:
            path = self._store_dir / f"{category}.jsonl"
            if not path.exists():
                continue
            entries: list[MemoryEntry] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entries.append(MemoryEntry.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning("Skipping malformed memory entry: %s", exc)
            self._entries[category] = entries
            total += len(entries)
        return total

    def count(self, category: str | None = None) -> int:
        if category:
            return len(self._entries.get(category, []))
        return sum(len(v) for v in self._entries.values())
