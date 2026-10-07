"""Similarity retrieval with time decay, plus a dependency-free hashing embedding."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from datetime import UTC, datetime

from idea2hypothesis.memory.store import MemoryEntry, MemoryStore

EmbedFn = Callable[[str], list[float]]
HASH_DIM = 256


def time_decay_weight(
    created_at: datetime,
    half_life_days: float = 90.0,
    max_age_days: float = 365.0,
    *,
    now: datetime | None = None,
) -> float:
    """Exponential decay in [0, 1]; entries older than ``max_age_days`` weigh 0."""
    now = now or datetime.now(UTC)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    age_days = (now - created_at).total_seconds() / 86400.0
    if age_days < 0:
        return 1.0
    if age_days > max_age_days:
        return 0.0
    return math.exp(-age_days * math.log(2) / half_life_days)


def confidence_update(
    current: float, delta: float, floor: float = 0.0, ceiling: float = 1.0
) -> float:
    return max(floor, min(ceiling, current + delta))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def hashing_embed(text: str, dim: int = HASH_DIM) -> list[float]:
    """Deterministic L2-normalised bag-of-words hashing vector (retrieval aid only)."""
    vec = [0.0] * dim
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.md5(token.encode(), usedforsecurity=False).hexdigest()
        vec[int(digest, 16) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 0 else vec


class MemoryRetriever:
    """Score = sim*cosine + decay*decay + conf*confidence + access*uses (weighted)."""

    def __init__(
        self,
        store: MemoryStore,
        half_life_days: float = 90.0,
        sim_weight: float = 0.5,
        decay_weight: float = 0.2,
        conf_weight: float = 0.2,
        access_weight: float = 0.1,
    ) -> None:
        self._store = store
        self._half_life_days = half_life_days
        self._sim_weight = sim_weight
        self._decay_weight = decay_weight
        self._conf_weight = conf_weight
        self._access_weight = access_weight

    def recall(
        self,
        query_embedding: list[float],
        category: str | None = None,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryEntry, float]]:
        entries = self._store.get_all(category)
        if not entries:
            return []
        max_access = max((e.access_count for e in entries), default=1) or 1
        now = datetime.now(UTC)
        scored: list[tuple[MemoryEntry, float]] = []
        for entry in entries:
            sim = cosine_similarity(query_embedding, entry.embedding)
            try:
                created = datetime.fromisoformat(entry.created_at)
            except (ValueError, TypeError):
                created = now
            decay = time_decay_weight(created, half_life_days=self._half_life_days, now=now)
            score = (
                self._sim_weight * sim
                + self._decay_weight * decay
                + self._conf_weight * entry.confidence
                + self._access_weight * (entry.access_count / max_access)
            )
            if score >= min_score:
                scored.append((entry, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        for entry, _ in scored[:top_k]:
            self._store.mark_accessed(entry.id)
        return scored[:top_k]

    def recall_by_text(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 5,
        embed_fn: EmbedFn | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        if embed_fn is None:
            return []
        return self.recall(embed_fn(query), category=category, top_k=top_k)

    def format_for_prompt(
        self, results: list[tuple[MemoryEntry, float]], max_chars: int = 3000
    ) -> str:
        parts: list[str] = []
        total = 0
        for i, (entry, score) in enumerate(results, 1):
            line = f"{i}. [{entry.category}] (relevance: {score:.2f}) {entry.content}"
            if total + len(line) > max_chars:
                break
            parts.append(line)
            total += len(line)
        return "\n".join(parts)
