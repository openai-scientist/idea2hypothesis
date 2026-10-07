"""Ideation memory: past research directions, hypotheses and the anti-patterns among them."""

from __future__ import annotations

from pathlib import Path

from idea2hypothesis.memory.retriever import EmbedFn, MemoryRetriever
from idea2hypothesis.memory.store import MemoryStore

CATEGORY = "ideation"


class IdeationMemory:
    """Records and retrieves research direction experiences.

    ``store`` may be a :class:`MemoryStore`, a directory path, or ``None`` when ``store_dir``
    is given (the constructor fix carried over from the baseline).
    """

    def __init__(
        self,
        store: MemoryStore | str | Path | None = None,
        retriever: MemoryRetriever | None = None,
        embed_fn: EmbedFn | None = None,
        *,
        store_dir: str | Path | None = None,
    ) -> None:
        effective_dir = store_dir
        if isinstance(store, (str, Path)):
            effective_dir = store
            store = None
        if store is None:
            if effective_dir is None:
                raise TypeError(
                    "IdeationMemory requires either 'store' (MemoryStore) or 'store_dir'."
                )
            store = MemoryStore(effective_dir)
            if (Path(effective_dir) / f"{CATEGORY}.jsonl").exists():
                store.load()
        self._store = store
        self._retriever = retriever or MemoryRetriever(store)
        self._embed_fn = embed_fn

    @property
    def store(self) -> MemoryStore:
        return self._store

    def save(self) -> None:
        self._store.save()

    def record_topic_outcome(
        self,
        topic: str,
        outcome: str,
        quality_score: float,
        run_id: str = "",
        reason: str = "",
    ) -> str:
        """Record how a topic went (``success``, ``failure`` or ``abandoned``)."""
        content = f"Topic: {topic}\nOutcome: {outcome}\nQuality: {quality_score:.1f}/10"
        metadata = {
            "type": "topic_outcome",
            "outcome": outcome,
            "quality_score": quality_score,
            "run_id": run_id,
            "reason": reason,
        }
        confidence = min(1.0, 0.3 + quality_score / 15.0)
        if outcome == "failure":
            confidence = max(0.5, confidence)  # failures are valuable too
        embedding = self._embed_fn(content) if self._embed_fn else []
        return self._store.add(CATEGORY, content, metadata, embedding, confidence)

    def record_hypothesis(
        self, hypothesis: str, feasible: bool, reason: str, run_id: str = ""
    ) -> str:
        outcome = "feasible" if feasible else "infeasible"
        content = f"Hypothesis: {hypothesis}\nAssessment: {outcome}\nReason: {reason}"
        metadata = {"type": "hypothesis", "feasible": feasible, "run_id": run_id}
        confidence = 0.6 if feasible else 0.7  # infeasible is more informative
        embedding = self._embed_fn(content) if self._embed_fn else []
        return self._store.add(CATEGORY, content, metadata, embedding, confidence)

    def recall_similar_topics(self, query: str, top_k: int = 5) -> str:
        results = self._retriever.recall_by_text(
            query, category=CATEGORY, top_k=top_k, embed_fn=self._embed_fn
        )
        if not results:
            return ""
        parts = ["### Past Research Directions (from memory)"]
        icons = {"success": "+", "failure": "-", "abandoned": "~"}
        for i, (entry, score) in enumerate(results, 1):
            outcome = entry.metadata.get("outcome", "unknown")
            quality = entry.metadata.get("quality_score", "?")
            parts.append(
                f"{i}. [{icons.get(outcome, '?')}] {entry.content.splitlines()[0]} "
                f"(score: {quality}, relevance: {score:.2f})"
            )
        return "\n".join(parts)

    def get_anti_patterns(self) -> list[str]:
        """Topic descriptions that previously failed, with the recorded reason if any."""
        failures: list[str] = []
        for entry in self._store.get_all(CATEGORY):
            if entry.metadata.get("outcome") == "failure":
                message = entry.content.splitlines()[0]
                reason = entry.metadata.get("reason", "")
                failures.append(f"{message} — {reason}" if reason else message)
        return failures
