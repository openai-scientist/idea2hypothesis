"""Ideation memory tests (ported from the baseline memory suite, ideation scope only)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from idea2hypothesis.memory import IdeationMemory, MemoryRetriever, MemoryStore, hashing_embed
from idea2hypothesis.memory.retriever import (
    confidence_update,
    cosine_similarity,
    time_decay_weight,
)
from idea2hypothesis.memory.store import VALID_CATEGORIES, MemoryEntry


def embed(text: str) -> list[float]:
    vec = [0.0] * 16
    for i, ch in enumerate(text[:16]):
        vec[i] = ord(ch) / 256.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "memory_store")


def test_only_ideation_memory_exists() -> None:
    assert VALID_CATEGORIES == ("ideation",)


def test_add_get_count_and_invalid_category(store: MemoryStore) -> None:
    entry_id = store.add("ideation", "content", {"k": "v"})
    assert store.count("ideation") == 1 and store.get(entry_id) is not None
    assert store.get("missing") is None
    with pytest.raises(ValueError, match="Invalid category"):
        store.add("experiment", "x")
    assert store.count() == 1 and len(store.get_all()) == 1


def test_confidence_updates_clamp(store: MemoryStore) -> None:
    entry_id = store.add("ideation", "c", confidence=0.95)
    assert store.update_confidence(entry_id, 0.5) and store.get(entry_id).confidence == 1.0
    assert store.update_confidence(entry_id, -5) and store.get(entry_id).confidence == 0.0
    assert not store.update_confidence("missing", 0.1)


def test_mark_accessed_counts_uses(store: MemoryStore) -> None:
    entry_id = store.add("ideation", "c")
    assert store.mark_accessed(entry_id) and store.mark_accessed(entry_id)
    assert store.get(entry_id).access_count == 2
    assert not store.mark_accessed("missing")


def test_capacity_keeps_the_most_confident(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "m", max_entries_per_category=2)
    store.add("ideation", "low", confidence=0.1)
    store.add("ideation", "high", confidence=0.9)
    store.add("ideation", "mid", confidence=0.5)
    assert sorted(e.content for e in store.get_all("ideation")) == ["high", "mid"]


def test_persistence_roundtrip_and_malformed_lines(tmp_path: Path) -> None:
    directory = tmp_path / "m"
    store = MemoryStore(directory)
    store.add("ideation", "Topic: A", {"run_id": "r1"}, embedding=[0.1, 0.2], confidence=0.8)
    store.save()
    path = directory / "ideation.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "not json\n\n", encoding="utf-8")
    loaded = MemoryStore(directory)
    assert loaded.load() == 1
    entry = loaded.get_all("ideation")[0]
    assert entry.content == "Topic: A" and entry.metadata == {"run_id": "r1"}
    assert entry.embedding == [0.1, 0.2] and entry.confidence == 0.8
    assert MemoryStore(tmp_path / "empty").load() == 0


def test_prune_removes_low_confidence_and_old_entries(store: MemoryStore) -> None:
    store.add("ideation", "keep", confidence=0.9)
    store.add("ideation", "low", confidence=0.1)
    old = store.add("ideation", "old", confidence=0.9)
    store.get(old).created_at = (datetime.now(UTC) - timedelta(days=400)).isoformat()
    assert store.prune() == 2
    assert [e.content for e in store.get_all()] == ["keep"]
    assert store.prune() == 0


def test_entry_dict_roundtrip_and_defaults() -> None:
    entry = MemoryEntry("i", "ideation", "c", {"a": 1}, [0.5], 0.7, "t1", "t2", 3)
    assert MemoryEntry.from_dict(entry.to_dict()) == entry
    blank = MemoryEntry.from_dict({})
    assert blank.confidence == 0.5 and blank.access_count == 0
    assert blank.metadata == {} and blank.embedding == []


def test_time_decay_and_confidence_update() -> None:
    now = datetime.now(UTC)
    assert time_decay_weight(now, now=now) == pytest.approx(1.0)
    assert time_decay_weight(now - timedelta(days=90), now=now) == pytest.approx(0.5, abs=0.01)
    assert time_decay_weight(now - timedelta(days=400), now=now) == 0.0
    assert time_decay_weight(now + timedelta(days=1), now=now) == 1.0
    assert time_decay_weight(datetime.now() - timedelta(days=90), now=now) < 1.0  # naive input
    assert confidence_update(0.5, 0.7) == 1.0 and confidence_update(0.5, -0.9) == 0.0


def test_cosine_similarity_edge_cases() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1], [1, 2]) == 0.0
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_hashing_embedding_is_deterministic_and_normalised() -> None:
    a = hashing_embed("sleep and exam performance")
    assert a == hashing_embed("Sleep and exam performance!")
    assert math.sqrt(sum(v * v for v in a)) == pytest.approx(1.0)
    assert hashing_embed("") == [0.0] * 256
    near = cosine_similarity(a, hashing_embed("sleep exam"))
    far = cosine_similarity(a, hashing_embed("quark gluon"))
    assert near > far


def test_retriever_ranks_and_formats(store: MemoryStore) -> None:
    retriever = MemoryRetriever(store)
    assert retriever.recall(embed("q")) == []
    store.add("ideation", "alpha topic", embedding=embed("alpha topic"))
    store.add("ideation", "zzzz other", embedding=embed("zzzz other"))
    results = retriever.recall(embed("alpha topic"), top_k=1)
    assert [e.content for e, _ in results] == ["alpha topic"]
    assert sum(e.access_count for e in store.get_all()) == 1
    assert retriever.recall_by_text("alpha topic", embed_fn=None) == []
    text = retriever.format_for_prompt(results)
    assert text.startswith("1. [ideation] (relevance:") and retriever.format_for_prompt([]) == ""
    assert retriever.format_for_prompt(results, max_chars=5) == ""


def test_ideation_topic_outcomes_and_anti_patterns(store: MemoryStore) -> None:
    memory = IdeationMemory(store, MemoryRetriever(store), embed_fn=embed)
    assert memory.record_topic_outcome("RL for robotics", "success", 8.0)
    memory.record_topic_outcome(
        "Bad direction", "failure", 1.0, run_id="r1", reason="NO_LITERATURE"
    )
    memory.record_topic_outcome("Good direction", "success", 9.0)
    assert store.count("ideation") == 3
    assert store.get_all()[1].metadata["outcome"] == "failure"
    assert store.get_all()[1].confidence >= 0.5
    assert memory.get_anti_patterns() == ["Topic: Bad direction — NO_LITERATURE"]


def test_ideation_records_hypotheses_and_recalls_similar_topics(store: MemoryStore) -> None:
    memory = IdeationMemory(store, embed_fn=embed)
    memory.record_hypothesis("H1: X beats Y", True, "validated")
    memory.record_hypothesis("H2: Z", False, "untestable")
    assert [e.confidence for e in store.get_all()] == [0.6, 0.7]
    memory.record_topic_outcome("sleep and exams", "success", 7.0)
    recalled = memory.recall_similar_topics("sleep and exams", top_k=1)
    assert recalled.startswith("### Past Research Directions") and "sleep and exams" in recalled


def test_recall_without_data_or_embedding_is_empty(store: MemoryStore) -> None:
    assert IdeationMemory(store).recall_similar_topics("anything") == ""


def test_init_with_store_dir(tmp_path: Path) -> None:
    """Regression kept from the baseline: constructing from a directory must work."""
    directory = tmp_path / "ideation_mem"
    memory = IdeationMemory(store_dir=str(directory))
    assert memory._store is not None and memory._retriever is not None
    memory.record_topic_outcome("Quantum AI", "success", 8.0)
    assert memory._store.count("ideation") == 1
    memory.save()
    assert IdeationMemory(directory).store.count("ideation") == 1  # path as first argument
    assert IdeationMemory(store_dir=directory).store.count("ideation") == 1


def test_init_requires_a_store_or_directory() -> None:
    with pytest.raises(TypeError, match="store_dir"):
        IdeationMemory()
