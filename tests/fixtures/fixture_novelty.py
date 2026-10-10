"""FixtureNoveltyJudge: a scripted novelty judge. It reads nothing; it returns set verdicts."""

from __future__ import annotations

from typing import Any


class FixtureNoveltyJudge:
    """Gives each hypothesis the verdict set for it ("new" by default).

    "tested" and "related" name the first paper listed with the hypothesis. ``fail`` makes it
    return None, as the stage does when no valid judgement comes back.
    """

    def __init__(self, verdicts: dict[str, str] | None = None, *, fail: bool = False) -> None:
        self.verdicts = verdicts or {}
        self.fail = fail
        self.calls: list[list[dict[str, Any]]] = []

    async def __call__(self, items: list[dict[str, Any]]) -> dict[str, dict[str, Any]] | None:
        self.calls.append(items)
        if self.fail:
            return None
        out: dict[str, dict[str, Any]] = {}
        for item in items:
            hid = str(item["hypothesis"]["id"])
            verdict = self.verdicts.get(hid, "new")
            ids = [item["papers"][0]["paper_id"]] if verdict != "new" and item["papers"] else []
            out[hid] = {"verdict": verdict, "paper_ids": ids, "reason": f"{verdict} by fixture"}
        return out
