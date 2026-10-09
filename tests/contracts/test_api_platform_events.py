"""Platform event stream: built only from real artifacts, in the shape the consumers read."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from idea2hypothesis.api.platform_events import GROUPS, PlatformEventLog, _merges
from idea2hypothesis.storage.runs import RunStore
from tests.fixtures.api_helpers import SERVER_KEY, Harness, harness

#: Event types the engine can emit; everything else of the original full_runner is dropped.
EMITTED = {
    "run.started",
    "run.plan",
    "run.status",
    "run.completed",
    "stage.started",
    "stage.completed",
    "step.started",
    "step.completed",
    "rule.checked",
    "scope.profile",
    "scope.goal",
    "scope.approved",
    "problem.subquestion",
    "problem.risk",
    "topic.evaluated",
    "search.strategy",
    "search.query",
    "search.sources",
    "literature.request",
    "literature.batch",
    "literature.collected",
    "literature.merged",
    "screen.criteria",
    "screen.scored",
    "screen.rejected",
    "screen.kept",
    "card.extracted",
    "synthesis.cluster",
    "synthesis.set_aside",
    "synthesis.tension",
    "synthesis.gap",
    "synthesis.overview",
    "synthesis.ranked",
    "debate.turn",
    "hypothesis.drafted",
    "hypothesis.checked",
    "hypothesis.selected",
    "map.node",
    "map.edge",
    "canvas.piece",
    "gate.opened",
    "gate.resolved",
    "agent.message",
}
DROPPED = {
    "skills.loaded",
    "scope.estimate",
    "scope.adjusted",
    "estimate.checked",
    "idea.set_aside",
}
STAGE_KEYS = {"scope", "search", "screen", "read", "synthesize", "r1-hypothesize", "map"}


@pytest.fixture(autouse=True)
def _server_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("I2H_SERVICE_KEY", SERVER_KEY)


async def _completed(h: Harness, mode: str = "auto") -> tuple[str, list[dict[str, Any]]]:
    run_id = (await h.start("p-1", review_mode=mode)).json()["popper_run_id"]
    await h.settle()
    return run_id, await h.events(run_id)


def _of(events: list[dict[str, Any]], type_: str) -> list[dict[str, Any]]:
    return [e for e in events if e["type"] == type_]


async def test_only_known_types_and_stage_keys_are_emitted(tmp_path: Path) -> None:
    async with harness(tmp_path, research={"hardware_advisory": True}) as h:
        _, events = await _completed(h, "full")
        for _ in range(2):  # scope and screen gates
            run_id = h.service.store.list_run_ids()[0]
            gate_id = _of(await h.events(run_id), "gate.opened")[-1]["payload"]["gate_id"]
            await h.client.post(f"/runs/{run_id}/gates/{gate_id}", json={"option_id": "approve"})
            await h.settle()
        events = await h.events(run_id)
    types = {e["type"] for e in events}
    assert types <= EMITTED, types - EMITTED
    assert not types & DROPPED
    assert {e["stage_key"] for e in events if e["stage_key"]} <= STAGE_KEYS
    assert {"scope", "r1-hypothesize"} <= {e["stage_key"] for e in events}
    for event in events:  # content events always carry their stage
        if event["type"].split(".")[0] in {"screen", "card", "synthesis", "hypothesis", "problem"}:
            assert event["stage_key"] in STAGE_KEYS


async def test_plan_and_stage_plans_use_the_ui_groups(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        _, events = await _completed(h, "copilot")
    plan = _of(events, "run.plan")[0]["payload"]["stages"]
    assert [(s["key"], s["stage"], s["pipeline"]) for s in plan] == [
        ("scope", "scope", [1, 2]),
        ("search", "search", [3, 4]),
        ("screen", "screen", [5]),
        ("read", "read", [6]),
        ("synthesize", "synthesize", [7]),
        ("r1-hypothesize", "hypothesize", [8]),
        ("map", "map", [9]),
    ]
    # copilot stops after screening and again once the hypotheses are written
    assert [s["has_gate"] for s in plan] == [False, False, True, False, False, True, False]
    assert [g.key for g in GROUPS] == [s["key"] for s in plan]
    started = {e["stage_key"]: e["payload"]["plan"] for e in _of(events, "stage.started")}
    assert list(started) == ["scope", "search", "screen"]  # stops at the gate
    screen_steps = [s["id"] for s in started["screen"]["steps"]]
    assert screen_steps == ["score", "reject", "shortlist", "screen_gate"]
    assert started["screen"]["steps"][-1]["gate"] == "screen"
    for plan_payload in started.values():
        assert {"key", "stage", "title", "has_gate", "pipeline", "purpose", "cast", "steps"} <= set(
            plan_payload
        )
        for step in plan_payload["steps"]:
            assert {"id", "title", "actor", "explain"} <= set(step)


async def test_steps_are_balanced_and_stages_complete_in_order(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        _, events = await _completed(h)
    for key in STAGE_KEYS:
        started = [
            e["payload"]["step_id"]
            for e in events
            if e["type"] == "step.started" and e["stage_key"] == key
        ]
        done = [
            e["payload"]["step_id"]
            for e in events
            if e["type"] == "step.completed" and e["stage_key"] == key
        ]
        assert started == done and started, key
    order = [e["stage_key"] for e in events if e["type"] == "stage.completed"]
    assert order == ["scope", "search", "screen", "read", "synthesize", "r1-hypothesize", "map"]
    assert events[-1]["type"] == "run.completed" and events[-1]["stage_key"] is None


async def test_content_comes_from_the_real_artifacts(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id, events = await _completed(h)
        art = h.service.store.artifacts(run_id)
        candidates = art.read_jsonl(4, "candidates.jsonl")
        shortlist = art.read_jsonl(5, "shortlist.jsonl")
        review = art.read_json(5, "review.json")
        synthesis = art.read_json(7, "synthesis.json")
        hypotheses = art.read_json(8, "hypotheses.json")["hypotheses"]
        tree = art.read_json(2, "problem_tree.json")
        meta = art.read_json(4, "search_meta.json")

    collected = _of(events, "literature.collected")[0]["payload"]
    assert (collected["raw"], collected["unique"], collected["duplicates"]) == (
        meta["raw"],
        meta["unique"],
        meta["duplicates"],
    )
    assert collected["unique"] == len(candidates) == 12
    merged = [e["payload"] for e in _of(events, "literature.merged")]
    merged_records = sum(len(m["records"]) - 1 for m in merged)
    assert collected["repeats"] == meta["duplicates"] - merged_records
    extra = [q["text"] for q in meta["queries_used"] if q["origin"] == "expansion"]
    assert collected["expansion"] == {
        "queries": len(extra),
        "hits": sum(sum(meta["per_query"][q].values()) for q in extra),
    }

    # one screening point per scored paper, taken from review.json
    points = [p for e in _of(events, "screen.scored") for p in e["payload"]["points"]]
    assert sorted(p["id"] for p in points) == sorted(d["paper_id"] for d in review["decisions"])
    assert len({p["id"] for p in points}) == len(points)
    by_id = {d["paper_id"]: d for d in review["decisions"]}
    assert all(p["relevance"] == by_id[p["id"]]["relevance_score"] for p in points)

    # rejected papers carry their reason and the shared word that made them match
    rejected = [e["payload"]["paper"] for e in _of(events, "screen.rejected")]
    assert len(rejected) == 3 and all(
        r["reason"] and r["false_friend"] == "sleep" for r in rejected
    )
    kept = [e["payload"]["paper"] for e in _of(events, "screen.kept")]
    assert [k["id"] for k in kept] == [s["paper_id"] for s in shortlist]
    assert all(k["reason"] and 0 <= k["relevance"] <= 1 and k["citation"] for k in kept)

    questions = [e["payload"]["sub_question"] for e in _of(events, "problem.subquestion")]
    assert [q["id"] for q in questions] == [q["id"] for q in tree["sub_questions"]]
    evaluation = _of(events, "topic.evaluated")[0]["payload"]
    assert evaluation["overall"] == 7.3 and set(evaluation["scores"]) == {
        "novelty",
        "specificity",
        "feasibility",
    }
    # every score says what earned it, and the advice is never empty
    assert set(evaluation["reasons"]) == set(evaluation["scores"])
    assert all(evaluation["reasons"].values()) and evaluation["advice"]

    cards = [e["payload"]["card"] for e in _of(events, "card.extracted")]
    assert sorted(c["paper_id"] for c in cards) == sorted(s["paper_id"] for s in shortlist)
    assert all(c["evidence_scope"] == "abstract" for c in cards)
    assert any(c["unknown_fields"] for c in cards)  # null stays unknown, never filled in

    gaps = [e["payload"]["gap"] for e in _of(events, "synthesis.gap")]
    assert [g["id"] for g in gaps] == [g["id"] for g in synthesis["gaps"]]
    card_ids = {c["id"] for c in cards}
    assert all(set(g["from"]) <= card_ids for g in gaps)
    # every card is in a cluster or set aside with a reason; none is left to sort
    clustered = [
        i for e in _of(events, "synthesis.cluster") for i in e["payload"]["cluster"]["card_ids"]
    ]
    asides = [e["payload"]["aside"] for e in _of(events, "synthesis.set_aside")]
    assert asides and all(a["reason"] for a in asides)
    assert set(clustered) | {i for a in asides for i in a["card_ids"]} == card_ids
    ranking = _of(events, "synthesis.ranked")[0]["payload"]["ranking"]
    assert [r["priority"] for r in ranking] == [1, 2]

    drafted = [e["payload"]["hypothesis"] for e in _of(events, "hypothesis.drafted")]
    assert [h["id"] for h in drafted] == [h["id"] for h in hypotheses]
    assert len({h["novelty"] for h in drafted}) == len(drafted)  # distinct, not one template
    assert len({h["rationale"] for h in drafted}) == len(drafted)
    assert all(h["falsify"]["text"] and h["gap"] in {g["id"] for g in gaps} for h in drafted)
    zones = {h["prediction"]: h["falsify"]["zone"] for h in drafted}
    assert zones.get("> 0") == [None, 0] and zones.get("< 0") == [0, None]
    selected = [e["payload"]["hypothesis_id"] for e in _of(events, "hypothesis.selected")]
    assert selected == [h["id"] for h in hypotheses]
    checks = _of(events, "hypothesis.checked")
    assert checks and all("feasibility" not in c["payload"] for c in checks)  # never invented
    assert all(set(c["payload"]["novelty"]) == {"novel", "closest", "similarity"} for c in checks)
    assert _of(events, "rule.checked")[0]["payload"]["rule"] == 5
    assert _of(events, "debate.turn")  # one turn per real perspective output


async def test_topic_rejected_by_the_evaluation_reports_the_rating_and_stops(
    tmp_path: Path,
) -> None:
    from tests.fixtures import FixtureLLM

    llm = FixtureLLM(
        overrides={
            "topic_evaluation": {
                "novelty": 2,
                "specificity": 2,
                "feasibility": 3,
                "overall": 2.4,
                "reasons": {"novelty": "n", "specificity": "s", "feasibility": "f"},
                "suggestion": "narrow it",
            }
        }
    )
    async with harness(tmp_path, llm=llm) as h:
        run_id, events = await _completed(h)
        state = await h.state(run_id)
    assert state["status"] == "failed" and "TOPIC_BELOW_THRESHOLD" in state["message"]
    rated = _of(events, "topic.evaluated")[0]["payload"]
    assert rated["overall"] < rated["threshold"] and rated["advice"] == "narrow it"
    assert events[-1]["type"] == "run.status" and events[-1]["payload"]["status"] == "failed"
    assert {e["stage_key"] for e in _of(events, "stage.started")} == {"scope"}


async def test_gate_payload_matches_what_the_studio_reads(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        _, events = await _completed(h, "copilot")
    gate = _of(events, "gate.opened")[0]["payload"]
    for key in (
        "id",
        "gate_id",
        "kind",
        "title",
        "why",
        "summary",
        "options",
        "stop_index",
        "stop_total",
        "droppable",
        "gate",
    ):
        assert key in gate, key
    assert gate["gate"]["gate_id"] == gate["gate_id"]
    assert all(
        {"id", "label", "description", "leads_to", "confirm_label"} <= set(o)
        for o in gate["options"]
    )
    assert [o["recommended"] for o in gate["options"] if o.get("recommended")] == [True]
    shortlist_ids = {r["paper_id"] for r in gate["shortlist"]}
    assert set(gate["droppable"]) == shortlist_ids
    statuses = [e["payload"]["status"] for e in _of(events, "run.status")]
    assert statuses == ["awaiting_review"]


async def test_log_is_private_state_free_and_heals_a_torn_tail(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id, events = await _completed(h)
        path = h.service.store.run_dir(run_id) / "platform_events.jsonl"
    raw = path.read_text(encoding="utf-8").splitlines()
    assert all("_core_seq" not in json.dumps(e) for e in events)  # internals never leak
    assert json.loads(raw[-1])["_core_seq"] > 0

    # a crash that left half a line, then a partly written batch
    lines = raw[:-1]
    path.write_text("\n".join(lines) + "\n" + raw[-1][:20], encoding="utf-8")
    healed = PlatformEventLog(RunStore(tmp_path / "runs")).read(run_id)
    assert [e["source_seq"] for e in healed] == list(range(1, len(healed) + 1))
    assert len(healed) == len(events) - 1

    # the next start re-projects from the core log, so no event is lost or duplicated
    async with harness(tmp_path) as h2:
        again = await h2.events(run_id)
        assert [e["source_seq"] for e in again] == list(range(1, len(again) + 1))
        assert [e["type"] for e in again] == [e["type"] for e in events]


async def test_parts_are_sent_while_the_stage_runs(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id, events = await _completed(h)
        rows = h.service.log._load(run_id)
        core = {e.seq: e for e in h.service.store.read_events(run_id)}
    by_type = lambda t: [r for r in rows if r["type"] == t]  # noqa: E731
    cards = by_type("card.extracted")
    # one progress event per card, each sent before the stage completed
    assert len({r["_core_seq"] for r in cards}) == len(cards) > 1
    assert all(core[r["_core_seq"]].type == "stage.progress" for r in cards)
    read_done = next(r for r in by_type("stage.completed") if r["stage_key"] == "read")
    assert all(r["source_seq"] < read_done["source_seq"] for r in cards)
    for kind in ("screen.scored", "literature.batch", "debate.turn", "hypothesis.drafted"):
        assert any(core[r["_core_seq"]].type == "stage.progress" for r in by_type(kind)), kind


async def test_no_part_is_sent_twice(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        _, events = await _completed(h)

    def ids(type_: str, pick) -> list[str]:
        return [pick(e["payload"]) for e in _of(events, type_)]

    for type_, pick in (
        ("card.extracted", lambda p: p["card"]["id"]),
        ("screen.rejected", lambda p: p["paper"]["id"]),
        ("screen.kept", lambda p: p["paper"]["id"]),
        ("debate.turn", lambda p: p["turn"]["id"]),
        ("hypothesis.drafted", lambda p: p["hypothesis"]["id"]),
        ("screen.criteria", lambda p: "criteria"),
    ):
        found = ids(type_, pick)
        assert found and len(found) == len(set(found)), type_
    cells = [(p["query_id"], p["source_id"]) for p in map(lambda e: e["payload"], events)
             if "query_id" in p and "hits" in p]  # fmt: skip
    assert len(cells) == len(set(cells))
    points = [p["id"] for e in _of(events, "screen.scored") for p in e["payload"]["points"]]
    assert len(points) == len(set(points))


async def test_every_narration_line_is_closed_and_comes_from_the_records(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id, events = await _completed(h)
        art = h.service.store.artifacts(run_id)
        summary = art.read_json(5, "review.json")["summary"]
        cards = len([n for n in art.list_files(6) if n.endswith(".json") and "cards/" in n])
    lines = _of(events, "agent.message")
    assert lines and all(e["payload"]["text"].strip() and e["actor"] for e in lines)
    last: dict[str, dict[str, Any]] = {}
    for e in lines:
        last[e["payload"]["message_id"]] = e["payload"]
    assert all(p.get("done") for p in last.values())  # no line is left "in progress"
    done = [p["text"] for p in last.values()]
    assert f"Kept {summary['kept']} of {summary['candidates']} papers." in done
    assert any(t.startswith(f"{cards} knowledge cards") for t in done)
    assert any(
        t.startswith(f"Card {cards} of {cards}:") for t in (e["payload"]["text"] for e in lines)
    )
    # a line closes before its step completes
    for i, e in enumerate(events):
        if e["type"] == "step.completed":
            step = e["payload"]["step_id"]
            mine = [x for x in lines if x["payload"]["message_id"].endswith(f".{step}")]
            assert all(x["payload"].get("done") or events.index(x) < i for x in mine)


async def test_a_restarted_stage_starts_its_ui_group_over(tmp_path: Path) -> None:
    from idea2hypothesis.llm.models import LLMTimeout
    from tests.fixtures import FixtureLLM

    state = {"failed": False}

    def fail_once(info) -> None:
        if info.key == "knowledge_extract" and info.index == 1 and not state["failed"]:
            state["failed"] = True
            raise LLMTimeout("transient")

    async with harness(tmp_path, llm=FixtureLLM(on_call=fail_once)) as h:
        _, events = await _completed(h)
    read_starts = [
        i for i, e in enumerate(_of(events, "stage.started")) if e["stage_key"] == "read"
    ]
    assert len(read_starts) == 2  # the UI drops the failed try's cards and receives them again
    last_start = max(i for i, e in enumerate(events) if e["type"] == "stage.started"
                     and e["stage_key"] == "read")  # fmt: skip
    after = [
        e["payload"]["card"]["id"] for e in events[last_start:] if e["type"] == "card.extracted"
    ]
    assert len(after) == len(set(after)) and after


async def test_a_debate_round_shows_critiques_answers_reviews_and_the_judge(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path, llm_settings={"debate_rounds": 1}) as h:
        run_id, events = await _completed(h)
        rows = h.service.log._load(run_id)
        core = {e.seq: e for e in h.service.store.read_events(run_id)}
    turns = [e["payload"]["turn"] for e in _of(events, "debate.turn")]
    by_id = {t["id"]: t for t in turns}
    assert len(by_id) == len(turns)
    phase = {p: [t for t in turns if t.get("phase") == p] for p in ("critique", "answer", "review")}
    # 3 perspectives x 3 hypotheses; each critic raises one fatal challenge and concedes one
    assert sum(1 for t in turns if t["stance"] == "propose") == 9
    assert [t["stance"] for t in phase["critique"]].count("challenge") == 3
    assert [t["stance"] for t in phase["critique"]].count("concede") == 3
    for t in phase["critique"]:
        parent = by_id[t["reply_to"]]
        assert parent["stance"] == "propose" and parent["about"] == t["about"]
        if t["stance"] == "challenge":
            assert (t["severity"], t["flaw"], t["field"]) == ("fatal", "unsupported", "statement")
    # the Theorist answers 2 (a rewrite, a defence), the Methodologist 1 (a rewrite)
    assert sorted(t["stance"] for t in phase["answer"]) == ["defend", "refine", "refine"]
    for t in phase["answer"]:
        assert t["answers"] and t["reply_to"] == t["answers"][0]
        for cid in t["answers"]:
            assert by_id[cid]["stance"] == "challenge" and by_id[cid]["about"] == t["about"]
    assert all(t["note"] for t in phase["answer"] if t["stance"] == "refine")
    # each critic reviews the answer to its challenge: two resolved, one stands as a caveat
    reviews = phase["review"]
    assert sorted(t["stance"] for t in reviews) == ["challenge", "concede", "concede"]
    stands = next(t for t in reviews if t["stance"] == "challenge")
    assert stands["severity"] == "caveat" and stands["actor"] == "skeptic"
    for t in reviews:
        assert by_id[t["reply_to"]]["phase"] == "answer"  # it replies to the answer it reviews
        assert by_id[t["answers"][0]]["actor"] == t["actor"]  # about its own challenge
    order = [t.get("phase") for t in turns]
    assert order.index("answer") > max(i for i, p in enumerate(order) if p == "critique")
    assert order.index("review") > max(i for i, p in enumerate(order) if p == "answer")
    judge = [t for t in turns if t["id"] == "judge"]
    assert len(judge) == 1 and judge[0]["actor"] == "pi" and "7/10" in judge[0]["text"]
    # the final set is built from candidates that survived, with what still stands on record
    hypotheses = [e["payload"]["hypothesis"] for e in _of(events, "hypothesis.drafted")]
    assert [h["from"] for h in hypotheses] == [["T1"], ["T2"], ["T3"]]
    assert [c["by"] for c in hypotheses[0]["caveats"]] == ["skeptic"]
    assert any("Debate caveat" in str(x) for x in hypotheses[0]["limitations"])
    assert all(not h["contested"] for h in hypotheses)
    # everything in the debate is sent while stage 8 runs
    sent = [r for r in rows if r["type"] == "debate.turn"]
    assert all(core[r["_core_seq"]].type == "stage.progress" for r in sent)
    lines = [e["payload"]["text"] for e in _of(events, "agent.message")]
    assert any("same model as the perspectives" in t for t in lines)  # no reviewer configured
    assert any(
        "answered each other in 1 round (3 challenged, 3 answered, 2 resolved on review, "
        "1 still standing)" in t
        for t in lines
    )
    assert any("reviewed 1 answer: 0 resolved, 1 still standing (0 fatal)" in t for t in lines)


async def test_tensions_carry_their_sides_and_the_hypotheses_that_settle_them(
    tmp_path: Path,
) -> None:
    async with harness(tmp_path) as h:
        _, events = await _completed(h)
    (tension,) = [e["payload"]["tension"] for e in _of(events, "synthesis.tension")]
    assert tension["id"] == "X1" and tension["between"] == ["C1", "C2"]
    assert [len(side["card_ids"]) for side in tension["sides"]] == [1, 1]
    hypotheses = [e["payload"]["hypothesis"] for e in _of(events, "hypothesis.drafted")]
    assert hypotheses[0]["tension_ids"] == ["X1"]
    assert all(h["tension_ids"] == [] for h in hypotheses[1:])
    lines = [e["payload"]["text"] for e in _of(events, "agent.message")]
    assert any("1 tension of the literature settled (X1)" in t for t in lines)


class _Sources:
    def exists(self, stage: int, name: str) -> bool:
        return True

    def read_json(self, stage: int, name: str) -> dict[str, Any]:
        return {
            "sources": [{"id": "openalex", "name": "OpenAlex"}, {"id": "arxiv", "name": "arXiv"}]
        }


def test_a_merged_paper_shows_each_source_record_and_the_one_kept() -> None:
    record = {"url": "", "retrieved_at": ""}
    merged = {
        "title": "Sparse attention",
        "source_records": [
            {"provider": "openalex", "source_id": "W9", "citations": 40, "has_doi": True, **record},
            {
                "provider": "arxiv",
                "source_id": "2301.1",
                "citations": 0,
                "has_doi": False,
                **record,
            },
        ],
    }
    single = {"title": "Alone", "source_records": merged["source_records"][:1]}
    old = {"title": "Before records kept citations", "source_records": [
        {"provider": "openalex", "source_id": "W1", **record},
        {"provider": "arxiv", "source_id": "2301.2", **record},
    ]}  # fmt: skip
    (event,) = _merges(_Sources(), [merged, single, old])  # type: ignore[arg-type]
    assert event == {
        "title": "Sparse attention",
        "records": [
            {"source": "OpenAlex", "record_id": "W9", "citations": 40, "has_doi": True},
            {"source": "arXiv", "record_id": "2301.1", "citations": 0, "has_doi": False},
        ],
        "kept": "OpenAlex",
        "kept_record": "W9",
    }


def test_every_domain_s_perspectives_speak_as_three_different_agents() -> None:
    import yaml

    from idea2hypothesis.api.platform_events import _ROLE_ACTORS, _idea

    roles_file = Path(__file__).parents[2] / "src/idea2hypothesis/prompts/hypothesis_roles.yaml"
    for domain, roles in yaml.safe_load(roles_file.read_text())["roles"].items():
        assert set(roles) <= set(_ROLE_ACTORS), domain
        actors = [_ROLE_ACTORS[r] for r in roles]
        assert len(set(actors)) == len(actors), domain  # no two share an agent or a thread
        assert len({_idea(r, 1) for r in roles}) == len(roles), domain


async def test_the_map_draws_each_relation_after_both_its_ends(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        run_id, events = await _completed(h)
        graph = h.service.store.artifacts(run_id).read_json(9, "semantic_graph.json")
    drawn = [e for e in events if e["stage_key"] == "map"]
    placed: set[str] = set()
    edges = []
    for e in drawn:
        if e["type"] == "map.node":
            placed.add(e["payload"]["node"]["id"])
        elif e["type"] == "map.edge":
            edge = e["payload"]["edge"]
            assert {edge["from"], edge["to"]} <= placed, edge["id"]
            edges.append(edge["id"])
    assert placed == {n["id"] for n in graph["entities"]}
    assert sorted(edges) == sorted(r["id"] for r in graph["relations"])
    pieces = [e["payload"]["piece"]["id"] for e in _of(drawn, "canvas.piece")]
    assert len(pieces) == 9 and pieces[0] == "puzzle"
    steps = [e["payload"]["step_id"] for e in _of(drawn, "step.completed")]
    assert steps == ["questions", "foundation", "reasoning", "contribution", "canvas"]
    summary = _of(drawn, "stage.completed")[0]["payload"]["summary"]
    assert f"{len(graph['entities'])} entities" in summary


async def test_cards_carry_their_quotes_and_the_topic_score_its_basis(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        _, events = await _completed(h)
    cards = [e["payload"]["card"] for e in _of(events, "card.extracted")]
    assert cards and all(c["quotes"]["findings"] for c in cards)
    assert all(set(c["quotes"]) <= {k for k in c if c[k]} for c in cards)
    score = _of(events, "topic.evaluated")[0]["payload"]
    assert score["basis"] == "model judgement before any literature search"
