"""Stage 8 debate: critique, answer and review rounds over the perspectives' hypotheses.

Each round has three phases:

* critique: every perspective challenges or concedes the others' hypotheses and says how serious
  each challenge is, ``fatal`` (with one of four named flaws) or ``caveat``;
* answer: every challenged author answers each challenge by revising the hypothesis, defending
  it or withdrawing it, and may add a replacement;
* review: every critic judges the answers to its own challenges (``resolved`` or ``stands``,
  never raising the severity) and examines the hypotheses added that round.

What still stands after the last round decides which candidates may enter the final set: a
candidate with a fatal objection standing is held back unless the set cannot be filled otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from idea2hypothesis.pipeline.contracts import Findings
from idea2hypothesis.pipeline.models import StageContext
from idea2hypothesis.stages.base import StageFailure, compact_json, gather_limited, request_json

STAGE = 8
#: How a critique answers one hypothesis of another perspective.
RESPONSE_STANCES = ("challenge", "concede")
#: How an author answers one challenge to its own hypothesis.
ANSWER_ACTIONS = ("revise", "defend", "withdraw")
SEVERITIES = ("fatal", "caveat")
#: The only reasons a challenge may be fatal.
FLAWS = ("unsupported", "unfalsifiable", "undecidable_test", "already_established")
VERDICTS = ("resolved", "stands")

Positions = dict[str, list[dict[str, Any]]]


def candidate_id(role: str, number: int) -> str:
    """How the final merge names a perspective's hypothesis, such as ``innovator-2``."""
    return f"{role}-{number}"


def _statement(hypothesis: dict[str, Any]) -> str:
    return str(hypothesis.get("statement", "")).strip()


def _number(value: Any) -> int | None:
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def numbered(hyps: list[dict[str, Any]], withdrawn: set[int] | None = None) -> str:
    gone = withdrawn or set()
    return "\n".join(
        f"{i}. WITHDRAWN" if i in gone else f"{i}. {compact_json(h)}" for i, h in enumerate(hyps, 1)
    )


def check_perspective(data: Any) -> Findings:
    """Light structural check of one role's output; the strict contract applies to the final set."""
    f = Findings()
    items = data.get("hypotheses") if isinstance(data, dict) else None
    if f.require(
        isinstance(items, list) and items, "the answer needs a non-empty 'hypotheses' list"
    ):
        for i, h in enumerate(items):
            f.require(
                isinstance(h, dict)
                and isinstance(h.get("statement"), str)
                and h["statement"].strip(),
                f"hypotheses[{i}] needs a statement",
            )
    return f


# -- objections -------------------------------------------------------------------------------


@dataclass
class Objection:
    """One challenge to one hypothesis, followed through its answer and review."""

    to: str
    hypothesis: int
    critic: str
    round: int
    severity: str
    text: str
    flaw: str | None = None
    field: str | None = None
    card_id: str | None = None
    #: Place in the critic's critique (names the challenge turn); None for a review challenge.
    response: int | None = None
    #: Place among the hypotheses added that round, for a challenge made in the review.
    added_item: int | None = None
    answer: dict[str, Any] | None = None
    review: dict[str, Any] | None = None
    #: ``stands``, ``resolved`` or ``withdrawn`` (the hypothesis was withdrawn).
    status: str = "stands"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, Any]:
        """What a final hypothesis carries of an objection that still stands."""
        return {
            "candidate": candidate_id(self.to, self.hypothesis),
            "from": self.critic,
            "severity": self.severity,
            "flaw": self.flaw,
            "field": self.field,
            "card_id": self.card_id,
            "text": (self.review or {}).get("text") or self.text,
        }

    def label(self) -> str:
        detail = f" ({self.flaw}, in {self.field})" if self.flaw else ""
        return f"{self.severity}{detail} from {self.critic}: {self.summary()['text']}"


def objection_problems(item: dict[str, Any], valid_refs: set[str], tag: str) -> list[str]:
    """Why a challenge's severity fields are not usable (empty when they are)."""
    severity = item.get("severity")
    if severity not in SEVERITIES:
        return [f"{tag}: severity must be one of {list(SEVERITIES)}"]
    if severity == "caveat":
        return []
    problems = []
    flaw = item.get("flaw")
    if flaw not in FLAWS:
        problems.append(f"{tag}: a fatal challenge names its flaw, one of {list(FLAWS)}")
    if not _text(item.get("field")):
        problems.append(f"{tag}: a fatal challenge names the hypothesis field that holds the flaw")
    if flaw == "already_established" and item.get("card_id") not in valid_refs:
        problems.append(
            f"{tag}: already_established needs card_id, the allowed reference that already shows it"
        )
    return problems


def _severity_fields(item: dict[str, Any]) -> dict[str, Any]:
    fatal = item.get("severity") == "fatal"
    flaw = item.get("flaw") if fatal else None
    return {
        "severity": item["severity"],
        "flaw": flaw,
        "field": _text(item.get("field")) or None if fatal else None,
        "card_id": item.get("card_id") if flaw == "already_established" else None,
    }


def parse_critique(
    data: Any,
    previous: Positions,
    role: str,
    withdrawn: dict[str, set[int]],
    valid_refs: set[str],
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """``(responses, unusable, problems)``: critiques that name another perspective and one of
    its standing hypotheses, how many did not, and what is wrong with a usable challenge."""
    raw = data.get("responses") if isinstance(data, dict) else None
    items = raw if isinstance(raw, list) else []
    kept: list[dict[str, Any]] = []
    problems: list[str] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        to = str(item.get("to") or "").strip().lower()
        n = _number(item.get("hypothesis"))
        text = _text(item.get("text"))
        if (
            to == role
            or to not in previous
            or n is None
            or not 1 <= n <= len(previous[to])
            or n in withdrawn.get(to, set())
            or item.get("stance") not in RESPONSE_STANCES
            or not text
        ):
            continue
        response: dict[str, Any] = {
            "to": to,
            "hypothesis": n,
            "stance": item["stance"],
            "text": text,
        }
        if item["stance"] == "challenge":
            found = objection_problems(item, valid_refs, f"responses[{i}]")
            problems += found
            if not found:
                response.update(_severity_fields(item))
        kept.append(response)
    return kept, len(items) - len(kept), problems


def check_critique(
    data: Any,
    previous: Positions,
    role: str,
    withdrawn: dict[str, set[int]],
    valid_refs: set[str],
) -> Findings:
    f = Findings()
    if not f.require(
        isinstance(data, dict) and isinstance(data.get("responses"), list),
        "the answer needs a 'responses' list",
    ):
        return f
    for problem in parse_critique(data, previous, role, withdrawn, valid_refs)[2]:
        f.error(problem)
    return f


def challenge_list(challenges: list[Objection]) -> str:
    def head(o: Objection) -> str:
        detail = f" {o.flaw} in {o.field}" if o.flaw else ""
        return f"to your hypothesis {o.hypothesis}, from {o.critic}, {o.severity}{detail}"

    return "\n".join(f"{n}. ({head(o)}) {o.text}" for n, o in enumerate(challenges, 1))


def check_answers(data: Any, before: list[dict[str, Any]], challenges: list[Any]) -> Findings:
    """Every challenge is answered once; a revision changes the statement it revises; a
    withdrawal lists the hypothesis it withdraws; the position keeps its numbering."""
    f = check_perspective(data)
    if not f.ok:
        return f
    hyps = data["hypotheses"]
    f.require(
        len(hyps) >= len(before),
        f"keep all {len(before)} hypotheses in their numbered order; new ones go at the end",
    )
    raw_withdrawn = data.get("withdrawn") or []
    withdrawn = [_number(n) for n in raw_withdrawn] if isinstance(raw_withdrawn, list) else [None]
    if not f.require(
        all(n is not None and 1 <= n <= len(before) for n in withdrawn),
        f"withdrawn must list hypothesis numbers from 1 to {len(before)}",
    ):
        withdrawn = []
    answers = data.get("answers")
    if not f.require(isinstance(answers, list), "the answer needs an 'answers' list"):
        return f
    seen: set[int] = set()
    for i, answer in enumerate(answers):
        n = _number(answer.get("challenge")) if isinstance(answer, dict) else None
        if n is None or not 1 <= n <= len(challenges):
            f.error(f"answers[{i}].challenge must be a number from 1 to {len(challenges)}")
            continue
        f.require(n not in seen, f"challenge {n} is answered more than once")
        seen.add(n)
        action = answer.get("action")
        f.require(action in ANSWER_ACTIONS, f"challenge {n}: action must be {ANSWER_ACTIONS}")
        f.require(_text(answer.get("text")), f"challenge {n}: give the reason in text")
        challenge = challenges[n - 1]
        k = challenge.hypothesis if isinstance(challenge, Objection) else challenge["hypothesis"]
        if action == "revise" and len(hyps) >= k and isinstance(hyps[k - 1], dict):
            f.require(
                _statement(hyps[k - 1]) != _statement(before[k - 1]),
                f"challenge {n}: you revise hypothesis {k} but its statement is unchanged",
            )
        if action == "withdraw":
            f.require(k in withdrawn, f"challenge {n}: you withdraw {k}; list {k} in withdrawn")
    missing = [n for n in range(1, len(challenges) + 1) if n not in seen]
    f.require(not missing, f"challenges {missing} have no answer; answer every challenge once")
    return f


def check_review(
    data: Any, items: list[Objection], added: list[tuple[str, int]], valid_refs: set[str]
) -> Findings:
    """Every answered challenge is reviewed once; an objection that stands says why and is never
    raised from caveat to fatal; challenges of added hypotheses follow the critique rules."""
    f = Findings()
    if not f.require(isinstance(data, dict), "the answer must be an object"):
        return f
    reviews = data.get("reviews")
    if not f.require(isinstance(reviews, list), "the answer needs a 'reviews' list"):
        return f
    seen: set[int] = set()
    for i, review in enumerate(reviews):
        n = _number(review.get("item")) if isinstance(review, dict) else None
        if n is None or not 1 <= n <= len(items):
            f.error(f"reviews[{i}].item must be a number from 1 to {len(items)}")
            continue
        f.require(n not in seen, f"item {n} is reviewed more than once")
        seen.add(n)
        verdict = review.get("verdict")
        if not f.require(verdict in VERDICTS, f"item {n}: verdict must be {list(VERDICTS)}"):
            continue
        if verdict == "resolved":
            continue
        f.require(_text(review.get("text")), f"item {n}: say what the answer leaves unaddressed")
        severity = review.get("severity", items[n - 1].severity)
        if not f.require(severity in SEVERITIES, f"item {n}: severity must be {list(SEVERITIES)}"):
            continue
        f.require(
            not (items[n - 1].severity == "caveat" and severity == "fatal"),
            f"item {n}: a caveat cannot be raised to fatal in the review",
        )
        if severity == "fatal":
            flaw = review.get("flaw") or items[n - 1].flaw
            f.require(flaw in FLAWS, f"item {n}: a fatal objection that stands names its flaw")
    missing = [n for n in range(1, len(items) + 1) if n not in seen]
    f.require(not missing, f"items {missing} have no review; review every item once")
    extra = data.get("added") or []
    if not f.require(isinstance(extra, list), "'added' must be a list"):
        return f
    seen_added: set[int] = set()
    for i, item in enumerate(extra):
        n = _number(item.get("item")) if isinstance(item, dict) else None
        if n is None or not 1 <= n <= len(added):
            f.error(f"added[{i}].item must be a number from 1 to {len(added)}")
            continue
        f.require(n not in seen_added, f"added item {n} is answered more than once")
        seen_added.add(n)
        stance = item.get("stance")
        if not f.require(
            stance in RESPONSE_STANCES, f"added item {n}: stance is challenge or concede"
        ):
            continue
        f.require(_text(item.get("text")), f"added item {n}: say why in text")
        if stance == "challenge":
            for problem in objection_problems(item, valid_refs, f"added item {n}"):
                f.error(problem)
    return f


def _answered_list(items: list[Objection], positions: Positions) -> str:
    blocks = []
    for n, o in enumerate(items, 1):
        answer = o.answer or {}
        detail = f" {o.flaw} in {o.field}" if o.flaw else ""
        now = positions[o.to][o.hypothesis - 1]
        blocks.append(
            f"{n}. Your challenge to {o.to} hypothesis {o.hypothesis} ({o.severity}{detail}): "
            f"{o.text}\n   Answer ({answer.get('action')}): {answer.get('text')}\n"
            f"   The hypothesis now reads: {compact_json(now)}"
        )
    return "\n".join(blocks) or "(none)"


def _added_list(added: list[tuple[str, int]], positions: Positions) -> str:
    return (
        "\n".join(
            f"{n}. {role} hypothesis {k}: {compact_json(positions[role][k - 1])}"
            for n, (role, k) in enumerate(added, 1)
        )
        or "(none)"
    )


# -- the debate -------------------------------------------------------------------------------


@dataclass
class Candidate:
    """A perspective's hypothesis as it stands after the debate."""

    id: str
    role: str
    number: int
    hypothesis: dict[str, Any]
    fatal: list[Objection] = field(default_factory=list)
    caveats: list[Objection] = field(default_factory=list)

    @property
    def standing(self) -> str:
        if self.fatal:
            return "FATAL"
        return "CAVEATS" if self.caveats else "CLEARED"

    def block(self) -> str:
        lines = [f"### {self.id} [{self.standing}]", compact_json(self.hypothesis)]
        if self.fatal or self.caveats:
            lines.append("Objections that stand:")
            lines += [f"- {o.label()}" for o in self.fatal + self.caveats]
        return "\n".join(lines)


@dataclass
class Debate:
    positions: Positions
    withdrawn: dict[str, set[int]]
    objections: list[Objection]
    record: dict[str, Any]
    assessment: str = ""

    def candidates(self) -> dict[str, Candidate]:
        """Every hypothesis not withdrawn, with the objections that still stand against it."""
        out: dict[str, Candidate] = {}
        for role, hyps in self.positions.items():
            for n, h in enumerate(hyps, 1):
                if n in self.withdrawn.get(role, set()):
                    continue
                standing = [
                    o
                    for o in self.objections
                    if o.to == role and o.hypothesis == n and o.status == "stands"
                ]
                out[candidate_id(role, n)] = Candidate(
                    candidate_id(role, n),
                    role,
                    n,
                    h,
                    fatal=[o for o in standing if o.severity == "fatal"],
                    caveats=[o for o in standing if o.severity == "caveat"],
                )
        return out


async def run_debate(
    ctx: StageContext,
    positions: Positions,
    variables: dict[str, Any],
    valid_refs: set[str],
    warnings: list[str],
) -> Debate:
    """Run ``llm.debate_rounds`` rounds, then the judge. With no rounds every hypothesis is a
    candidate with nothing against it."""
    rounds = ctx.config.llm.debate_rounds
    current = {role: list(hyps) for role, hyps in positions.items()}
    withdrawn: dict[str, set[int]] = {role: set() for role in current}
    objections: dict[tuple[str, int, str], Objection] = {}
    record: dict[str, Any] = {"rounds": rounds, "roles": sorted(current), "concessions": {}}
    common = {
        "valid_refs": variables["valid_refs"],
        "valid_gaps": variables["valid_gaps"],
        "synthesis_json": variables["synthesis_json"],
    }
    if rounds <= 0:
        return Debate(current, withdrawn, [], record)
    for r in range(1, rounds + 1):
        if len(current) < 2:
            break
        previous = {role: list(hyps) for role, hyps in current.items()}
        critiques = await _critiques(ctx, r, previous, withdrawn, common, valid_refs, warnings)
        aimed: dict[str, list[Objection]] = {}
        for critic, responses in critiques.items():
            for k, resp in enumerate(responses, 1):
                key = (resp["to"], resp["hypothesis"], critic)
                if resp["stance"] == "concede":
                    record["concessions"].setdefault(critic, []).append(resp["text"])
                    if key in objections:  # an earlier objection by this critic is dropped
                        objections[key].status = "resolved"
                    continue
                o = Objection(
                    resp["to"], resp["hypothesis"], critic, r, resp["severity"], resp["text"],
                    flaw=resp["flaw"], field=resp["field"], card_id=resp["card_id"], response=k,
                )  # fmt: skip
                objections[key] = o
                aimed.setdefault(o.to, []).append(o)
        docs = await _answers(ctx, r, previous, withdrawn, aimed, common, warnings)
        for role, doc in docs.items():
            current[role] = doc["hypotheses"]
            withdrawn[role] |= set(doc["withdrawn"])
            for a in doc["answers"]:
                o = aimed[role][a["challenge"] - 1]
                o.answer = {"action": a["action"], "text": a["text"]}
                if a["action"] == "withdraw":
                    o.status = "withdrawn"
        await _reviews(ctx, r, current, withdrawn, objections, docs, common, valid_refs, warnings)
    for o in objections.values():
        if o.hypothesis in withdrawn.get(o.to, set()):
            o.status = "withdrawn"
    record["withdrawn"] = {role: sorted(ns) for role, ns in withdrawn.items() if ns}
    record["answers"] = {}
    for o in objections.values():
        if o.answer:
            tally = record["answers"].setdefault(o.to, dict.fromkeys(ANSWER_ACTIONS, 0))
            tally[o.answer["action"]] += 1
    record["objections"] = [o.to_dict() for o in objections.values()]
    debate = Debate(current, withdrawn, list(objections.values()), record)
    debate.assessment = await _judge(ctx, debate, warnings)
    ctx.artifacts.write_json(STAGE, "perspectives/debate_record.json", debate.record)
    if debate.record.get("rankings"):
        await ctx.progress("judged", file="perspectives/debate_record.json")
    return debate


async def _critiques(
    ctx: StageContext,
    r: int,
    previous: Positions,
    withdrawn: dict[str, set[int]],
    common: dict[str, str],
    valid_refs: set[str],
    warnings: list[str],
) -> dict[str, list[dict[str, Any]]]:
    async def critique(role: str) -> tuple[str, list[dict[str, Any]]]:
        others = "\n\n---\n\n".join(
            f"### {other}\n{numbered(previous[other], withdrawn[other])}"
            for other in previous
            if other != role
        )
        prompt = ctx.prompts.render(
            "debate_critique",
            role=role,
            own_position=numbered(previous[role], withdrawn[role]),
            others=others,
            **common,
        )
        try:
            data, _ = await request_json(
                ctx, prompt, label=f"debate {role} r{r} critique",
                validate=lambda d: check_critique(d, previous, role, withdrawn, valid_refs),
            )  # fmt: skip
        except StageFailure as exc:
            warnings.append(f"debate round {r}: {role} made no critique ({exc.message})")
            return role, []
        responses, unusable, _ = parse_critique(data, previous, role, withdrawn, valid_refs)
        if unusable:
            warnings.append(
                f"debate round {r}: {unusable} critiques of {role} named no standing "
                "hypothesis of another perspective and were left out"
            )
        name = f"perspectives/{role}.r{r}.critique.json"
        ctx.artifacts.write_json(
            STAGE, name, {"role": role, "round": r, "phase": "critique", "responses": responses}
        )
        await ctx.progress("critique", file=name, role=role, round=r)
        return role, responses

    results = await gather_limited(
        [lambda role=role: critique(role) for role in previous], ctx.config.runtime.concurrency
    )
    return dict(results)


async def _answers(
    ctx: StageContext,
    r: int,
    previous: Positions,
    withdrawn: dict[str, set[int]],
    aimed: dict[str, list[Objection]],
    common: dict[str, str],
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    async def answer(role: str) -> tuple[str, dict[str, Any] | None]:
        challenges = aimed[role]
        before = previous[role]
        prompt = ctx.prompts.render(
            "debate_answer",
            role=role,
            own_position=numbered(before, withdrawn[role]),
            challenges=challenge_list(challenges),
            **common,
        )
        try:
            data, _ = await request_json(
                ctx, prompt, label=f"debate {role} r{r} answer",
                validate=lambda d: check_answers(d, before, challenges),
            )  # fmt: skip
        except StageFailure as exc:
            warnings.append(
                f"debate round {r}: {role} left {len(challenges)} challenges unanswered "
                f"and kept its position ({exc.message})"
            )
            return role, None
        hyps = [h for h in data["hypotheses"] if isinstance(h, dict)]
        # A hypothesis withdrawn in an earlier round was shown as WITHDRAWN; it keeps its text.
        hyps = [
            before[i - 1] if i in withdrawn[role] and i <= len(before) else h
            for i, h in enumerate(hyps, 1)
        ]
        gone = sorted({n for n in (_number(x) for x in data.get("withdrawn") or []) if n})
        answers = []
        for a in data["answers"]:
            n = int(_number(a["challenge"]) or 0)
            o = challenges[n - 1]
            answers.append(
                {
                    "challenge": n, "from": o.critic, "response": o.response,
                    "hypothesis": o.hypothesis, "action": a["action"],
                    "text": str(a["text"]).strip(),
                }
            )  # fmt: skip
        revised = [
            i
            for i, h in enumerate(hyps[: len(before)], 1)
            if _statement(h) != _statement(before[i - 1]) and i not in gone
        ]
        doc = {
            "role": role,
            "round": r,
            "phase": "answer",
            "hypotheses": hyps,
            "answers": answers,
            "revised": revised,
            "added": list(range(len(before) + 1, len(hyps) + 1)),
            "withdrawn": gone,
        }
        name = f"perspectives/{role}.r{r}.json"
        ctx.artifacts.write_json(STAGE, name, doc)
        await ctx.progress("perspective", file=name, role=role, round=r)
        return role, doc

    outcomes = await gather_limited(
        [lambda role=role: answer(role) for role in previous if aimed.get(role)],
        ctx.config.runtime.concurrency,
    )
    return {role: doc for role, doc in outcomes if doc is not None}


async def _reviews(
    ctx: StageContext,
    r: int,
    current: Positions,
    withdrawn: dict[str, set[int]],
    objections: dict[tuple[str, int, str], Objection],
    docs: dict[str, dict[str, Any]],
    common: dict[str, str],
    valid_refs: set[str],
    warnings: list[str],
) -> None:
    """Every critic reviews the answers to its challenges of this round and the added
    hypotheses; a challenge whose author gave no answer stands without review."""

    def items_of(critic: str) -> list[Objection]:
        return [
            o
            for o in objections.values()
            if o.critic == critic and o.round == r and o.answer and o.status == "stands"
        ]

    def added_for(critic: str) -> list[tuple[str, int]]:
        return [
            (role, k) for role, doc in docs.items() if role != critic for k in doc["added"]
        ]  # fmt: skip

    async def review(critic: str) -> None:
        items, added = items_of(critic), added_for(critic)
        prompt = ctx.prompts.render(
            "debate_review",
            role=critic,
            answered=_answered_list(items, current),
            added=_added_list(added, current),
            valid_refs=common["valid_refs"],
            synthesis_json=common["synthesis_json"],
        )
        try:
            data, _ = await request_json(
                ctx, prompt, label=f"debate {critic} r{r} review",
                validate=lambda d: check_review(d, items, added, valid_refs),
            )  # fmt: skip
        except StageFailure as exc:
            warnings.append(
                f"debate round {r}: {critic} did not review its {len(items)} answered "
                f"challenges; they stand as raised ({exc.message})"
            )
            return
        reviews = []
        for rev in data["reviews"]:
            n = int(_number(rev["item"]) or 0)
            o = items[n - 1]
            if rev["verdict"] == "resolved":
                o.status = "resolved"
                o.review = {"verdict": "resolved", "text": _text(rev.get("text"))}
            else:
                o.severity = rev.get("severity", o.severity)
                o.flaw = (rev.get("flaw") or o.flaw) if o.severity == "fatal" else None
                o.review = {"verdict": "stands", "text": _text(rev.get("text"))}
            reviews.append({**o.to_dict(), "item": n})
        challenges_added = []
        for k, item in enumerate(data.get("added") or [], 1):
            n = int(_number(item["item"]) or 0)
            role, number = added[n - 1]
            entry: dict[str, Any] = {
                "to": role, "hypothesis": number, "stance": item["stance"],
                "text": _text(item.get("text")), "item": n,
            }  # fmt: skip
            if item["stance"] == "challenge":
                entry.update(_severity_fields(item))
                objections[(role, number, critic)] = Objection(
                    role, number, critic, r, entry["severity"], entry["text"],
                    flaw=entry["flaw"], field=entry["field"], card_id=entry["card_id"],
                    added_item=k,
                )  # fmt: skip
            challenges_added.append(entry)
        name = f"perspectives/{critic}.r{r}.review.json"
        doc = {
            "role": critic,
            "round": r,
            "phase": "review",
            "reviews": reviews,
            "added": challenges_added,
        }
        ctx.artifacts.write_json(STAGE, name, doc)
        await ctx.progress("review", file=name, role=critic, round=r)

    critics = [c for c in current if items_of(c) or added_for(c)]
    await gather_limited(
        [lambda c=c: review(c) for c in critics], ctx.config.runtime.concurrency
    )  # fmt: skip


async def _judge(ctx: StageContext, debate: Debate, warnings: list[str]) -> str:
    judge = ctx.reviewer or ctx.llm
    debate.record["independent_judge"] = ctx.reviewer is not None
    if ctx.reviewer is None:
        warnings.append("no reviewer model configured; the debate judge is not independent")
    candidates = debate.candidates()
    combined = "\n\n---\n\n".join(
        f"### Perspective: {role}\n"
        + "\n\n".join(c.block() for c in candidates.values() if c.role == role)
        for role in debate.positions
        if any(c.role == role for c in candidates.values())
    )
    prompt = ctx.prompts.render("debate_judge", perspectives=combined)
    try:
        data, _ = await request_json(
            ctx, prompt, label="debate judge", llm=judge, validate=_check_rankings
        )
    except StageFailure as exc:
        warnings.append(f"debate judge failed: {exc.message}")
        return ""
    debate.record["rankings"] = data["rankings"]
    return "Independent reviewer assessment:\n" + compact_json(data["rankings"])


def _check_rankings(data: Any) -> Findings:
    f = Findings()
    rankings = data.get("rankings") if isinstance(data, dict) else None
    f.require(
        isinstance(rankings, list) and rankings, "the answer needs a non-empty 'rankings' list"
    )
    return f
