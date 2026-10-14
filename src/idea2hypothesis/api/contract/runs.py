"""Contract runs: create, state, events, gate answers and run control.

:class:`ContractRuns` belongs to one :class:`~idea2hypothesis.api.service.RunService` and
shares its store, run control, background tasks and webhook delivery. Every contract run has
its own :class:`Services`, built from the models of its request; the keys exist only there.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import TYPE_CHECKING, Any

from idea2hypothesis.api.contract.artifacts import ArtifactPublisher
from idea2hypothesis.api.contract.errors import ContractError
from idea2hypothesis.api.contract.events import ContractEventLog, ContractProjector
from idea2hypothesis.api.contract.llm import (
    LlmFactory,
    UnconfiguredLLM,
    services_for_request,
)
from idea2hypothesis.api.contract.mapping import run_state
from idea2hypothesis.api.contract.models import (
    GateAnswer,
    RunCreateRequest,
    RunState,
)
from idea2hypothesis.api.webhooks import CallbackRejected, validate_callback_url
from idea2hypothesis.literature.search import build_literature
from idea2hypothesis.llm.models import LLMConfigError
from idea2hypothesis.pipeline import gates
from idea2hypothesis.pipeline.models import GateAnswer as CoreAnswer
from idea2hypothesis.pipeline.models import RunRequest, RunStatus, Services
from idea2hypothesis.pipeline.runner import (
    RunStateError,
    apply_gate_answer,
    cancel_run,
    new_run_id,
    pause_run,
    resume_pipeline,
    run_pipeline,
)
from idea2hypothesis.pipeline.services import build_services
from idea2hypothesis.storage.runs import RunNotFoundError, utc_now

if TYPE_CHECKING:
    from idea2hypothesis.api.service import RunService

HYPOTHESIS_STAGE = "hypothesis"
TOPIC_MIN, TOPIC_MAX, MAX_DOMAINS = 12, 2000, 6
#: How long pause and cancel wait for the worker to reach a safe point before answering.
SETTLE_TIMEOUT_SEC = 30.0
ANSWER_KEYS = ("action", "option_id", "text", "findings", "params")


def _invalid(message: str, code: str = "VALIDATION_ERROR") -> ContractError:
    return ContractError(422, code, message)


def _id_list(params: dict[str, Any], key: str) -> tuple[str, ...]:
    value = params.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise _invalid(f"params.{key} must be a list of strings")
    return tuple(value)


class ContractRuns:
    """Everything the ``/v1`` routes do with runs."""

    def __init__(
        self,
        service: RunService,
        log: ContractEventLog,
        llm_factory: LlmFactory,
    ) -> None:
        self.service = service
        self.store = service.store
        self.log = log
        self.llm_factory = llm_factory
        self.publisher = ArtifactPublisher(self.store, service.deliveries, service.sleep)
        self.projector = ContractProjector(
            self.store, log, self.publisher, on_closed=self._stage_closed
        )
        self.projector.listeners.append(service.deliveries.notify)
        self.services: dict[str, Services] = {}
        self._base: Services | None = None
        self._ids: set[str] = set()
        self._locks: dict[str, asyncio.Lock] = {}

    # -- lookup ---------------------------------------------------------------

    def is_contract(self, run_id: str) -> bool:
        if run_id in self._ids:
            return True
        meta = self.service.deliveries.read_meta(run_id)
        if meta and meta.get("api") == "contract":
            self._ids.add(run_id)
            return True
        return False

    def require(self, run_id: str) -> dict[str, Any]:
        """The run record of a contract run (404 ``RUN_NOT_FOUND`` otherwise)."""
        try:
            record = self.store.read_run(run_id)
        except (RunNotFoundError, ValueError) as exc:
            raise ContractError(404, "RUN_NOT_FOUND", "run not found") from exc
        if not self.is_contract(run_id):
            raise ContractError(404, "RUN_NOT_FOUND", "run not found")
        return record

    def services_of(self, run_id: str) -> Services:
        services = self.services.get(run_id)
        if services is None:
            raise ContractError(
                409, "RUN_CLOSED", "the run's model credentials are gone; start a new run"
            )
        return services

    def base_services(self) -> Services:
        """Services without a model of their own; each run swaps in its request's models."""
        if self._base is None:
            injected = self.service._injected  # noqa: SLF001
            if injected is not None:
                base = injected
            else:
                base = build_services(
                    self.service.config, llm=UnconfiguredLLM(), events=self.service,
                    store=self.store,
                )  # fmt: skip
                base.control = self.service.control
            self._base = dataclasses.replace(
                base, events=self.service, control=self.service.control
            )
        return self._base

    def phase(self, run_id: str) -> str:
        return self.projector.timeline(run_id).phase or "drafting"

    # -- create -----------------------------------------------------------------

    async def create(self, req: RunCreateRequest) -> tuple[RunState, bool]:
        """Start (or find) the run of ``req.stage_run_id``; ``True`` when newly created."""
        if req.stage_key != HYPOTHESIS_STAGE:
            raise _invalid(f"stage_key {req.stage_key!r} is not served by this engine")
        topic, domains = self._params(req)
        try:
            events_url = validate_callback_url(req.callback_url, self.service.config.api)
        except CallbackRejected as exc:
            raise _invalid(str(exc)) from exc
        async with self.service._create_lock:  # noqa: SLF001
            existing = self.store.find_by_platform(req.stage_run_id)
            if existing:
                return await self.state(existing), False
            try:
                services = self._services_for_request(req)
            except LLMConfigError as exc:
                raise _invalid(str(exc)) from exc
            run_id = self.store.claim_platform_id(req.stage_run_id, new_run_id())
            if self.store.exists(run_id):
                return await self.state(run_id), False
            self.store.run_dir(run_id).mkdir(parents=True, exist_ok=True)
            self.service.deliveries.write_meta(
                run_id,
                {
                    "api": "contract",
                    "stage_run_id": req.stage_run_id,
                    "platform_run_id": req.platform_run_id,
                    "callback_url": req.callback_url,
                    "events_url": events_url,
                    "created_at": utc_now(),
                },
            )
            self._ids.add(run_id)
            self.services[run_id] = services
            request = RunRequest(
                topic=topic,
                domains=tuple(domains),
                review_mode=req.interaction_mode,
                run_id=run_id,
                platform_run_id=req.stage_run_id,
                budget_usd=float(req.budget_usd),
                review_phase=True,
            )
            await self.service._start(run_pipeline(request, services), run_id)  # noqa: SLF001
        return await self.state(run_id), True

    @staticmethod
    def _params(req: RunCreateRequest) -> tuple[str, list[str]]:
        params = req.inputs.params
        topic = params.get("topic")
        if not isinstance(topic, str) or not TOPIC_MIN <= len(topic.strip()) <= TOPIC_MAX:
            raise _invalid(f"inputs.params.topic must be {TOPIC_MIN} to {TOPIC_MAX} characters")
        domains = params.get("domains", [])
        if (
            not isinstance(domains, list)
            or len(domains) > MAX_DOMAINS
            or not all(isinstance(d, str) for d in domains)
        ):
            raise _invalid(f"inputs.params.domains must be at most {MAX_DOMAINS} strings")
        return topic.strip(), [d.strip() for d in domains if d.strip()]

    def _services_for_request(self, req: RunCreateRequest) -> Services:
        services = services_for_request(self.base_services(), req.models, self.llm_factory)
        if self.service._injected is None:  # noqa: SLF001
            # Literature keys come from the request, never from the environment.
            keys = {name: value.get_secret_value() for name, value in req.integrations.items()}
            literature = build_literature(
                services.config.literature,
                cache_root=services.config.cache_root,
                api_keys=keys,
            )
            services = dataclasses.replace(services, literature=literature)
        return services

    # -- state and events ---------------------------------------------------------

    async def state(self, run_id: str) -> RunState:
        record = self.require(run_id)
        await self.projector.project(run_id)
        record = self.store.read_run(run_id)
        meta = self.service.deliveries.read_meta(run_id) or {}
        gate = record.get("gate") or {}
        question_payload = (
            gates.gate_payload(gates.gate_by_kind(gate["kind"]), self.store.artifacts(run_id))
            if gate.get("status") == "open" and gate["kind"] == gates.SCREEN
            else None
        )
        return run_state(
            run_id,
            record,
            meta,
            self.projector.timeline(run_id),
            self.log.last_seq(run_id),
            self.publisher.latest_id(run_id, "i2h.hypotheses/v1"),
            question_payload,
        )

    async def events(self, run_id: str, after: int, limit: int) -> list[dict[str, Any]]:
        self.require(run_id)
        await self.projector.project(run_id)
        return self.log.read(run_id, after, limit)

    async def project_all(self) -> None:
        for run_id in self.store.list_run_ids():
            if self.is_contract(run_id):
                await self.projector.project(run_id)

    async def _stage_closed(self, run_id: str) -> None:
        await self.service.deliveries.platform_closed(run_id)

    # -- gate answers ---------------------------------------------------------------

    def _lock(self, run_id: str) -> asyncio.Lock:
        return self._locks.setdefault(run_id, asyncio.Lock())

    @staticmethod
    def _normalised(body: GateAnswer) -> dict[str, Any]:
        doc = body.model_dump(mode="json", include=set(ANSWER_KEYS))
        doc["text"] = doc["text"] or ""
        return doc

    def _core_answer(self, gate: dict[str, Any], body: GateAnswer) -> CoreAnswer:
        note = body.text or ""
        if gate["kind"] == gates.HYPOTHESES:
            if body.action == "approve":
                return CoreAnswer(
                    gates.APPROVE,
                    _id_list(body.params, "dropped"),
                    note,
                    _id_list(body.params, "kept"),
                )
            if body.action == "comment":
                if not body.findings:
                    raise _invalid("a comment needs at least one finding")
                return CoreAnswer(gates.REJECT, (), note)
            raise _invalid("an artifact review accepts approve or comment", "GATE_ACTION_INVALID")
        if body.action != "answer":
            raise _invalid("a question gate accepts answer", "GATE_ACTION_INVALID")
        if body.option_id == "drop" and gate["kind"] == gates.SCREEN:
            dropped = _id_list(body.params, "dropped")
            if not dropped:
                raise _invalid("drop needs at least one paper in params.dropped")
            return CoreAnswer(gates.APPROVE, dropped, note)
        if body.option_id == "approve":
            return CoreAnswer(gates.APPROVE, _id_list(body.params, "dropped"), note)
        if body.option_id == "reject":
            return CoreAnswer(gates.REJECT, (), note)
        raise _invalid("option_id is not one of the gate's options", "GATE_OPTION_INVALID")

    async def answer_gate(self, run_id: str, gate_id: str, body: GateAnswer) -> RunState:
        record = self.require(run_id)
        services = self.services_of(run_id)
        normalised = self._normalised(body)
        async with self._lock(run_id):
            record = self.store.read_run(run_id)
            answers: dict[str, Any] = record.get("contract_gate_answers") or {}
            if gate_id in answers:
                if answers[gate_id] != normalised:
                    raise ContractError(409, "GATE_NOT_OPEN", "the gate was answered differently")
                return await self.state(run_id)
            gate = record.get("gate")
            if not gate or gate.get("status") != "open" or gate["gate_id"] != gate_id:
                raise ContractError(409, "GATE_NOT_OPEN", "the gate is not open")
            core = self._core_answer(gate, body)
            if body.action == "comment":
                art = self.store.artifacts(run_id)
                art.write_run_json(
                    f"revision_before.r{int(record['attempt'])}.json",
                    {"artifact_id": self.publisher.latest_id(run_id, "i2h.hypotheses/v1"),
                     "markdown": art.read_text(8, "hypotheses.md")},
                )
                art.write_run_json(
                    f"researcher_findings.r{int(record['attempt'])}.json",
                    [f.model_dump(mode="json") for f in body.findings],
                )
            actors = {**(record.get("contract_gate_actors") or {}), gate_id: body.actor}
            self.store.update_run(
                run_id,
                contract_gate_answers={**answers, gate_id: normalised},
                contract_gate_actors=actors,
            )
            try:
                result = await apply_gate_answer(run_id, gate_id, core, services)
            except gates.GateError as exc:
                self.store.update_run(run_id, contract_gate_answers=answers)
                raise _invalid(str(exc)) from exc
            if result.status is not RunStatus.FAILED:
                self.service.spawn(resume_pipeline(run_id, services))
        return await self.state(run_id)

    # -- control --------------------------------------------------------------------

    async def _settle(self, run_id: str) -> None:
        """Wait until the worker has taken a requested pause or cancel."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SETTLE_TIMEOUT_SEC
        # No event is raised when a worker ends, so poll.
        while self.service.control.is_active(run_id) and loop.time() < deadline:  # noqa: ASYNC110
            await asyncio.sleep(0.01)

    async def pause(self, run_id: str) -> RunState:
        self.require(run_id)
        try:
            await pause_run(run_id, self.service._control_view, "user")  # type: ignore[arg-type]  # noqa: SLF001
        except RunStateError as exc:
            raise ContractError(409, "RUN_CLOSED", "the run is finished") from exc
        await self._settle(run_id)
        return await self.state(run_id)

    async def resume(self, run_id: str) -> RunState:
        record = self.require(run_id)
        status = RunStatus(record["status"])
        if status.terminal:
            raise ContractError(409, "RUN_CLOSED", "the run is finished")
        control = self.service.control
        if control.is_active(run_id):
            control.clear(run_id)  # a pause that was requested but not yet taken
        elif not (
            status is RunStatus.AWAITING_REVIEW
            and (record.get("gate") or {}).get("status") == "open"
        ):
            self.service.spawn(resume_pipeline(run_id, self.services_of(run_id)))
        return await self.state(run_id)

    async def cancel(self, run_id: str) -> RunState:
        record = self.require(run_id)
        status = RunStatus(record["status"])
        if status is not RunStatus.CANCELLED:
            try:
                await cancel_run(run_id, self.service._control_view, "user")  # type: ignore[arg-type]  # noqa: SLF001
            except RunStateError as exc:
                raise ContractError(409, "RUN_CLOSED", "the run is finished") from exc
            await self._settle(run_id)
        return await self.state(run_id)

