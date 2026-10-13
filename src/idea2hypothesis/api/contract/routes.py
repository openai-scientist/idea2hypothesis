"""HTTP routes for the engine wire contract."""

from typing import Annotated

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from idea2hypothesis import __version__
from idea2hypothesis.api.contract.capabilities import capabilities
from idea2hypothesis.api.contract.models import (
    CONTRACT_VERSION,
    Capabilities,
    GateAnswer,
    Health,
    RunCreateRequest,
    RunCreateResponse,
    RunState,
)

router = APIRouter()


def _runs(request: Request):  # noqa: ANN202 - the service module imports this one
    return request.app.state.run_service.contract


@router.get("/health", response_model=Health, response_model_exclude_none=True)
async def health(request: Request) -> Health:
    service = request.app.state.run_service
    return Health(
        status="ok",
        engine_version=__version__,
        contract_version=CONTRACT_VERSION,
        active_runs=sum(not task.done() for task in service._tasks),
    )


@router.get("/capabilities", response_model=Capabilities)
async def get_capabilities() -> Capabilities:
    return capabilities()


@router.post("/runs", response_model=RunCreateResponse, status_code=201)
async def create_run(body: RunCreateRequest, request: Request) -> JSONResponse:
    state, created = await _runs(request).create(body)
    response = RunCreateResponse(engine_run_id=state.engine_run_id, phase=state.phase,
                                 created=created)  # fmt: skip
    return JSONResponse(response.model_dump(mode="json"), status_code=201 if created else 200)


@router.get("/runs/{engine_run_id}", response_model=RunState)
async def get_run(engine_run_id: str, request: Request) -> RunState:
    return await _runs(request).state(engine_run_id)


@router.get("/runs/{engine_run_id}/events")
async def get_events(
    engine_run_id: str,
    request: Request,
    after: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=500)] = 500,
) -> dict[str, list[dict]]:
    return {"events": await _runs(request).events(engine_run_id, after, limit)}


@router.post("/runs/{engine_run_id}/gates/{gate_id}", response_model=RunState)
async def answer_gate(
    engine_run_id: str, gate_id: str, body: GateAnswer, request: Request
) -> RunState:
    return await _runs(request).answer_gate(engine_run_id, gate_id, body)


@router.post("/runs/{engine_run_id}/pause", response_model=RunState)
async def pause_run(engine_run_id: str, request: Request) -> RunState:
    return await _runs(request).pause(engine_run_id)


@router.post("/runs/{engine_run_id}/resume", response_model=RunState)
async def resume_run(engine_run_id: str, request: Request) -> RunState:
    return await _runs(request).resume(engine_run_id)


@router.post("/runs/{engine_run_id}/cancel", response_model=RunState)
async def cancel_run(engine_run_id: str, request: Request) -> RunState:
    return await _runs(request).cancel(engine_run_id)
