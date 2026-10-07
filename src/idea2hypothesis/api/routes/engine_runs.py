"""Platform BE engine routes (``/runs``): start, state, events, gates and run control."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request, Response, status

from idea2hypothesis.api.schemas import (
    EventItem,
    EventsBatchResponse,
    GateAnswerRequest,
    OkResponse,
    RunCreateRequest,
    RunCreateResponse,
    RunStateResponse,
)
from idea2hypothesis.api.service import RunService, ServiceError

router = APIRouter(tags=["Engine Runs (Platform BE Integration)"])


def get_service(request: Request) -> RunService:
    service: RunService = request.app.state.run_service
    return service


@router.post(
    "/runs",
    status_code=status.HTTP_201_CREATED,
    response_model=RunCreateResponse,
    summary="Start a research run (idempotent by platform_run_id)",
)
async def start_run(
    req: RunCreateRequest, response: Response, service: RunService = Depends(get_service)
) -> Any:
    """Starts the 8-stage run. A repeated ``platform_run_id`` returns the existing run (200).

    Events are delivered to ``callback_url`` (must be in ``api.callback_allowed_hosts``) with
    the server's own ``X-Service-Key``.
    """
    state, created = await service.create_platform_run(req)
    if not created:
        response.status_code = status.HTTP_200_OK
    return RunCreateResponse(**{k: v for k, v in state.items() if k != "last_source_seq"})


@router.get(
    "/runs",
    response_model=RunStateResponse,
    summary="Find a run by platform_run_id",
)
async def find_run(
    platform_run_id: str = Query(..., description="Platform run id"),
    service: RunService = Depends(get_service),
) -> Any:
    run_id = service.store.find_by_platform(platform_run_id)
    if run_id is None:
        raise ServiceError(404, "Run with given platform_run_id not found")
    return RunStateResponse(**service.state(run_id))


@router.get(
    "/runs/{popper_run_id}",
    response_model=RunStateResponse,
    summary="Run state by run id or platform_run_id",
)
async def get_run_status(popper_run_id: str, service: RunService = Depends(get_service)) -> Any:
    return RunStateResponse(**service.state(service.resolve(popper_run_id)))


@router.get(
    "/runs/{popper_run_id}/events",
    response_model=EventsBatchResponse,
    summary="Replay events after a source_seq (sync / recovery)",
)
async def get_run_events(
    popper_run_id: str,
    after_source_seq: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=1000),
    service: RunService = Depends(get_service),
) -> EventsBatchResponse:
    run_id = service.resolve(popper_run_id)
    rows = service.events(run_id, after_source_seq, limit)
    return EventsBatchResponse(events=[EventItem(**row) for row in rows])


@router.post(
    "/runs/{popper_run_id}/gates/{gate_id}",
    response_model=OkResponse,
    summary="Submit the human decision for an open gate",
)
async def submit_gate_answer(
    popper_run_id: str,
    gate_id: str,
    answer: GateAnswerRequest,
    service: RunService = Depends(get_service),
) -> OkResponse:
    """``option_id``: ``approve``, ``drop`` (with ``dropped`` paper ids) or ``reject``."""
    run_id = service.resolve(popper_run_id)
    message = await service.answer_gate(run_id, gate_id, answer)
    return OkResponse(message=message)


@router.post(
    "/runs/{popper_run_id}/pause", response_model=OkResponse, summary="Pause at a safe point"
)
async def pause_run(popper_run_id: str, service: RunService = Depends(get_service)) -> OkResponse:
    await service.pause(service.resolve(popper_run_id))
    return OkResponse(message="Run paused")


@router.post(
    "/runs/{popper_run_id}/resume", response_model=OkResponse, summary="Resume a paused run"
)
async def resume_run(popper_run_id: str, service: RunService = Depends(get_service)) -> OkResponse:
    message = await service.resume(service.resolve(popper_run_id))
    return OkResponse(message=message)


@router.post("/runs/{popper_run_id}/cancel", response_model=OkResponse, summary="Cancel a run")
async def cancel_run(popper_run_id: str, service: RunService = Depends(get_service)) -> OkResponse:
    message = await service.cancel(service.resolve(popper_run_id))
    return OkResponse(message=message)
