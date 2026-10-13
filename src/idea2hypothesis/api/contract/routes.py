"""HTTP routes for the engine wire contract."""

from fastapi import APIRouter, Request

from idea2hypothesis import __version__
from idea2hypothesis.api.contract.capabilities import capabilities
from idea2hypothesis.api.contract.models import CONTRACT_VERSION, Capabilities, Health

router = APIRouter()


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
