"""Contract routes preserve the wire boundary and legacy behavior."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import jsonschema
import pytest

from idea2hypothesis.api.app import create_app
from tests.conftest import make_services
from tests.fixtures.api_helpers import SERVER_KEY, harness

VENDOR = Path(__file__).parent / "vendor"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("I2H_SERVICE_KEY", SERVER_KEY)
    services = make_services(tmp_path)
    return create_app(services.config, services=services)


async def test_health_liveness_without_key(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://i2h") as c:
        response = await c.get("/v1/health")
        assert response.status_code == 200
        body = response.json()
        jsonschema.validate(body, json.loads((VENDOR / "schema/Health.json").read_text()))
        assert body["status"] == "ok"
        assert "checks" not in body and "last_error" not in body


async def test_capabilities_valid(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://i2h") as c:
        response = await c.get("/v1/capabilities", headers={"X-Service-Key": SERVER_KEY})
        assert response.status_code == 200
        body = response.json()
        jsonschema.validate(body, json.loads((VENDOR / "schema/Capabilities.json").read_text()))
        assert body["engine_key"] == "i2h" and body["auto_revision"] is False
        assert body["stage_keys"] == ["hypothesis"]
        assert body["profiles"] == [{"domain": None, "research_type": None}]
        assert len(body["steps"]) == 9
        assert [o["schema"] for o in body["output_schemas"] if o["primary"]] == [
            "i2h.hypotheses/v1"
        ]


@pytest.mark.parametrize("path", ["/v1/runs", "/v1/capabilities"])
async def test_contract_routes_require_service_key(app, path):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://i2h") as c:
        response = await (c.post(path, json={}) if path.endswith("runs") else c.get(path))
        assert response.status_code == 401
        assert response.json()["code"] == "SERVICE_KEY_INVALID"


async def test_contract_auth_closed_without_configured_key(app, monkeypatch):
    monkeypatch.delenv("I2H_SERVICE_KEY")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://i2h") as c:
        response = await c.get("/v1/capabilities")
        assert response.status_code == 503
        assert response.json()["code"] == "SERVICE_KEY_NOT_CONFIGURED"
        assert (await c.get("/v1/health")).status_code == 200


async def test_legacy_runs_route_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("I2H_SERVICE_KEY", SERVER_KEY)
    async with harness(tmp_path) as h:
        response = await h.start("legacy-contract-mount")
        assert response.status_code == 201
        assert "popper_run_id" in response.json()


def test_vendor_matches_management():
    source = Path(__file__).resolve().parents[3] / "management/docs/contract"
    if not source.is_dir():
        pytest.skip("Adjacent management contract documentation is absent")
    # ``pi/`` is the PI service's contract; engines do not vendor it.
    expected = {
        p.relative_to(source): p.read_bytes()
        for p in source.rglob("*") if p.is_file() and p.relative_to(source).parts[0] != "pi"
    }
    actual = {
        p.relative_to(VENDOR): p.read_bytes()
        for p in VENDOR.rglob("*") if p.is_file() and p.name != "SOURCE"
    }
    assert actual == expected, "Refresh vendor documents and SOURCE from management"
