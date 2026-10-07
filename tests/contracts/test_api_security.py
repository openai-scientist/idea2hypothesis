"""Inbound service key, callback allow-list, app startup and the OpenAPI surface."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml

from tests.fixtures.api_helpers import CALLBACK, SERVER_KEY, harness, run_body


async def test_service_key_is_enforced_on_protected_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("I2H_SERVICE_KEY", SERVER_KEY)
    async with harness(tmp_path, send_key=False) as h:
        protected = [
            ("POST", "/runs"),
            ("GET", "/runs/x"),
            ("GET", "/runs/x/events"),
            ("POST", "/runs/x/pause"),
            ("GET", "/api/phase1/runs"),
            ("POST", "/api/phase1/start"),
            ("GET", "/api/stage3/abc/queries"),
            ("GET", "/api/health/bedrock"),
            ("GET", "/api/health/bedrock/models"),
        ]
        for method, path in protected:
            response = await h.client.request(method, path, json={} if method == "POST" else None)
            assert response.status_code == 401, (method, path)
        wrong = await h.client.get("/runs/x", headers={"X-Service-Key": "nope"})
        assert wrong.status_code == 401
        right = await h.client.get("/runs/x", headers={"X-Service-Key": SERVER_KEY})
        assert right.status_code == 404  # authenticated, then simply unknown

        for open_path in ("/api/health", "/docs", "/openapi.json"):
            assert (await h.client.get(open_path)).status_code == 200, open_path


async def test_no_key_configured_leaves_routes_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("I2H_SERVICE_KEY", raising=False)
    async with harness(tmp_path, send_key=False) as h:
        assert (await h.client.get("/runs/x")).status_code == 404
        assert (await h.client.get("/api/phase1/runs")).status_code == 200


async def test_callback_urls_are_checked_against_the_allow_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("I2H_SERVICE_KEY", raising=False)
    async with harness(tmp_path) as h:
        for bad in (
            "http://evil.example/hook",
            "ftp://platform.test/x",
            "platform.test/x",
            "http://169.254.169.254/latest",
            "http:///nohost",
        ):
            response = await h.client.post("/runs", json=run_body("p-bad", callback_url=bad))
            assert response.status_code == 422, bad
        assert h.service.store.list_run_ids() == []

        ok = await h.client.post(
            "/runs", json=run_body("p-ok", review_mode="auto", callback_url=CALLBACK + "/events")
        )
        assert ok.status_code == 201
        await h.settle()
        urls = {str(r.url) for r in h.platform.requests}
        assert urls == {CALLBACK + "/events"}  # not doubled


async def test_empty_allow_list_rejects_every_callback_unless_allow_any(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("I2H_SERVICE_KEY", raising=False)
    async with harness(tmp_path, api={"callback_allowed_hosts": []}) as h:
        response = await h.client.post("/runs", json=run_body("p-1"))
        assert response.status_code == 422 and "callback_allowed_hosts" in response.text
    async with harness(tmp_path / "any", api={"callback_allow_any": True}) as h2:
        response = await h2.client.post(
            "/runs", json=run_body("p-1", review_mode="auto", callback_url="http://other.test/x")
        )
        assert response.status_code == 201
        await h2.settle()


async def test_openapi_lists_engine_stage_and_bedrock_routes(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        paths = (await h.client.get("/openapi.json")).json()["paths"]
        expected = [
            "/runs",
            "/runs/{popper_run_id}",
            "/runs/{popper_run_id}/events",
            "/runs/{popper_run_id}/gates/{gate_id}",
            "/runs/{popper_run_id}/pause",
            "/runs/{popper_run_id}/resume",
            "/runs/{popper_run_id}/cancel",
            "/api/health",
            "/api/health/bedrock",
            "/api/health/bedrock/models",
            "/api/phase1/start",
            "/api/phase1/status",
            "/api/phase1/stop",
            "/api/phase1/runs",
            "/api/phase1/runs/{run_id}/summary",
            "/api/phase1/runs/{run_id}/checkpoint",
            "/api/phase1/runs/{run_id}/health-overview",
            "/api/phase1/runs/{run_id}",
            "/api/stage1/run",
            "/api/stage1/{run_id}/goal",
            "/api/stage1/{run_id}/hardware",
            "/api/stage2/{run_id}/problem-tree",
            "/api/stage2/{run_id}/evaluation",
            "/api/stage3/{run_id}/plan",
            "/api/stage3/{run_id}/queries",
            "/api/stage3/{run_id}/sources",
            "/api/stage4/{run_id}/candidates",
            "/api/stage4/{run_id}/download-bibtex",
            "/api/stage4/{run_id}/references-text",
            "/api/stage4/{run_id}/stats",
            "/api/stage5/{run_id}/shortlist",
            "/api/stage5/{run_id}/approve",
            "/api/stage6/{run_id}/cards",
            "/api/stage6/{run_id}/cards-merged",
            "/api/stage6/{run_id}/cards/{card_name}",
            "/api/stage7/{run_id}/synthesis",
            "/api/stage8/{run_id}/hypotheses",
            "/api/stage8/{run_id}/novelty",
            "/api/stage8/{run_id}/perspectives",
            "/api/stage8/{run_id}/perspectives/{filename}",
        ]
        for n in range(1, 9):
            expected += [f"/api/stage{n}/{{run_id}}/decision", f"/api/stage{n}/{{run_id}}/health"]
            if n > 1:
                expected.append(f"/api/stage{n}/{{run_id}}/run")
        for path in expected:
            assert path in paths, path
        # the chat / projects / voice routes of the original server are gone
        assert not [p for p in paths if "chat" in p or "projects" in p or "voice" in p]


async def test_lazy_app_reads_the_config_named_by_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from idea2hypothesis.api import app as app_module

    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"storage": {"runs_root": str(tmp_path / "r")}}), encoding="utf-8"
    )
    monkeypatch.setenv("I2H_CONFIG", str(path))
    lazy = app_module._LazyApp()
    built = lazy.build()
    async with built.router.lifespan_context(built):
        transport = httpx.ASGITransport(app=lazy)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            health = await client.get("/api/health")
            assert health.status_code == 200 and health.json()["status"] == "ok"
            assert (await client.get("/docs")).status_code == 200
    assert (tmp_path / "r").is_dir()
    assert isinstance(app_module.app, app_module._LazyApp)


async def test_missing_llm_credentials_give_a_clear_503_not_a_fake_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from idea2hypothesis.api.app import create_app
    from idea2hypothesis.config import load_config

    monkeypatch.delenv("I2H_SERVICE_KEY", raising=False)
    monkeypatch.delenv("I2H_LLM_API_KEY", raising=False)
    config = load_config(
        {
            "storage": {"runs_root": str(tmp_path / "runs")},
            "api": {"callback_allowed_hosts": ["platform.test"]},
        }
    )
    app = create_app(config)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/api/health")).status_code == 200
            response = await client.post("/runs", json=run_body("p-1"))
            assert response.status_code == 503
            assert response.json()["detail"]["code"] == "LLM_NOT_CONFIGURED"
            assert "I2H_LLM_API_KEY" in response.json()["detail"]["message"]
            found = await client.get("/runs", params={"platform_run_id": "p-1"})
            assert found.status_code == 404
    assert list((tmp_path / "runs").glob("*")) == []  # nothing claimed, nothing created


async def test_finished_platform_runs_cannot_be_edited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("I2H_SERVICE_KEY", raising=False)
    async with harness(tmp_path) as h:
        run_id = (await h.start("p-1", review_mode="auto")).json()["popper_run_id"]
        await h.settle()
        shortlist = (await h.client.get(f"/api/stage5/{run_id}/shortlist")).json()
        response = await h.client.put(f"/api/stage5/{run_id}/shortlist", json=shortlist[:-1])
        assert response.status_code == 409 and response.json()["detail"]["code"] == "RUN_FINISHED"
        assert (await h.client.delete(f"/api/phase1/runs/{run_id}")).status_code == 409
