"""Bedrock diagnostics routes with boto3 mocked (no network, no credentials needed)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import boto3
import pytest

from tests.fixtures.api_helpers import SERVER_KEY, harness

SECRET = "wJalrXUtnFEMI/FAKE/secretkey"
KEY_ID = "AKIAFAKEKEYID000000"


class FakeClient:
    """Records constructor kwargs; ``converse`` and ``list_foundation_models`` are scripted."""

    created: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        FakeClient.created.append(kwargs)
        self.converse_error: Exception | None = None

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.last = kwargs
        return {
            "output": {"message": {"content": [{"text": " Hello from the fake model. "}]}},
            "usage": {"inputTokens": 7, "outputTokens": 9},
        }

    def list_foundation_models(self) -> dict[str, Any]:
        return {
            "modelSummaries": [
                {
                    "providerName": "Anthropic",
                    "modelId": "anthropic.claude-x",
                    "modelName": "Claude X",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "inferenceTypesSupported": ["ON_DEMAND"],
                },
                {"providerName": "Amazon", "modelId": "amazon.titan-y", "modelName": "Titan Y"},
            ]
        }


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("I2H_SERVICE_KEY", SERVER_KEY)
    FakeClient.created = []
    monkeypatch.setattr(boto3, "client", lambda **kw: FakeClient(**kw))


async def test_health_check_with_credentials_in_the_body(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        response = await h.client.post(
            "/api/health/bedrock",
            json={
                "aws_access_key_id": KEY_ID,
                "aws_secret_access_key": SECRET,
                "aws_region": "us-west-2",
                "model_id": "anthropic.claude-x",
            },
        )
        body = response.json()
        assert response.status_code == 200
        assert body["status"] == "ok" and body["reply_preview"] == "Hello from the fake model."
        assert body["credentials_source"] == "request_body" and body["region"] == "us-west-2"
        assert body["token_usage"] == {"inputTokens": 7, "outputTokens": 9}
        assert body["latency_ms"] >= 0
        assert FakeClient.created[0]["aws_access_key_id"] == KEY_ID
        assert SECRET not in response.text  # secrets are never echoed


async def test_health_check_reads_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", KEY_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
    async with harness(tmp_path) as h:
        body = (await h.client.get("/api/health/bedrock", params={"model_id": "m"})).json()
        assert body["status"] == "ok" and body["credentials_source"] == "environment_vars"


async def test_missing_credentials_are_reported_in_the_body(tmp_path: Path) -> None:
    async with harness(tmp_path) as h:
        response = await h.client.get("/api/health/bedrock")
        body = response.json()
        assert response.status_code == 200 and body["status"] == "error"
        assert (
            body["credentials_source"] == "aws_default_profile" and "Missing AWS" in body["error"]
        )
        assert FakeClient.created == []


async def test_provider_errors_are_diagnosed_and_scrubbed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Denied(FakeClient):
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(f"AccessDeniedException for key {KEY_ID} secret {SECRET}")

    monkeypatch.setattr(boto3, "client", lambda **kw: Denied(**kw))
    async with harness(tmp_path) as h:
        response = await h.client.post(
            "/api/health/bedrock",
            json={
                "aws_access_key_id": KEY_ID,
                "aws_secret_access_key": SECRET,
            },
        )
        body = response.json()
        assert body["status"] == "error" and "AccessDenied" in body["error"]
        assert SECRET not in response.text and "Model access" in body["troubleshooting_tip"]


async def test_invoke_model_fallback_for_models_without_converse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Legacy(FakeClient):
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("ValidationException: unsupported by converse")

        def invoke_model(self, **kwargs: Any) -> dict[str, Any]:
            self.invoked = kwargs
            payload = {"content": [{"text": "legacy reply"}], "usage": {"input_tokens": 1}}
            return {"body": io.BytesIO(json.dumps(payload).encode())}

    monkeypatch.setattr(boto3, "client", lambda **kw: Legacy(**kw))
    async with harness(tmp_path) as h:
        body = (
            await h.client.post(
                "/api/health/bedrock",
                json={
                    "aws_access_key_id": KEY_ID,
                    "aws_secret_access_key": SECRET,
                    "model_id": "anthropic.claude-old",
                },
            )
        ).json()
        assert body["status"] == "ok" and body["reply_preview"] == "legacy reply"


async def test_model_listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with harness(tmp_path) as h:
        assert (await h.client.get("/api/health/bedrock/models")).status_code == 400
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", KEY_ID)
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET)
        response = await h.client.get("/api/health/bedrock/models", params={"region": "eu-west-1"})
        body = response.json()
        assert response.status_code == 200
        assert body["region"] == "eu-west-1" and body["total_models"] == 2
        assert body["providers_count"] == 2
        assert body["models_by_provider"]["Anthropic"][0]["model_id"] == "anthropic.claude-x"
