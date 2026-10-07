"""AWS Bedrock connectivity diagnostics (``/api/health/bedrock``).

Ported from the original server. Credentials may be sent in the request body or come from the
environment; they are never logged and never echoed back (error text is scrubbed). boto3 is
imported lazily (``bedrock`` extra) and called in a worker thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health/bedrock", tags=["AWS Bedrock AI Health Check"])

DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_PROMPT = "Hello! Please confirm AWS Bedrock connectivity with a one-sentence greeting."


class BedrockHealthCheckRequest(BaseModel):
    aws_access_key_id: str | None = Field(
        default=None,
        description="AWS access key id; falls back to the AWS_ACCESS_KEY_ID environment variable",
    )
    aws_secret_access_key: str | None = Field(
        default=None,
        description="AWS secret key; falls back to the AWS_SECRET_ACCESS_KEY environment variable",
    )
    aws_session_token: str | None = Field(
        default=None, description="Session token for temporary credentials (optional)"
    )
    aws_region: str = Field(default="us-east-1", description="AWS region of the model")
    model_id: str = Field(default=DEFAULT_MODEL, description="Bedrock model id")
    test_prompt: str = Field(default=DEFAULT_PROMPT, description="Short prompt for the test call")


class BedrockHealthCheckResponse(BaseModel):
    status: str = Field(description="'ok' or 'error'")
    provider: str = Field(default="AWS Bedrock")
    model_id: str
    region: str
    latency_ms: float
    reply_preview: str
    credentials_source: str
    token_usage: dict[str, Any] | None = None
    troubleshooting_tip: str | None = None
    error: str | None = None


def _resolve_credentials(
    key_id: str | None, secret: str | None, token: str | None, region: str | None
) -> tuple[str | None, str | None, str | None, str, str]:
    env = os.environ
    resolved_region = (
        region or env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION") or "us-east-1"
    )
    if key_id and secret:
        source = "request_body"
    elif env.get("AWS_ACCESS_KEY_ID"):
        source = "environment_vars"
    else:
        source = "aws_default_profile"
    return (
        key_id or env.get("AWS_ACCESS_KEY_ID"),
        secret or env.get("AWS_SECRET_ACCESS_KEY"),
        token or env.get("AWS_SESSION_TOKEN"),
        resolved_region,
        source,
    )


def _scrub(message: str, *secrets: str | None) -> str:
    for secret in secrets:
        if secret:
            message = message.replace(secret, "***")
    return message


def _invoke_bedrock_test(
    key_id: str | None,
    secret: str | None,
    token: str | None,
    region: str,
    model_id: str,
    prompt: str,
) -> tuple[str, dict[str, Any], float]:
    """Call the model once and measure latency (blocking; run in a thread)."""
    import boto3
    from botocore.config import Config

    boto_cfg = Config(
        region_name=region,
        retries={"max_attempts": 2, "mode": "standard"},
        connect_timeout=10,
        read_timeout=30,
    )
    kwargs: dict[str, Any] = {"service_name": "bedrock-runtime", "config": boto_cfg}
    if key_id and secret:
        kwargs["aws_access_key_id"] = key_id
        kwargs["aws_secret_access_key"] = secret
        if token:
            kwargs["aws_session_token"] = token
    client = boto3.client(**kwargs)
    start = time.perf_counter()
    try:
        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 60, "temperature": 0.5},
        )
        latency = round((time.perf_counter() - start) * 1000, 2)
        content = response.get("output", {}).get("message", {}).get("content", [])
        return (content[0].get("text", "") if content else ""), response.get("usage", {}), latency
    except Exception as converse_err:
        text = str(converse_err)
        if "ValidationException" not in text and "unsupported" not in text.lower():
            raise
        logger.info("Converse API unsupported for %s, falling back to invoke_model", model_id)
        lower = model_id.lower()
        if "anthropic" in lower:
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 60,
                "messages": [{"role": "user", "content": prompt}],
            }
        elif "titan" in lower:
            payload = {
                "inputText": prompt,
                "textGenerationConfig": {"maxTokenCount": 60, "temperature": 0.5},
            }
        else:
            raise
        resp = client.invoke_model(modelId=model_id, body=json.dumps(payload))
        latency = round((time.perf_counter() - start) * 1000, 2)
        body = json.loads(resp["body"].read().decode("utf-8"))
        if "anthropic" in lower:
            return body.get("content", [{}])[0].get("text", ""), body.get("usage", {}), latency
        return body.get("results", [{}])[0].get("outputText", ""), {}, latency


def _advice(message: str, model_id: str, region: str) -> str:
    if "AccessDeniedException" in message:
        return (
            f"Access denied: enable model access for '{model_id}' in the AWS console "
            "(Amazon Bedrock > Model access) and check that the IAM policy allows "
            "'bedrock:InvokeModel'."
        )
    if "UnrecognizedClientException" in message or "InvalidClientTokenId" in message:
        return "The access key id does not exist or has been disabled in IAM."
    if "SignatureDoesNotMatch" in message:
        return "The secret access key is incorrect; check it."
    if "ResourceNotFoundException" in message or "ValidationException" in message:
        return (
            f"Model '{model_id}' may not be available in region '{region}'. "
            "Try 'us-east-1' or 'us-west-2'."
        )
    if "EndpointConnectionError" in message:
        return f"Cannot reach the Bedrock endpoint in region '{region}'; check network or VPN."
    return "Check the AWS IAM permissions and the Bedrock quotas of the account."


@router.post(
    "", response_model=BedrockHealthCheckResponse, summary="Check Bedrock connectivity (POST)"
)
async def check_bedrock_health(req: BedrockHealthCheckRequest) -> BedrockHealthCheckResponse:
    """Sends one short prompt to the model and measures latency. Credentials come from the body
    or the environment; with none, the answer is an ``error`` body (not an HTTP error)."""
    key_id, secret, token, region, source = _resolve_credentials(
        req.aws_access_key_id, req.aws_secret_access_key, req.aws_session_token, req.aws_region
    )
    if not key_id or not secret:
        return BedrockHealthCheckResponse(
            status="error",
            model_id=req.model_id,
            region=region,
            latency_ms=0.0,
            reply_preview="",
            credentials_source=source,
            error=(
                "Missing AWS credentials: send aws_access_key_id and aws_secret_access_key "
                "in the body or set them in the environment."
            ),
            troubleshooting_tip="Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.",
        )
    try:
        reply, usage, latency = await asyncio.to_thread(
            _invoke_bedrock_test, key_id, secret, token, region, req.model_id, req.test_prompt
        )
    except Exception as exc:  # noqa: BLE001 - reported in the response body
        message = _scrub(str(exc), key_id, secret, token)
        logger.warning("Bedrock health check failed: %s", type(exc).__name__)
        return BedrockHealthCheckResponse(
            status="error",
            model_id=req.model_id,
            region=region,
            latency_ms=0.0,
            reply_preview="",
            credentials_source=source,
            error=message,
            troubleshooting_tip=_advice(message, req.model_id, region),
        )
    return BedrockHealthCheckResponse(
        status="ok",
        model_id=req.model_id,
        region=region,
        latency_ms=latency,
        reply_preview=reply.strip(),
        credentials_source=source,
        token_usage=usage,
        troubleshooting_tip="Bedrock is reachable; the API key and model id are usable.",
    )


@router.get(
    "",
    response_model=BedrockHealthCheckResponse,
    summary="Check Bedrock with environment credentials (GET)",
)
async def check_bedrock_health_env(
    model_id: str = Query(DEFAULT_MODEL, description="Model id to test"),
    region: str = Query("us-east-1", description="AWS region"),
) -> BedrockHealthCheckResponse:
    return await check_bedrock_health(
        BedrockHealthCheckRequest(model_id=model_id, aws_region=region)
    )


def _list_models(key_id: str, secret: str, token: str | None, region: str) -> dict[str, Any]:
    import boto3
    from botocore.config import Config

    client = boto3.client(
        service_name="bedrock",
        region_name=region,
        config=Config(region_name=region, connect_timeout=10, read_timeout=20),
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        aws_session_token=token or None,
    )
    summaries = client.list_foundation_models().get("modelSummaries", [])
    by_provider: dict[str, list[dict[str, Any]]] = {}
    for m in summaries:
        by_provider.setdefault(m.get("providerName", "Other"), []).append(
            {
                "model_id": m.get("modelId"),
                "model_name": m.get("modelName"),
                "input_modalities": m.get("inputModalities", []),
                "output_modalities": m.get("outputModalities", []),
                "inference_types": m.get("inferenceTypesSupported", []),
            }
        )
    return {
        "region": region,
        "total_models": len(summaries),
        "providers_count": len(by_provider),
        "models_by_provider": by_provider,
    }


@router.get("/models", summary="List the foundation models available in a region")
async def list_bedrock_foundation_models(
    region: str = Query("us-east-1", description="AWS region to list"),
) -> dict[str, Any]:
    key_id, secret, token, resolved, _ = _resolve_credentials(None, None, None, region)
    if not key_id or not secret:
        raise HTTPException(
            status_code=400, detail="No AWS credentials in the environment to reach Bedrock."
        )
    try:
        return await asyncio.to_thread(_list_models, key_id, secret, token, resolved)
    except Exception as exc:  # noqa: BLE001
        message = _scrub(str(exc), key_id, secret, token)
        raise HTTPException(
            status_code=500,
            detail={"error": message, "troubleshooting_tip": _advice(message, "N/A", resolved)},
        ) from exc
