"""Shared stage helpers: failures, JSON requests with bounded repair, small formatters."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from idea2hypothesis.llm.models import ChatMessage, LLMError, LLMPort, LLMResponse
from idea2hypothesis.llm.parsing import extract_json_object
from idea2hypothesis.pipeline.contracts import Findings
from idea2hypothesis.pipeline.models import StageContext
from idea2hypothesis.prompts.loader import RenderedPrompt
from idea2hypothesis.storage.runs import utc_now

logger = logging.getLogger(__name__)

Validator = Callable[[dict[str, Any]], Findings]
T = TypeVar("T")


class StageFailure(Exception):
    """A stage cannot produce a valid result; the run stops with this code."""

    def __init__(self, code: str, message: str, warnings: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.warnings = warnings


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1)


def bullet_list(items: tuple[str, ...] | list[str], empty: str = "none") -> str:
    return "; ".join(items) if items else empty


async def request_json(
    ctx: StageContext,
    prompt: RenderedPrompt,
    *,
    label: str,
    validate: Validator | None = None,
    llm: LLMPort | None = None,
    max_tokens: int | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Ask for one JSON object; on a parse or contract failure ask the model to repair it.

    Returns ``(data, warnings)``. At most ``runtime.max_retries`` repair rounds are made; the
    final failure raises :class:`StageFailure` ``LLM_OUTPUT_INVALID`` (nothing is fabricated).
    Provider errors propagate to the runner's bounded stage retry.

    Every call is logged with what was sent and what came back (see :func:`_log_call`).
    """
    client = llm or ctx.llm
    messages = [ChatMessage.user(prompt.user)]
    rounds = 1 + max(0, ctx.config.runtime.max_retries)
    problems: list[str] = []
    tokens = max_tokens or prompt.max_tokens
    temperature = _temperature(ctx, prompt, independent=llm is not None and llm is not ctx.llm)
    for attempt in range(rounds):
        await ctx.safe_point()
        sent = list(messages)
        started, clock = utc_now(), time.monotonic()
        try:
            response = await client.chat(
                messages,
                system=prompt.system or None,
                json_mode=prompt.json_mode,
                max_tokens=tokens,
                temperature=prompt.temperature,
            )
        except LLMError as exc:
            _log_call(
                ctx, label, attempt, prompt, sent, tokens, temperature, started, clock,
                outcome="provider_error", problems=[f"{type(exc).__name__}: {exc}"],
            )  # fmt: skip
            raise
        try:
            data = extract_json_object(response.text)
        except ValueError as exc:
            problems = [f"the reply was not a valid JSON object ({exc})"]
        else:
            findings = validate(data) if validate else Findings()
            if findings.ok:
                _log_call(
                    ctx, label, attempt, prompt, sent, tokens, temperature, started, clock,
                    outcome="accepted", response=response, problems=list(findings.warnings),
                )  # fmt: skip
                return data, tuple(findings.warnings)
            problems = findings.errors
        _log_call(
            ctx, label, attempt, prompt, sent, tokens, temperature, started, clock,
            outcome="rejected", response=response, problems=problems,
        )  # fmt: skip
        logger.warning(
            "%s: invalid model output (attempt %d/%d): %s", label, attempt + 1, rounds, problems
        )
        if attempt + 1 < rounds:
            messages = [
                ChatMessage.user(prompt.user),
                ChatMessage.assistant(response.text),
                ChatMessage.user(
                    "Your previous reply was rejected for these reasons:\n- "
                    + "\n- ".join(problems[:12])
                    + "\nReturn the corrected JSON object only."
                ),
            ]
    raise StageFailure(
        "LLM_OUTPUT_INVALID",
        f"{label}: the model did not return a valid answer after {rounds} attempt(s): "
        + "; ".join(problems[:6]),
    )


def _temperature(ctx: StageContext, prompt: RenderedPrompt, *, independent: bool) -> float:
    """The temperature the call runs at: the prompt's own, else the model's configured one."""
    if prompt.temperature is not None:
        return float(prompt.temperature)
    cfg = ctx.config.llm.reviewer if independent and ctx.config.llm.reviewer else ctx.config.llm
    return float(cfg.temperature)


def _log_call(
    ctx: StageContext,
    label: str,
    attempt: int,
    prompt: RenderedPrompt,
    messages: list[ChatMessage],
    max_tokens: int | None,
    temperature: float,
    started: str,
    clock: float,
    *,
    outcome: str,
    response: LLMResponse | None = None,
    problems: list[str] | None = None,
) -> None:
    """Store one model call as sent and as answered, with why it was accepted or rejected.

    ``outcome`` is ``accepted``, ``rejected`` (unparseable or failing the stage contract; a repair
    round follows if any is left) or ``provider_error``. A log that cannot be written never stops
    the stage; the failure is logged instead.
    """
    record: dict[str, Any] = {
        "schema_version": 1,
        "label": label,
        "stage": int(ctx.stage),
        "attempt": ctx.attempt,
        "try": ctx.try_index,
        "round": attempt + 1,
        "outcome": outcome,
        "problems": problems or [],
        "started_at": started,
        "duration_ms": round((time.monotonic() - clock) * 1000),
        "request": {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": prompt.json_mode,
            "system": prompt.system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        },
        "response": None,
    }
    if response is not None:
        record["response"] = {
            "model": response.model,
            "text": response.text,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "cost_usd": response.cost_usd,
            "finish_reason": response.finish_reason,
            "truncated": response.truncated,
        }
    try:
        ctx.artifacts.write_llm_call(int(ctx.stage), ctx.attempt, label, record)
    except OSError:
        logger.exception("could not store the model call log of %s", label)


async def gather_limited(factories: Sequence[Callable[[], Awaitable[T]]], limit: int) -> list[T]:
    """Run coroutine factories with bounded concurrency, results in input order.

    The first exception cancels the remaining work and is re-raised unchanged.
    """
    semaphore = asyncio.Semaphore(max(1, limit))

    async def run(factory: Callable[[], Awaitable[T]]) -> T:
        async with semaphore:
            return await factory()

    tasks = [asyncio.ensure_future(run(f)) for f in factories]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
