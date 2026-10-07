"""Robust extraction of JSON objects from model output, and reasoning-tag stripping."""

from __future__ import annotations

import json
import re
from typing import Any

# --- reasoning artefacts -------------------------------------------------

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_THINK_STRAY_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_BRACKET_THINKING_RE = re.compile(
    r"\[thinking\].*?(?=\n\n(?!\[thinking\])|\n(?:#{1,3}\s)|\n```|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_PLAN_BLOCK_RE = re.compile(r"\[plan\].*?(?=\n\n|\Z)", re.DOTALL)


def strip_thinking_tags(text: str) -> str:
    """Remove ``<think>`` blocks and bracket-style reasoning markers.

    Clean input is returned byte-for-byte unchanged.
    """
    if not text:
        return text
    result = text
    if "think" in result.lower():
        result = _THINK_BLOCK_RE.sub("", result)
        result = _THINK_UNCLOSED_RE.sub("", result)
        result = _THINK_STRAY_CLOSE_RE.sub("", result)
    if "[thinking]" in result.lower():
        result = _BRACKET_THINKING_RE.sub("", result)
        result = re.sub(r"^\[thinking\].*$", "", result, flags=re.MULTILINE | re.IGNORECASE)
    if "[plan]" in result.lower():
        result = _PLAN_BLOCK_RE.sub("", result)
    if result == text:
        return result
    return re.sub(r"\n{3,}", "\n\n", result).strip()


# --- JSON extraction -----------------------------------------------------


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract and merge valid top-level JSON objects from model output.

    Handles markdown fences, preamble or trailing commentary, and several objects emitted
    one after another (merged key by key). Raises :class:`ValueError` when nothing parses.
    """
    cleaned = strip_thinking_tags(text).strip()

    if cleaned.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", cleaned)
        stripped = re.sub(r"\s*```$", "", stripped).strip()
        try:
            result = json.loads(stripped)
        except json.JSONDecodeError:
            result = None
        if isinstance(result, dict):
            return result

    decoder = json.JSONDecoder()
    pos = 0
    found: list[dict[str, Any]] = []
    while pos < len(cleaned):
        brace = cleaned.find("{", pos)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(cleaned, idx=brace)
        except json.JSONDecodeError:
            pos = brace + 1
            continue
        if isinstance(obj, dict):
            found.append(obj)
        pos = end

    if found:
        merged: dict[str, Any] = {}
        for obj in found:
            merged.update(obj)
        return merged

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            result = json.loads(match.group(0))
        except json.JSONDecodeError:
            result = None
        if isinstance(result, dict):
            return result

    raise ValueError(f"Could not extract valid JSON from LLM output (length {len(cleaned)})")
