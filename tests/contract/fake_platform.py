"""A minimal Platform for the engine contract tests.

It serves the three callback routes of a stage run (events, artifacts, telemetry), enforces the
published rules (service key, contiguous ``source_seq``, ``EVENT_GAP``, ``STAGE_CLOSED``,
``SHA256_MISMATCH``) and validates every body it receives against the vendored JSON Schema.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from typing import Any

import httpx
import jsonschema

VENDOR = Path(__file__).parent / "vendor"
ROUTE = re.compile(r"^/internal/engines/(?P<engine>[^/]+)/runs/(?P<run>[^/]+)/(?P<kind>[a-z]+)$")
TERMINAL = ("run.published", "run.failed", "run.cancelled")


def schema(name: str) -> dict[str, Any]:
    return json.loads((VENDOR / "schema" / f"{name}.json").read_text(encoding="utf-8"))


def validate(name: str, document: Any) -> None:
    jsonschema.Draft202012Validator(schema(name)).validate(document)


@dataclass
class StoredArtifact:
    artifact_id: str
    meta: dict[str, Any]
    content: bytes


@dataclass
class FakePlatform:
    service_key: str = "server-secret-key"
    expected: dict[str, int] = field(default_factory=dict)
    events: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    artifacts: dict[str, list[StoredArtifact]] = field(default_factory=dict)
    telemetry: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    bodies: list[bytes] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)
    closed: set[str] = field(default_factory=set)
    #: ``stage_run_id`` to a ``delivered_seq``: the next events call answers EVENT_GAP and the
    #: Platform forgets everything after that sequence number.
    gaps: dict[str, int] = field(default_factory=dict)
    next_artifact: int = 1

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    # -- views ----------------------------------------------------------------

    def types(self, run: str) -> list[str]:
        return [e["type"] for e in self.events.get(run, [])]

    def phases(self, run: str) -> list[str]:
        """Phase sequence: every ``run.phase_changed`` plus the phase of a terminal event."""
        out: list[str] = []
        for event in self.events.get(run, []):
            is_phase = event["type"] == "run.phase_changed" or event["type"] in TERMINAL
            if is_phase and (not out or out[-1] != event["phase"]):
                out.append(event["phase"])
        return out

    def uploaded(self, run: str, schema_id: str) -> list[StoredArtifact]:
        return [a for a in self.artifacts.get(run, []) if a.meta["schema"] == schema_id]

    # -- handler --------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(request.content)
        match = ROUTE.match(request.url.path)
        if request.method != "POST" or match is None:
            return _error(404, "NOT_FOUND", "no such callback route")
        if request.headers.get("X-Service-Key") != self.service_key:
            return _error(401, "SERVICE_KEY_INVALID", "invalid service key")
        run, kind = match["run"], match["kind"]
        if run in self.closed:
            return _error(409, "STAGE_CLOSED", "the stage run is closed")
        if kind == "events":
            return self._events(run, request)
        if kind == "artifacts":
            return self._artifact(run, request)
        if kind == "telemetry":
            return self._telemetry(run, request)
        return _error(404, "NOT_FOUND", "no such callback route")

    def _events(self, run: str, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        validate("EventBatch", body)
        expected = self.expected.setdefault(run, 1)
        if run in self.gaps:
            delivered = self.gaps.pop(run)
            self.expected[run] = delivered + 1
            self.events[run] = [e for e in self.events.get(run, []) if e["source_seq"] <= delivered]
            return httpx.Response(
                409,
                json={"code": "EVENT_GAP", "message": "gap", "delivered_seq": delivered},
            )
        accepted = skipped = 0
        for event in body["events"]:
            if event["source_seq"] < expected:
                skipped += 1
            elif event["source_seq"] > expected:
                return httpx.Response(
                    409,
                    json={"code": "EVENT_GAP", "message": "gap", "delivered_seq": expected - 1},
                )
            else:
                self.events.setdefault(run, []).append(event)
                expected += 1
                accepted += 1
        self.expected[run] = expected
        return httpx.Response(200, json={"accepted": accepted, "skipped": skipped})

    def _artifact(self, run: str, request: httpx.Request) -> httpx.Response:
        header = b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n"
        message = BytesParser(policy=default).parsebytes(header + request.content)
        parts = {p.get_param("name", header="content-disposition"): p for p in message.iter_parts()}
        meta = json.loads(parts["meta"].get_payload(decode=True))
        validate("ArtifactMeta", meta)
        content = parts["file"].get_payload(decode=True)
        if hashlib.sha256(content).hexdigest() != meta["sha256"]:
            return _error(422, "SHA256_MISMATCH", "sha256 does not match the file")
        if meta["schema"] == "review.findings/v1":
            validate("ReviewFindings", json.loads(content))
        artifact_id = f"art-{self.next_artifact}"
        self.next_artifact += 1
        self.artifacts.setdefault(run, []).append(StoredArtifact(artifact_id, meta, content))
        return httpx.Response(201, json={"artifact_id": artifact_id})

    def _telemetry(self, run: str, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        validate("TelemetryBatch", body)
        self.telemetry.setdefault(run, []).extend(body["records"])
        return httpx.Response(200, json={"accepted": len(body["records"])})


def _error(status: int, code: str, message: str) -> httpx.Response:
    return httpx.Response(status, json={"code": code, "message": message})
