"""Pushing artifacts to the Platform before any event that names their ids."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from difflib import unified_diff
from typing import Any

import httpx

from idea2hypothesis.api.contract.models import ArtifactMeta, RevisionDiff
from idea2hypothesis.api.contract.review import review_findings, revision_response
from idea2hypothesis.api.webhooks import DeliveryManager, callback_key
from idea2hypothesis.pipeline import events as ev
from idea2hypothesis.pipeline import gates
from idea2hypothesis.pipeline.events import Event
from idea2hypothesis.storage.runs import RunStore

logger = logging.getLogger(__name__)

I2H_HYPOTHESES = "i2h.hypotheses/v1"
I2H_EVIDENCE = "i2h.evidence/v1"
REVIEW_FINDINGS = "review.findings/v1"
ARGUMENT_MAP = "i2h.argument_map/v1"
REVISION_RESPONSE = "revision.response/v1"
REVISION_DIFF = "revision.diff/v1"
REGISTRY = "pushed_artifacts.json"
CALLBACK_ARTIFACTS = "/artifacts"
JSON_TYPE = "application/json"


class PushFailed(Exception):
    """An artifact could not be delivered; the events that need its id wait."""

    def __init__(self, message: str, *, closed: bool = False, retryable: bool = True) -> None:
        super().__init__(message)
        self.closed = closed
        self.retryable = retryable


async def push_artifact(
    client: httpx.AsyncClient, url: str, service_key: str | None, meta: ArtifactMeta, content: bytes
) -> str:
    """Upload ``content`` (multipart ``meta`` + ``file``); returns the Platform artifact id."""
    headers = {"X-Service-Key": service_key} if service_key else {}
    response = await client.post(
        url,
        files={
            "meta": (None, meta.model_dump_json(by_alias=True), JSON_TYPE),
            "file": (meta.filename, content, meta.media_type),
        },
        headers=headers,
    )
    if response.status_code == 409 and _code(response) == "STAGE_CLOSED":
        raise PushFailed("the Platform closed the stage run", closed=True, retryable=False)
    if response.status_code != 201:
        retryable = response.status_code >= 500 or response.status_code in (408, 429)
        raise PushFailed(
            f"HTTP {response.status_code}: {response.text[:200]}", retryable=retryable
        )
    return str(response.json()["artifact_id"])


def _code(response: httpx.Response) -> str:
    try:
        return str(response.json().get("code", ""))
    except (ValueError, AttributeError):
        return ""


def _json_bytes(doc: Any) -> bytes:
    return (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


EVIDENCE_CARD_FIELDS = (
    "card_id", "paper_id", "title", "year", "venue", "doi", "url",
    "data", "metrics", "findings", "limitations",
)  # fmt: skip


def evidence_document(
    hypotheses: list[dict[str, Any]], cards: list[dict[str, Any]]
) -> dict[str, Any]:
    """The knowledge cards the published hypotheses cite, each with the hypotheses citing it."""
    by_id = {str(c["card_id"]): c for c in cards if c.get("card_id")}
    cited: dict[str, list[str]] = {}
    for h in hypotheses:
        for ref in h.get("evidence_refs") or []:
            ids = cited.setdefault(str(ref), [])
            if str(h["id"]) not in ids:
                ids.append(str(h["id"]))
    return {
        "schema_version": I2H_EVIDENCE,
        "cards": [
            {**{k: by_id[ref].get(k) for k in EVIDENCE_CARD_FIELDS}, "hypothesis_ids": ids}
            for ref, ids in cited.items()
            if ref in by_id
        ],
    }


class ArtifactPublisher:
    """Uploads the artifacts a run produces and remembers their Platform ids.

    ``pushed_artifacts.json`` (run level, so no stage manifest covers it) lists every upload by
    schema, version and sha256; an identical document is never uploaded twice, also after a
    restart.
    """

    def __init__(
        self,
        store: RunStore,
        deliveries: DeliveryManager,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.store = store
        self.deliveries = deliveries
        self._sleep = sleep

    # -- registry -----------------------------------------------------------

    def entries(self, run_id: str) -> list[dict[str, Any]]:
        doc = self.store.artifacts(run_id).read_run_json(REGISTRY) or {}
        return list(doc.get("artifacts", []))

    def latest_id(self, run_id: str, schema: str) -> str | None:
        found = [e for e in self.entries(run_id) if e["schema"] == schema]
        return str(found[-1]["artifact_id"]) if found else None

    # -- upload ---------------------------------------------------------------

    async def ensure_pushed(
        self,
        run_id: str,
        schema: str,
        content: bytes,
        *,
        filename: str,
        primary: bool = False,
        review_round: int | None = None,
    ) -> str:
        sha = hashlib.sha256(content).hexdigest()
        known = self.entries(run_id)
        for entry in known:
            if entry["schema"] == schema and entry["sha256"] == sha:
                return str(entry["artifact_id"])
        version = 1 + max((int(e["version"]) for e in known if e["schema"] == schema), default=0)
        meta = ArtifactMeta(
            schema=schema, version=version, filename=filename, media_type=JSON_TYPE,
            sha256=sha, primary=primary, review_round=review_round,
        )  # fmt: skip
        artifact_id = await self._upload(run_id, meta, content)
        entry = {
            "schema": schema, "version": version, "artifact_id": artifact_id, "sha256": sha,
            "primary": primary, "review_round": review_round,
        }  # fmt: skip
        self.store.artifacts(run_id).write_run_json(REGISTRY, {"artifacts": [*known, entry]})
        return artifact_id

    async def _upload(self, run_id: str, meta: ArtifactMeta, content: bytes) -> str:
        info = self.deliveries.read_meta(run_id)
        if info is None:
            raise PushFailed("the run has no callback", retryable=False)
        url = info["callback_url"].rstrip("/") + CALLBACK_ARTIFACTS
        api = self.deliveries.api
        failure = ""
        for attempt in range(1, api.delivery_max_attempts + 1):
            try:
                async with self.deliveries.client() as client:
                    return await push_artifact(client, url, callback_key(api), meta, content)
            except PushFailed as exc:
                if not exc.retryable:
                    raise
                failure = str(exc)
            except httpx.HTTPError as exc:
                failure = f"{type(exc).__name__}: {exc}"[:300]
            if attempt < api.delivery_max_attempts:
                await self._sleep(min(2.0 ** (attempt - 1), api.delivery_backoff_max_sec))
        raise PushFailed(failure)

    # -- documents ------------------------------------------------------------

    async def push_hypotheses(self, run_id: str) -> str:
        """The hypotheses, then the evidence cards they cite (a non-primary artifact)."""
        art = self.store.artifacts(run_id)
        doc = art.read_json(8, "hypotheses.json")
        artifact_id = await self.ensure_pushed(
            run_id, I2H_HYPOTHESES, _json_bytes({"schema": I2H_HYPOTHESES, **doc}),
            filename="hypotheses.json", primary=True,
        )  # fmt: skip
        try:
            cards = [
                art.read_json(6, name)
                for name in art.list_files(6)
                if name.startswith("cards/") and name.endswith(".json")
            ]
            await self.ensure_pushed(
                run_id, I2H_EVIDENCE, _json_bytes(evidence_document(doc["hypotheses"], cards)),
                filename="evidence.json",
            )
        except Exception:  # noqa: BLE001 - the evidence is optional; the hypotheses are delivered
            logger.exception("run %s: the evidence artifact was not delivered", run_id)
        return artifact_id

    async def push_review(self, run_id: str) -> dict[str, str]:
        """Hypotheses first (the findings name them), then the findings of the review.

        After a revision round the generator's answer to the researcher's findings goes up too.
        """
        art = self.store.artifacts(run_id)
        hypotheses_id = await self.push_hypotheses(run_id)
        review = art.read_json(8, "review.json")
        round_ = int(review["round"]) - 1
        previous = art.read_run_json(f"revision_before.r{round_}.json")
        if previous and art.exists(8, "revision_responses.json"):
            before = previous["artifact_id"]
            answer = revision_response(
                art.read_json(8, "revision_responses.json"),
                round_=round_, before=before, after=hypotheses_id,
            )  # fmt: skip
            diff = RevisionDiff(
                round=round_, artifact_before=before, artifact_after=hypotheses_id,
                patch="".join(unified_diff(
                    previous["markdown"].splitlines(keepends=True),
                    art.read_text(8, "hypotheses.md").splitlines(keepends=True),
                    fromfile="before/hypotheses.md", tofile="after/hypotheses.md",
                )),
            )
            await self.ensure_pushed(
                run_id, REVISION_DIFF, _json_bytes(diff.model_dump(mode="json")),
                filename="revision_diff.json", review_round=diff.round,
            )
            await self.ensure_pushed(
                run_id, REVISION_RESPONSE, _json_bytes(answer.model_dump(mode="json")),
                filename="revision_response.json", review_round=answer.round,
            )  # fmt: skip
        findings = review_findings(review, artifact_id=hypotheses_id)
        findings_id = await self.ensure_pushed(
            run_id, REVIEW_FINDINGS, _json_bytes(findings.model_dump(mode="json")),
            filename="review_findings.json", review_round=findings.round,
        )  # fmt: skip
        return {"hypotheses": hypotheses_id, "findings": findings_id}

    async def push_final(self, run_id: str) -> dict[str, str]:
        """The approved set (a new version when the researcher dropped or kept hypotheses) and
        the argument map drawn from it."""
        hypotheses_id = await self.push_hypotheses(run_id)
        art = self.store.artifacts(run_id)
        argument_map = art.read_json(9, "argument_map.json")
        await self.ensure_pushed(
            run_id, ARGUMENT_MAP, _json_bytes({"schema": ARGUMENT_MAP, **argument_map}),
            filename="argument_map.json",
        )  # fmt: skip
        return {"hypotheses": hypotheses_id}

    async def before(self, run_id: str, event: Event) -> dict[str, str]:
        """Upload what ``event`` refers to; returns the ids the mapping puts on the wire."""
        if event.type == ev.REVIEW_COMPLETED:
            return await self.push_review(run_id)
        if event.type == ev.RUN_COMPLETED:
            return await self.push_final(run_id)
        if event.type == ev.GATE_OPENED and event.data.get("kind") == gates.HYPOTHESES:
            return {"hypotheses": await self.push_hypotheses(run_id)}
        return {}
