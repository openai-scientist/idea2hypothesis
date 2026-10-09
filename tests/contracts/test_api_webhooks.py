"""Webhook delivery: FIFO with a cursor, bounded retries, 409 stop, resume after restart."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from tests.fixtures.api_helpers import SERVER_KEY, FakePlatform, harness, run_body


@pytest.fixture(autouse=True)
def _server_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("I2H_SERVICE_KEY", SERVER_KEY)


def _cursor(tmp_path: Path, run_id: str) -> dict:
    return json.loads((tmp_path / "runs" / run_id / "delivery.json").read_text(encoding="utf-8"))


async def test_events_are_delivered_in_order_with_the_server_key(tmp_path: Path) -> None:
    platform = FakePlatform()
    async with harness(tmp_path, platform=platform) as h:
        response = await h.client.post(
            "/runs", json=run_body("p-1", review_mode="auto"), headers={"X-Service-Key": SERVER_KEY}
        )
        run_id = response.json()["popper_run_id"]
        await h.settle()
        events = await h.events(run_id)

        assert platform.seqs == list(range(1, len(events) + 1))
        assert platform.received == events  # exactly the persisted envelopes, in order
        assert {r.url.path for r in platform.requests} == {
            "/internal/runs/00000000-0000-0000-0000-000000000001/events"
        }
        assert {r.headers["x-service-key"] for r in platform.requests} == {SERVER_KEY}
        assert max(len(json.loads(r.content)["events"]) for r in platform.requests) <= 100
        cursor = _cursor(tmp_path, run_id)
        assert cursor["delivered_source_seq"] == len(events) and cursor["status"] == "idle"


async def test_the_callers_key_is_never_forwarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    platform = FakePlatform()
    async with harness(tmp_path, platform=platform) as h:
        response = await h.client.post(
            "/runs", json=run_body("p-1", review_mode="auto"), headers={"X-Service-Key": SERVER_KEY}
        )
        assert response.status_code == 201
        await h.settle()
    monkeypatch.delenv("I2H_SERVICE_KEY")  # no server key: nothing to send, caller key unused
    platform2 = FakePlatform()
    async with harness(tmp_path / "second", platform=platform2) as h2:
        response = await h2.client.post(
            "/runs",
            json=run_body("p-1", review_mode="auto"),
            headers={"X-Service-Key": "caller-supplied-key"},
        )
        assert response.status_code == 201
        await h2.settle()
        assert platform2.requests and all(
            "x-service-key" not in r.headers for r in platform2.requests
        )


async def test_separate_callback_key_is_sent_on_deliveries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("I2H_CALLBACK_KEY", "callback-only-key")
    platform = FakePlatform()
    async with harness(tmp_path, platform=platform) as h:
        response = await h.client.post(
            "/runs", json=run_body("p-1", review_mode="auto"), headers={"X-Service-Key": SERVER_KEY}
        )
        assert response.status_code == 201
        await h.settle()
        assert platform.requests and all(
            r.headers["x-service-key"] == "callback-only-key" for r in platform.requests
        )


async def test_retries_then_succeeds_without_loss_or_reordering(tmp_path: Path) -> None:
    platform = FakePlatform(script=[503, 500])
    async with harness(tmp_path, platform=platform) as h:
        run_id = (await h.start("p-1", review_mode="auto")).json()["popper_run_id"]
        await h.settle()
        events = await h.events(run_id)
        assert platform.seqs == list(range(1, len(events) + 1))
        assert len(platform.requests) >= 3
        assert _cursor(tmp_path, run_id)["delivered_source_seq"] == len(events)


async def test_exhausted_retries_keep_the_cursor_and_mark_delivery_failed(tmp_path: Path) -> None:
    platform = FakePlatform(script=[500] * 500)  # the platform stays down (max attempts is 3)
    async with harness(tmp_path, platform=platform) as h:
        run_id = (await h.start("p-1", review_mode="auto")).json()["popper_run_id"]
        await h.settle()
        cursor = _cursor(tmp_path, run_id)
        assert cursor["status"] == "delivery_failed" and cursor["delivered_source_seq"] == 0
        assert platform.received == []
        assert (await h.state(run_id))["status"] == "completed"
        assert len(await h.events(run_id)) > 0  # replayable through GET


async def test_a_409_stops_delivery_and_cancels_the_run(tmp_path: Path) -> None:
    platform = FakePlatform(closed=True)
    async with harness(tmp_path, platform=platform) as h:
        run_id = (await h.start("p-1")).json()["popper_run_id"]
        await h.settle()
        cursor = _cursor(tmp_path, run_id)
        assert cursor["status"] == "stopped_by_platform"
        assert h.service.store.read_run(run_id)["status"] == "cancelled"
        assert len(platform.requests) == 1  # no retry after 409
        state = await h.state(run_id)
        assert state["status"] == "failed" and state["message"] == "Cancelled by user"


async def test_delivery_resumes_from_the_cursor_after_a_restart(tmp_path: Path) -> None:
    platform = FakePlatform()
    # first process: accept the first batch(es), then the platform goes away
    async with harness(tmp_path, platform=platform) as h:
        platform.script = []

        original = platform.handle
        calls = {"n": 0}

        def flaky(request):  # accept only the first request
            calls["n"] += 1
            if calls["n"] > 1:
                return httpx.Response(503)
            return original(request)

        h.service.deliveries._transport = httpx.MockTransport(flaky)
        run_id = (await h.start("p-1", review_mode="auto")).json()["popper_run_id"]
        await h.settle()
        total = len(await h.events(run_id))
        delivered = _cursor(tmp_path, run_id)["delivered_source_seq"]
        assert 0 < delivered < total
        assert _cursor(tmp_path, run_id)["status"] in ("delivery_failed", "active")
    before = list(platform.received)
    assert before and platform.seqs == list(range(1, len(before) + 1))

    # second process: same storage, healthy platform; startup resumes from the cursor
    async with harness(tmp_path, platform=platform) as h2:
        await h2.settle()
        assert platform.seqs == list(range(1, total + 1))  # no gap, no duplicate delivery
        assert platform.received[: len(before)] == before
        assert _cursor(tmp_path, run_id)["delivered_source_seq"] == total


async def test_gate_answers_continue_delivery_in_sequence(tmp_path: Path) -> None:
    platform = FakePlatform()
    async with harness(tmp_path, platform=platform) as h:
        run_id = (await h.start("p-1")).json()["popper_run_id"]
        await h.settle()
        opened = [e for e in platform.received if e["type"] == "gate.opened"]
        assert len(opened) == 1  # the BE learns about the gate through the webhook
        gate_id = opened[0]["payload"]["gate_id"]
        await h.client.post(f"/runs/{run_id}/gates/{gate_id}", json={"option_id": "approve"})
        await h.settle()
        opened = [e for e in platform.received if e["type"] == "gate.opened"]
        assert [e["payload"]["kind"] for e in opened] == ["screen", "hypotheses"]
        gate_id = opened[-1]["payload"]["gate_id"]
        await h.client.post(f"/runs/{run_id}/gates/{gate_id}", json={"option_id": "approve"})
        await h.settle()
        assert platform.types[-1] == "run.completed"
        assert platform.seqs == list(range(1, len(platform.seqs) + 1))
