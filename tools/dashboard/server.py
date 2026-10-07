"""Developer dashboard server: static UI, live ``data.js`` and a "run phase 1" proxy.

Developer tool, not part of the ``idea2hypothesis`` package and not a research CLI. It reads
``runs/`` from disk and starts runs through the HTTP API (``POST /api/phase1/start``); it never
runs the engine itself.

Environment (all optional)::

    I2H_RUNS_DIR      runs directory, default ./runs
    I2H_API_URL       API base URL, default http://127.0.0.1:8001
    I2H_SERVICE_KEY   sent to the API as X-Service-Key when set (stays on the server side)
    DASHBOARD_HOST    default 127.0.0.1
    DASHBOARD_PORT    default 8090
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import build_data

DASHBOARD_DIR = Path(__file__).resolve().parent
BLOCKED_SUFFIXES = (".py", ".pyc")
LOG_TAIL = 40

SETTINGS: dict[str, Any] = {
    "runs_dir": build_data.default_runs_dir(),
    "api_url": os.environ.get("I2H_API_URL", "http://127.0.0.1:8001").rstrip("/"),
    "active_run_id": None,
}


def _format_event(event: dict[str, Any]) -> str:
    data = event.get("data") or {}
    detail = data.get("message") or data.get("code") or data.get("reason") or data.get("kind") or ""
    stage = f" stage {event['stage']}" if event.get("stage") else ""
    suffix = f": {detail}" if detail else ""
    return f"[{event.get('timestamp', '')}] {event.get('type', '')}{stage}{suffix}"


def current_run_id() -> str | None:
    if SETTINGS["active_run_id"]:
        return SETTINGS["active_run_id"]
    runs = build_data.list_runs(SETTINGS["runs_dir"])
    return runs[0]["run_id"] if runs else None


def status_payload() -> dict[str, Any]:
    run_id = current_run_id()
    record = build_data._json(SETTINGS["runs_dir"] / run_id / "run.json") if run_id else None
    if not isinstance(record, dict):
        return {"run_id": run_id, "status": None, "is_running": False, "error": None}
    err = record.get("error")
    return {
        "run_id": run_id,
        "topic": record.get("topic"),
        "status": record["status"],
        "is_running": record["status"] == "running",
        "current_stage": record.get("current_stage"),
        "error": f"{err.get('code')}: {err.get('message')}" if err else None,
    }


def logs_payload() -> dict[str, Any]:
    run_id = current_run_id()
    if not run_id:
        return {"logs": []}
    events = build_data._jsonl(SETTINGS["runs_dir"] / run_id / "events.jsonl")
    return {"logs": [_format_event(e) for e in events[-LOG_TAIL:]]}


def start_run(topic: str, domains: list[str]) -> tuple[int, dict[str, Any]]:
    """Forward to ``POST {I2H_API_URL}/api/phase1/start``."""
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("I2H_SERVICE_KEY")
    if key:
        headers["X-Service-Key"] = key
    body = json.dumps({"topic": topic, "domains": domains}).encode("utf-8")
    request = urllib.request.Request(
        f"{SETTINGS['api_url']}/api/phase1/start", data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - local API
            payload = json.loads(response.read().decode("utf-8") or "{}")
            status = response.status
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        return exc.code, {"error": f"API returned HTTP {exc.code}: {detail}"}
    except (urllib.error.URLError, OSError) as exc:
        return 502, {"error": f"cannot reach the API at {SETTINGS['api_url']}: {exc}"}
    if isinstance(payload, dict) and payload.get("run_id"):
        SETTINGS["active_run_id"] = payload["run_id"]
    return status, payload if isinstance(payload, dict) else {"result": payload}


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/status":
            return self._json(200, status_payload())
        if parsed.path == "/api/logs":
            return self._json(200, logs_payload())
        if parsed.path == "/api/runs":
            runs = build_data.list_runs(SETTINGS["runs_dir"])
            return self._json(200, [{k: v for k, v in r.items() if k != "mtime"} for r in runs])
        if parsed.path == "/data.js":
            query = urllib.parse.parse_qs(parsed.query)
            run_id = (query.get("run") or [None])[0] or current_run_id()
            data = build_data.build_data(SETTINGS["runs_dir"], run_id)
            body = build_data.render_js(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None
        if parsed.path.endswith(BLOCKED_SUFFIXES):
            return self._json(404, {"error": "not found"})
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/run-phase-1":
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._json(400, {"error": "invalid JSON body"})
            topic = str(payload.get("topic", "")).strip()
            if not topic:
                return self._json(400, {"error": "topic is required"})
            domains = [str(d) for d in payload.get("domains") or []]
            status, result = start_run(topic, domains)
            return self._json(status, result)
        if parsed.path == "/api/rebuild-data":
            out = build_data.write_data_js(SETTINGS["runs_dir"], current_run_id())
            return self._json(200, {"status": "rebuilt", "file": str(out)})
        return self._json(404, {"error": "not found"})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="idea2hypothesis developer dashboard")
    parser.add_argument("--host", default=os.environ.get("DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", "8090")))
    parser.add_argument("--runs-dir", type=Path, default=SETTINGS["runs_dir"])
    parser.add_argument("--api-url", default=SETTINGS["api_url"])
    args = parser.parse_args(argv)
    SETTINGS["runs_dir"] = args.runs_dir.resolve()
    SETTINGS["api_url"] = args.api_url.rstrip("/")
    with http.server.ThreadingHTTPServer((args.host, args.port), Handler) as httpd:
        print(f"dashboard on http://{args.host}:{args.port}  runs={SETTINGS['runs_dir']}")
        print(f"phase 1 runs are started via {SETTINGS['api_url']}/api/phase1/start")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
