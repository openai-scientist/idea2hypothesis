# idea2hypothesis

idea2hypothesis turns a research idea into evidence-linked, falsifiable hypotheses. It runs eight
stages: scope the topic, search real literature, screen it, extract evidence cards, synthesise
research gaps and generate hypotheses. Every output is written to disk as JSON (plus a Markdown
rendering for people) and every claim points back to a stored record.

The project was extracted from the stage 1 to 8 part of an earlier autonomous research pipeline;
see [INSPIRE.md](INSPIRE.md) for the attribution.
It has no CLI: the HTTP API (used by the Platform backend and Swagger) and the Python library are
the entry points.

## What it does not do

* It never substitutes made-up output. If the model, a literature source or a credential is
  missing, the run fails with a clear error code; there are no template papers, scores or cards.
* Knowledge cards are built from abstracts and say so (`evidence_scope: "abstract"`); unknown fields
  are `null`.
* The novelty report is a heuristic assessment against retrieved papers, not proof of novelty.
* There are no experiment, code-generation or paper-writing stages. The pipeline ends at
  stage 8.

## Stages

| # | Stage | Main outputs |
| --- | --- | --- |
| 1 | `TOPIC_INIT` | `goal.json`, `goal.md`, `hardware_profile.json` (optional advisory) |
| 2 | `PROBLEM_DECOMPOSE` | `problem_tree.json`, `problem_tree.md`, `topic_evaluation.json` |
| 3 | `SEARCH_STRATEGY` | `search_plan.yaml`, `queries.json`, `sources.json` |
| 4 | `LITERATURE_COLLECT` | `candidates.jsonl`, `references.bib`, `search_meta.json` |
| 5 | `LITERATURE_SCREEN` | `shortlist.jsonl`, `screen_meta.json`, `review.json` |
| 6 | `KNOWLEDGE_EXTRACT` | `cards/*.json`, `cards/*.md`, `knowledge_meta.json` |
| 7 | `SYNTHESIS` | `synthesis.json`, `synthesis.md` |
| 8 | `HYPOTHESIS_GEN` | `hypotheses.json`, `hypotheses.md`, `perspectives/`, `novelty_report.json` |

Contracts and pass conditions are in [docs/stage-contracts.md](docs/stage-contracts.md).
Review modes: `auto` and `light` have no manual gate (`light` adds advisory quality notes),
`copilot` opens a gate after stage 5, `full` also gates after stage 2.

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[api,bedrock,dev]"
```

The core package needs only `pyyaml` and `httpx`. Extras:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `api` | fastapi, uvicorn, pydantic | HTTP API and Swagger |
| `bedrock` | boto3 | `llm.provider: bedrock` |
| `dev` | pytest, pytest-asyncio, ruff, build | development |

## Configure

Copy [configs/example.yaml](configs/example.yaml) and export the variables named in
[.env.example](.env.example) (the package does not read `.env` files itself). Configuration
has eight sections: `research`, `llm`, `literature`, `prompts`, `runtime`, `review`, `storage`,
`api`. Unknown keys and wrong types fail at load time and name the offending key. Secrets are never
written to YAML or to run snapshots: the configuration holds only the names of environment
variables or AWS profiles.

| Variable | Used for |
| --- | --- |
| `I2H_LLM_API_KEY` | OpenAI-compatible provider key (name set by `llm.api_key_env`) |
| `AWS_REGION`, `AWS_PROFILE` | Bedrock region and profile |
| `S2_API_KEY` | optional Semantic Scholar key (`literature.s2_api_key_env`) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Bedrock access keys (alternative to a profile) |
| `I2H_SERVICE_KEY` | key the Platform backend must send as `X-Service-Key` (`api.service_key_env`) |
| `I2H_CALLBACK_KEY` | key sent on webhook deliveries (`api.callback_key_env`; defaults to `I2H_SERVICE_KEY`) |
| `I2H_CONFIG` | config path used by `tools/serve_api.py` and `uvicorn ...:app` |

## Run the API

```bash
export I2H_CONFIG=configs/example.yaml            # PowerShell: $env:I2H_CONFIG = "configs/example.yaml"
python tools/serve_api.py                          # honours PORT and HOST; loads ./.env if present
# or: uvicorn idea2hypothesis.api.app:app --port 8001
```

For a local run with the Platform BE (Docker) and FE use `configs/platform-local.yaml`; see
[docs/platform-integration.md](docs/platform-integration.md#local-setup-with-ai-research-platform-be-in-docker--fe).

Swagger UI is at `http://127.0.0.1:8001/docs` and the health check at `/api/health`. Routes (Platform
`/runs` API, per-stage `/api/phase1` and `/api/stage1` to `/api/stage8` routes, Bedrock
diagnostics), event types and webhook behaviour are described in
[docs/platform-integration.md](docs/platform-integration.md).

## Use as a library

```python
import asyncio

from idea2hypothesis.config import load_config
from idea2hypothesis.pipeline.models import RunRequest
from idea2hypothesis.pipeline.runner import resume_pipeline, run_pipeline
from idea2hypothesis.pipeline.services import build_services

config = load_config("configs/example.yaml")
services = build_services(config)      # raises LLMConfigError if credentials are missing


async def main() -> None:
    result = await run_pipeline(
        RunRequest(topic="Effect of sleep duration on exam performance", review_mode="auto"),
        services,
    )
    print(result.status, result.completed_stages)
    # A run waiting on a gate, paused or interrupted continues with:
    # result = await resume_pipeline(result.run_id, services)


asyncio.run(main())
```

Tests inject fixture LLM and literature ports instead; see `tests/fixtures/`.

## Run output

Runs are written under `storage.runs_root` (default `runs/`, gitignored):

```text
runs/<run-id>/
  run.json  config.snapshot.json  prompts.snapshot.json  checkpoint.json  events.jsonl
  stage-01/ ... stage-08/        (each with manifest.json)
  attempts/<n>/                  (outputs superseded by a rejected gate)
```

Resume reuses only valid artifacts of the same run and attempt, never passes an unanswered gate, and
is a no-op once stage 8 has completed.

## Developer tools

* `tools/serve_api.py` starts the API with uvicorn.
* `tools/dashboard/` is a small run viewer (`python tools/dashboard/server.py`, default
  `http://127.0.0.1:8090`). It reads `runs/` and starts runs through the API. See
  [docs/architecture.md](docs/architecture.md).

## Checks

```bash
python -m ruff check .
python -m pytest -q
python -m build
```

All tests run offline with fixture ports. A live smoke test against a real model and the literature
providers needs your own credentials and is not part of the suite. CI runs the same checks on
Linux and Windows (Python 3.11 and 3.13).

## Documentation

* [docs/architecture.md](docs/architecture.md): layers, data flow, storage, state machine.
* [docs/stage-contracts.md](docs/stage-contracts.md): per-stage inputs, outputs, pass conditions.
* [docs/platform-integration.md](docs/platform-integration.md): HTTP API, events, webhooks.
* [docs/pipeline-stage-guide.md](docs/pipeline-stage-guide.md): what each stage does and why.
* [docs/llm-reliability-benchmark.md](docs/llm-reliability-benchmark.md): rubric for comparing models.
* [docs/migration.md](docs/migration.md): changes from the code this project was extracted from.
* [docs/kalvin-changes.md](docs/kalvin-changes.md): where each KalvinKhanh change now lives.
* [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE), which keeps the original Aiming Lab notice.
