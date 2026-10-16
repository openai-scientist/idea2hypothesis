# idea2hypothesis

idea2hypothesis turns a research idea into evidence-linked, falsifiable hypotheses. It runs nine
stages: scope the topic, search real literature, screen it, extract evidence cards, synthesise
research gaps, generate hypotheses and map the argument as a semantic graph and a research canvas.
Every output is written to disk as JSON (plus a Markdown rendering for people) and every claim
points back to a stored record.

The entry points are the HTTP API (used by the Platform backend and Swagger) and the Python
library. See [AGENTS.md](AGENTS.md) for the development rules, commands and checks.

## Origin

Stages 1 to 8 are adapted from [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw)
(ARC) by Aiming Lab (MIT); stage names and numbers follow ARC. The prompts, the literature provider
logic, the ideation memory store, the LLM client and the AWS Bedrock adapter were extracted from ARC
and cleaned up. Stage 9 is new. The ARC package is never imported, and CI fails on legacy ARC names.

## What it does not do

* It never substitutes made-up output. If the model, a literature source or a credential is
  missing, the run fails with a clear error code; there are no template papers, scores or cards.
* Knowledge cards are built from abstracts and say so (`evidence_scope: "abstract"`); unknown fields
  are `null`.
* The novelty report is a heuristic assessment against retrieved papers, not proof of novelty.
* There are no experiment, code-generation or paper-writing stages. The pipeline ends at
  stage 9 (`ARGUMENT_MAP`).

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
| 9 | `ARGUMENT_MAP` | `argument_map.json`, `semantic_graph.json`, `research_canvas.json` |

Pass conditions are enforced in `src/idea2hypothesis/pipeline/contracts.py`.
Review modes: `auto` and `light` have no manual gate (`light` adds advisory quality notes),
`copilot` opens a gate after stage 5 and after stage 8, `full` also gates after stage 2.

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[api,bedrock,dev]"
```

The core package needs only `pyyaml` and `httpx`. Extras (lazy-import rules are in
[AGENTS.md](AGENTS.md)):

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
| name set by `literature.openalex_api_key_env` | optional OpenAlex key; empty by default, so none is read |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Bedrock access keys (alternative to a profile) |
| `I2H_SERVICE_KEY` | key the Platform backend must send as `X-Service-Key` (`api.service_key_env`) |
| `I2H_CALLBACK_KEY` | key sent on webhook deliveries (`api.callback_key_env`; defaults to `I2H_SERVICE_KEY`) |
| `I2H_CONFIG` | config path used by `tools/serve_api.py` and `uvicorn ...:app` |
| `I2H_ENV_FILE` | env file that `tools/serve_api.py` loads instead of `./.env` |
| `HOST`, `PORT` | bind address of `tools/serve_api.py` (default: `api.host`, `api.port`) |
| `DASHBOARD_HOST`, `DASHBOARD_PORT` | bind address of the run viewer (default `127.0.0.1`, `8090`) |
| `I2H_API_URL` | API the run viewer calls (default `http://127.0.0.1:8001`) |
| `I2H_RUNS_DIR` | runs directory the run viewer reads (default `runs`) |

## Run the API

```bash
export I2H_CONFIG=configs/example.yaml            # PowerShell: $env:I2H_CONFIG = "configs/example.yaml"
python tools/serve_api.py                          # honours PORT and HOST; loads ./.env or I2H_ENV_FILE
# or: uvicorn idea2hypothesis.api.app:app --port 8001
```

For a local run with the Platform BE (Docker) and FE use `configs/platform-local.yaml`.

Swagger UI is at `http://127.0.0.1:8001/docs` and the health check at `/api/health`. Besides the
Platform service below, the API has the older `/runs` routes, the `/api/phase1` routes (start all
nine stages) and per-stage `/api/stage1` to `/api/stage8` routes (stage 9 has no per-stage route),
and Bedrock diagnostics.

## Platform service

The Platform backend (`management`) drives the engine through the versioned contract under `/v1`
(`src/idea2hypothesis/api/contract/routes.py`):

| Route | Purpose |
| --- | --- |
| `GET /v1/health` | liveness |
| `GET /v1/capabilities` | engine capabilities |
| `POST /v1/runs` | create a run (201) |
| `GET /v1/runs/{engine_run_id}` | run state |
| `GET /v1/runs/{engine_run_id}/events` | lifecycle and activity events |
| `POST /v1/runs/{engine_run_id}/gates/{gate_id}` | answer a review gate |
| `POST /v1/runs/{engine_run_id}/pause`, `/resume`, `/cancel` | run control |

Callers send the key named by `api.service_key_env` (default `I2H_SERVICE_KEY`) as the
`X-Service-Key` header. The engine delivers webhooks to callback URLs whose host is listed in
`api.callback_allowed_hosts` (empty by default); the callback key comes from `I2H_CALLBACK_KEY`.

The contract is owned by management (`../management/docs/contract/`). Its `schema/`, `examples/` and
README are copied into `tests/contract/vendor/`, with the management commit in
`tests/contract/vendor/SOURCE`. Do not edit the copy by hand; refresh it as described in
[AGENTS.md](AGENTS.md).

### Artifacts, review and send-back

Each delivery of the hypotheses also delivers `i2h.evidence/v1`, a non-primary artifact for the next
stage. It holds the knowledge cards that the delivered hypotheses cite (`evidence_refs`), once each:

```json
{"schema_version": "i2h.evidence/v1",
 "cards": [{"card_id": "card-p-1", "paper_id": "p-1", "title": "...", "year": 2024, "venue": null,
            "doi": null, "url": null, "data": "...", "metrics": "...", "findings": "...",
            "limitations": "...", "hypothesis_ids": ["H1", "H3"]}]}
```

A revision or a dropped hypothesis delivers a new version. The other outputs are the primary
`i2h.hypotheses/v1`, `review.findings/v1`, `revision.response/v1`, `revision.diff/v1` and
`i2h.argument_map/v1`.

The independent review converges. Each finding says when it is due: protocol details that
experiment design fixes (tool versions, seeds, replication counts, sample sizes, compute caps,
dataset revisions, exact fault lists) are `later` and become minor findings with the prefix
`[for experiment design]`. After a researcher comment, the reviewer sees the previous round's
findings and the hypotheses the revision changed (compared by content, by id, against the set the
previous round reviewed). It answers every earlier blocking finding with `resolved` or
`maintained`; a maintained finding stays blocking and names the earlier id. A new blocking finding
on a hypothesis the revision did not change is only major, with the prefix `[unchanged]`.

Capabilities declare `reuses_prior_run = true`. When `inputs.prior_engine_run_id` names a run in the
local store with the same topic and valid stages 1-7, the new run copies those stages, does not
ask their gates again, and starts at stage 8 with the guidance and feedback lines and the earlier
final hypotheses as the set to revise (hypotheses the lines do not concern stay as they are).
Otherwise the run executes every stage, and one activity says why the prior run was not reused.

The run's domain picks the hypothesis roles: the first name in `inputs.params.domains`, then the
request profile's domain, that has a role set (`ml`, `hep`, `biology`, `se`); otherwise
`prompts.domain` of the configuration. `se` is empirical software engineering and uses the `ml`
role names (innovator, pragmatist, contrarian).

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
  stage-01/ ... stage-09/        (each with manifest.json)
  attempts/<n>/                  (outputs superseded by a rejected gate)
```

Resume reuses only valid artifacts of the same run and attempt, never passes an unanswered gate, and
is a no-op once stage 9 has completed.

## Developer tools

* `tools/serve_api.py` starts the API with uvicorn.
* `tools/dashboard/` is a small run viewer (`python tools/dashboard/server.py`, default
  `http://127.0.0.1:8090`). It reads `runs/` and starts runs through the API.

## License

MIT. See [LICENSE](LICENSE), which keeps the original Aiming Lab notice.
