# Migration from the AutoResearchClaw based code

This repository replaces the stage 1 to 8 part of the `researchclaw` package (as it stood at
commit `3bdf8f4` of AutoResearchClaw, including KalvinKhanh's additions). There is no
compatibility layer: no `researchclaw`, `arc` or `sibyl` alias is shipped, and nothing imports
the old package. File-level mapping is in [kalvin-changes.md](kalvin-changes.md).

## Breaking changes

### Package, entry points and install

| Before | Now |
| --- | --- |
| `import researchclaw...` | `import idea2hypothesis...` (`src/` layout, hatchling) |
| `python -m researchclaw run ...`, `researchclaw` command, setup wizard | removed; use the HTTP API or `run_pipeline` / `resume_pipeline` |
| `scripts/phase1_fastapi_server.py`, `start_engine_8001.bat`, `start_fastapi_swagger.bat` | `python tools/serve_api.py` (env `I2H_CONFIG`, `PORT`) or `uvicorn idea2hypothesis.api.app:app` |
| one large dependency set | core: `pyyaml`, `httpx`; extras `api`, `bedrock`, `dev` |
| `arxiv` Python package | dropped; arXiv is queried through its Atom API with `httpx` and `xml.etree` |
| `config.arc.yaml` with many sections | `configs/example.yaml` with eight sections (below) |

### Configuration

Only `research`, `llm`, `literature`, `prompts`, `runtime`, `review`, `storage` and `api` exist.
Unknown keys and wrong types are errors that name the key (for example `llm.timeout_sec`).
Fields without a consumer (experiment, sandbox, Docker/SSH, paper, export, MetaClaw/OpenClaw,
knowledge base, notifications, security and other ARC settings) are gone. Credentials are never
stored: keys are referenced by environment variable name (`llm.api_key_env`,
`literature.s2_api_key_env`, `api.service_key_env`) and Bedrock by region/profile.

| Old | New |
| --- | --- |
| `research.topic`, `research.domains` | `research.topic`, `research.domains`; `research.constraints` added |
| `research.quality_threshold` | `research.min_relevance`, `research.min_quality` (both in `[0, 1]`) |
| hard-coded topic score threshold | `research.min_topic_score` |
| provider presets, ACP agent bridge | `llm.provider` is `openai_compatible` or `bedrock`; other vendors via `llm.base_url` |
| `literature_search.sources` | `literature.sources` (`openalex`, `semantic_scholar`, `arxiv`) |
| mode / HITL settings | `review.mode`: `auto`, `light`, `copilot` (default), `full` |
| `artifacts/` output directory | `storage.runs_root` (default `runs/`) |

### Run layout and ids

| Before | Now |
| --- | --- |
| `artifacts/rc-YYYYMMDD-HHMMSS-<hash>/` | `runs/i2h-YYYYMMDD-HHMMSS-<8 hex>/` |
| `stage-01/decision.json`, `stage_health.json` | `checkpoint.json`, per-stage `manifest.json`, `events.jsonl` |
| `hitl/session.json`, `interventions.jsonl` | gate state in `checkpoint.json` and `run.json`, decisions in `review.json` (`human_review`) |
| engine runs kept in memory | persisted run store; `platform_run_id` idempotency survives restarts |

The Platform wire field `popper_run_id` is kept and carries the new `run_id`. Stage route ids accept
`^[A-Za-z0-9][A-Za-z0-9_-]{2,80}$`, so older ids still address old directories if copied under
`runs/`, but old runs have no checkpoint or manifest and cannot be resumed.

### Behaviour that was removed on purpose

The old stage code and the end-to-end generator could emit content that was not evidence. All of it
is gone; the run fails instead.

* Template goal and problem decomposition when the model was unavailable.
* Default hypotheses (`_default_hypotheses`) and template knowledge cards.
* Placeholder papers when retrieval returned nothing; LLM-invented candidate papers; a web agent
  that added unverified candidates.
* Injection of a fixed list of "seminal" papers into the candidate pool (the seminal data is
  gone; stage 5 no longer has a seminal-paper rule).
* Fabricated screening scores for papers the model did not score, and the minimum shortlist size
  padding (`_MIN_SHORTLIST`). Unscored papers are excluded with a reason.
* The IdeaWorkshop framework in synthesis and the tournament framework in hypothesis generation.
* Automatic PyTorch installation and SSH hardware probing with host-key checking disabled.
  Hardware detection is local and read-only (NVIDIA, Apple MPS or CPU) and runs only with
  `research.hardware_advisory: true`.
* In the Platform engine: random numbers, hard-coded counts (214 / 359 / 145), hard-coded passing
  `rule.checked` events, automatic `scope.approved`, always-`"0.00"` cost, the 30 minute webhook
  drop and unauthenticated inbound calls.

### Results that now exist or changed

* JSON is the contract; Markdown is rendered from it. Added JSON outputs that used to be Markdown
  only: `goal.json`, `problem_tree.json`, `synthesis.json`, `hypotheses.json`; new files:
  `review.json` (a decision and reason for every candidate), `screen_meta.json`,
  `knowledge_meta.json`, `topic_evaluation.json` (already existed, now schema-checked) and
  per-stage `manifest.json`.
* Every JSON artifact has `schema_version`; every paper has `source_records`; cards say
  `evidence_scope` and use `null` for unknown fields; `novelty_report.json` is labelled
  `kind: "novelty_assessment"` with a disclaimer.
* Resume reuses only valid completed stages of the same run and attempt. Previously a finished run
  restarted from stage 1 when resumed; now resume after stage 8 is a no-op.
* Rejecting a gate starts a new attempt and moves the superseded stage outputs to
  `attempts/<n>/`.
* `cost_usd` is a number only when `llm.pricing` is configured; otherwise it is `null` (or absent).
  `budget_usd` is enforced only with pricing.
* The inbound `X-Service-Key` is verified when the key named by `api.service_key_env` is set, and
  webhook callbacks are accepted only for hosts in `api.callback_allowed_hosts`
  (`api.callback_allow_any` opts out). Webhook delivery is persisted with a cursor and retried a
  bounded number of times; failures leave the events replayable through `GET /runs/{id}/events`.
* The phase 1 routes (`/api/phase1/*`, `/api/stage1` to `/api/stage8`) were never mounted in the
  old application; they are mounted now and run the same engine as `/runs`.
* The Platform multipart `/runs` and `/runs/{id}/review` flow is not implemented (it was not before
  either); see [platform-integration.md](platform-integration.md).

### Platform event types

Event types and payload fields that the frontend and backend read are kept, but they are filled
only from real artifacts. Types that only existed to display fabricated data were dropped or
changed. The exact list of kept, changed and dropped event types is maintained in
[platform-integration.md](platform-integration.md).

### Removed subsystems

Experiment design and execution, code generation, iterative refinement, paper writing and review,
LaTeX/Overleaf, Docker/SSH/Colab sandboxes, domain agents, MetaClaw/OpenClaw bridges, cron and
message adapters, the evolution/knowledge-base framework, trends, calendar, voice, collaboration
and project scheduler, MCP server, chat/projects/websocket routes, the website and showcase,
ARC-Bench, tunnel launchers and binaries, scratch inspection scripts and parallel runners.
Experiment and writing memory were removed; only ideation memory remains (its constructor fix is
kept: store, path or `store_dir`).

## Upgrading checklist

1. Install `pip install -e ".[api,bedrock]"` and set the environment variables from
   `.env.example`.
2. Port your YAML to `configs/example.yaml`; fix the errors the loader reports.
3. Replace CLI invocations with calls to `POST /runs` (Platform) or `POST /api/phase1/start`.
4. Point consumers of run files at `runs/<run-id>/stage-NN/` and read the `.json` files.
5. If you relied on fabricated results (demo mode), use `tests/fixtures/` Fixture ports in tests
   only; the engine has no demo mode.
