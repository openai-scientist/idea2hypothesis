# Platform integration (HTTP adapter)

The HTTP adapter lives in `idea2hypothesis.api` and needs the `api` extra
(`pip install "idea2hypothesis[api]"`). The core engine never imports it.

```text
uvicorn idea2hypothesis.api.app:app      # config file from $I2H_CONFIG (default configs/example.yaml)
```

or in code: `create_app(config)` with `config = load_config(path)`.

Both route groups (`/runs...` for Platform BE and `/api/phase1`, `/api/stageN` for review in
Swagger) drive **one engine** (`pipeline.runner`) through `api.service.RunService`. A run started
from either side can be inspected, paused, resumed or gated from the other.

## Configuration (`api:` section)

| Key | Default | Meaning |
| --- | --- | --- |
| `host`, `port` | `127.0.0.1`, `8001` | Used by `tools/serve_api.py`. |
| `service_key_env` | `I2H_SERVICE_KEY` | Env var holding the key callers must send (BE `POPPER_SERVICE_KEY`). |
| `callback_key_env` | `I2H_CALLBACK_KEY` | Env var holding the key sent on webhooks (BE `POPPER_CALLBACK_KEY`); falls back to the service key when unset. |
| `callback_allowed_hosts` | `[]` | Hosts (or `host:port`) accepted in `callback_url`. Empty rejects every callback. |
| `callback_allow_any` | `false` | Skip the allow-list (development only). |
| `delivery_max_attempts` | `5` | Attempts per batch before delivery is marked `delivery_failed`. |
| `delivery_backoff_max_sec` | `60` | Cap of the exponential back-off (1 s, 2 s, 4 s, ...). |
| `cors_origins` | `[]` | Allowed CORS origins. |

## Authentication

- If the environment variable named by `api.service_key_env` is set and non-empty, every request
  to `/runs*`, `/api/phase1*`, `/api/stage1..8*` and `/api/health/bedrock*` must carry a matching
  `X-Service-Key` (constant-time comparison); otherwise the answer is `401`.
  `/api/health`, `/docs`, `/redoc` and `/openapi.json` stay open. The Bedrock diagnostics are
  protected because they can use the server's AWS credentials.
- If the variable is unset, nothing is enforced (development).
- Outbound webhooks send `X-Service-Key` with the value of `api.callback_key_env` (or, when that
  variable is unset, `api.service_key_env`). A key supplied by a caller is never stored or forwarded.

## Local setup with ai-research-platform (BE in Docker + FE)

`configs/platform-local.yaml` is ready for this setup: Bedrock LLM, API on `0.0.0.0:8001` (the BE
container calls `POPPER_BASE_URL=http://host.docker.internal:8001`) and callbacks allowed to
`localhost`, `127.0.0.1` and `host.docker.internal` (the BE builds callback URLs from
`PUBLIC_BASE_URL`, default `http://localhost:8000`).

1. Create `.env` in the repository root (gitignored) from `.env.example`:
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` (or `AWS_PROFILE`),
   `I2H_SERVICE_KEY` = BE `POPPER_SERVICE_KEY`, `I2H_CALLBACK_KEY` = BE `POPPER_CALLBACK_KEY`.
2. Start the engine: `I2H_CONFIG=configs/platform-local.yaml python tools/serve_api.py`
   (PowerShell: `$env:I2H_CONFIG = "configs/platform-local.yaml"; python tools/serve_api.py`).
   `tools/serve_api.py` loads `.env` without overriding variables already set.
3. Check `GET http://localhost:8001/api/health` and, with the service key header,
   `GET /api/health/bedrock`.
4. Start the BE and FE as usual and create a run from the FE. Engine state is in `runs/<run-id>/`
   (`events.jsonl`, `platform_events.jsonl`, `delivery.json` for webhook progress).

## Engine runs (Platform BE) - no prefix

| Method | Route | Body / query | Response |
| --- | --- | --- | --- |
| POST | `/runs` | `RunCreateRequest` | `201` `RunCreateResponse` (new) or `200` (same `platform_run_id` again) |
| GET | `/runs?platform_run_id=` | | `RunStateResponse`, `404` if unknown |
| GET | `/runs/{id}` | `id` = `popper_run_id` or `platform_run_id` | `RunStateResponse` |
| GET | `/runs/{id}/events` | `after_source_seq>=0`, `limit 1..1000` | `{"events": [EventItem]}` |
| POST | `/runs/{id}/gates/{gate_id}` | `GateAnswerRequest` | `{"status":"ok","message":...}` |
| POST | `/runs/{id}/pause` | | `{"status":"ok","message":"Run paused"}`; `409 RUN_FINISHED` |
| POST | `/runs/{id}/resume` | | `{"status":"ok",...}`; `409 RUN_FINISHED` |
| POST | `/runs/{id}/cancel` | | `{"status":"ok",...}`; cancelling twice is `200`; `409 RUN_FINISHED` on completed/failed |

Schemas (`api/schemas.py`):

- `RunCreateRequest`: `platform_run_id` (str), `topic` (12-1000 chars), `domains` (<= 6),
  `review_mode` (`auto|light|copilot|full`, default `copilot`), `budget_usd` (decimal > 0, default
  `5.00`), `callback_url`.
- `RunCreateResponse`: `popper_run_id` (= engine `run_id`), `status`, `cost_usd`, `message`.
- `RunStateResponse`: the same plus `last_source_seq`. `status` is one of `running`, `paused`,
  `awaiting_review`, `completed`, `failed`. A cancelled run is reported as `failed` with message
  `Cancelled by user` (the existing consumer has no `cancelled` status).
- `cost_usd` is a decimal string **only when the configured pricing priced at least one call**;
  otherwise it is `null` (never `"0.00"`). `budget_usd` is enforced when a cost is known: the run
  pauses with reason `budget_exceeded`. With no `llm.pricing` there is no monetary cap.
- `GateAnswerRequest`: `option_id` = `approve`, `drop` (needs `dropped`: paper ids from the
  shortlist) or `reject`; optional `note`. Re-sending the same option for an answered gate is a
  `200` ("already resolved"), a different option is `409 GATE_NOT_OPEN`, an unknown gate id `404`,
  invalid ids or options `422`.

Persistence: run state, the platform-id index, Platform events and the delivery cursor live under
`storage.runs_root`; a restarted process serves the same ids and event sequence. On startup every run
left `running` is set to `paused` (`interrupted`, event `run.status paused`) and is **not**
restarted automatically; call `POST /runs/{id}/resume`.

`callback_url` must be http(s) with a host in `api.callback_allowed_hosts`, otherwise `422`.
If the engine services cannot be built (for example `I2H_LLM_API_KEY` is unset) `POST /runs`
answers `503 LLM_NOT_CONFIGURED` and creates nothing.

### Review modes and gates

| Mode | Gates |
| --- | --- |
| `auto`, `light` | none (`light` adds `advisories` to `stage.completed`) |
| `copilot` | `screen` gate after stage 5, `hypotheses` gate after stage 8 |
| `full` | `scope` gate after stage 2, `screen` gate after stage 5, `hypotheses` gate after stage 8 |

`reject` at the screen gate reruns stages 3-9 (attempt + 1, new gate id `gate-s05-a2`);
`reject` at the scope gate reruns stages 1-9; `reject` at the hypotheses gate reruns stages 8-9.
A gate approved before the rollback point is not asked again. Approving never bypasses
validation or an empty shortlist.

The `hypotheses` gate (`gate-s08-a<attempt>`) carries `droppable` (hypothesis ids), `keepable`
(candidates the set leaves out, held back or not used, by thread id, such as `T2`), `hypotheses`,
`held_back`, `not_used` and `stage_key` (the UI group the answer belongs to). Its answer is
`approve`, `drop` with `dropped` and/or `kept`, or `reject` with a note; `kept` ids that were not
set aside answer 422. Each left-out candidate is also sent as `idea.set_aside` with its reason and `kind` (`held_back` or `not_used`), and
a merged hypothesis carries `merge_note` (candidate ids written as their threads).

## Event stream

Envelope: `{source_seq, type, stage_key, actor, payload}`. `source_seq` is contiguous from 1 per
run. Events are persisted to `runs/<id>/platform_events.jsonl` before delivery, built only from
real core events and the artifacts of the stage that produced them (`api/platform_events.py`).

UI groups (`stage_key`): `scope` = stages 1-2, `search` = 3-4, `screen` = 5, `read` = 6,
`synthesize` = 7, `r1-hypothesize` = 8 (the key the run studio reads; `hypothesize` is the UI
`stage` value in `run.plan`/`stage.started`), `map` = 9.

### Parts of a stage are sent as soon as they exist

Long stages announce each part of their result once it is persisted (core `stage.progress`), and
the parts are translated and delivered at once instead of when the stage completes:

| Stage | Sent while it runs |
| --- | --- |
| 4 collect | `literature.request` + `literature.batch` after each query (planned queries only) |
| 5 screen | `screen.criteria` and the `prefiltered` rejections first, then per scored batch its `screen.scored` points and `screen.rejected` papers; `screen.kept` follows the final, capped shortlist |
| 6 read | one `card.extracted` per card |
| 8 hypothesize | `debate.turn` threads, one per hypothesis of each perspective; in each debate round every role's critiques as soon as they are written, then every challenged role's answers; `hypothesis.drafted` and `rule.checked` before the novelty check |

When the stage completes only what was not sent yet follows, so no part is sent twice. If a
transient model error restarts stage 5, 6, 7 or 8, its `stage.started` is sent again: the run
studio starts that group over and the parts are sent again. Cards, scored batches and
perspectives are kept per attempt under `runs/<id>/partial/`, so a paused or restarted stage
reuses them instead of asking the model again.

`agent.message` (`message_id`, `text`, `done`) says what the current step is doing and, when it
ends, what it produced. A line with the same `message_id` replaces the previous one; `done: true`
closes it before its `step.completed`. Every number and name in it comes from the run's records
(queries, scored batches, cards, artifacts); nothing is invented.

### Emitted types and their sources

| Type | Built from |
| --- | --- |
| `run.started`, `run.plan` | run record and review mode (`has_gate` per group) |
| `stage.started`, `step.started`, `step.completed`, `stage.completed` | core stage events; `summary` from stage artifacts, plus `warnings`/`advisories` |
| `scope.profile` | `hardware_profile.json` (only with `research.hardware_advisory`) |
| `scope.goal` | `goal.json` (fields title/problem/objective/scope/success) |
| `scope.approved` | the scope gate approval (`full` mode only) |
| `problem.subquestion`, `problem.risk`, `topic.evaluated` | `problem_tree.json`, `topic_evaluation.json` (the rating is also sent when it stops the run; `basis` says it is the model's judgement before any literature search) |
| `search.strategy`, `search.query`, `search.sources` | `search_plan.yaml`, `queries.json`, `sources.json`; `delay_ms` is `literature.inter_query_delay_sec` |
| `literature.request`, `literature.batch`, `literature.collected` | `search_meta.json` per-query hits and totals; expansion queries have no cell, so `literature.collected.expansion` gives their count and hits |
| `literature.merged` | `candidates.jsonl`: one per paper found as several source records, each record with its own `citations` and `has_doi`; `kept` / `kept_record` name the record whose metadata was kept (most citations, then the longer abstract) |
| `screen.criteria`, `screen.scored`, `screen.rejected`, `screen.kept` | `review.json`, `shortlist.jsonl` (one point per scored paper, every rejection with its reason; `decision` is `rejected`, `below_cutoff`, `unscored` or `prefiltered`; `screen.criteria.candidates` is the number of candidates) |
| `card.extracted` | `cards/*.json` (`id` is the card id used by clusters and gaps; unknown fields are empty strings and listed in `unknown_fields`; `quotes` maps each filled field to the abstract passages that back it, checked word for word) |
| `synthesis.cluster`, `.tension`, `.gap`, `.overview`, `.ranked` | `synthesis.json`; from synthesis schema 2 a tension also has `id` (`X1`) and `sides` (`claim`, `card_ids`) |
| `synthesis.set_aside` | `synthesis.json` `set_aside`: cards in no cluster, as `{aside: {id, card_ids, reason}}`; with the clusters they account for every card |
| `debate.turn` | `perspectives/*.json`: each hypothesis of a perspective is a thread (`about`: `T1`, `M2`, `S1` for the Theorist, Methodologist and Skeptic) opened by a `propose` turn. With `llm.debate_rounds`, a round's critiques (`phase: "critique"`) become `challenge` and `concede` turns whose `reply_to` is the proposal they answer; then each author's answers (`phase: "answer"`) follow: a revision is a `refine` turn with the new statement, `note` (the author's reason) and `answers` (ids of the challenges it answers, the first also in `reply_to`); a defence is a `defend` turn and a withdrawal a `concede` turn by the author, each replying to the challenge it answers; `added` hypotheses are new `propose` turns. A turn's `answers` are the only link from an answer to a challenge; a `refine` without them was not written in reply to a challenge. Every challenge carries `severity` (`fatal` or `caveat`) and, when fatal, `flaw`, `field` and for `already_established` `card_id`. Each critic's review (`phase: "review"`) follows: a `concede` turn when the answer resolves its objection, a `challenge` turn at the severity that still stands when it does not, both with `answers` naming the challenge and `reply_to` the answer they review; then its critique of the hypotheses added that round. Engine perspective names in the text (innovator, pragmatist, contrarian) are replaced by the agents' names. The judge's ranking from `debate_record.json` is a `test` turn by `pi` (id `judge`, `about` `Verdict`) |
| `hypothesis.drafted`, `hypothesis.selected` | `hypotheses.json` (`falsify.zone` derived from the validated `prediction`; for `≈ 0` the zone is `[null, null]` and `falsify.within` gives the band `[-margin, margin]` the result must land in; `merge_note` on a hypothesis merged from several threads; `tension_ids` lists the synthesis tensions the hypothesis settles; `from` the debate threads it is built from; `contested` and `caveats` the objections that still stand, each with `by` the agent, `about` the thread, `severity`, `flaw`, `text`; `kept_by_reviewer` when kept at the gate). A candidate kept at the hypotheses gate is sent as `hypothesis.drafted` and `hypothesis.selected` after `gate.resolved` |
| `idea.set_aside` | each held-back candidate of `hypotheses.json`, with the objection that holds it back |
| `hypothesis.checked` | `novelty_report.json` (`novelty` only; heuristic) |
| `rule.checked` | rule 5, only after `hypotheses.json` passed the falsification contract |
| `map.node`, `map.edge` | `semantic_graph.json`: `{node}` is an entity, `{edge}` a relation; steps `questions`, `foundation`, `reasoning` and `contribution` each send their entities, then the relations they complete, so an edge never arrives before both its ends |
| `canvas.piece` | `research_canvas.json`, one `{piece}` per canvas piece (step `canvas`) |
| `gate.opened`, `gate.resolved` | core gate events; `gate.opened` carries `gate_id`, `kind`, `droppable`, `options`, `summary`, `stop_index/stop_total` and the shortlist / scope data |
| `run.status` | `awaiting_review` (before `gate.opened`), `running` (resume), `paused` (with `reason`), `failed` (with `reason` incl. error code, also for cancel) |
| `run.completed` | final usage; `cost_usd` only when priced |
| `agent.message` | progress and artifacts of the current step (see above) |

### Dropped or changed relative to the previous engine

| Previous type | Now | Reason |
| --- | --- | --- |
| `agent.message` | built from the run's records | the old text was generated narration; the new lines only report what a step is doing and what it produced |
| `skills.loaded`, `scope.estimate`, `scope.adjusted` | dropped | hardcoded / no source |
| `estimate.checked`, `rule.checked` (rule 6, rule 5 always `pass`) | dropped / only real rule 5 | were unconditional |
| `idea.set_aside` | dropped | no source in stage 8 output |
| `scope.approved` | only after a human scope approval | was emitted automatically |
| `hypothesis.checked.feasibility` | omitted | the engine does not assess feasibility; the run studio shows the novelty card only when both parts exist (a studio change is needed to show novelty alone) |
| `screen.scored.points` | real 0-1 scores | were random |
| `literature.collected` numbers | real totals | were hardcoded 359/214/145 |
| `gate.opened` id `gate-1` | `gate-s05-a<attempt>` / `gate-s02-a<attempt>` | ids are attempt-specific |
| cancel | `run.status failed` reason `cancelled: ...` | consumer has no cancelled status |

## Webhook delivery

- One task per run reads `platform_events.jsonl` in order and POSTs `{"events": [...]}`
  (<= 100 per request) to `<callback_url>/events` (a URL already ending in `/events` is used as is)
  with the callback key as `X-Service-Key`.
- The cursor `delivered_source_seq` is stored in `runs/<id>/delivery.json` after every accepted
  batch; delivery is at-least-once (consumers drop `source_seq <= last`).
- Network errors and non-2xx answers are retried with exponential back-off, at most
  `delivery_max_attempts` times in a row; then the state is `delivery_failed` and the cursor stays.
  New events start a fresh attempt; the events remain available through `GET /runs/{id}/events`.
- `409` from the consumer means the run is closed there: delivery stops (`stopped_by_platform`) and
  the run is cancelled.
- On startup delivery resumes from the cursor for every run that is behind.
- Delivery never blocks or fails the research run.

## Stage review routes (Swagger)

All routes take `run_id` matching `^[A-Za-z0-9][A-Za-z0-9_-]{2,80}$` (`400` otherwise, `404` if
unknown). Reads return Markdown/YAML text by default; `?format=json` returns the JSON artifact.

| Group | Routes |
| --- | --- |
| `0. Phase 1` | `POST /api/phase1/start` (`topic`, `domains`, `llm_provider`, `model`, `quality_threshold` (ignored), `auto_approve`), `GET /api/phase1/status`, `POST /api/phase1/stop`, `GET /api/phase1/runs`, `GET /api/phase1/runs/{id}/summary`, `.../checkpoint`, `.../health-overview`, `DELETE /api/phase1/runs/{id}` |
| Stage 1 | `POST /api/stage1/run` (creates the run), `GET\|PUT /api/stage1/{id}/goal`, `GET .../hardware` |
| Stage 2 | `POST /api/stage2/{id}/run`, `GET\|PUT .../problem-tree`, `GET .../evaluation` |
| Stage 3 | `POST .../run`, `GET\|PUT .../plan` (YAML), `GET\|PUT .../queries` (list of strings), `GET .../sources` |
| Stage 4 | `POST .../run`, `GET .../candidates?limit=`, `GET .../download-bibtex`, `GET .../references-text`, `GET .../stats` |
| Stage 5 | `POST .../run?auto_approve=`, `GET\|PUT .../shortlist`, `POST .../approve?reason=` |
| Stage 6 | `POST .../run`, `GET .../cards`, `.../cards-merged`, `.../cards/{card_name}` |
| Stage 7 | `POST .../run`, `GET\|PUT .../synthesis` |
| Stage 8 | `POST .../run`, `GET\|PUT .../hypotheses`, `GET .../novelty`, `.../perspectives`, `.../perspectives/{filename}` |
| Every stage | `GET /api/stageN/{id}/decision`, `GET /api/stageN/{id}/health` (from the checkpoint and events; `404` until the stage ran) |

Behaviour that differs from the previous (never mounted) implementation:

- `POST /api/stageN/{id}/run` executes exactly that stage through `runner.execute_stage`, then leaves
  the run `paused` with reason `stage_run`; `POST /runs/{id}/resume` continues the remaining stages.
  Rerunning a stage archives its outputs and the later ones under `attempts/<n>/`.
  Missing inputs answer `409 MISSING_INPUT`; a failed stage answers `200` with
  `status: "failed"` and an `error` object.
- `PUT` bodies are `{"content": "<json text>"}`: the JSON object (or a part, merged over the stored
  document) of the artifact being edited; the Markdown is re-rendered. The result is validated with
  the stage contract; violations give `422 {"detail": {"message", "errors": [...]}}` and nothing is
  changed. A successful edit rewrites the stage manifest, archives and invalidates every later stage
  (attempt + 1) and records the edit in `run.json` (`human_edits`). `PUT .../queries` takes a JSON
  list of strings (new queries get strategy `manual`, linked to all sub-questions).
  `PUT .../shortlist` takes full shortlist rows; removed papers become `dropped_by_reviewer`.
  Edits are refused while the run executes (`409 RUN_BUSY`) and on finished Platform runs.
- `approve` answers the open screen gate (`409 GATE_NOT_OPEN` otherwise) and continues the run.
- `auto_approve=false` on `phase1/start` uses `copilot`; on stage 5 it switches an `auto`/`light`
  run to `copilot` so the gate opens when the run resumes.
- `decision` returns `PASSED|FAILED|AWAITING_REVIEW|APPROVED|REJECTED`; the old `decision.json` and
  `stage_health.json` files no longer exist.
- Runs are no longer limited to one at a time.
- `llm_provider` accepts `bedrock` or `openai`; `gemini` is not supported (`422`).

## Bedrock diagnostics (`/api/health/bedrock`)

| Method | Route | Notes |
| --- | --- | --- |
| POST | `/api/health/bedrock` | `BedrockHealthCheckRequest` (`aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`, `aws_region`, `model_id`, `test_prompt`); credentials from the body or `AWS_*` variables |
| GET | `/api/health/bedrock?model_id=&region=` | environment credentials |
| GET | `/api/health/bedrock/models?region=` | foundation-model listing grouped by provider; `400` without credentials |

Answers are `BedrockHealthCheckResponse` (`status` `ok|error`, `latency_ms`, `reply_preview`,
`credentials_source`, `token_usage`, `troubleshooting_tip`, `error`). A failed call is an HTTP
`200` with `status: "error"`; secrets are never logged and are scrubbed from error text. boto3 runs
in a worker thread (`bedrock` extra).

## Not supported

- Platform BE's multipart `POST /runs` (research markdown + dataset), `POST /runs/{id}/review`
  and `review_sequence` frame reviews. They belong to the experiment/paper phases that are not part of
  this engine; the JSON `POST /runs` is the only start path.
- Routes of the original server for chat, projects, voice, pipeline control and websockets.
- Multi-process deployment: state is filesystem based and a run has one worker in one process.
