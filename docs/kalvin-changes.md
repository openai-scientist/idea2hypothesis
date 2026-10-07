# KalvinKhanh changes and where they live now

This repository starts from the AutoResearchClaw tree at commit `3bdf8f4` plus two uncommitted
files of the same author at that time (a robust JSON cleaner in `llm_pipeline_generator.py` and
`scratch/test_parse_robust.py`). The table maps every changed file that is still in scope to its
new location and the test that checks the behaviour. Files that were out of scope or only existed
to patch other repositories are listed with the reason. Backups of the original history exist
outside this repository.

Commits (author KalvinKhanh):

| Commit | Summary |
| --- | --- |
| `c55d06d` | Phase 1 FastAPI/Swagger suite, AWS Bedrock integration, dashboard, stage guide |
| `9e5fe83` | Engine Runs API (`/runs`) for the Platform backend |
| `3f49fe2` | End-to-end 6-group runner, LLM generator, event/frontend contract, reliability docs, scratch cases |
| `730d883` | Reliability benchmark and pipeline architecture documentation update |
| `f5450ee` | Reduction to the 8-stage idea to hypothesis flow, ideation memory constructor fix |
| `3bdf8f4` | Structure clean-up (removed legacy frontend, paper-writing samples) |
| `5ce5408` | Merge of the Platform webhook runner branch (`engine_runs.py`, launcher) |

`src/idea2hypothesis/api/*` paths refer to the Platform/Swagger layer; its tests are the HTTP
contract tests under `tests/contracts/` (files named `test_api_*.py`).

## LLM and Bedrock

| Commit | Old file | New location | Behaviour kept | Test |
| --- | --- | --- | --- | --- |
| `c55d06d` | `researchclaw/llm/bedrock_adapter.py` | `src/idea2hypothesis/llm/bedrock.py` | Bedrock Converse client, throttling and credential errors mapped to typed errors; boto3 optional and called off the event loop | `tests/unit/test_llm.py` (`test_bedrock_*`) |
| `c55d06d` | `researchclaw/llm/client.py`, `llm/__init__.py` | `src/idea2hypothesis/llm/openai_compatible.py`, `llm/chain.py`, `llm/retry.py`, `llm/factory.py`, `llm/models.py` | OpenAI-compatible calls, retry/backoff, model fallback chain, reasoning-tag stripping on by default (upstream fix `be4ba47`, carried because the client was reworked in the same files); provider presets and ACP bridge dropped | `tests/unit/test_llm.py` |
| uncommitted | `llm_pipeline_generator._clean_json_markdown` | `src/idea2hypothesis/llm/parsing.py` (`extract_json_object`) | fence stripping, several top-level objects merged, trailing commentary tolerated | `tests/unit/test_parsing.py` (`test_baseline_regression_extra_data_between_objects`) |

## API and Platform backend

| Commit | Old file | New location | Behaviour kept | Test |
| --- | --- | --- | --- | --- |
| `c55d06d` | `server/app.py` | `src/idea2hypothesis/api/app.py` | FastAPI app with Swagger, CORS | `tests/contracts/test_api_*.py` |
| `c55d06d` | `server/routes/phase1.py` (never mounted) | `src/idea2hypothesis/api/routes/stages.py` | all `/api/phase1/*` and `/api/stage1` to `/api/stage8` routes, now mounted and backed by the real engine | `tests/contracts/test_api_*.py` |
| `c55d06d`, `3f49fe2` | `server/routes/bedrock.py` | `src/idea2hypothesis/api/routes/bedrock.py` | Bedrock health, connectivity and model listing | `tests/contracts/test_api_*.py` |
| `9e5fe83`, `3f49fe2`, `5ce5408` | `server/routes/engine_runs.py` | `src/idea2hypothesis/api/routes/engine_runs.py`, `api/service.py`, `api/webhooks.py`, `storage/runs.py` | `/runs` contract, `platform_run_id` idempotency, event replay, gates, pause/resume/cancel, webhook batches. State is now persisted; delivery has a cursor, bounded retry and an allow-list; `X-Service-Key` is verified | `tests/contracts/test_api_*.py`, `tests/integration/test_gates_and_resume.py` |
| `9e5fe83` | `engine_runs._legacy_pipeline` | dropped | dead code with hard-coded events | n/a |

## Pipeline and research logic

| Commit | Old file | New location | Behaviour kept / changed | Test |
| --- | --- | --- | --- | --- |
| `3f49fe2` | `pipeline/full_runner.py` | `src/idea2hypothesis/pipeline/runner.py` plus `src/idea2hypothesis/api/platform_events.py` | the Platform event stream is produced from real artifacts of the shared engine; the simulated progress, random scores and hard-coded counts were removed | `tests/integration/test_full_run.py`, `tests/contracts/test_api_*.py` |
| `3f49fe2` | `pipeline/llm_pipeline_generator.py` | `src/idea2hypothesis/stages/*.py`, `prompts/stages.yaml` | prompt rules kept: out-of-scope topic guard, topic evaluation, short queries, false-friend rejection, falsifiability, quantified estimand, directional diversity, distinct novelty and rationale, evidence lineage. Generated papers and random assembly removed | `tests/unit/test_prompts.py` (`test_reliability_rules_are_in_the_prompts`), `tests/contracts/test_stage_contracts.py` |
| `3bdf8f4` | `pipeline/full_pipeline_data.json` | dropped | unreferenced static mock payload | n/a |
| `f5450ee` | `pipeline/stages.py` | `src/idea2hypothesis/pipeline/models.py` | exactly stages 1 to 8 | `tests/integration/test_full_run.py` |
| `f5450ee` | `pipeline/contracts.py` | `src/idea2hypothesis/pipeline/contracts.py` | per-stage input/output contracts, now with JSON validators | `tests/contracts/test_stage_contracts.py` |
| `f5450ee` | `pipeline/executor.py`, `pipeline/runner.py` | `pipeline/runner.py`, `pipeline/gates.py`, `pipeline/control.py` | 8-stage execution, gates, resume; resume after stage 8 is a no-op | `tests/integration/test_gates_and_resume.py`, `tests/integration/test_stops_and_failures.py` |
| `f5450ee` | removed `_analysis`, `_code_generation`, `_execution`, `_experiment_design`, `_paper_writing`, `_review_publish` | not ported | stages 9+ are out of scope | n/a |

## Memory

| Commit | Old file | New location | Behaviour kept | Test |
| --- | --- | --- | --- | --- |
| `f5450ee` | `memory/ideation_memory.py` | `src/idea2hypothesis/memory/ideation.py` (+ `store.py`, `retriever.py`) | constructor accepts a store, a path or `store_dir`; topic outcomes and anti-patterns recorded after a run | `tests/unit/test_memory.py` (`test_init_with_store_dir`, `test_init_requires_a_store_or_directory`), `tests/integration/test_options.py` (`test_ideation_memory_is_recorded_and_recalled`) |
| `f5450ee` | `memory/experiment_memory.py`, `memory/writing_memory.py` | dropped | experiment and writing memory are out of scope | `test_only_ideation_memory_exists` |
| `f5450ee` | `tests/test_memory_system.py` | `tests/unit/test_memory.py` | store, retriever, decay and ideation cases | `tests/unit/test_memory.py` |

## Dashboard and tools

| Commit | Old file | New location | Notes |
| --- | --- | --- | --- |
| `c55d06d` | `dashboard/index.html`, `app.js`, `style.css` | `tools/dashboard/index.html`, `app.js`, `style.css` | rewritten to render only what a run contains; hard-coded demo text and numbers removed; reads the `runs/<run-id>/stage-NN` layout |
| `c55d06d` | `dashboard/data.js` (10k line snapshot) | not shipped | generated by `build_data.py` or served live by `server.py`; `tools/dashboard/data.js` is gitignored |
| `c55d06d` | `scripts/dashboard_server.py` | `tools/dashboard/server.py` | "Run phase 1" calls `POST {I2H_API_URL}/api/phase1/start`; no subprocess, no key entry in the browser |
| `c55d06d` | `scripts/build_dashboard_data.py` | `tools/dashboard/build_data.py` | no absolute paths; `--runs-dir` / `I2H_RUNS_DIR` |
| `c55d06d`, `9e5fe83` | `scripts/phase1_fastapi_server.py`, `start_fastapi_swagger.bat`, `start_engine_8001.bat` | `tools/serve_api.py` | `I2H_CONFIG`, `PORT` |
| `c55d06d` | `scripts/tail_logs.py`, `view_live_logs.bat` | dropped | the dashboard shows the run event log instead |
| `c55d06d` | `start_*_tunnel.bat`, `start_ngrok_dashboard.bat`, `start_public_dashboard.bat`, `bin/cloudflared.exe`, `bin/ngrok.exe` | dropped | tunnel launchers and binaries are not source; kept in the backup bundle |

## Documentation

| Commit | Old file | New location |
| --- | --- | --- |
| `c55d06d` | `docs/pipeline_stage_1_to_8_guide.md` | `docs/pipeline-stage-guide.md` (rewritten for this code base; stage 9+ section removed) |
| `3f49fe2`, `730d883` | `docs/FE_RESEARCH_PIPELINE_6_STAGES_ARCHITECTURE.md` | `docs/llm-reliability-benchmark.md` (rubrics kept, prompts and sample payloads replaced by the current contracts) |
| `c55d06d`, `3bdf8f4` | `paper_writing_*`, `frontend-legacy/`, `run_hep_pipeline.sh` | not ported (out of scope) |

## Scratch scripts (`3f49fe2` and uncommitted)

| Old file | Disposition |
| --- | --- |
| `scratch/test_parse_robust.py` | `tests/unit/test_parsing.py` |
| `scratch/test_contract_hypo_diversity.py`, `test_hypo_zones.py` | intent kept on real data: distinct novelty and rationale, prediction values and diversity warning in `pipeline/contracts.py`; `tests/contracts/test_stage_contracts.py::test_stage8_*`, `tests/integration/test_options.py::test_duplicate_novelty_text_is_sent_back_for_repair` |
| `scratch/test_pipeline_screen_output.py` | intent kept: rejected papers carry reasons, one decision per candidate; `tests/integration/test_full_run.py::test_off_topic_papers_are_rejected_with_reasons`, `tests/contracts/test_stage_contracts.py::test_stage5_*` |
| `scratch/test_assemble.py`, `test_counts.py`, `test_dynamic_screen_spread.py`, `test_screen_pipeline.py`, `test_screen_points.py` | dropped: they validated random "screen point" assembly and fixed counts that fabricated data. Replaced by real per-paper decisions in `review.json` |
| `scratch/test_llm_gen.py`, `test_step1.py`, `test_scope_eval.py`, `test_stage_prompts.py`, `test_live_llm.py` | dropped: manual live-LLM probes of the old generator; live smoke testing needs your own credentials and is not part of the suite |
| `scratch/test_pipeline_run.py`, `run_phase5_e2e.py`, `check_run_events.py` | dropped: drove the old runner or the Platform stack by hand; covered by the integration and API contract tests |
| `scratch/test_parse_mock.py`, `set_fe_api.py`, `patch_*.py`, `transpile_dump.js`, `update_internal_popper.py` | dropped: one-off patches or probes for the frontend and backend repositories (already applied there) |
| `scratch/inspect_*.py` | dropped: inspection scripts |
| `scratch/test_phase3_events_ingest.py` | belongs to the Platform backend repository |
| `tests/test_rc_contracts.py`, `test_rc_stages.py`, `test_rc_runner.py`, `test_rc_executor.py` (`3bdf8f4`) | relevant cases re-expressed in `tests/contracts/` and `tests/integration/` |

No email addresses or personal data are carried over.
