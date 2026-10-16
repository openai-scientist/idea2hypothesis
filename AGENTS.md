# Agent and contributor guide

Instructions for coding agents and humans working in this repository.

## Rules

* Never read, print or commit secrets. Configuration and snapshots hold only the *names* of
  environment variables and AWS profiles. Keep `.env` out of git; `.env.example` lists names only.
* Never add fallback or fake output in `src/`. A missing model, credential or literature result must
  end in a clear error (`LLMConfigError`, `NO_LITERATURE`, `EMPTY_SHORTLIST`, ...), not a template.
  Test doubles live only under `tests/`: the helpers in `tests/fixtures/` (`Fixture*` ports,
  `FakePlatform`, `Harness`, `ContractHarness`) and `tests/contract/fake_platform.py`.
* Do not add a CLI (`cli.py`, `__main__.py`, `[project.scripts]`). The API and the Python library
  are the entry points; `tools/` holds developer helpers only.
* Do not commit or push unless the user asks. Do not rewrite history.

## Component boundaries

| Path | Owns | Must not |
| --- | --- | --- |
| `src/idea2hypothesis/stages/` | research logic, one module per stage | call the API, webhooks or storage layout directly |
| `src/idea2hypothesis/pipeline/` | runner, contracts, gates, events, models, ports | contain prompts or provider code |
| `src/idea2hypothesis/literature/` | providers, dedup, BibTeX, novelty | depend on the API or pipeline |
| `src/idea2hypothesis/llm/` | providers, retry, parsing, usage and pricing | know which stage is running |
| `src/idea2hypothesis/prompts/` | prompt YAML and loader | contain pipeline logic |
| `src/idea2hypothesis/storage/` | run store, artifacts, checkpoints, events (filesystem) | import stages |
| `src/idea2hypothesis/memory/` | ideation memory only | hold experiment or writing memory |
| `src/idea2hypothesis/resources/` | read-only hardware advisory | install packages or reach other machines |
| `src/idea2hypothesis/api/` | HTTP, worker lifecycle, Platform mapping, webhooks | generate hypotheses itself |
| `tools/` | `serve_api.py`, `dashboard/` | be imported by the package |
| `tests/` | unit, integration, contract tests and fixtures | touch the network |

No `utils/` package and no catch-all `_helpers.py`: helpers sit next to the component that owns them.

## Evidence rules

* Every paper has an identity and provenance (`source_records`: provider, source id, URL, time).
* Every card, gap and hypothesis references stored records (`card_id`, `paper_id`, `gap_id`,
  sub-question ids). Check references with `pipeline/contracts.py`.
* Cards are abstract-level; do not claim full-text reading. Unknown fields are `null`.
* Unscored papers are excluded with a reason; they never receive default scores.
* The novelty report is a heuristic assessment, labelled as such.
* Hypotheses need a falsification criterion that states a concrete failing observation.
* Numbers in synthesis must come from cards (the contract warns on unsupported figures).

## Commands

Install is in [README.md](README.md#install). Tests must work offline. During implementation,
use a focused behavioral test after each edit; do not repeat the full suite for every small edit.
Run affected coverage against the target branch before pushing, and run `ruff` plus the full
suite once for final handoff (repeat only after a relevant change or unresolved failure).

```bash
python -m ruff check .
python tools/test_feedback.py --files tests/unit/test_parsing.py  # scoped edit loop
python tools/test_feedback.py --base origin/main                # affected, before push
python -m pytest -q
uv run --with pytest-xdist pytest -q -p no:cacheprovider -n 8  # full suite in parallel, ~2.5 min
python -m build
python tools/serve_api.py          # needs I2H_CONFIG or configs/example.yaml
python tools/dashboard/server.py   # run viewer
```

CI runs the same checks on Ubuntu and Windows (Python 3.11 and 3.13), audits the wheel (no tests,
tools or runs; prompt YAML present) and imports the core wheel in an environment without extras.
PRs use the same affected selector as the local command. Main, release branches/tags, scheduled
and manual runs retain the full suite. `gate` is the always-running required-check entry point.
Lanes live in `tools/test_feedback.py`. Shared core, fixtures, runtime assets,
configuration, tooling/CI, unknown paths and unavailable diffs select all tests. When adding a
test file, register its owner in `tools/test_feedback.py`; API consumers include
`test_api_idle.py` and `test_stage_helpers.py` despite their unit directory.

## Change rules

* New stage output fields need a check in `pipeline/contracts.py` and an update to the stage prompt
  in `prompts/stages.yaml`.
* New configuration keys go through `config.py` (typed, unknown keys rejected) and
  `configs/example.yaml`. Never add a key without a consumer.
* Prompts are packaged assets. After editing YAML, `tests/unit/test_prompts.py` must pass and the
  wheel must include the file.
* Add a test with every change. Unit tests cover single components, `tests/integration/` runs the
  pipeline with fixture ports, `tests/contracts/` checks stage artifacts and API contracts.

## Platform contract tests

`tests/contract/` runs the vendored management contract against the `/v1` service (see "Platform
service" in the README). Run it with `python -m pytest tests/contract -q`. Management owns the
contract. To refresh `tests/contract/vendor/`, copy `schema/`, `examples/` and `README.md` from
`../management/docs/contract/` and write the management commit hash into `vendor/SOURCE`. Never edit
the vendored files by hand. The vendored README describes the management repo; its commands do not
apply here.

## Dependencies

Core depends only on `pyyaml` and `httpx`. FastAPI, uvicorn and pydantic belong to the `api` extra
and boto3 to `bedrock`; import them lazily or only under `api/` and `llm/bedrock.py` so that
`import idea2hypothesis` works without extras. Ask before adding a dependency.

## Style

Ruff with line length 100. JSON and JSONL files are written atomically (temporary file then
`os.replace`) through `storage/`.
