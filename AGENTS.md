# Agent and contributor guide

Instructions for coding agents and humans working in this repository.

## Rules

* **Do not use Astra unless the user explicitly asks for it.**
* Never read, print or commit secrets. Configuration and snapshots hold only the *names* of
  environment variables and AWS profiles. Keep `.env` out of git; `.env.example` lists names only.
* Never add fallback or fake output in `src/`. A missing model, credential or literature result must
  end in a clear error (`LLMConfigError`, `NO_LITERATURE`, `EMPTY_SHORTLIST`, ...), not a template.
  Test doubles live only in `tests/fixtures/` and are named `Fixture*`.
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

```bash
pip install -e ".[api,bedrock,dev]"
python -m ruff check .
python -m pytest -q
python -m build
python tools/serve_api.py          # needs I2H_CONFIG or configs/example.yaml
python tools/dashboard/server.py   # run viewer
```

Run `ruff` and `pytest` before finishing a change. Tests must work offline.

## Dependencies

Core depends only on `pyyaml` and `httpx`. FastAPI, uvicorn and pydantic belong to the `api` extra
and boto3 to `bedrock`; import them lazily or only under `api/` and `llm/bedrock.py` so that
`import idea2hypothesis` works without extras. Ask before adding a dependency.

## Style

Python 3.11+, type hints, ruff with line length 100, small functions, brief docstrings. JSON and
JSONL files are written atomically (temporary file then `os.replace`) through `storage/`.
