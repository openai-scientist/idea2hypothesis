# Contributing

Thanks for helping with idea2hypothesis. Read [AGENTS.md](AGENTS.md) first for the component
boundaries and the no-fabrication rule.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[api,bedrock,dev]"
# Real providers: export the variables listed in .env.example in your shell.
# The package does not read .env files; never commit one.
```

## Before opening a pull request

```bash
python -m ruff check .
python -m pytest -q
python -m build
```

CI runs the same steps on Ubuntu and Windows with Python 3.11 and 3.13, audits the wheel (no tests,
tools or runs; prompt YAML present) and imports the core wheel in an environment without extras.

## Guidelines

* Keep the change focused; update the docs that describe the behaviour you touched
  (`docs/stage-contracts.md` for artifacts, `docs/platform-integration.md` for HTTP behaviour,
  `docs/architecture.md` for structure).
* Add a test with the change. Unit tests cover single components, `tests/integration/` runs
  the pipeline with fixture ports, `tests/contracts/` checks stage artifacts and API contracts.
  Tests must not use the network or real credentials.
* New stage output fields need a contract check in `pipeline/contracts.py`, an update to the stage
  prompt in `prompts/stages.yaml` and a note in `docs/stage-contracts.md`.
* New configuration keys go through `config.py` (typed, unknown keys rejected) and
  `configs/example.yaml`; never add a key without a consumer.
* Prompts are packaged assets. After editing YAML, check `tests/unit/test_prompts.py` still passes
  and that the wheel includes the file.
* Commit messages: short imperative subject (for example `fix(stage5): keep unscored papers out of
  the shortlist`).

## Reporting problems

Open an issue with the run id, the stage and error code from `run.json`, and the relevant lines of
`events.jsonl`. Remove any credentials before sharing files.
