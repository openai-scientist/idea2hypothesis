# Stage contracts

Each stage reads named artifacts of earlier stages, writes its own files into
`runs/<run-id>/stage-NN/` and must satisfy the pass conditions below before the run continues. The
rules live in `src/idea2hypothesis/pipeline/contracts.py` (`CONTRACTS`, `validate_stage`) and are
checked

* by the stage itself, which asks the model once per `runtime.max_retries` to repair an invalid
  answer before anything is written,
* by the runner after the stage (a violation fails the stage with the contract's error code),
* on resume, to decide whether a completed stage is still valid,
* when an artifact is edited through the API.

JSON artifacts carry `"schema_version": 1`. JSONL rows are covered by the version in the stage
`manifest.json`. Markdown files are rendered from the JSON and are never read back by later
stages.

Every completed stage also has `manifest.json` (`schema_version`, `run_id`, `stage`, `attempt`,
`files[{path, bytes, sha256}]`). A stage counts as valid on resume only if the manifest belongs to
this run and every listed file still matches its hash.

## Overview

| Stage | Name | Needs | Writes | Error code on contract failure |
| --- | --- | --- | --- | --- |
| 1 | `TOPIC_INIT` | topic, domains, constraints | `goal.json`, `goal.md`, `hardware_profile.json` (when `research.hardware_advisory`) | `INVALID_GOAL` |
| 2 | `PROBLEM_DECOMPOSE` | `stage-01/goal.json` | `problem_tree.json`, `problem_tree.md`, `topic_evaluation.json` | `INVALID_PROBLEM_TREE` |
| 3 | `SEARCH_STRATEGY` | `stage-02/problem_tree.json` | `search_plan.yaml`, `queries.json`, `sources.json` | `INVALID_SEARCH_PLAN` |
| 4 | `LITERATURE_COLLECT` | `stage-03/queries.json` | `candidates.jsonl`, `references.bib`, `search_meta.json` | `NO_LITERATURE` |
| 5 | `LITERATURE_SCREEN` | `stage-04/candidates.jsonl` | `shortlist.jsonl`, `screen_meta.json`, `review.json` | `EMPTY_SHORTLIST` |
| 6 | `KNOWLEDGE_EXTRACT` | `stage-05/shortlist.jsonl` | `cards/<card_id>.json`, `cards/<card_id>.md`, `knowledge_meta.json` | `INVALID_CARDS` |
| 7 | `SYNTHESIS` | `stage-06/knowledge_meta.json` and cards, `stage-02/problem_tree.json` | `synthesis.json`, `synthesis.md` | `INVALID_SYNTHESIS` |
| 8 | `HYPOTHESIS_GEN` | `stage-07/synthesis.json`, cards | `hypotheses.json`, `hypotheses.md`, `perspectives/`, `novelty_report.json` (when `research.novelty_check`) | `INVALID_HYPOTHESES` |

Stage 8 is the end of the pipeline. Other failure codes a run can end with:
`TOPIC_NOT_RESEARCHABLE` (stage 1), `TOPIC_BELOW_THRESHOLD` (stage 2, overall topic score below
`research.min_topic_score`), `NO_CARDS` (stage 6), `NO_PERSPECTIVES` (stage 8),
`LLM_OUTPUT_INVALID` (model output still invalid after repair), `LLM_CONFIG`, `LLM_TIMEOUT`,
`LLM_RATE_LIMITED`, `LLM_ERROR`, `MISSING_INPUT`, `STAGE_ERROR`, `INTERNAL_ERROR`.

## Stage 1: TOPIC_INIT

`goal.json`: `topic`, `domains`, `research_constraints`, `researchable` (bool), and when researchable
`working_title`, `problem`, `objective`, `scope`, `smart_goal`, `constraints`, `success_criteria`
(non-empty list), optional `novel_angle` and `benchmark`; when not researchable a
`rejection_reason`. A topic judged not researchable is written and the run fails with
`TOPIC_NOT_RESEARCHABLE`; stages 2 to 8 never run.

`hardware_profile.json` (advisory only): `has_gpu`, `gpu_type`, `gpu_name`, `vram_mb`, `tier`,
`warning`. It is read-only detection; nothing is installed.

## Stage 2: PROBLEM_DECOMPOSE

`problem_tree.json`: `sub_questions` (at least 3), each with unique `id`, `text`, integer
`priority`, `goal_link`, and optional `tests`, `covers`; `priority_ranking` (ids in priority order);
`risks`. `topic_evaluation.json`: `novelty`, `specificity`, `feasibility` and `overall` in
`[0, 10]`, `threshold`, `suggestion`. `overall` is computed from the three scores.

## Stage 3: SEARCH_STRATEGY

`search_plan.yaml`: at least 2 `search_strategies`, each with `name`, non-empty `queries` and
`sub_question_ids` that exist in the problem tree; `filters.min_year`. `queries.json`: rows
`{id, text, strategy, sub_question_ids}` with distinct, shortened query text and `year_min`.
`sources.json`: the configured providers (`id`, `name`, `type`, `url`, `status: "configured"`).

## Stage 4: LITERATURE_COLLECT

`candidates.jsonl`: one deduplicated real paper per line:
`paper_id` (stable, derived from DOI, else arXiv id, else normalised title), `title`, `authors`,
`year`, `abstract`, `venue`, `citation_count`, `doi`, `arxiv_id`, `url`, `cite_key` and
`source_records[{provider, source_id, url, retrieved_at}]`. Records from several providers for the
same paper are merged and keep every source record. `references.bib` has exactly one entry per
candidate. `search_meta.json`: `queries_used` (plan and expansion), `year_min`, `raw`, `unique`,
`duplicates`, `dropped_without_title`, `per_source` (requests, papers, errors), `per_query`,
`errors`. A source that fails is recorded here and surfaced as a warning; zero papers from all
sources fails the stage with `NO_LITERATURE` and stages 5 to 8 do not run.

## Stage 5: LITERATURE_SCREEN

`review.json`: `rules`, `thresholds`, `summary` (`candidates`, `kept`, `rejected`, `unscored`,
`prefiltered`), `decisions` (one per candidate: `paper_id`, `decision` in `kept | rejected |
unscored | prefiltered | dropped_by_reviewer`, `reason`, `relevance_score`, `quality_score`,
`false_friend`), `human_review`. Papers that never received scores (`unscored`, `prefiltered`) carry
`null` scores and are excluded; they never get default values. `shortlist.jsonl`: the kept
candidates plus `relevance_score`, `quality_score` in `[0, 1]` and `keep_reason`.
`screen_meta.json`: `outcome`, counts, `keywords`, batches.

Pass conditions: every candidate has a decision and a reason; kept papers in `review.json` equal
the shortlist; the shortlist is a non-empty subset of the candidates. An empty shortlist fails with
`EMPTY_SHORTLIST`. A reviewer who drops papers at the gate updates these three files and marks the
decision `dropped_by_reviewer`.

## Stage 6: KNOWLEDGE_EXTRACT

One card per shortlisted paper that has an abstract (`skipped` in `knowledge_meta.json` lists the
others with a reason). `cards/<card_id>.json`: `card_id` = `card-<paper_id>`, `paper_id`, `title`,
`cite_key`, `year`, `venue`, `doi`, `arxiv_id`, `url`, `evidence_scope` (`abstract` or `full_text`), and `problem`, `method`, `data`, `metrics`,
`findings`, `limitations`, each text or `null`. Nothing is filled with template text. Every card
also has a `.md` rendering. `knowledge_meta.json`: `shortlist_size`, `cards`, `evidence_scope`,
`skipped`.

## Stage 7: SYNTHESIS

`synthesis.json`: `clusters` (`id`, `title`, `card_ids` that exist) and `gaps` (at least 2, each with
unique `id`, `text`, `sub_question_ids` that exist and `card_ids` that exist), plus the model's
overview and tensions. Numbers in the text that do not appear in any card are reported as warnings
(`synthesis mentions '...' which does not appear in any card`).

## Stage 8: HYPOTHESIS_GEN

`hypotheses.json`: `hypotheses` (at least 2) with `id`, `statement`, `gap_id` (a real gap),
`sub_question_ids`, `evidence_refs` (card ids or shortlisted paper ids that resolve), `exposure`,
`outcome`, `estimand`, `method`, `conditions`, `prediction` (`> 0`, `< 0` or `≠ 0`),
`falsification_criteria` (at least 25 characters and a concrete failing observation such as a
threshold or interval), `limitations`, `rationale`, `novelty`, `risk`. `novelty` and `rationale`
must differ between hypotheses (near-duplicates fail the stage). `perspectives/` holds the
per-role generations (`<role>.json`), debate rounds (`<role>.r<N>.json`) and `debate_record.json`
when `llm.debate_rounds > 0`; at least one perspective output must exist or the stage fails with
`NO_PERSPECTIVES`. `disagreements` lists unresolved points between perspectives.

`novelty_report.json` (when enabled): `kind: "novelty_assessment"`, a `disclaimer` stating it is a
heuristic assessment and not proof of novelty, `novelty_score`, `assessment`, `recommendation`,
`similar_papers`, `per_hypothesis[{hypothesis_id, closest_paper}]`, search coverage and errors.

## Cross-stage references

```text
sub-question (stage 2) -> query (3) -> paper (4) -> screening decision (5) -> card (6)
                                                         -> gap (7) -> hypothesis (8)
```

A reference that does not resolve to a stored record is a contract error. This is what makes a
hypothesis traceable to its evidence.

## Edits and invalidation

Editing an artifact through the API validates it against the same rules; the changed stage is
re-manifested and every later stage is moved to `attempts/<n>/` before it can run again, so a stale
downstream output is never read after an upstream edit. A rejected gate does the same for stages
from the rollback point (stage 3 for the screening gate, stage 1 for the scope gate) with a new
attempt number.
