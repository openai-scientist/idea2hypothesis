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
| 9 | `ARGUMENT_MAP` | `stage-08/hypotheses.json`, `stage-07/synthesis.json` | `argument_map.json`, `semantic_graph.json`, `research_canvas.json` | `INVALID_ARGUMENT_MAP` |

Stage 9 is the end of the pipeline. Other failure codes a run can end with:
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
`[0, 10]`, `threshold`, `reasons` (one sentence per score saying what in the topic or goal earned
it) and `suggestion` (the change that would most raise the weakest score, or a refined topic when
the run stops). The prompt anchors each score band; novelty is judged before any search, from the
model's knowledge. `overall` is computed from the three scores.

## Stage 3: SEARCH_STRATEGY

`search_plan.yaml`: at least 2 `search_strategies`, each with `name`, non-empty `queries` and
`sub_question_ids` that exist in the problem tree; `filters.min_year`. `queries.json`: rows
`{id, text, strategy, sub_question_ids}` with distinct, shortened query text and `year_min`.
`sources.json`: the configured providers (`id`, `name`, `type`, `url`, `status: "configured"`).

## Stage 4: LITERATURE_COLLECT

`candidates.jsonl`: one deduplicated real paper per line:
`paper_id` (stable, derived from DOI, else arXiv id, else normalised title), `title`, `authors`,
`year`, `abstract`, `venue`, `citation_count`, `doi`, `arxiv_id`, `url`, `cite_key` and
`source_records[{provider, source_id, url, retrieved_at, citations, has_doi}]` (`citations` and
`has_doi` are what that source's own record said). Records from several providers for the same
paper are merged and keep every source record; the first is the one whose metadata was kept (most
citations, then the longer abstract), and the others fill its gaps. `references.bib` has exactly one entry per
candidate. `search_meta.json`: `queries_used` (plan and expansion), `year_min`, `raw`, `unique`,
`duplicates`, `dropped_without_title`, `per_source` (requests, papers, errors), `per_query`,
`errors`. A source that fails is recorded here and surfaced as a warning; zero papers from all
sources fails the stage with `NO_LITERATURE` and stages 5 to 8 do not run.

## Stage 5: LITERATURE_SCREEN

`review.json`: `rules`, `thresholds` (`min_relevance`, `min_quality`, `max_shortlist`), `summary`
(`candidates`, `kept`, `rejected`, `below_cutoff`, `unscored`, `prefiltered`), `decisions` (one per
candidate: `paper_id`, `decision` in `kept | rejected | below_cutoff | unscored | prefiltered |
dropped_by_reviewer`, `reason`, `relevance_score`, `quality_score`, `false_friend`),
`human_review`. With `research.max_shortlist` above 0, only that many papers (best relevance, then
quality) are kept; the others that cleared both bars are `below_cutoff` with their scores and a
reason. Papers that never received scores (`unscored`, `prefiltered`) carry
`null` scores and are excluded; they never get default values. `shortlist.jsonl`: the kept
candidates plus `relevance_score`, `quality_score` in `[0, 1]` and `keep_reason`.
`screen_meta.json`: `outcome`, counts, `keywords`, batches, `no_abstract` (how many of the
`prefiltered` papers had no abstract). `review.json` `reviewer_view` lists
what the reviewer model saw of each paper (`fields`, `abstract_max_chars`); a decision on an
abstract longer than that carries `abstract_cut_at`.

Pass conditions: every candidate has a decision and a reason; kept papers in `review.json` equal
the shortlist; the shortlist is a non-empty subset of the candidates. An empty shortlist fails with
`EMPTY_SHORTLIST`. A reviewer who drops papers at the gate updates these three files and marks the
decision `dropped_by_reviewer`.

## Stage 6: KNOWLEDGE_EXTRACT

One card per shortlisted paper that has an abstract and whose card its abstract can back
(`skipped` in `knowledge_meta.json` lists the others with a reason). `cards/<card_id>.json`:
`schema_version` (2), `card_id` = `card-<paper_id>`, `paper_id`, `title`, `cite_key`, `year`,
`venue`, `doi`, `arxiv_id`, `url`, `evidence_scope` (`abstract` or `full_text`), and `problem`,
`method`, `data`, `metrics`, `findings`, `limitations`, each text or `null`, and `quotes`:
`{field: [passage, ...]}` for every filled field. Nothing is filled with template text. Every card
also has a `.md` rendering. `knowledge_meta.json`: `shortlist_size`, `cards`, `evidence_scope`,
`quoted`, `skipped`.

Pass conditions for schema 2 cards: every filled field has 1 to 6 quotes, each at least four words
long and found in the paper's abstract (from the shortlist) after normalisation: Unicode NFKC,
curly quotes and dash variants folded, HTML tags removed, case folded, white space collapsed,
surrounding quote marks and ellipses stripped. A `null` field has no quotes. Schema 1 cards,
written before quotes existed, are checked without them.

## Stage 7: SYNTHESIS

`synthesis.json`: `clusters` (`id`, `title`, `card_ids` that exist) and `gaps` (at least 2, each with
unique `id`, `text`, `sub_question_ids` that exist and `card_ids` that exist), plus the model's
overview and tensions. Every card sent to the model is accounted for: it is in exactly one cluster,
or in `set_aside` (`id`, `card_ids`, `reason`) when it bears on no school of thought; a card in
neither fails the answer and the model is asked again. Numbers in the text that do not appear in any card are reported as warnings
(`synthesis mentions '...' which does not appear in any card`).

From `schema_version` 2 every tension (a point where cards disagree) has an `id` (`X1`, `X2`, ...),
`between` (cluster ids that exist), `text` and exactly two `sides`, each with a `claim` and the
`card_ids` behind it (at least one existing card per side, no card on both sides). Cards that agree
give an empty `tensions` list. Syntheses of schema 1 have tensions without ids or sides and are
read as before.

## Stage 8: HYPOTHESIS_GEN

`hypotheses.json`: `hypotheses` (between `research.min_hypotheses` and `research.max_hypotheses`,
default 3 to 6, never fewer than 2) with `id`, `statement`, `gap_id` (a real gap),
`sub_question_ids`, `evidence_refs` (card ids or shortlisted paper ids that resolve), `exposure`,
`outcome`, `estimand`, `method`, `conditions`, `prediction` (`> 0`, `< 0`, `≠ 0`, or `≈ 0` for a negligible effect, which then needs
`equivalence_margin`, a positive number in the outcome's unit),
`falsification_criteria` (at least 25 characters and a concrete failing observation such as a
threshold or interval), `limitations`, `rationale`, `novelty`, `risk`. `novelty` and `rationale`
must differ between hypotheses (near-duplicates fail the stage). `tension_ids` lists the synthesis
tensions a hypothesis settles; each id must exist, and when the synthesis has tensions with ids at
least one hypothesis must settle one (otherwise the answer is sent back). `open_tensions` lists the
tensions no hypothesis of the set settles. `disagreements` lists unresolved points between
perspectives.

Every final hypothesis lists in `from` the debate candidates it is built from (`<role>-<number>`,
such as `innovator-2`); each must exist and not be withdrawn. A candidate with a fatal objection
that still stands may be used only when the set cannot reach `min_hypotheses` from the others;
otherwise the answer is sent back. The stage then records, from the debate, what still stands
against each hypothesis's sources: `contested` (fatal objections: `candidate`, `from` the critic,
`severity`, `flaw`, `field`, `card_id`, `text`) and `caveats` (same fields; each caveat is also
added to `limitations` as "Debate caveat from the <role> perspective: ..."). `held_back` lists the
candidates with a fatal objection standing that the set does not use (`candidate`, `role`,
`number`, `hypothesis`, `objections`). `not_used` lists every other candidate the set leaves out
(`candidate`, `role`, `number`, `hypothesis`, `reason`, `of`, `text`, `caveats`): the merge must
give each one a `reason`, `duplicate` (with `of`, the final hypothesis that already tests its
claim, with the same `prediction`; two `≈ 0` claims that differ only in margin are one study, so
the one left out is a duplicate and its `text` names both margins) or `over_limit` (only when the
set has `max_hypotheses`), or the answer is sent back.
A hypothesis built from two or more candidates needs `merge_note` (what it takes from each), and
candidates whose predictions differ (including a benefit and a negligible effect, or two negligible
effects with different `equivalence_margin`) are never merged; `merge_note` is kept only on a hypothesis with two or more sources. After the hypotheses gate,
`human_review` holds the reviewer's `dropped` and `kept`, and a kept candidate (held back or not
used) joins the set under a new id with `kept_by_reviewer` (the reviewer's note); a held-back one
keeps its objections in `contested`.

`perspectives/` holds the per-role generations (`<role>.json`) and, when `llm.debate_rounds > 0`,
each round in three phases plus `debate_record.json`:

* `<role>.r<N>.critique.json` (`phase: "critique"`): `responses` to the other roles (`to`,
  `hypothesis` number, `stance` `challenge` or `concede`, `text`). Every challenge has `severity`
  `fatal` or `caveat`; a fatal one names its `flaw` (`unsupported`, `unfalsifiable`,
  `undecidable_test` or `already_established`) and the hypothesis `field` that holds it, and
  `already_established` names in `card_id` an allowed reference that already shows the claim. A
  challenge missing any of these is sent back. Responses that name no standing hypothesis of
  another role are left out with a warning.
* `<role>.r<N>.json` (`phase: "answer"`), written only for a role that was challenged: `answers`,
  one per challenge it received (`challenge` number, `from` and `response`: the critic and the
  place of the challenge in its `responses`, `hypothesis`, `action` `revise`, `defend` or
  `withdraw`, `text`), the updated `hypotheses` (numbering kept, new ones at the end), `revised`,
  `added` and `withdrawn` (hypothesis numbers). An answer that leaves a challenge unanswered, says
  `revise` without changing any field of the hypothesis (the field the challenge is about need
  not be the statement), or withdraws a hypothesis it does not list in
  `withdrawn`, or leaves a negligible-effect (`≈ 0`) hypothesis without a positive
  `equivalence_margin` is sent back (a perspective's own proposals follow the same margin
  rule). A role whose answer still fails keeps its position, a warning names
  the unanswered challenges, and they stand as raised.
* `<role>.r<N>.review.json` (`phase: "review"`), for each critic with answered challenges or with
  hypotheses added that round by others: `reviews`, one per answered challenge (the challenge with
  its `answer`, and `review`: `verdict` `resolved` or `stands` with `text`); a challenge that
  stands keeps its severity or is lowered from fatal to caveat, never raised, and a fatal one
  keeps a flaw; then `added`, the critique of the added hypotheses (same rules as a critique).
  A review that skips an answered challenge or raises a caveat is sent back; a critic whose
  review still fails leaves its challenges standing as raised.
* `debate_record.json`: `rounds`, `roles`, `concessions`, `answers` (per role, counts per action),
  `withdrawn` (per role, numbers), `objections` (every challenge with its answer, review and
  `status`: `stands`, `resolved` or `withdrawn`), `independent_judge` and the judge's
  `rankings`. Withdrawn hypotheses take no further part: critiques of them are dropped, and
  neither the judge nor the final merge sees them. The judge and the merge see every candidate
  with the objections that still stand against it.

At least one perspective output must exist or the stage fails with `NO_PERSPECTIVES`. Rounds
written by earlier versions hold `responses` and the revised `hypotheses` in one
`<role>.r<N>.json` and are read as before.

`novelty_report.json` (when enabled): `kind: "novelty_assessment"`, a `disclaimer` stating it is a
heuristic assessment and not proof of novelty, `method`, `novelty_score` (null when nothing was
judged), `assessment`, `recommendation`, `papers_compared`, `similar_papers` (the papers a verdict
names, each with `hypothesis_id` and `verdict`), `similar_papers_found` (papers judged to test a
hypothesis), `per_hypothesis[{hypothesis_id, verdict, reason, closest_paper, papers_read}]`
(`verdict` is `tested`, `related`, `new`, or null when the papers could not be judged;
`closest_paper.similarity` is the share of the hypothesis's keywords that paper holds),
`search_queries` (one keyword query per hypothesis, 2-7 plain words without quotes, field
prefixes or AND/OR/NOT, checked by `check_novelty_queries`), search coverage, `search_errors`
and `judge_errors`. `check_novelty_judgements` requires one verdict per hypothesis, naming only
papers given for it: at least one for `tested` and `related`, none for `new`.
`search_coverage` is `full`, `partial`, `run_corpus_only` (the search returned nothing, so only
the stage 4 pool was compared; the recommendation is then at most `proceed_with_caution` and the
stage warns) or `insufficient`.

## Stage 9: ARGUMENT_MAP

`argument_map.json`: `assessment: "model judgement, not reviewed"`, `evidence_links[{card_id,
claim_id, relation, rationale}]` with `relation` `supports`, `contradicts` or `unrelated`, every
card of every cluster judged exactly once against its own cluster; `rationales[{claim_id,
hypothesis_id, polarity, rationale}]` with `polarity` `supports` or `challenges`, known ids, no pair
twice, and at least one claim per hypothesis.

`semantic_graph.json`: `ontology`, `entities[{id, type, code, label, text, stage, ...}]` and
`relations[{id, from, to, relation, polarity?, status, rationale, provenance}]`. Ids are unique,
types are the seven entity types, each relation keeps the direction its vocabulary gives (for
example `addresses` reads from a hypothesis to a gap; a flipped edge is an error), `status` is
`stated`, `derived` or `unreviewed`, and `rationale` and `provenance` are not empty.

`research_canvas.json`: `pieces` holds each of the nine pieces (`puzzle`, `audience`, `question`,
`theory`, `setting`, `design`, `findings`, `contributions`, `boundaries`) once; a `filled` piece has
items.

## Cross-stage references

```text
sub-question (stage 2) -> query (3) -> paper (4) -> screening decision (5) -> card (6)
                                                         -> gap (7) -> hypothesis (8)
                                                         -> argument map (9)
```

A reference that does not resolve to a stored record is a contract error. This is what makes a
hypothesis traceable to its evidence.

## Edits and invalidation

Editing an artifact through the API validates it against the same rules; the changed stage is
re-manifested and every later stage is moved to `attempts/<n>/` before it can run again, so a stale
downstream output is never read after an upstream edit. A rejected gate does the same for stages
from the rollback point (stage 3 for the screening gate, stage 1 for the scope gate, stage 8 for
the hypotheses gate) with a new attempt number. A gate before the rollback point that was
approved stays approved in the new attempt (`carried_from` names the original), since the stages
it approved do not run again.
