# Inspiration and attribution

idea2hypothesis stands on [AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) (ARC)
by Aiming Lab, released under the MIT License.

## What was inherited

ARC is an autonomous research pipeline that goes from a topic to a paper. This project keeps only
the first eight stages of that pipeline, the part that turns an idea into hypotheses, and
rebuilds them as an independent package:

| Stage | ARC stage name (kept) | Inherited idea |
| --- | --- | --- |
| 1 | `TOPIC_INIT` | structured research goal, hardware advisory |
| 2 | `PROBLEM_DECOMPOSE` | prioritised sub-question tree, topic evaluation |
| 3 | `SEARCH_STRATEGY` | multi-strategy plan with short academic queries |
| 4 | `LITERATURE_COLLECT` | OpenAlex, Semantic Scholar and arXiv retrieval, deduplication, BibTeX |
| 5 | `LITERATURE_SCREEN` | relevance and quality screening with a review gate |
| 6 | `KNOWLEDGE_EXTRACT` | one structured evidence card per paper |
| 7 | `SYNTHESIS` | thematic clusters and research gaps |
| 8 | `HYPOTHESIS_GEN` | multi-perspective generation, optional debate, novelty assessment |

Prompts, the literature provider logic, the ideation memory store, and the LLM client and AWS
Bedrock adapter ideas were extracted from ARC and cleaned up. Stage numbers and names follow ARC.

## What is different

* Independent package `idea2hypothesis` with a `src/` layout and a small dependency set; no ARC
  package, alias or configuration is shipped.
* No fabricated output: template goals, placeholder papers, invented seminal papers and default
  screening scores were removed. A failed dependency stops the run with an error.
* JSON and JSONL artifacts with `schema_version`, manifests and attempt versioning are the
  contract; Markdown is rendered from them.
* Crash-safe checkpointing, resume that never repeats valid work, review gates by mode.
* HTTP API and Platform backend adapter built on the same engine; no command line interface.
* Everything after hypothesis generation (experiments, code generation, paper writing,
  sandboxes, external agent bridges, the website and benchmarks) is out of scope.

The ARC license notice is preserved in [LICENSE](LICENSE). Detailed changes are listed in
[docs/migration.md](docs/migration.md), and the changes made by KalvinKhanh before extraction
are traced in [docs/kalvin-changes.md](docs/kalvin-changes.md).
