# LMLM Audit

![Tests](badges/tests.svg)
![Coverage](badges/coverage.svg)

This repository contains the code for [*Auditing Forgetting in Limited Memory
Language Models*](https://arxiv.org/abs/2607.00605).

The original audit was built for relational LMLMs. We are now extending it to
[Co-LMLM](https://arxiv.org/abs/2607.07707), where facts are stored as free-form
text and retrieved with continuous keys.

## What are we testing?

LMLMs are designed to keep factual knowledge in an external memory. In theory,
deleting a fact from that memory should make the model forget it. In practice,
the answer may still be available through the model's parameters or through a
different, related memory entry.

For each fact, we run the model in three settings:

- `FULL`: the memory is unchanged and retrieval is enabled.
- `DEL-ON`: the target entry is hidden, but retrieval remains enabled.
- `DEL-OFF`: the target entry is hidden and retrieval is disabled.

Comparing these runs helps us distinguish parametric memory from answers that
are recovered through retrieval. The audit also saves the retrieved entries so
we can inspect where an answer came from.

## Current status

The repository supports both backends through one model-agnostic audit:

- the original rel-LMLM audit and a Co-LMLM backend on the public model and
  index interfaces, behind a registry so new models drop in without touching
  the audit core;
- a shared audit format that does not require subject-relation pairs;
- non-destructive deletion (filtering selected entry or source IDs at search
  time), retrieval traces, query-embedding sidecars, and the cross-state
  forgetting metrics L(f) and R(f);
- an oracle smoke-test mode that uses the entry retrieved during `FULL` as the
  deletion target;
- materialized deletion closures (`--closure geometric,semantic,provenance`)
  built from the `FULL` pass, with per-entry attribution artifacts and a
  run-time semantic backstop;
- entanglement sweeps (`--radius-grid 0.95:0.70:0.05`) that measure deletion
  efficacy against collateral damage on neighbor facts
  (`--neighbor-mode cosine|same-source`) and report per-fact operating curves
  and the entanglement gap G(f);
- a representational-leakage probe (run automatically as part of an audit)
  that fits a linear readout on frozen query embeddings over a fact-disjoint
  split and reports L_rep and Δ_rep against the behavioral DEL-OFF baseline;
- an adversarial-closure evaluation (`--adversarial`) that injects synthetic
  survivor entries just outside the deletion radius, reports the evasion rate
  Ev(ρ, ε) per value template and topology, and scores a geometry-only margin
  predictor (AUROC) for retrieval-mediated leakage;
- the closure, sweep, and adversarial evaluations on rel-LMLM too (geometric
  and semantic predicates only — relational triples carry no provenance),
  giving the relational G(f) baseline.

Both backends emit the same result and metric schema, so a rel-vs-Co-LMLM
comparison is a join over their output CSVs.

The Co-LMLM backend is unit-tested but has not yet been run against the full
released checkpoint and index. The closure's semantic and provenance predicates
begin to address the research goal of identifying all memory entries that
express a fact, rather than a single oracle entry.

## Repository structure

The code separates the model-agnostic audit from the model backends:

- `src/lmlm_audit/` — the audit itself, which knows nothing about any specific
  model:
  - `core/` — the abstract backend interface, database states, forgetting
    metrics, entanglement/probe/neighbor analysis, answer-equivalence, and
    prompt/example handling.
  - `interventions/` — the backend-agnostic deletion and attack machinery:
    the deletion closure, the non-destructive filtering search, the support
    judge, and the adversarial survivor construction. These operate on any
    model through a generic search-index interface.
  - `registry.py` — the backend registry the CLI dispatches through.
  - `cli/` — the `lmlm-audit` entry point, the audit runner, and reporting.
- `src/models/` — one subpackage per audited model. Each follows the same
  template so models stay consistent, and each registers a backend with
  `lmlm_audit.registry`, so a new model slots in with no changes to the audit
  core:
  - `__init__.py` — registers the model's `BackendSpec` (how to build the
    backend, its search index, job grouping, and argument validation).
  - `backend.py` — the `*AuditBackend` class (implements `generate`), loading,
    and output parsing.
  - `adapter.py` — how the model plugs into the audit's deletion/search
    machinery: `build_search_index` and any support-judge override.

  A model may add internal modules where it genuinely does more — e.g.
  `models/rel_lmlm/database.py` holds the relational retriever and its
  deletion-aware database manager, which Co-LMLM delegates to its upstream
  package and so does not need.

Databases and prompt sets are under `data/`: the released LMLM database with
six prompt types, and three custom domains (countries, politicians, sports)
with base/alias/collision/noise variants.

## Setup

The project uses Python 3.12 and [uv](https://docs.astral.sh/uv/).

For the original rel-LMLM audit, place the upstream LMLM repository at
`../LMLM`, then run:

```bash
uv sync
uv run pytest
```

## Running the rel-LMLM audit

```bash
uv run lmlm-audit \
  --database-path data/custom_databases/countries/base.json \
  --prompt-files data/custom_databases/countries/prompts/base/prompts_direct_questions.jsonl \
  --output-dir outputs/audit \
  --wandb-activation off
```

The runner writes one JSONL result file per prompt set, along with per-state and
cross-state metric CSVs.

The audit deletes non-destructively (it filters retrieval results at query
time and never mutates the triple database), which matches how upstream builds
its retriever: the FAISS index is constructed once on first use and never
rebuilt. Two consequences to keep in mind: do not mutate the base database
mid-session (the frozen index would go stale), and the upstream on-disk
retriever cache is keyed only on `<database_name>_<triple_count>`, so two
different databases that share a name and triple count collide in that cache —
give distinct databases distinct names.

## Running the Co-LMLM audit

Co-LMLM and rel-LMLM both use the Python package name `lmlm`, so they should be
kept in separate environments. Run this command from the public Co-LMLM
environment after downloading its model and index:

```bash
cd /path/to/Co-LMLM

PYTHONPATH=/path/to/HALOCoLMLM/src:src \
uv run python -m lmlm_audit.run_audit \
  --backend co-lmlm \
  --co-lmlm-source-path . \
  --co-lmlm-model-path /path/to/CoLMLM-360M-FW \
  --index-path /path/to/co-lmlm-wiki-index \
  --entries-db-path /path/to/co-lmlm-wiki-index/entries.db \
  --prompt-files /path/to/prompts.jsonl \
  --bootstrap-oracle-from-full \
  --output-dir /path/to/results
```

The audit always runs all three states (`FULL`, `DEL-ON`, `DEL-OFF`) and
auto-detects device, dtype, and attention implementation — there are no flags
for those.

An example prompt file is available at
[data/colmlm/prompts_smoke.example.jsonl](data/colmlm/prompts_smoke.example.jsonl).
For a proper experiment, each prompt should use a reviewed deletion manifest
rather than relying on the oracle bootstrap option.

For the full FineWeb-Edu + Wikipedia index, the backend memory-maps the ~59 GB
FAISS file by default (set `LMLM_FAISS_MMAP=0` to force it into RAM) and
auto-uses the SQLite faiss-id mapping when a `.db` ships alongside the index.
Use `--nprobe` to raise IVF recall (the geometric closure depends on
approximate IVFPQ search; the value used is recorded in each closure manifest).

## Sweeps, closures, and the adversarial evaluation

`--closure`, `--radius-grid`, and `--adversarial` attach to the same
`lmlm-audit` command and work on both backends (drop `provenance` for
rel-LMLM). For example, an entanglement sweep on a custom rel-LMLM database:

```bash
uv run lmlm-audit \
  --database-path data/custom_databases/countries/base.json \
  --prompt-files data/custom_databases/countries/prompts/base/prompts_direct_questions.jsonl \
  --closure geometric,semantic \
  --radius-grid 0.95:0.70:0.05 \
  --neighbor-mode cosine \
  --output-dir outputs/sweep --wandb-activation off
```

## Representational-leakage probe

A standard audit run automatically fits the representational-leakage probe on
its own FULL query embeddings (offline, no GPU, no extra command) and writes
`<prompt>_probe_per_fact.csv` and `<prompt>_probe_summary.csv` next to the
results — reporting L_rep, the behavioral DEL-OFF baseline L, and Δ_rep. It is
skipped for prompt files with too few facts to fit a fact-disjoint probe.

## Comparing rel-LMLM and Co-LMLM

The two backends can't share a process (their `lmlm` packages collide) and
usually live on different machines — rel-LMLM locally, Co-LMLM on a cluster
with the large index. So run each backend separately with the same evaluation
flags into its own `--output-dir`. Because both emit the same result and metric
schema (`cross_state_metrics.csv`, `entanglement_gaps.csv`, …), the comparison
is a join over those CSVs by fact and backend — use whatever analysis tool you
prefer (pandas, a notebook, etc.).

## Papers

- [Auditing Forgetting in Limited Memory Language Models](https://arxiv.org/abs/2607.00605)
- [Pre-training Limited Memory Language Models with Internal and External
  Knowledge](https://arxiv.org/abs/2505.15962)
- [Co-LMLM](https://arxiv.org/abs/2607.07707)

## Citation

```bibtex
@misc{lmlmauditing,
  title         = {Auditing Forgetting in Limited Memory Language Models},
  author        = {Raeesi, Arya and Roed, Hanna},
  year          = {2026},
  eprint        = {2607.00605},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2607.00605},
  doi           = {10.48550/arXiv.2607.00605}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
