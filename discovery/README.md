# discovery/ -- company discovery & one-time harvest

**This is a separate, off-to-the-side program. It is NOT part of Hiredra's live apply pipeline**
(`hiredra/sourcing`, `hiredra/filter.py`, `data/hiredra.db`) and nothing here is imported by it.

## What it does

Finds every company that posts India-or-remote-India-eligible roles across our three ATS
(greenhouse, lever, ashby), so Path 1 (`config/companies.yml`) and Path 2 (discovery) can be
closed out with a durable company list instead of a hand-picked one, and freezes the full
collected job corpus so the funnel can be tuned against it later without ever re-fetching.

It reuses the existing sourcing/filter code unchanged (the ATS providers, the http layer's
pacing/cooldowns, the orchestrator's `runs` table + `--resume`, `dedup`, `ScoringSink`, and
`hiredra.filter.run_funnel`) — this file is orchestration glue over those pieces, plus the
open-jobs ingest and the CSV/parquet export.

## Why a separate database

The harvest corpus is large (100k+ jobs expected) and would pollute `data/hiredra.db`, which
tracks jobs through the live apply stages. Everything here reads/writes only
`data/discovery/discovery.db` — a completely separate file, same schema (reused via
`hiredra.storage.db.init_db()`), never touching the canonical one.

## Running it (phases, in order)

```sh
pip install -r discovery/requirements.txt   # duckdb, on top of the root requirements.txt

python -m discovery.harvest pool         # covered-vs-not-covered split -> data/discovery/not_covered.yml
python -m discovery.harvest ingest       # pulls + ingests open-jobs' greenhouse/lever/ashby parquet (~1.2 GB)
python -m discovery.harvest self-fetch   # fetches the not-covered companies live (paced, resumable, --resume built in)
python -m discovery.harvest funnel       # runs the existing location -> seniority -> role-type funnel
python -m discovery.harvest export       # writes companies.csv + raw.parquet, backs up the db once
```

Each phase is independent and safe to re-run:
- `ingest` is idempotent (`upsert_job` dedups by `job_id`).
- `self-fetch` is the orchestrator's own `--resume` (per-company persist, a `runs` row).
- `funnel` only ever touches `status='fetched'` rows.
- `export` refuses to overwrite `raw.parquet` or `discovery.db.bak` if they already exist —
  those are the frozen corpus. `companies.csv` is the regenerable deliverable and does get
  overwritten on a fresh export.

## Outputs (`data/discovery/`, git-ignored, same as `data/`)

- `discovery.db` — the working store during processing.
- `discovery.db.bak` — the frozen corpus, written once by `export` and never overwritten again.
  Run any later filter-tuning experiment against a **copy** of this file, never the original.
- `exports/raw.parquet` — the same corpus as one portable file (via DuckDB's sqlite scanner).
- `exports/companies.csv` — the real deliverable: `ats,company,tag` where `tag` is `confirmed`
  (at least one job with a definite India/remote-eligible location match) or `unknown` (kept only
  because the location stage couldn't tell, never a confirmed match). Feeds `config/companies.wide.yml`
  as a follow-up step outside this program.

## Deliberately not built here

- **The career-ops reverse-ATS sweep** (`scan-ats-full.mjs`-style discovery of boards nobody told us
  about) — flagged as a bigger lift (real new discovery logic, not dataset ingestion) and deferred.
- Anything that touches `config/companies.wide.yml` or the live pipeline directly — this program's
  job ends at `companies.csv`; wiring it back in is a separate, reviewed step.
