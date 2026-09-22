-- Hiredra SQLite storage layer -- build sub-slice 2a.1.
-- Applied (idempotently) by hiredra.storage.db.init_db().
-- Persistence only: no sourcing / filtering / scoring logic lives here.

PRAGMA journal_mode = WAL;   -- concurrent dashboard reads while the pipeline writes

-- ---------------------------------------------------------------
-- jobs -- one row per unique posting (rows arrive already deduped).
-- Columns 1:1 with the Job object (docs/sourcing.md sec. 2), plus
-- the pipeline status, the scoring-ledger space, outcome logging,
-- and ingestion metadata.
--
-- Re-sighting rule (enforced in db.upsert_job, not here): a repeat
-- fetch of an existing job_id bumps last_seen_at and appends a
-- 'seen' event only. It NEVER regresses status and NEVER clobbers
-- score / score_features / outcome*. The Job content columns are
-- written once, on first insert.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS jobs (
    -- identity (Job sec. 2)
    job_id            TEXT PRIMARY KEY,        -- stable source ID; falls back to url
    url               TEXT NOT NULL UNIQUE,    -- listing link; a dedup key
    title             TEXT NOT NULL,
    company           TEXT NOT NULL,
    description       TEXT,                    -- raw JD text
    years_required    INTEGER,                 -- best-effort parse from source.
                                               -- NULL = unknown. 0 = an explicitly
                                               -- zero-experience role (a real value,
                                               -- never a stand-in for unknown).

    -- provenance / downstream (Job sec. 2)
    source            TEXT NOT NULL,           -- 'greenhouse', 'jobspy-linkedin', ...
    location          TEXT,
    date_posted       TEXT,                    -- ISO-8601 date from source, else NULL

    -- optional-if-free (Job sec. 2; NULL when the source omits it)
    job_type          TEXT,                    -- fulltime / intern / contract
    salary            TEXT,                    -- raw salary string as given
    apply_url         TEXT,                    -- when different from url
    is_remote         INTEGER,                 -- 0 / 1 / NULL (no bool in SQLite)

    -- pipeline status (docs/sourcing.md sec. 10, CLAUDE.md sec. 7)
    status            TEXT NOT NULL DEFAULT 'fetched'
                          CHECK (status IN (
                              'fetched','filtered','extracted','scored','tailored',
                              'queued','applied','outcome','dropped')),
    status_detail     TEXT,                    -- why we're here (esp. drop reason)
    status_changed_at TEXT NOT NULL,           -- ISO-8601 UTC

    -- skill extraction (build step 3; NULL until step 3 runs). Also NULL
    -- (domain_bucket/legitimacy) on a too-thin-to-call skip -- extraction_
    -- confidence is 'low' in that case, never guessed as anything else.
    skills                 TEXT,               -- JSON array of {name, requirement}
    domain_bucket          TEXT,               -- agent_llm|backend|data|ml|generic_swe|other
    legitimacy             TEXT,               -- legitimate|suspect|likely_ghost
    extraction_confidence  TEXT,               -- high|low

    -- scoring-ledger space (build step 6 finalizes; NULL until then)
    score             REAL,
    score_features    TEXT,                    -- JSON: domain bucket, coverage,
                                               -- gap_distance, fit vector, skills

    -- outcome logging (docs/scoring.md sec. 7; filled after 'applied')
    outcome           TEXT,                    -- silence | rejected | reply |
                                               -- interview | offer | hired
    outcome_at        TEXT,
    outcome_detail    TEXT,

    -- ingestion metadata
    first_seen_at     TEXT NOT NULL,           -- first time this job_id was fetched (ISO-8601 UTC)
    last_seen_at      TEXT NOT NULL,           -- most recent re-fetch
    updated_at        TEXT NOT NULL,           -- last write of any kind
    first_seen_run    TEXT                     -- runs.run_id of the run that first fetched it (v2)
);

CREATE INDEX IF NOT EXISTS idx_jobs_status        ON jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_source        ON jobs (source);
CREATE INDEX IF NOT EXISTS idx_jobs_company_title ON jobs (company, title);   -- dedup path 2
CREATE INDEX IF NOT EXISTS idx_jobs_outcome       ON jobs (outcome);
CREATE INDEX IF NOT EXISTS idx_jobs_date_posted   ON jobs (date_posted);

-- ---------------------------------------------------------------
-- job_events -- append-only. One row per re-sighting / status
-- change / outcome / note. Never updated, never deleted. This is
-- what makes the dashboard's funnel-velocity and yield-over-time
-- queries possible (mirrors career-ops' status-log.tsv,
-- docs/repos.md sec. 1c).
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES jobs (job_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('seen','status','outcome','note')),
    from_status TEXT,
    to_status   TEXT,
    detail      TEXT,
    at          TEXT NOT NULL                  -- ISO-8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_events_job  ON job_events (job_id);
CREATE INDEX IF NOT EXISTS idx_events_at   ON job_events (at);
CREATE INDEX IF NOT EXISTS idx_events_kind ON job_events (kind);

-- ---------------------------------------------------------------
-- runs (v2) -- one row per orchestrator run. Per-run rollups the
-- dashboard/learning-loop need; job_events only has per-job history.
-- run_id = <YYYYMMDDThhmmssZ>-<4hex>. started_at/finished_at are
-- ordinary ISO-8601 UTC. detail = JSON RunReport (incl. the list of
-- companies fetched so far, and jobspy_done, for --resume).
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    kind             TEXT NOT NULL CHECK (kind IN ('scoring','backfill')),
    config_name      TEXT,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,                     -- NULL = crashed / still running
    status           TEXT NOT NULL DEFAULT 'running'
                         CHECK (status IN ('running','ok','partial','failed')),
    fetched          INTEGER NOT NULL DEFAULT 0,
    collapsed_in_run INTEGER NOT NULL DEFAULT 0,
    matched_in_store INTEGER NOT NULL DEFAULT 0,
    persisted_new    INTEGER NOT NULL DEFAULT 0,
    touched          INTEGER NOT NULL DEFAULT 0,
    detail           TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_started ON runs (started_at);
CREATE INDEX IF NOT EXISTS idx_runs_open    ON runs (kind, config_name, finished_at);

-- ---------------------------------------------------------------
-- cooldowns (v2) -- cross-process durability for CooldownRegistry.
-- `until` is EPOCH SECONDS (deadline arithmetic wants a float, not
-- an ISO string parsed on every request); `tripped_at` is ISO for
-- the human. Key = a host or "jobspy-<site>".
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cooldowns (
    key        TEXT PRIMARY KEY,
    until      REAL NOT NULL,
    tripped_at TEXT NOT NULL,
    reason     TEXT
);

-- ---------------------------------------------------------------
-- skill_vocab (v3, extended in v4) -- the growing canonical skill
-- vocabulary AND the Step 4 graph's node table (docs/graph.md sec 2:
-- one embedded index, not a graph plus a vector store). name =
-- canonical, case-folded. Step 3 wrote (name, first_seen_at,
-- seen_count); v4 adds the rest of the Skill object:
--   vector     float32 little-endian BLOB, L2-normalized, NULL until
--              embedded (768 x 4 = 3072 bytes at the configured dims)
--   embed_tag  the version tag the vector was made under
--              (model|dims|prefix -- hiredra.graph.config); a row whose
--              tag differs from the live config is STALE and is
--              re-embedded before scoring
--   mine       1 = the owner has it (overlay from the master profile;
--              deferred -- nothing sets it yet), else a market/JD node
--   level      only if mine (docs/graph.md sec 2)
--   source     where the node first came from. 'profile' is here now
--              because a CHECK can't be altered without a table rebuild
--   edges      JSON list, ALWAYS '[]' through Phase A (docs/graph.md sec 3)
-- seen_count counts sightings in OUR jobs only: market-harvested nodes
-- start at 0, so it stays meaningful for yield analysis.
-- No index on the v4 columns on purpose: init_db runs this script BEFORE
-- migrations, so an index on a column a v3 DB doesn't have yet would
-- fail; the table is a few thousand rows and every scan of it is cheap.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skill_vocab (
    name          TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL,
    seen_count    INTEGER NOT NULL DEFAULT 1,
    vector        BLOB,
    embed_tag     TEXT,
    mine          INTEGER NOT NULL DEFAULT 0 CHECK (mine IN (0, 1)),
    level         TEXT CHECK (level IS NULL OR level IN ('production','built-with','learned','interview-ready')),
    source        TEXT NOT NULL DEFAULT 'jd' CHECK (source IN ('market','jd','profile')),
    edges         TEXT NOT NULL DEFAULT '[]'
);

-- ---------------------------------------------------------------
-- skill_alias (v4) -- a spelling that was merged into a canonical node
-- ("fast api" -> "fastapi"). A table, not a JSON list on the node, so
-- "have I seen this exact string before?" is one indexed lookup: that
-- is what makes a merged spelling a cache hit instead of a fresh embed
-- call. Invariant (enforced by the helpers in hiredra.storage.db): an
-- alias string is never also a skill_vocab.name -- a string is a node or
-- an alias, not both. Aliases are always flat (one hop to a node).
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS skill_alias (
    alias      TEXT PRIMARY KEY,
    canonical  TEXT NOT NULL REFERENCES skill_vocab (name) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_skill_alias_canonical ON skill_alias (canonical);
