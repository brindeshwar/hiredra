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
                              'fetched','filtered','scored','tailored',
                              'queued','applied','outcome','dropped')),
    status_detail     TEXT,                    -- why we're here (esp. drop reason)
    status_changed_at TEXT NOT NULL,           -- ISO-8601 UTC

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
    first_seen_at     TEXT NOT NULL,           -- first run that fetched this job_id
    last_seen_at      TEXT NOT NULL,           -- most recent run that re-fetched it
    updated_at        TEXT NOT NULL            -- last write of any kind
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
