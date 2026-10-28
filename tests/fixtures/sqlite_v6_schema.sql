-- The SQLite schema Hiredra used until the move to Postgres (user_version 6), read from data/hiredra.db's
-- sqlite_master on 2026-10-29. Kept only so tests/test_pg_import.py can build a legacy file to import.
CREATE TABLE cooldowns (
    key        TEXT PRIMARY KEY,
    until      REAL NOT NULL,
    tripped_at TEXT NOT NULL,
    reason     TEXT
);

CREATE TABLE job_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES jobs (job_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('seen','status','outcome','note')),
    from_status TEXT,
    to_status   TEXT,
    detail      TEXT,
    at          TEXT NOT NULL                  -- ISO-8601 UTC
);

CREATE TABLE jobs (
    job_id            TEXT PRIMARY KEY,
    url               TEXT NOT NULL UNIQUE,
    title             TEXT NOT NULL,
    company           TEXT NOT NULL,
    description       TEXT,
    years_required    INTEGER,
    source            TEXT NOT NULL,
    location          TEXT,
    date_posted       TEXT,
    job_type          TEXT,
    salary            TEXT,
    apply_url         TEXT,
    is_remote         INTEGER,
    status            TEXT NOT NULL DEFAULT 'fetched'
                          CHECK (status IN (
                              'fetched','filtered','extracted','scored','tailored',
                              'queued','applied','outcome','dropped')),
    status_detail     TEXT,
    status_changed_at TEXT NOT NULL,
    skills                 TEXT,
    domain_bucket          TEXT,
    legitimacy             TEXT,
    extraction_confidence  TEXT,
    score             REAL,
    score_features    TEXT,
    outcome           TEXT,
    outcome_at        TEXT,
    outcome_detail    TEXT,
    first_seen_at     TEXT NOT NULL,
    last_seen_at      TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    first_seen_run    TEXT
, company_slug TEXT, list_tag TEXT, extraction_version TEXT, review_reason TEXT, released_at TEXT, scored_at TEXT, score_version TEXT, score_embed_tag TEXT, score_skills_hash TEXT, score_profile_hash TEXT);

CREATE TABLE runs (
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

CREATE TABLE skill_alias (
    alias      TEXT PRIMARY KEY,
    canonical  TEXT NOT NULL REFERENCES skill_vocab (name) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);

CREATE TABLE skill_vocab (
    name          TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL,
    seen_count    INTEGER NOT NULL DEFAULT 1
, vector BLOB, embed_tag TEXT, mine INTEGER NOT NULL DEFAULT 0 CHECK (mine IN (0, 1)), level TEXT CHECK (level IS NULL OR level IN ('production','built-with','learned','interview-ready')), source TEXT NOT NULL DEFAULT 'jd' CHECK (source IN ('market','jd','profile')), edges TEXT NOT NULL DEFAULT '[]');

CREATE INDEX idx_events_at   ON job_events (at);

CREATE INDEX idx_events_job  ON job_events (job_id);

CREATE INDEX idx_events_kind ON job_events (kind);

CREATE INDEX idx_jobs_company_title ON jobs (company, title);

CREATE INDEX idx_jobs_date_posted   ON jobs (date_posted);

CREATE INDEX idx_jobs_outcome       ON jobs (outcome);

CREATE INDEX idx_jobs_source        ON jobs (source);

CREATE INDEX idx_jobs_status        ON jobs (status);

CREATE INDEX idx_jobs_status_score  ON jobs (status, score);

CREATE INDEX idx_runs_open    ON runs (kind, config_name, finished_at);

CREATE INDEX idx_runs_started ON runs (started_at);

CREATE INDEX idx_skill_alias_canonical ON skill_alias (canonical);
