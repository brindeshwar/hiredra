-- 0001_init: Hiredra's first Postgres schema (docs/freehire-merge/05-schema.md, approved 2026-10-28).
-- Applied once by hiredra.pg.migrate, inside one transaction, and recorded in schema_migrations.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS schema_migrations (  -- the runner creates it first, too
    version    text PRIMARY KEY,                 -- '0001_init'
    applied_at timestamptz NOT NULL DEFAULT now()
);

-- ── Companies and boards (D2.1, D2.4, D2.5, D5.1, D5.2, D5.5, D2.3) ─────────────────────────
CREATE TABLE companies (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    slug        text NOT NULL UNIQUE,            -- normalised name (D6.5 companyname)
    name        text NOT NULL,                   -- display name
    website     text,
    tags        text[] NOT NULL DEFAULT '{}',    -- collections: yc, unicorn, indian-roots ... (D2.5)
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX companies_tags ON companies USING gin (tags);

CREATE TABLE company_slug_aliases (                -- other spellings that mean the same employer (D2.4)
    alias       text PRIMARY KEY,
    company_id  bigint NOT NULL REFERENCES companies (id),
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE boards (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    provider       text NOT NULL,                -- greenhouse | lever | ashby | workday | ...
    board          text NOT NULL,                -- the slug / tenant on that platform
    region         text NOT NULL DEFAULT '',     -- e.g. Lever EU, Workday data centre
    company_id     bigint REFERENCES companies (id),
    careers_url    text,
    status         text NOT NULL DEFAULT 'pending'
                       CHECK (status IN ('pending', 'active', 'retired', 'rejected')),   -- D5.5
    status_reason  text,
    provenance     text NOT NULL,                -- curated | openjobs | ats-inventory | atsdetect | yc | orphan (D2.2)
    list_tag       text,                         -- today's companies.yml tag: confirmed | unknown
    cadence        text NOT NULL DEFAULT 'daily' CHECK (cadence IN ('daily', 'weekly')),  -- D2.3 (b)
    next_due_at    timestamptz NOT NULL DEFAULT now(),                                    -- D5.1
    last_eligible_at timestamptz,                -- last crawl that produced an India/remote-eligible job (D2.3)
    created_at     timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX boards_identity ON boards (provider, lower(board), region);
CREATE INDEX boards_due ON boards (next_due_at) WHERE status IN ('pending', 'active');

CREATE TABLE board_health (                         -- D5.2: 3 failures, then 6h x 2^(f-3), cap 24h; success heals
    board_id              bigint PRIMARY KEY REFERENCES boards (id),
    consecutive_failures  integer NOT NULL DEFAULT 0,
    cooldown_until        timestamptz,
    last_success_at       timestamptz,
    last_failure_at       timestamptz,
    last_error            text,
    last_yield_at         timestamptz             -- last crawl whose listing named any posting
);

-- ── Runs (D5.4) ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE runs (
    run_id           text PRIMARY KEY,           -- kept as today: '20261017T163834Z-cdc4'
    kind             text NOT NULL CHECK (kind IN ('crawl', 'scoring', 'backfill')),
    config_name      text,
    started_at       timestamptz NOT NULL,
    finished_at      timestamptz,
    status           text NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'ok', 'partial', 'failed')),
    fetched          integer NOT NULL DEFAULT 0,
    collapsed_in_run integer NOT NULL DEFAULT 0,
    matched_in_store integer NOT NULL DEFAULT 0,
    persisted_new    integer NOT NULL DEFAULT 0,
    touched          integer NOT NULL DEFAULT 0,
    detail           jsonb
);

CREATE TABLE board_crawls (                          -- D5.4: one row per board per run
    run_id       text NOT NULL REFERENCES runs (run_id),
    board_id     bigint NOT NULL REFERENCES boards (id),
    started_at   timestamptz NOT NULL,
    duration_ms  integer,
    status       text NOT NULL CHECK (status IN ('ok', 'partial', 'failed', 'gone', 'blocked')),
    fetched      integer NOT NULL DEFAULT 0,
    new          integer NOT NULL DEFAULT 0,
    changed      integer NOT NULL DEFAULT 0,     -- content_hash moved (D6.2)
    closed       integer NOT NULL DEFAULT 0,     -- soft-closed by this crawl (D5.3)
    eligible     integer NOT NULL DEFAULT 0,     -- India/remote-eligible jobs seen (feeds cadence, D2.3)
    error        text,
    PRIMARY KEY (run_id, board_id)
);

-- ── Jobs ────────────────────────────────────────────────────────────────────────────────────
CREATE TABLE jobs (
    id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source             text NOT NULL,            -- greenhouse | lever | ashby | jobspy-indeed | ...
    external_id        text NOT NULL,            -- the platform's id; board-prefixed where only tenant-unique (Q4)
    job_key            text GENERATED ALWAYS AS (source || ':' || external_id) STORED UNIQUE,  -- today's job_id
    board_id           bigint REFERENCES boards (id),      -- NULL for JobSpy
    company_id         bigint REFERENCES companies (id),
    url                text NOT NULL UNIQUE,
    apply_url          text,
    title              text NOT NULL,
    company            text NOT NULL,            -- the name as the posting gives it
    location           text,                     -- raw
    countries          text[] NOT NULL DEFAULT '{}',   -- hiredra.location.parse, for filters and dedup (Q3)
    is_remote          boolean,
    work_mode          text,                     -- remote | hybrid | onsite (structured where the ATS gives it)
    date_posted        date,
    job_type           text,                     -- raw, as given
    employment_type    text,                     -- mapped: full_time | part_time | internship | contract
    salary             text,                     -- raw string, as today
    salary_min         numeric,
    salary_max         numeric,
    salary_currency    text,
    salary_period      text,                     -- year | month | hour
    description        text,                     -- plain text for the funnel and the LLM (Q1)
    description_html   text,                     -- sanitized HTML for the desktop app (D6.7)
    content_hash       text,                     -- D6.2: re-extract / re-embed only when it moves
    role_fingerprint   text,                     -- D6.3, includes the country set (Q3)
    duplicate_of       bigint REFERENCES jobs (id),       -- D6.3: hidden copy of a canonical job
    hydration_attempted_at timestamptz,          -- D3.4: snippet-then-detail retry window
    years_required     integer,
    -- pipeline state
    status             text NOT NULL DEFAULT 'fetched' CHECK (status IN (
                           'fetched', 'filtered', 'extracted', 'scored', 'tailored',
                           'queued', 'applied', 'outcome', 'dropped')),
    status_detail      text,
    status_changed_at  timestamptz NOT NULL DEFAULT now(),
    -- extraction (the skills themselves live in job_skills)
    domain_bucket          text,
    legitimacy             text,
    extraction_confidence  text,
    extraction_version     text,
    -- scoring
    score              real,
    score_features     jsonb,
    review_reason      text[],
    released_at        timestamptz,
    scored_at          timestamptz,
    score_version      text,
    score_embed_tag    text,
    score_skills_hash  text,
    score_profile_hash text,
    -- outcome
    outcome            text,
    outcome_at         timestamptz,
    outcome_detail     text,
    -- lifecycle
    first_seen_run     text REFERENCES runs (run_id),
    first_seen_at      timestamptz NOT NULL DEFAULT now(),
    last_seen_at       timestamptz NOT NULL DEFAULT now(),
    closed_at          timestamptz,              -- D5.3 soft close; reopened (NULL) when listed again
    updated_at         timestamptz NOT NULL DEFAULT now(),
    search             tsvector GENERATED ALWAYS AS (
                           setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
                           setweight(to_tsvector('english', coalesce(company, '')), 'B') ||
                           setweight(to_tsvector('english', coalesce(description, '')), 'C')) STORED,
    UNIQUE (source, external_id)
);
CREATE INDEX jobs_status         ON jobs (status);
CREATE INDEX jobs_ranked         ON jobs (status, score DESC) WHERE closed_at IS NULL;
CREATE INDEX jobs_board_open     ON jobs (board_id, last_seen_at) WHERE closed_at IS NULL;  -- the close sweep
CREATE INDEX jobs_company        ON jobs (company_id);
CREATE INDEX jobs_fingerprint    ON jobs (role_fingerprint);
CREATE INDEX jobs_date_posted    ON jobs (date_posted);
CREATE INDEX jobs_outcome        ON jobs (outcome);
CREATE INDEX jobs_countries      ON jobs USING gin (countries);
CREATE INDEX jobs_search         ON jobs USING gin (search);
CREATE INDEX jobs_title_trgm     ON jobs USING gin (title gin_trgm_ops);

CREATE TABLE job_skills (                            -- D1.6: today's jobs.skills JSON, one row per skill
    job_id       bigint NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    skill        text NOT NULL,                  -- canonical name; no FK, new skills arrive before the graph has them
    requirement  text NOT NULL CHECK (requirement IN ('required', 'good_to_have')),
    position     smallint NOT NULL,              -- the extraction's order
    PRIMARY KEY (job_id, skill)
);
CREATE INDEX job_skills_skill ON job_skills (skill);

CREATE TABLE job_events (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id       bigint NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    kind         text NOT NULL CHECK (kind IN ('seen', 'status', 'outcome', 'note')),
    from_status  text,
    to_status    text,
    detail       text,
    at           timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX job_events_job ON job_events (job_id, at);
CREATE INDEX job_events_kind ON job_events (kind, at);

-- ── Skill graph (D1.7, D1.15) ───────────────────────────────────────────────────────────────
CREATE TABLE skill_vocab (
    name           text PRIMARY KEY,
    description    text,                         -- freehire descriptions.tsv; the embedding input is "name: description"
    first_seen_at  timestamptz NOT NULL DEFAULT now(),
    seen_count     integer NOT NULL DEFAULT 1,
    vector         vector(1024),                 -- Qwen3-Embedding-0.6B (D1.15)
    embed_tag      text,
    mine           boolean NOT NULL DEFAULT false,
    level          text CHECK (level IS NULL OR level IN ('production', 'built-with', 'learned', 'interview-ready')),
    source         text NOT NULL DEFAULT 'jd' CHECK (source IN ('market', 'jd', 'profile')),
    edges          jsonb NOT NULL DEFAULT '[]'
);
CREATE INDEX skill_vocab_vector ON skill_vocab USING hnsw (vector vector_cosine_ops);

CREATE TABLE skill_alias (
    alias       text PRIMARY KEY,
    canonical   text NOT NULL REFERENCES skill_vocab (name) ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX skill_alias_canonical ON skill_alias (canonical);

-- ── API quotas (Q5) ─────────────────────────────────────────────────────────────────────────
CREATE TABLE api_usage (                             -- today's data/usage/*.json and embed_usage.json
    provider  text NOT NULL,                         -- gemini | nvidia | ...
    model     text NOT NULL,
    day       date NOT NULL,                         -- the provider's quota day
    calls     integer NOT NULL DEFAULT 0,
    tokens    bigint NOT NULL DEFAULT 0,
    PRIMARY KEY (provider, model, day)
);
