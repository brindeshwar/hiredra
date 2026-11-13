-- 0008_artifacts: every file the system writes for a job (a resume, a cover letter, their PDFs, the payload, the
-- claim-gate report, later the form scan and the answers), and every model call the writing steps make (sizes and
-- outcome only, never the text). Plan contract 3.3 v7, in Postgres.
CREATE TABLE artifacts (
    id             bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id         bigint NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    kind           text NOT NULL,          -- cv_docx|cv_pdf|cover_docx|cover_pdf|payload|claims|form_scan|answers
    path           text NOT NULL,          -- relative to the repo root, under data/
    sha256         text NOT NULL,
    model          text,                   -- NULL for a file written by code alone
    prompt_version text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    used_at        timestamptz             -- the human used it (uploaded it with an application)
);
CREATE INDEX artifacts_job ON artifacts (job_id, kind);

CREATE TABLE llm_calls (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    at          timestamptz NOT NULL DEFAULT now(),
    model       text NOT NULL,
    stage       text NOT NULL,             -- tailor|cover|answers
    job_id      bigint REFERENCES jobs (id) ON DELETE SET NULL,
    ok          boolean NOT NULL,
    error_class text,
    in_chars    integer,
    out_chars   integer,
    latency_ms  integer
);
CREATE INDEX llm_calls_at ON llm_calls (at);
