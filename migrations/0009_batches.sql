-- 0009_batches: groups of up to 10 prepared applications for the human to review together, the human's decision on
-- each, and the attention items (what only the human can answer, by label, never a value). Plan contract 3.3 v8.
CREATE TABLE batches (
    batch_id    text PRIMARY KEY,                     -- B0001 ...
    created_at  timestamptz NOT NULL DEFAULT now(),
    status      text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'in_review', 'closed'))
);

CREATE TABLE batch_items (
    batch_id    text NOT NULL REFERENCES batches (batch_id) ON DELETE CASCADE,
    job_id      bigint NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    position    smallint NOT NULL,
    decision    text CHECK (decision IN ('applied', 'skipped', 'deferred', 'closed')),
    decided_at  timestamptz,
    PRIMARY KEY (batch_id, job_id)
);
CREATE INDEX batch_items_job ON batch_items (job_id);

CREATE TABLE attention (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id       bigint NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
    category     text NOT NULL,      -- work_auth|salary|notice|relocation|demographic|legal|gov_id|judgment|captcha|...
    field_label  text NOT NULL,      -- the label only, never a value
    created_at   timestamptz NOT NULL DEFAULT now(),
    resolved_at  timestamptz
);
CREATE INDEX attention_job ON attention (job_id);
