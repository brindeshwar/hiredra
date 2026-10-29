-- 0004_search_on_demand: full-text search stays Postgres (decision D1, decisions #34), but nothing is stored for it
-- until a screen searches. The stored `search` tsvector was 540 MB of the 1.33 GB database (more than the
-- descriptions it was built from), and its GIN index and the title trigram index had never been used.
--
-- job_tsvector() is the one definition of a job's search text. When the app's job list needs search:
--   CREATE INDEX jobs_search ON jobs USING gin (job_tsvector(title, company, description));
--   ... WHERE job_tsvector(title, company, description) @@ websearch_to_tsquery('english', $1)
-- A query must use the same expression to hit that index. Typo-tolerant title search would be
--   CREATE INDEX jobs_title_trgm ON jobs USING gin (title gin_trgm_ops);  (pg_trgm stays installed)
DROP INDEX IF EXISTS jobs_search;
DROP INDEX IF EXISTS jobs_title_trgm;
ALTER TABLE jobs DROP COLUMN search;

CREATE FUNCTION job_tsvector(title text, company text, description text) RETURNS tsvector
    LANGUAGE sql IMMUTABLE PARALLEL SAFE
    RETURN setweight(to_tsvector('english'::regconfig, coalesce(title, '')), 'A')
        || setweight(to_tsvector('english'::regconfig, coalesce(company, '')), 'B')
        || setweight(to_tsvector('english'::regconfig, coalesce(description, '')), 'C');
