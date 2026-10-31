-- 0005_skill_source: a job's skills come from two producers (decision D10.3). The extraction LLM, and after it
-- freehire's skill dictionary (hiredra/skilltag.py), which adds the skills the LLM did not list as good_to_have.
-- Each row says which one found it, so a dictionary skill can be told apart, audited and re-derived for free.
-- jobs.dictionary_version stamps the dictionary + stoplists that produced a job's dictionary rows (NULL = never
-- run), separate from extraction_version: re-running the dictionary costs nothing and never makes a job stale
-- for the LLM.
ALTER TABLE job_skills ADD COLUMN source text NOT NULL DEFAULT 'llm' CHECK (source IN ('llm', 'dictionary'));
ALTER TABLE jobs ADD COLUMN dictionary_version text;
