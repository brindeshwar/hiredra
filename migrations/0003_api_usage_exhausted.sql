-- 0003_api_usage_exhausted: what today's quota files also record (Q5, plan unit DB3b). Google reporting a model's
-- daily quota gone must stop every later run that day without another API call, so the day-row carries it.
ALTER TABLE api_usage
    ADD COLUMN exhausted_at timestamptz,   -- Google said the daily quota is gone (NULL = not today)
    ADD COLUMN note         text;          -- Google's own wording, for the refusal message
