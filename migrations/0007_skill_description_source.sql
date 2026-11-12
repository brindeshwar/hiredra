-- 0007_skill_description_source: one home for skill descriptions (graph audit, 2026-11-12). Where each node's
-- description came from, and the exact text its vector was made from, so a changed description re-embeds that one
-- skill instead of the whole index.
ALTER TABLE skill_vocab
    ADD COLUMN description_source text CHECK (description_source IN ('override', 'freehire', 'generated', 'declined')),
    ADD COLUMN embed_input text;   -- "name" or "name: description", exactly as embedded
