-- 0002_cooldowns: today's crawler cooldowns, kept as they are until the sourcing unit replaces them with
-- board_health (owner's call 2026-10-29: the side-by-side run needs identical crawl behaviour).
-- key: a host, or a JobSpy (term, site); until: epoch seconds, as CooldownRegistry keeps it.
CREATE TABLE cooldowns (
    key        text PRIMARY KEY,
    until      double precision NOT NULL,
    tripped_at timestamptz NOT NULL,
    reason     text
);
