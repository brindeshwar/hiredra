"""SQLite storage layer for Hiredra (build sub-slice 2a.1).

Single source of truth for jobs, their pipeline status, the scoring ledger, and
outcomes. Every later stage reads and updates rows here; cross-run dedup and
status tracking work against it.

This module is persistence only — no sourcing, filtering, or scoring logic.

Conventions
-----------
* All timestamps are ISO-8601 UTC strings (second precision). They sort
  lexicographically, so range queries are plain string comparisons.
* ``years_required`` is ``None`` when unknown and ``0`` only for an explicitly
  zero-experience role. The two are never conflated.
* A re-sight (``upsert_job`` on an existing ``job_id``) bumps ``last_seen_at``
  and appends a ``seen`` event. It never regresses ``status`` and never clobbers
  ``score`` / ``score_features`` / ``outcome*``; Job content columns are written
  once, on first insert.
"""

from __future__ import annotations

import secrets
import sqlite3
import time
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# data/hiredra.db at the repo root (this file lives at hiredra/storage/db.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH: Path = _REPO_ROOT / "data" / "hiredra.db"

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 2

#: run statuses (see schema `runs.status`).
RUN_STATUSES: tuple[str, ...] = ("running", "ok", "partial", "failed")

#: Ordered pipeline statuses plus the off-path ``dropped`` (funnel exit — kept in
#: the DB so cross-run dedup never re-processes it).
STATUSES: tuple[str, ...] = (
    "fetched",
    "filtered",
    "scored",
    "tailored",
    "queued",
    "applied",
    "outcome",
    "dropped",
)

#: Documented outcome vocabulary (docs/scoring.md §7). NOT enforced by the DB —
#: the learning loop (v2) may extend it.
OUTCOMES: tuple[str, ...] = (
    "silence",
    "rejected",
    "reply",
    "interview",
    "offer",
    "hired",
)

#: Job-object columns written by ``upsert_job`` (docs/sourcing.md §2). Written
#: once, on first insert; a re-sight does not overwrite them.
_JOB_COLUMNS: tuple[str, ...] = (
    "job_id",
    "url",
    "title",
    "company",
    "description",
    "years_required",
    "source",
    "location",
    "date_posted",
    "job_type",
    "salary",
    "apply_url",
    "is_remote",
)

_REQUIRED_JOB_FIELDS = ("url", "title", "company", "source")


def _now() -> str:
    """Current UTC time as an ISO-8601 string, second precision.

    Tests monkeypatch this module attribute to get a deterministic clock.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    """``<YYYYMMDDThhmmssZ>-<4hex>`` derived from ``_now()``.

    Compact basic-format ISO (no colons → filename/URL safe), explicit ``Z``,
    lexicographically sortable by start time, plus a random suffix so same-second
    starts don't collide. Deterministic under a monkeypatched ``_now`` except the
    hex.
    """
    compact = (
        _now().replace("-", "").replace(":", "").replace("+0000", "Z").replace("+00:00", "Z")
    )
    return f"{compact}-{secrets.token_hex(2)}"


# ─────────────────────────────────────────────────────────────────────────────
# connection / setup
# ─────────────────────────────────────────────────────────────────────────────
def connect(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open and return a tuned connection.

    Creates the parent directory if needed. ``row_factory`` is ``sqlite3.Row``
    and foreign keys are enforced. Pass ``":memory:"`` for an ephemeral DB.
    """
    if str(path) == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Apply ``schema.sql`` (idempotent) + any pending migrations; stamp the version.

    A brand-new DB gets everything from ``schema.sql`` directly. An existing DB
    at an older ``user_version`` is migrated with guarded ``ALTER TABLE``s (each
    checks the column is actually absent, so re-running is harmless).
    """
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    if version < 2:
        _migrate_v2(conn)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def _migrate_v2(conn: sqlite3.Connection) -> None:
    """v1 → v2: add ``jobs.first_seen_run`` (``runs``/``cooldowns`` come from
    ``schema.sql``'s ``CREATE TABLE IF NOT EXISTS``)."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    if "first_seen_run" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN first_seen_run TEXT")


# ─────────────────────────────────────────────────────────────────────────────
# writes
# ─────────────────────────────────────────────────────────────────────────────
def upsert_job(
    conn: sqlite3.Connection,
    job: Mapping[str, Any],
    *,
    first_seen_run: str | None = None,
) -> str:
    """Insert a new job, or re-sight an existing one. Returns the ``job_id``.

    ``job_id`` falls back to ``job["url"]`` when absent. ``url``, ``title``,
    ``company`` and ``source`` are required. ``years_required``, if given, must
    be an ``int`` (``0`` is valid; unknown is ``None``).

    New row: every Job column written, ``first_seen_run`` set (if given),
    ``status`` defaults to ``fetched``, ``first_seen_at``/``last_seen_at``/
    ``updated_at`` set, a ``seen`` event appended.

    Existing row (matched by ``job_id``): only ``last_seen_at``/``updated_at``
    are bumped and a ``seen`` event appended. Status, the scoring ledger, the
    outcome fields, and the Job content columns are all left untouched — a flaky
    re-fetch must never corrupt a good row or move a job backwards through the
    funnel.
    """
    job_id = job.get("job_id") or job.get("url")
    if not job_id:
        raise ValueError("job needs a 'job_id' or a 'url'")
    job_id = str(job_id)

    for field in _REQUIRED_JOB_FIELDS:
        if not job.get(field):
            raise ValueError(f"job {job_id!r} is missing required field {field!r}")

    yr = job.get("years_required")
    if yr is not None and not isinstance(yr, int):
        raise TypeError(
            "years_required must be an int or None "
            "(0 = explicitly zero-experience role; unknown = None)"
        )

    now = _now()
    with conn:
        exists = conn.execute(
            "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()

        if exists is None:
            cols = (*_JOB_COLUMNS, "status_changed_at", "first_seen_at",
                    "last_seen_at", "updated_at", "first_seen_run")
            values = [job_id if c == "job_id" else job.get(c) for c in _JOB_COLUMNS]
            values += [now, now, now, now, first_seen_run]
            placeholders = ", ".join("?" for _ in cols)
            try:
                conn.execute(
                    f"INSERT INTO jobs ({', '.join(cols)}) VALUES ({placeholders})",
                    values,
                )
            except sqlite3.IntegrityError as err:
                raise ValueError(
                    f"url {job.get('url')!r} is already stored under a different "
                    f"job_id (dedup should have merged these upstream)"
                ) from err
            _add_event(conn, job_id, "seen", detail="first sighting", at=now)
        else:
            _touch(conn, job_id, detail="re-sighted", now=now)

    return job_id


def set_status(
    conn: sqlite3.Connection,
    job_id: str,
    to_status: str,
    detail: str | None = None,
) -> None:
    """Move a job to ``to_status`` and append a ``status`` event.

    This is the deliberate stage-transition API; callers own the state machine.
    (The automatic no-regression guard is on ``upsert_job``, not here.)
    """
    if to_status not in STATUSES:
        raise ValueError(f"unknown status {to_status!r}; must be one of {STATUSES}")

    now = _now()
    with conn:
        row = conn.execute(
            "SELECT status FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no job with job_id {job_id!r}")
        from_status = row["status"]
        conn.execute(
            "UPDATE jobs SET status = ?, status_detail = ?, status_changed_at = ?, "
            "updated_at = ? WHERE job_id = ?",
            (to_status, detail, now, now, job_id),
        )
        _add_event(
            conn, job_id, "status",
            from_status=from_status, to_status=to_status, detail=detail, at=now,
        )


def set_years_required(
    conn: sqlite3.Connection, job_id: str, years: int | None
) -> None:
    """Store the filter funnel's extracted years-required feature.

    Not a status change and not an event — ``years_required`` is a stored
    feature (docs/scoring.md §1), written once by the seniority gate regardless
    of whether the gate then keeps or drops the job.
    """
    with conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no job with job_id {job_id!r}")
        conn.execute(
            "UPDATE jobs SET years_required = ?, updated_at = ? WHERE job_id = ?",
            (years, _now(), job_id),
        )


def record_outcome(
    conn: sqlite3.Connection,
    job_id: str,
    outcome: str,
    detail: str | None = None,
    at: str | None = None,
) -> None:
    """Record a post-application outcome and append an ``outcome`` event.

    Does not change ``status`` — when a job moves to the ``outcome`` stage is
    the tracking slice's decision (build step 11). ``outcome`` is free text; the
    ``OUTCOMES`` tuple is the documented vocabulary, not a DB constraint.
    """
    now = _now()
    with conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no job with job_id {job_id!r}")
        conn.execute(
            "UPDATE jobs SET outcome = ?, outcome_at = ?, outcome_detail = ?, "
            "updated_at = ? WHERE job_id = ?",
            (outcome, at or now, detail, now, job_id),
        )
        _add_event(
            conn, job_id, "outcome",
            detail=f"{outcome}: {detail}" if detail else outcome, at=now,
        )


def add_note(conn: sqlite3.Connection, job_id: str, text: str) -> None:
    """Append a free-text ``note`` event. Changes no column."""
    with conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no job with job_id {job_id!r}")
        _add_event(conn, job_id, "note", detail=text, at=_now())


def touch_job(
    conn: sqlite3.Connection, job_id: str, *, run_id: str | None = None
) -> None:
    """Record a re-sighting (dedup's cross-run match).

    Bumps ``last_seen_at``/``updated_at`` and appends a ``seen`` event. Does not
    change ``status``, ``first_seen_run``, or any Job column. Raises ``KeyError``
    if unknown. ``run_id`` (when given) is noted in the event detail.
    """
    now = _now()
    detail = "re-sighted (dedup match)" + (f" [{run_id}]" if run_id else "")
    with conn:
        row = conn.execute(
            "SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"no job with job_id {job_id!r}")
        _touch(conn, job_id, detail=detail, now=now)


def _touch(
    conn: sqlite3.Connection, job_id: str, *, detail: str, now: str
) -> None:
    conn.execute(
        "UPDATE jobs SET last_seen_at = ?, updated_at = ? WHERE job_id = ?",
        (now, now, job_id),
    )
    _add_event(conn, job_id, "seen", detail=detail, at=now)


def _add_event(
    conn: sqlite3.Connection,
    job_id: str,
    kind: str,
    *,
    from_status: str | None = None,
    to_status: str | None = None,
    detail: str | None = None,
    at: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO job_events (job_id, kind, from_status, to_status, detail, at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (job_id, kind, from_status, to_status, detail, at or _now()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# reads
# ─────────────────────────────────────────────────────────────────────────────
def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row is not None else None


def job_exists(conn: sqlite3.Connection, job_id: str) -> bool:
    """True if ``job_id`` is already stored — the cross-run dedup check (§8)."""
    return (
        conn.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        is not None
    )


def iter_jobs(
    conn: sqlite3.Connection,
    *,
    status: str | None = None,
    source: str | None = None,
) -> Iterator[dict]:
    """Yield job rows (oldest first), optionally filtered by status and/or source."""
    clauses: list[str] = []
    params: list[Any] = []
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if source is not None:
        clauses.append("source = ?")
        params.append(source)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    cur = conn.execute(
        f"SELECT * FROM jobs{where} ORDER BY first_seen_at, job_id", params
    )
    for row in cur:
        yield dict(row)


def get_events(conn: sqlite3.Connection, job_id: str) -> list[dict]:
    """All events for a job, oldest first."""
    cur = conn.execute(
        "SELECT * FROM job_events WHERE job_id = ? ORDER BY id", (job_id,)
    )
    return [dict(r) for r in cur]


def status_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """``{status: count}`` over every known status — the dashboard funnel snapshot."""
    counts = dict.fromkeys(STATUSES, 0)
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
    ):
        counts[row["status"]] = row["n"]
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# runs (v2) — the orchestrator's run bookkeeping. See docs/sourcing.md §9-10.
# ─────────────────────────────────────────────────────────────────────────────
def open_run(
    conn: sqlite3.Connection, run_id: str, *, kind: str, config_name: str | None
) -> str:
    """Insert the ``running`` row. Returns ``started_at`` (the caller often wants it)."""
    if kind not in ("scoring", "backfill"):
        raise ValueError(f"unknown run kind {kind!r}")
    now = _now()
    with conn:
        conn.execute(
            "INSERT INTO runs (run_id, kind, config_name, started_at, status) "
            "VALUES (?, ?, ?, ?, 'running')",
            (run_id, kind, config_name, now),
        )
    return now


def checkpoint_run(conn: sqlite3.Connection, run_id: str, *, detail: str) -> None:
    """Update only ``detail`` (the JSON progress blob) — a mid-run checkpoint."""
    with conn:
        conn.execute("UPDATE runs SET detail = ? WHERE run_id = ?", (detail, run_id))


def finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    status: str,
    fetched: int = 0,
    collapsed_in_run: int = 0,
    matched_in_store: int = 0,
    persisted_new: int = 0,
    touched: int = 0,
    detail: str | None = None,
) -> None:
    if status not in RUN_STATUSES or status == "running":
        raise ValueError(f"cannot finish a run with status {status!r}")
    with conn:
        conn.execute(
            "UPDATE runs SET finished_at = ?, status = ?, fetched = ?, "
            "collapsed_in_run = ?, matched_in_store = ?, persisted_new = ?, "
            "touched = ?, detail = COALESCE(?, detail) WHERE run_id = ?",
            (_now(), status, fetched, collapsed_in_run, matched_in_store,
             persisted_new, touched, detail, run_id),
        )


def unfinished_run(
    conn: sqlite3.Connection, kind: str, config_name: str | None
) -> dict | None:
    """Most recent ``running`` row for this (kind, config_name) — for ``--resume``."""
    row = conn.execute(
        "SELECT * FROM runs WHERE kind = ? AND config_name IS ? "
        "AND finished_at IS NULL ORDER BY started_at DESC LIMIT 1",
        (kind, config_name),
    ).fetchone()
    return dict(row) if row is not None else None


def sweep_stale_runs(
    conn: sqlite3.Connection, kind: str, config_name: str | None
) -> int:
    """Mark still-``running`` rows for this (kind, config_name) as ``failed``.

    Called ONLY on a fresh (non-``--resume``) start, before ``open_run``. Scoped
    to the same operation so a scoring start never fails an unfinished backfill.
    Assumes at most one live run per (kind, config_name) — true for the manual
    single-user CLI; add a lock/heartbeat if concurrent same-config runs appear.
    Returns the count swept.
    """
    with conn:
        cur = conn.execute(
            "UPDATE runs SET status = 'failed', finished_at = ? "
            "WHERE kind = ? AND config_name IS ? AND finished_at IS NULL",
            (_now(), kind, config_name),
        )
        return cur.rowcount


def list_runs(conn: sqlite3.Connection, *, limit: int = 20) -> list[dict]:
    cur = conn.execute(
        "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
    )
    return [dict(r) for r in cur]


# ─────────────────────────────────────────────────────────────────────────────
# cooldowns (v2) — cross-process durability for CooldownRegistry (docs §5).
# ─────────────────────────────────────────────────────────────────────────────
def load_cooldowns(conn: sqlite3.Connection) -> dict[str, float]:
    """``{key: until}`` (epoch seconds) for cooldowns that haven't expired."""
    now = time.time()
    return {
        r["key"]: r["until"]
        for r in conn.execute("SELECT key, until FROM cooldowns")
        if r["until"] > now
    }


def save_cooldown(
    conn: sqlite3.Connection, key: str, until: float, reason: str | None = None
) -> None:
    """Write-through from ``CooldownRegistry.on_trip``. Extends, never shortens.

    NOTE: called on the same thread as the ``trip()`` that produced ``until``.
    When the fetch loop is parallelized (deferred), a worker thread must NOT call
    this directly — marshal the save to the main thread. The single-connection /
    main-thread rule is load-bearing.
    """
    with conn:
        conn.execute(
            "INSERT INTO cooldowns (key, until, tripped_at, reason) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET until = max(until, excluded.until), "
            "tripped_at = excluded.tripped_at, reason = excluded.reason",
            (key, until, _now(), reason),
        )


def clear_cooldown(conn: sqlite3.Connection, key: str | None = None) -> None:
    """Drop one cooldown, or all of them (manual override)."""
    with conn:
        if key is None:
            conn.execute("DELETE FROM cooldowns")
        else:
            conn.execute("DELETE FROM cooldowns WHERE key = ?", (key,))
