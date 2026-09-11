"""Schema v1 -> v2 migration (build sub-slice 2a.8a)."""

from __future__ import annotations

import sqlite3

import pytest

from hiredra.storage import db


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _cols(conn, table) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_fresh_db_is_v2(tmp_path):
    conn = db.connect(tmp_path / "fresh.db")
    db.init_db(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert {"jobs", "job_events", "runs", "cooldowns"} <= _tables(conn)
    assert "first_seen_run" in _cols(conn, "jobs")
    conn.close()


def test_init_db_is_idempotent(tmp_path):
    conn = db.connect(tmp_path / "x.db")
    db.init_db(conn)
    db.init_db(conn)  # must not raise
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    conn.close()


def _make_v1_db(path) -> sqlite3.Connection:
    """A minimal v1 jobs table (no first_seen_run, no runs/cooldowns), user_version=1."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, url TEXT NOT NULL UNIQUE, title TEXT NOT NULL,
            company TEXT NOT NULL, description TEXT, years_required INTEGER,
            source TEXT NOT NULL, location TEXT, date_posted TEXT,
            job_type TEXT, salary TEXT, apply_url TEXT, is_remote INTEGER,
            status TEXT NOT NULL DEFAULT 'fetched', status_detail TEXT,
            status_changed_at TEXT NOT NULL, score REAL, score_features TEXT,
            outcome TEXT, outcome_at TEXT, outcome_detail TEXT,
            first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE job_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
            kind TEXT NOT NULL, from_status TEXT, to_status TEXT, detail TEXT, at TEXT NOT NULL
        );
        PRAGMA user_version = 1;
        """
    )
    conn.execute(
        "INSERT INTO jobs (job_id, url, title, company, source, status_changed_at, "
        "first_seen_at, last_seen_at, updated_at) VALUES "
        "('legacy:1','u://legacy','Old Role','OldCo','greenhouse','t','t','t','t')"
    )
    conn.commit()
    return conn


def test_v1_to_v2_migration_preserves_data(tmp_path):
    path = tmp_path / "v1.db"
    conn = _make_v1_db(path)
    conn.close()

    conn = db.connect(path)
    db.init_db(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert {"runs", "cooldowns"} <= _tables(conn)
    row = conn.execute("SELECT * FROM jobs WHERE job_id='legacy:1'").fetchone()
    assert row["title"] == "Old Role"
    assert "first_seen_run" in row.keys() and row["first_seen_run"] is None  # NULL for legacy

    db.init_db(conn)  # idempotent re-run
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    conn.close()


def test_migrate_v2_is_safe_on_a_db_that_already_has_the_column(tmp_path):
    conn = db.connect(tmp_path / "y.db")
    db.init_db(conn)                 # v2, column present
    db._migrate_v2(conn)             # calling it again must not raise
    conn.close()
