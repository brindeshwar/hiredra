"""Schema v3 -> v4 migration (build sub-slice 4.2): skill_vocab becomes the
graph's node table and skill_alias is added. Same discipline as the v3 rebuild:
a populated old-shape DB, data survives, and job_events' FK into jobs is proven
intact afterwards."""

from __future__ import annotations

import sqlite3

import pytest

from hiredra.storage import db

_NOW = "2026-01-01T00:00:00+00:00"

_V3_EXTRA_DDL = """
CREATE TABLE job_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES jobs (job_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('seen','status','outcome','note')),
    from_status TEXT,
    to_status   TEXT,
    detail      TEXT,
    at          TEXT NOT NULL
);
CREATE TABLE skill_vocab (
    name          TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL,
    seen_count    INTEGER NOT NULL DEFAULT 1
);
"""

_V4_SKILL_COLS = {"vector", "embed_tag", "mine", "level", "source", "edges"}


def _cols(conn, table) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _make_v3_db(path) -> sqlite3.Connection:
    """A genuine v3-shape DB: v3 jobs table, job_events with its FK, the 3-column
    skill_vocab, populated, stamped user_version=3."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(db._JOBS_V3_TABLE_DDL)
    conn.executescript(db._JOBS_V3_INDEX_DDL)
    conn.executescript(_V3_EXTRA_DDL)
    for i in (1, 2):
        conn.execute(
            "INSERT INTO jobs (job_id, url, title, company, source, status, "
            "status_changed_at, first_seen_at, last_seen_at, updated_at) "
            "VALUES (?, ?, 'Backend Engineer', 'Acme', 'greenhouse', 'extracted', ?, ?, ?, ?)",
            (f"j{i}", f"https://x/{i}", _NOW, _NOW, _NOW, _NOW),
        )
    for job_id, detail in (("j1", "first"), ("j1", "second"), ("j2", "third")):
        conn.execute(
            "INSERT INTO job_events (job_id, kind, detail, at) VALUES (?, 'note', ?, ?)",
            (job_id, detail, _NOW),
        )
    conn.execute("INSERT INTO skill_vocab (name, first_seen_at, seen_count) VALUES ('python', ?, 5)", (_NOW,))
    conn.execute("INSERT INTO skill_vocab (name, first_seen_at, seen_count) VALUES ('fast api', ?, 1)", (_NOW,))
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    return conn


def test_v3_to_v4_preserves_rows_and_applies_defaults(tmp_path):
    path = tmp_path / "v3.db"
    _make_v3_db(path).close()

    conn = db.connect(path)
    db.init_db(conn)

    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION  # v3 -> v4 -> v5 in one init_db
    assert _V4_SKILL_COLS <= _cols(conn, "skill_vocab")

    py = db.get_skill_vocab(conn, "python")
    assert py["seen_count"] == 5 and py["first_seen_at"] == _NOW  # untouched
    # existing (Step-3, real-JD) rows pick up the v4 defaults
    assert py["vector"] is None and py["embed_tag"] is None
    assert py["mine"] == 0 and py["level"] is None
    assert py["source"] == "jd" and py["edges"] == "[]"
    assert db.get_skill_vocab(conn, "fast api")["source"] == "jd"

    assert conn.execute("SELECT COUNT(*) FROM skill_alias").fetchone()[0] == 0
    conn.close()


def test_v3_to_v4_leaves_jobs_and_job_events_intact(tmp_path):
    path = tmp_path / "v3.db"
    _make_v3_db(path).close()

    conn = db.connect(path)
    db.init_db(conn)

    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 3
    assert [e["detail"] for e in db.get_events(conn, "j1")] == ["first", "second"]

    # the FK-survival check: job_events still references jobs (not a renamed-away
    # table), nothing is orphaned, and the DB is structurally sound
    fks = conn.execute("PRAGMA foreign_key_list(job_events)").fetchall()
    assert [(r["table"], r["from"], r["to"], r["on_delete"]) for r in fks] == [("jobs", "job_id", "job_id", "CASCADE")]
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    # and the FK still bites and cascades
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO job_events (job_id, kind, at) VALUES ('ghost', 'note', 't')")
    conn.execute("DELETE FROM jobs WHERE job_id = 'j2'")
    assert conn.execute("SELECT COUNT(*) FROM job_events WHERE job_id = 'j2'").fetchone()[0] == 0
    conn.close()


def test_migrated_skill_vocab_has_the_same_shape_as_a_fresh_one(tmp_path):
    path = tmp_path / "v3.db"
    _make_v3_db(path).close()
    migrated = db.connect(path)
    db.init_db(migrated)
    fresh = db.connect(tmp_path / "fresh.db")
    db.init_db(fresh)

    def shape(c):
        return [tuple(r) for r in c.execute("PRAGMA table_info(skill_vocab)")]

    assert shape(migrated) == shape(fresh)
    assert _cols(migrated, "skill_alias") == _cols(fresh, "skill_alias")
    migrated.close()
    fresh.close()


@pytest.mark.parametrize(
    "column,bad",
    [("source", "'bogus'"), ("mine", "2"), ("level", "'expert'")],
)
def test_migrated_table_enforces_the_v4_checks(tmp_path, column, bad):
    path = tmp_path / "v3.db"
    _make_v3_db(path).close()
    conn = db.connect(path)
    db.init_db(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(f"INSERT INTO skill_vocab (name, first_seen_at, {column}) VALUES ('x', 't', {bad})")
    conn.close()


def test_v4_migration_is_idempotent_and_resumes_a_partial_run(tmp_path):
    path = tmp_path / "v3.db"
    conn = _make_v3_db(path)
    # a half-finished earlier attempt: two of the six columns already added
    conn.execute("ALTER TABLE skill_vocab ADD COLUMN vector BLOB")
    conn.execute("ALTER TABLE skill_vocab ADD COLUMN embed_tag TEXT")
    conn.close()

    conn = db.connect(path)
    db.init_db(conn)
    assert _V4_SKILL_COLS <= _cols(conn, "skill_vocab")
    db.init_db(conn)  # and again: must not raise or duplicate anything
    db._migrate_v4(conn)
    assert conn.execute("SELECT COUNT(*) FROM skill_vocab").fetchone()[0] == 2
    conn.close()


def test_fresh_db_gets_v4_shape_directly(tmp_path):
    conn = db.connect(tmp_path / "fresh.db")
    db.init_db(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert _V4_SKILL_COLS <= _cols(conn, "skill_vocab")
    assert {"alias", "canonical", "created_at"} == _cols(conn, "skill_alias")
    conn.close()


def test_v3_to_v4_fk_guard_fires_on_a_pre_existing_orphan(tmp_path):
    """The closing foreign_key_check is live, not decorative: an orphaned
    job_event (already present before migrating) aborts the migration loudly
    and leaves the DB un-stamped, rather than blessing a broken database."""
    path = tmp_path / "v3.db"
    conn = _make_v3_db(path)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("INSERT INTO job_events (job_id, kind, at) VALUES ('ghost', 'note', 't')")
    conn.commit()
    conn.close()

    conn = db.connect(path)
    with pytest.raises(RuntimeError, match="v4 migration broke a foreign key"):
        db.init_db(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 3  # not stamped
    conn.close()
