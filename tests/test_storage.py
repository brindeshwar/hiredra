"""Tests for the SQLite storage layer (build sub-slice 2a.1)."""

from __future__ import annotations

import itertools

import pytest

from hiredra.storage import db


@pytest.fixture()
def conn():
    c = db.connect(":memory:")
    db.init_db(c)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def fake_clock(monkeypatch):
    """Deterministic, monotonically increasing timestamps."""
    counter = itertools.count(1)
    monkeypatch.setattr(
        db, "_now", lambda: f"2026-09-09T00:00:{next(counter):02d}+00:00"
    )


def _job(**over):
    base = {
        "job_id": "gh-1",
        "url": "https://boards.greenhouse.io/acme/jobs/1",
        "title": "Backend Engineer",
        "company": "Acme",
        "description": "Build things.",
        "years_required": 0,
        "source": "greenhouse",
        "location": "Remote",
        "date_posted": "2026-09-01",
        "job_type": "fulltime",
        "salary": None,
        "apply_url": None,
        "is_remote": 1,
    }
    base.update(over)
    return base


# ── schema / setup ───────────────────────────────────────────────────────────
def test_init_db_is_idempotent_and_versioned(conn):
    db.init_db(conn)  # second run must not raise
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"jobs", "job_events"} <= tables


# ── insert ───────────────────────────────────────────────────────────────────
def test_insert_new_job_defaults_to_fetched_and_logs_seen(conn):
    job_id = db.upsert_job(conn, _job())
    assert job_id == "gh-1"

    row = db.get_job(conn, "gh-1")
    assert row["status"] == "fetched"
    assert row["first_seen_at"] == row["last_seen_at"]
    assert row["score"] is None and row["outcome"] is None

    events = db.get_events(conn, "gh-1")
    assert [e["kind"] for e in events] == ["seen"]


def test_job_id_falls_back_to_url(conn):
    job = _job()
    del job["job_id"]
    job_id = db.upsert_job(conn, job)
    assert job_id == job["url"]


@pytest.mark.parametrize("missing", ["url", "title", "company", "source"])
def test_missing_required_field_raises(conn, missing):
    job = _job()
    job[missing] = None
    with pytest.raises(ValueError):
        db.upsert_job(conn, job)


# ── years_required: 0 vs None ────────────────────────────────────────────────
def test_years_required_zero_is_stored_as_zero(conn):
    db.upsert_job(conn, _job(job_id="z", url="u://z", years_required=0))
    assert db.get_job(conn, "z")["years_required"] == 0


def test_years_required_unknown_is_null(conn):
    db.upsert_job(conn, _job(job_id="n", url="u://n", years_required=None))
    assert db.get_job(conn, "n")["years_required"] is None


def test_years_required_non_int_rejected(conn):
    with pytest.raises(TypeError):
        db.upsert_job(conn, _job(years_required="unknown"))


# ── re-sight rule ────────────────────────────────────────────────────────────
def test_resight_bumps_last_seen_and_logs_but_does_not_regress(conn):
    db.upsert_job(conn, _job())
    db.set_status(conn, "gh-1", "scored", detail="score=0.8")
    db.record_outcome(conn, "gh-1", "interview")
    first_seen = db.get_job(conn, "gh-1")["first_seen_at"]

    # a repost tries to change content + would reset status
    db.upsert_job(conn, _job(title="DIFFERENT", status="fetched", score=99))

    row = db.get_job(conn, "gh-1")
    assert row["status"] == "scored"            # never regressed
    assert row["score"] is None                 # ledger space untouched
    assert row["outcome"] == "interview"        # outcome untouched
    assert row["title"] == "Backend Engineer"   # content written once
    assert row["first_seen_at"] == first_seen
    assert row["last_seen_at"] > first_seen

    kinds = [e["kind"] for e in db.get_events(conn, "gh-1")]
    assert kinds == ["seen", "status", "outcome", "seen"]


def test_duplicate_url_under_new_job_id_raises(conn):
    db.upsert_job(conn, _job())
    with pytest.raises(ValueError):
        db.upsert_job(conn, _job(job_id="gh-2"))  # same url, different id


# ── status transitions ──────────────────────────────────────────────────────
def test_set_status_records_from_and_to(conn):
    db.upsert_job(conn, _job())
    db.set_status(conn, "gh-1", "filtered")
    db.set_status(conn, "gh-1", "dropped", detail="seniority: 5+ yrs")

    row = db.get_job(conn, "gh-1")
    assert row["status"] == "dropped"
    assert row["status_detail"] == "seniority: 5+ yrs"

    status_events = [e for e in db.get_events(conn, "gh-1") if e["kind"] == "status"]
    assert (status_events[0]["from_status"], status_events[0]["to_status"]) == (
        "fetched", "filtered",
    )
    assert (status_events[-1]["from_status"], status_events[-1]["to_status"]) == (
        "filtered", "dropped",
    )


def test_set_status_rejects_unknown_status(conn):
    db.upsert_job(conn, _job())
    with pytest.raises(ValueError):
        db.set_status(conn, "gh-1", "banana")


def test_writes_on_missing_job_raise(conn):
    for call in (
        lambda: db.set_status(conn, "nope", "filtered"),
        lambda: db.record_outcome(conn, "nope", "silence"),
        lambda: db.add_note(conn, "nope", "hi"),
    ):
        with pytest.raises(KeyError):
            call()


# ── reads ────────────────────────────────────────────────────────────────────
def test_job_exists_and_iter_and_counts(conn):
    db.upsert_job(conn, _job(job_id="a", url="u://a", source="greenhouse"))
    db.upsert_job(conn, _job(job_id="b", url="u://b", source="jobspy-linkedin"))
    db.set_status(conn, "b", "filtered")

    assert db.job_exists(conn, "a") and not db.job_exists(conn, "zzz")
    assert [j["job_id"] for j in db.iter_jobs(conn, source="greenhouse")] == ["a"]
    assert [j["job_id"] for j in db.iter_jobs(conn, status="filtered")] == ["b"]

    counts = db.status_counts(conn)
    assert counts["fetched"] == 1 and counts["filtered"] == 1
    assert set(counts) == set(db.STATUSES)


def test_record_outcome_sets_fields_and_keeps_status(conn):
    db.upsert_job(conn, _job())
    db.set_status(conn, "gh-1", "applied")
    db.record_outcome(conn, "gh-1", "offer", detail="verbal", at="2026-10-01T00:00:00+00:00")

    row = db.get_job(conn, "gh-1")
    assert row["status"] == "applied"           # unchanged
    assert row["outcome"] == "offer"
    assert row["outcome_at"] == "2026-10-01T00:00:00+00:00"
    assert row["outcome_detail"] == "verbal"


# ── skill extraction (build step 3) ─────────────────────────────────────────
def test_set_extraction_stores_all_four_fields(conn):
    db.upsert_job(conn, _job())
    db.set_extraction(
        conn, "gh-1",
        skills='[{"name": "python", "requirement": "required"}]',
        domain_bucket="backend",
        legitimacy="legitimate",
        extraction_confidence="high",
    )
    row = db.get_job(conn, "gh-1")
    assert row["skills"] == '[{"name": "python", "requirement": "required"}]'
    assert row["domain_bucket"] == "backend"
    assert row["legitimacy"] == "legitimate"
    assert row["extraction_confidence"] == "high"


def test_set_extraction_allows_null_bucket_and_legitimacy_for_a_skip(conn):
    db.upsert_job(conn, _job())
    db.set_extraction(
        conn, "gh-1", skills="[]", domain_bucket=None, legitimacy=None,
        extraction_confidence="low",
    )
    row = db.get_job(conn, "gh-1")
    assert row["domain_bucket"] is None and row["legitimacy"] is None
    assert row["skills"] == "[]" and row["extraction_confidence"] == "low"


def test_set_extraction_on_missing_job_raises(conn):
    with pytest.raises(KeyError):
        db.set_extraction(
            conn, "nope", skills="[]", domain_bucket=None, legitimacy=None,
            extraction_confidence="low",
        )


# ── skill_vocab (build step 3) ──────────────────────────────────────────────
def test_touch_skill_vocab_inserts_new_entry(conn):
    db.touch_skill_vocab(conn, "fastapi")
    row = db.get_skill_vocab(conn, "fastapi")
    assert row is not None and row["seen_count"] == 1


def test_touch_skill_vocab_bumps_seen_count_on_repeat(conn):
    db.touch_skill_vocab(conn, "fastapi")
    db.touch_skill_vocab(conn, "fastapi")
    db.touch_skill_vocab(conn, "fastapi")
    assert db.get_skill_vocab(conn, "fastapi")["seen_count"] == 3


def test_get_skill_vocab_is_case_sensitive_no_folding(conn):
    db.touch_skill_vocab(conn, "fastapi")
    assert db.get_skill_vocab(conn, "FastAPI") is None  # folding is the caller's job


def test_get_skill_vocab_unknown_returns_none(conn):
    assert db.get_skill_vocab(conn, "nonexistent") is None


# ── v3 migration ─────────────────────────────────────────────────────────────
def test_migrate_v3_rebuilds_jobs_and_preserves_job_events_fk():
    """A real pre-v3 DB: old `jobs` shape (no extraction columns, no
    'extracted' in the status CHECK) plus a job_events row referencing it,
    stamped at user_version=2. init_db() must rebuild `jobs` in place without
    losing data or breaking job_events' foreign key into it."""
    raw = db.connect(":memory:")
    raw.executescript("""
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            description TEXT,
            years_required INTEGER,
            source TEXT NOT NULL,
            location TEXT,
            date_posted TEXT,
            job_type TEXT,
            salary TEXT,
            apply_url TEXT,
            is_remote INTEGER,
            status TEXT NOT NULL DEFAULT 'fetched'
                CHECK (status IN ('fetched','filtered','scored','tailored',
                                   'queued','applied','outcome','dropped')),
            status_detail TEXT,
            status_changed_at TEXT NOT NULL,
            score REAL,
            score_features TEXT,
            outcome TEXT,
            outcome_at TEXT,
            outcome_detail TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            first_seen_run TEXT
        );
        CREATE TABLE job_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES jobs (job_id) ON DELETE CASCADE,
            kind TEXT NOT NULL CHECK (kind IN ('seen','status','outcome','note')),
            from_status TEXT,
            to_status TEXT,
            detail TEXT,
            at TEXT NOT NULL
        );
    """)
    raw.execute(
        "INSERT INTO jobs (job_id, url, title, company, source, status, "
        "status_changed_at, first_seen_at, last_seen_at, updated_at) "
        "VALUES ('j1', 'https://x/1', 'Backend Engineer', 'Acme', 'greenhouse', "
        "'filtered', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00', "
        "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
    )
    raw.execute(
        "INSERT INTO job_events (job_id, kind, detail, at) VALUES "
        "('j1', 'note', 'a pre-migration note', '2026-01-01T00:00:00+00:00')"
    )
    raw.execute("PRAGMA user_version = 2")
    raw.commit()

    db.init_db(raw)  # runs _migrate_v3

    assert raw.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION  # v2 -> v3 -> v4 in one init_db
    cols = {r[1] for r in raw.execute("PRAGMA table_info(jobs)")}
    assert {"skills", "domain_bucket", "legitimacy", "extraction_confidence"} <= cols

    row = db.get_job(raw, "j1")
    assert row["status"] == "filtered"    # pre-existing data survived the rebuild
    assert row["skills"] is None          # new columns default NULL, never guessed
    assert row["domain_bucket"] is None

    events = db.get_events(raw, "j1")
    assert len(events) == 1 and events[0]["detail"] == "a pre-migration note"

    assert raw.execute("PRAGMA foreign_key_check").fetchall() == []

    db.set_status(raw, "j1", "extracted")  # the new CHECK actually accepts it
    assert db.get_job(raw, "j1")["status"] == "extracted"

    raw.close()


def test_migrate_v3_is_a_noop_on_an_already_current_db(conn):
    db.init_db(conn)  # conn fixture already ran init_db once; run again
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
