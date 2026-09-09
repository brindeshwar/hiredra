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
