"""Tests for the storage layer (build sub-slice 2a.1; on Postgres since plan unit DB3). They need the pg_url
fixture (conftest.py): a migrated scratch database, emptied before each test."""

from __future__ import annotations

import itertools
import struct
from datetime import date, datetime, timezone

import pytest

from hiredra.storage import db

JID = "greenhouse:gh-1"


@pytest.fixture()
def conn(pg_url):
    c = db.connect(pg_url)
    db.init_db(c)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def fake_clock(monkeypatch):
    """Deterministic, monotonically increasing timestamps."""
    counter = itertools.count(1)
    monkeypatch.setattr(db, "_now", lambda: f"2026-09-09T00:00:{next(counter):02d}+00:00")


def _at(second: int) -> datetime:
    return datetime(2026, 9, 9, 0, 0, second, tzinfo=timezone.utc)


def _job(**over):
    base = {
        "job_id": JID,
        "url": "https://boards.greenhouse.io/acme/jobs/1",
        "title": "Backend Engineer, Pune",
        "company": "Acme Inc.",
        "description": "Build things.",
        "years_required": 0,
        "source": "greenhouse",
        "location": "Pune, India",
        "date_posted": "2026-09-01",
        "job_type": "fulltime",
        "salary": None,
        "apply_url": None,
        "is_remote": True,
    }
    base.update(over)
    return base


# ── schema / setup ───────────────────────────────────────────────────────────
def test_init_db_is_idempotent(conn):
    db.init_db(conn)  # second run must not raise
    tables = {r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
    assert {"jobs", "job_events", "job_skills", "cooldowns"} <= tables


def test_rows_read_by_position_by_name_and_as_a_dict(conn):
    row = conn.execute("SELECT 1 AS a, 'x' AS b").fetchone()
    assert (row[0], row["b"], dict(row), row.keys()) == (1, "x", {"a": 1, "b": "x"}, ["a", "b"])
    assert row == (1, "x")


def test_connections_autocommit(conn, pg_url):
    db.upsert_job(conn, _job())
    other = db.connect(pg_url)
    try:
        assert db.job_exists(other, JID)  # visible to a second connection at once: no open transaction left
    finally:
        other.close()


# ── insert ───────────────────────────────────────────────────────────────────
def test_insert_new_job_defaults_to_fetched_and_logs_seen(conn):
    job_id = db.upsert_job(conn, _job())
    assert job_id == JID

    row = db.get_job(conn, JID)
    assert row["status"] == "fetched"
    assert row["first_seen_at"] == row["last_seen_at"] == _at(1)
    assert row["score"] is None and row["outcome"] is None and row["skills"] is None
    assert (row["source"], row["external_id"]) == ("greenhouse", "gh-1")
    assert (row["date_posted"], row["is_remote"]) == (date(2026, 9, 1), True)

    events = db.get_events(conn, JID)
    assert [(e["job_id"], e["kind"], e["at"]) for e in events] == [(JID, "seen", _at(1))]


def test_a_new_job_gets_its_company_countries_and_identity_hashes(conn):
    db.upsert_job(conn, _job())
    db.upsert_job(conn, _job(job_id="lever:x", url="u://x", source="lever", company="Acme"))
    a, b = db.get_job(conn, JID), db.get_job(conn, "lever:x")
    assert a["company_id"] == b["company_id"]  # "Acme Inc." and "Acme" are one employer
    assert conn.execute("SELECT slug FROM companies").fetchall() == [("acme",)]
    assert a["countries"] == ["IN"]
    assert a["content_hash"] and a["role_fingerprint"] == b["role_fingerprint"]  # same role, title clause stripped


def test_job_id_falls_back_to_source_and_url(conn):
    job = _job()
    del job["job_id"]
    assert db.upsert_job(conn, job) == f"greenhouse:{job['url']}"


def test_a_job_id_must_carry_its_source(conn):
    with pytest.raises(ValueError, match="must start with its source"):
        db.upsert_job(conn, _job(job_id="gh-1"))
    with pytest.raises(ValueError, match="must start with its source"):
        db.upsert_job(conn, _job(job_id="lever:gh-1"))


@pytest.mark.parametrize("missing", ["url", "title", "company", "source"])
def test_missing_required_field_raises(conn, missing):
    job = _job()
    job[missing] = None
    with pytest.raises(ValueError):
        db.upsert_job(conn, job)


# ── years_required: 0 vs None ────────────────────────────────────────────────
def test_years_required_zero_is_stored_as_zero(conn):
    db.upsert_job(conn, _job(job_id="greenhouse:z", url="u://z", years_required=0))
    assert db.get_job(conn, "greenhouse:z")["years_required"] == 0


def test_years_required_unknown_is_null(conn):
    db.upsert_job(conn, _job(job_id="greenhouse:n", url="u://n", years_required=None))
    assert db.get_job(conn, "greenhouse:n")["years_required"] is None


def test_years_required_non_int_rejected(conn):
    with pytest.raises(TypeError):
        db.upsert_job(conn, _job(years_required="unknown"))


# ── re-sight rule ────────────────────────────────────────────────────────────
def test_resight_bumps_last_seen_and_logs_but_does_not_regress(conn):
    db.upsert_job(conn, _job())
    db.set_status(conn, JID, "scored", detail="score=0.8")
    db.record_outcome(conn, JID, "interview")
    first_seen = db.get_job(conn, JID)["first_seen_at"]

    # a repost tries to change content + would reset status
    db.upsert_job(conn, _job(title="DIFFERENT", status="fetched", score=99))

    row = db.get_job(conn, JID)
    assert row["status"] == "scored"  # never regressed
    assert row["score"] is None  # ledger space untouched
    assert row["outcome"] == "interview"  # outcome untouched
    assert row["title"] == "Backend Engineer, Pune"  # content written once
    assert row["first_seen_at"] == first_seen
    assert row["last_seen_at"] > first_seen

    kinds = [e["kind"] for e in db.get_events(conn, JID)]
    assert kinds == ["seen", "status", "outcome", "seen"]


def test_a_liveness_gone_job_listed_again_reopens(conn):
    db.upsert_job(conn, _job())
    db.set_status(conn, JID, "dropped", detail="liveness: gone (not in the board listing)")
    db.upsert_job(conn, _job())
    assert db.get_job(conn, JID)["status"] == "filtered"


def test_duplicate_url_under_new_job_id_raises(conn):
    db.upsert_job(conn, _job())
    with pytest.raises(ValueError):
        db.upsert_job(conn, _job(job_id="greenhouse:gh-2"))  # same url, different id
    assert not db.job_exists(conn, "greenhouse:gh-2")  # and nothing of it was kept


# ── status transitions ──────────────────────────────────────────────────────
def test_set_status_records_from_and_to(conn):
    db.upsert_job(conn, _job())
    db.set_status(conn, JID, "filtered")
    db.set_status(conn, JID, "dropped", detail="seniority: 5+ yrs")

    row = db.get_job(conn, JID)
    assert row["status"] == "dropped"
    assert row["status_detail"] == "seniority: 5+ yrs"

    status_events = [e for e in db.get_events(conn, JID) if e["kind"] == "status"]
    assert (status_events[0]["from_status"], status_events[0]["to_status"]) == ("fetched", "filtered")
    assert (status_events[-1]["from_status"], status_events[-1]["to_status"]) == ("filtered", "dropped")


def test_set_status_rejects_unknown_status(conn):
    db.upsert_job(conn, _job())
    with pytest.raises(ValueError):
        db.set_status(conn, JID, "banana")


def test_writes_on_missing_job_raise(conn):
    for call in (
        lambda: db.set_status(conn, "greenhouse:nope", "filtered"),
        lambda: db.record_outcome(conn, "greenhouse:nope", "silence"),
        lambda: db.add_note(conn, "greenhouse:nope", "hi"),
        lambda: db.touch_job(conn, "greenhouse:nope"),
        lambda: db.set_years_required(conn, "greenhouse:nope", 1),
    ):
        with pytest.raises(KeyError):
            call()


# ── reads ────────────────────────────────────────────────────────────────────
def test_job_exists_and_iter_and_counts(conn):
    db.upsert_job(conn, _job(job_id="greenhouse:a", url="u://a", source="greenhouse"))
    db.upsert_job(conn, _job(job_id="jobspy-linkedin:b", url="u://b", source="jobspy-linkedin"))
    db.set_status(conn, "jobspy-linkedin:b", "filtered")

    assert db.job_exists(conn, "greenhouse:a") and not db.job_exists(conn, "greenhouse:zzz")
    assert [j["job_id"] for j in db.iter_jobs(conn, source="greenhouse")] == ["greenhouse:a"]
    assert [j["job_id"] for j in db.iter_jobs(conn, status="filtered")] == ["jobspy-linkedin:b"]

    counts = db.status_counts(conn)
    assert counts["fetched"] == 1 and counts["filtered"] == 1
    assert set(counts) == set(db.STATUSES)


def test_record_outcome_sets_fields_and_keeps_status(conn):
    db.upsert_job(conn, _job())
    db.set_status(conn, JID, "applied")
    db.record_outcome(conn, JID, "offer", detail="verbal", at="2026-10-01T00:00:00+00:00")

    row = db.get_job(conn, JID)
    assert row["status"] == "applied"  # unchanged
    assert row["outcome"] == "offer"
    assert row["outcome_at"] == datetime(2026, 10, 1, tzinfo=timezone.utc)
    assert row["outcome_detail"] == "verbal"


# ── skill extraction (build step 3) ─────────────────────────────────────────
def test_set_extraction_stores_all_four_fields(conn):
    db.upsert_job(conn, _job())
    db.set_extraction(
        conn,
        JID,
        skills=[{"name": "python", "requirement": "required"}, {"name": "aws", "requirement": "good_to_have"}],
        domain_bucket="backend",
        legitimacy="legitimate",
        extraction_confidence="high",
    )
    row = db.get_job(conn, JID)
    assert row["skills"] == [
        {"name": "python", "requirement": "required", "source": "llm"},
        {"name": "aws", "requirement": "good_to_have", "source": "llm"},
    ]
    assert row["domain_bucket"] == "backend"
    assert row["legitimacy"] == "legitimate"
    assert row["extraction_confidence"] == "high"


def test_set_extraction_keeps_a_repeated_skill_once_and_replaces_on_rewrite(conn):
    db.upsert_job(conn, _job())
    skills = [{"name": "go", "requirement": "required"}, {"name": "go", "requirement": "good_to_have"}]
    db.set_extraction(conn, JID, skills=skills, domain_bucket=None, legitimacy=None, extraction_confidence="high")
    assert db.get_job(conn, JID)["skills"] == [{"name": "go", "requirement": "required", "source": "llm"}]
    db.set_skills(conn, JID, [{"name": "rust", "requirement": "required"}], extraction_version="v2")
    row = db.get_job(conn, JID)
    assert (row["skills"], row["extraction_version"]) == (
        [{"name": "rust", "requirement": "required", "source": "llm"}],
        "v2",
    )
    with pytest.raises(ValueError):
        db.set_skills(conn, JID, [], status="scored")  # only extraction columns


def test_set_extraction_allows_null_bucket_and_legitimacy_for_a_skip(conn):
    db.upsert_job(conn, _job())
    db.set_extraction(conn, JID, skills=[], domain_bucket=None, legitimacy=None, extraction_confidence="low")
    row = db.get_job(conn, JID)
    assert row["domain_bucket"] is None and row["legitimacy"] is None
    assert row["skills"] == [] and row["extraction_confidence"] == "low"  # [] = extracted, none (None = not yet)


def test_set_extraction_on_missing_job_raises(conn):
    with pytest.raises(KeyError):
        db.set_extraction(
            conn, "greenhouse:nope", skills=[], domain_bucket=None, legitimacy=None, extraction_confidence="low"
        )


def test_set_score_stores_the_ledger_as_json_and_a_list(conn):
    db.upsert_job(conn, _job())
    db.set_score(conn, JID, score=-0.527244, features={"coverage": 0.5}, review_reason=["suspect"],
                 score_version="v1", embed_tag="t", skills_hash="h", profile_hash="p")  # fmt: skip
    row = db.get_job(conn, JID)
    assert (row["score"], row["score_features"], row["review_reason"]) == (-0.527244, {"coverage": 0.5}, ["suspect"])


# ── runs / cooldowns ─────────────────────────────────────────────────────────
def test_runs_round_trip_with_a_json_detail(conn):
    db.open_run(conn, "R1", kind="scoring", config_name=None)
    db.checkpoint_run(conn, "R1", detail='{"companies_done": ["a"]}')
    assert db.unfinished_run(conn, "scoring", None)["detail"] == {"companies_done": ["a"]}
    db.finish_run(conn, "R1", status="ok", fetched=3)
    assert db.unfinished_run(conn, "scoring", None) is None
    assert db.list_runs(conn)[0]["fetched"] == 3


def test_an_expired_cooldown_is_not_loaded(conn):
    db.save_cooldown(conn, "old", 1_000.0, "long ago")
    db.save_cooldown(conn, "live", 4_000_000_000.0, "429")
    assert db.load_cooldowns(conn) == {"live": 4_000_000_000.0}


def test_iter_jobs_is_oldest_first(conn):
    for key in ("b", "a", "c"):  # the fake clock makes each one a second younger
        db.upsert_job(conn, _job(job_id=f"greenhouse:{key}", url=f"u://{key}"))
    assert [j["job_id"] for j in db.iter_jobs(conn)] == ["greenhouse:b", "greenhouse:a", "greenhouse:c"]


def test_cooldowns_extend_never_shorten(conn):
    db.save_cooldown(conn, "host", 4_000_000_000.0, "429")
    db.save_cooldown(conn, "host", 3_000_000_000.0, "429 again")
    assert db.load_cooldowns(conn) == {"host": 4_000_000_000.0}
    db.clear_cooldown(conn)
    assert db.load_cooldowns(conn) == {}


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


def test_vectors_round_trip_bit_for_bit(conn):
    db.touch_skill_vocab(conn, "go")
    values = (0.1, -1 / 3, 1e-8, 0.123456789)
    blob = struct.pack("<4f", *values)
    db.set_skill_vectors(conn, [("go", blob)], "tag")
    assert db.get_skill_vocab(conn, "go")["vector"] == blob  # float32 -> pgvector float4 -> float32: exact
    with pytest.raises(KeyError):
        db.set_skill_vectors(conn, [("go", blob), ("missing", blob)], "tag2")
    assert db.get_skill_vocab(conn, "go")["embed_tag"] == "tag"  # the whole chunk rolled back
