"""Tests for persist_jobs (build sub-slice 2a.4)."""

from __future__ import annotations

import httpx
import pytest

from hiredra.sourcing.job import Job
from hiredra.sourcing.persist import PersistResult, persist_jobs
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers.greenhouse import GreenhouseProvider
from hiredra.storage import db


@pytest.fixture()
def conn(pg_url):
    c = db.connect(pg_url)
    db.init_db(c)
    yield c
    c.close()


def _jobs():
    return [
        Job(job_id="greenhouse:1", url="u://1", title="A", company="C", source="greenhouse"),
        Job(job_id="greenhouse:2", url="u://2", title="B", company="C", source="greenhouse"),
    ]


def test_counts_new_vs_reseen(conn):
    r1 = persist_jobs(conn, _jobs())
    assert (r1.inserted, r1.reseen, r1.total) == (2, 0, 2)

    r2 = persist_jobs(conn, _jobs())
    assert (r2.inserted, r2.reseen, r2.total) == (0, 2, 2)


def test_persisted_rows_start_fetched(conn):
    persist_jobs(conn, _jobs())
    assert [j["status"] for j in db.iter_jobs(conn)] == ["fetched", "fetched"]


def test_result_is_immutable():
    r = PersistResult(inserted=1, reseen=0)
    with pytest.raises(Exception):
        r.inserted = 5  # type: ignore[misc]


def test_provider_to_persist_round_trip(conn, http_factory, greenhouse_payload):
    http = http_factory(lambda req: httpx.Response(200, json=greenhouse_payload))
    jobs = GreenhouseProvider().fetch(CompanyEntry(name="Acme", extra={"slug": "acme"}), FetchContext(http=http))
    result = persist_jobs(conn, jobs)

    assert result.inserted == len(jobs) == 2
    stored = db.get_job(conn, "greenhouse:12345")
    assert stored["source"] == "greenhouse"
    assert stored["status"] == "fetched"
    assert stored["title"] == "Backend Engineer"
    assert stored["is_remote"] is None
    assert Job.from_row(stored) == jobs[0]


def _row(i, **over):
    return {"job_id": f"greenhouse:{i}", "url": f"u://{i}", "title": f"Role {i}", "company": "Acme",
            "source": "greenhouse", "description": f"About role {i}", "location": "Pune, India", **over}  # fmt: skip


def test_a_batch_saves_exactly_what_saving_one_by_one_saves(conn):
    job = _row(1, is_remote=1, years_required=0, salary_min=10, date_posted="2026-11-01")
    db.open_run(conn, "r", kind="scoring", config_name=None)
    db.upsert_job(conn, job, first_seen_run="r")
    one = dict(db.get_job(conn, "greenhouse:1"))
    conn.execute("TRUNCATE jobs RESTART IDENTITY CASCADE")
    assert db.upsert_jobs(conn, [job], first_seen_run="r") == 1
    many = dict(db.get_job(conn, "greenhouse:1"))
    timing = {"id", "status_changed_at", "first_seen_at", "last_seen_at", "updated_at"}
    assert {k: v for k, v in one.items() if k not in timing} == {k: v for k, v in many.items() if k not in timing}


def test_each_new_job_gets_its_first_sighting_event(conn):
    db.upsert_jobs(conn, [_row(1), _row(2)])
    rows = conn.execute("SELECT count(*), min(detail), max(detail) FROM job_events WHERE kind = 'seen'").fetchone()
    assert tuple(rows) == (2, "first sighting", "first sighting")


def test_a_copy_of_a_role_earlier_in_the_batch_points_at_it(conn):
    # D6.3: the same role listed twice on one board, a copy per city in the same country
    same = {"title": "Engineer", "description": "Build it."}
    db.upsert_jobs(conn, [_row(1, **same), _row(2, **same, location="Mumbai, India"), _row(3)])
    first = db.get_job(conn, "greenhouse:1")
    assert first["duplicate_of"] is None
    assert db.get_job(conn, "greenhouse:2")["duplicate_of"] == first["id"]
    assert db.get_job(conn, "greenhouse:3")["duplicate_of"] is None


def test_a_copy_of_a_stored_role_points_at_it_but_not_at_a_closed_one(conn):
    same = {"title": "Engineer", "description": "Build it."}
    db.upsert_job(conn, _row(1, **same))
    stored = db.get_job(conn, "greenhouse:1")
    db.upsert_jobs(conn, [_row(2, **same)])
    assert db.get_job(conn, "greenhouse:2")["duplicate_of"] == stored["id"]

    conn.execute("UPDATE jobs SET closed_at = now()")
    db.upsert_jobs(conn, [_row(3, **same)])
    assert db.get_job(conn, "greenhouse:3")["duplicate_of"] is None


def test_new_and_re_sighted_jobs_are_counted_apart_and_a_repeat_is_a_re_sighting(conn):
    db.upsert_job(conn, _row(1))
    assert db.upsert_jobs(conn, [_row(1), _row(2), _row(2)]) == 1
    seen = conn.execute("SELECT detail FROM job_events ORDER BY id").fetchall()
    assert [r[0] for r in seen] == ["first sighting", "first sighting", "re-sighted", "re-sighted"]


def test_a_job_dropped_as_gone_and_listed_again_is_reopened(conn):
    db.upsert_job(conn, _row(1))
    db.set_status(conn, "greenhouse:1", "filtered")
    db.set_status(conn, "greenhouse:1", "dropped", detail="liveness: gone")
    db.upsert_jobs(conn, [_row(1)])
    assert db.get_job(conn, "greenhouse:1")["status"] == "filtered"


def test_a_url_stored_under_another_id_keeps_the_jobs_before_it_and_names_it(conn):
    db.upsert_job(conn, _row(9, url="u://taken"))
    with pytest.raises(ValueError, match="u://taken"):
        db.upsert_jobs(conn, [_row(1), _row(2, url="u://taken"), _row(3)])
    assert db.get_job(conn, "greenhouse:1") is not None
    assert db.get_job(conn, "greenhouse:2") is None


def test_a_board_larger_than_one_batch_is_saved_whole(conn, monkeypatch):
    monkeypatch.setattr(db, "_INSERT_BATCH", 3)
    assert db.upsert_jobs(conn, [_row(i) for i in range(8)]) == 8
    assert conn.execute("SELECT count(DISTINCT id) FROM jobs").fetchone()[0] == 8
    assert conn.execute("SELECT count(*) FROM job_events").fetchone()[0] == 8


def test_a_later_single_insert_does_not_collide_with_batched_ids(conn):
    db.upsert_jobs(conn, [_row(1), _row(2)])
    db.upsert_job(conn, _row(3))
    assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 3


def test_a_job_the_store_cannot_take_stops_the_board_before_anything_is_written(conn):
    with pytest.raises(ValueError, match="title"):
        db.upsert_jobs(conn, [_row(1), _row(2, title="")])
    assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0
