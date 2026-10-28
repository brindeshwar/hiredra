"""Sinks (build sub-slice 2a.8a)."""

from __future__ import annotations

import pytest

from hiredra.sourcing.job import Job
from hiredra.sourcing.sink import BackfillSink, ScoringSink, SinkStats
from hiredra.storage import db


@pytest.fixture()
def conn(pg_url):
    c = db.connect(pg_url)
    db.init_db(c)
    db.open_run(c, "R1", kind="scoring", config_name=None)  # jobs.first_seen_run references runs
    yield c
    c.close()


def _job(**over) -> Job:
    base = {
        "url": "u://1",
        "title": "Backend Engineer",
        "company": "Acme",
        "source": "greenhouse",
        "description": "x",
        "date_posted": "2026-09-01",
    }
    base.update(over)
    return Job(**base)


def test_scoring_sink_persists_new_and_touches_seen(conn):
    sink = ScoringSink(conn, run_id="R1")

    sink.consume(
        [
            _job(job_id="greenhouse:1"),
            _job(job_id="greenhouse:2", url="u://2", title="Data Engineer"),  # distinct role
        ]
    )
    s1 = sink.stats()
    assert (s1.fetched, s1.persisted_new, s1.touched) == (2, 2, 0)
    assert db.get_job(conn, "greenhouse:1")["first_seen_run"] == "R1"

    # a second batch: one already stored (-> touched), one genuinely new role
    sink.consume(
        [
            _job(job_id="greenhouse:1"),
            _job(job_id="greenhouse:3", title="Site Reliability Engineer", url="u://3"),
        ]
    )
    s2 = sink.stats()
    assert (s2.fetched, s2.persisted_new, s2.touched) == (4, 3, 1)

    kinds = [e["kind"] for e in db.get_events(conn, "greenhouse:1")]
    assert kinds.count("seen") == 2
    assert any("R1" in (e["detail"] or "") for e in db.get_events(conn, "greenhouse:1"))


def test_scoring_sink_does_not_close_the_connection(conn):
    ScoringSink(conn, "R1").consume([_job(job_id="greenhouse:1")])
    conn.execute("SELECT 1")  # still usable


def test_backfill_sink_collapses_and_holds_no_conn():
    sink = BackfillSink()
    sink.consume(
        [
            _job(job_id="jobspy-indeed:1"),
            _job(job_id="jobspy-indeed:1"),  # exact dup -> collapsed
            _job(job_id="jobspy-indeed:2", url="u://2", title="Data Engineer"),
        ]
    )
    assert len(sink.jobs) == 2
    assert sink.stats() == SinkStats(fetched=3, collapsed_in_run=1)
