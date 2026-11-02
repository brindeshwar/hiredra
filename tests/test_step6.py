"""Build step 6, slice 6.1: one role, one job to judge. D6.3: a copy of a role already stored and open (a city each,
a repost) points at it and the funnel drops it first. D6.4: the company's own posting of a role wins over an
aggregator's copy even when the copy came first; the aggregator set is every feed, not only JobSpy. And the sink
reads the store for dedup once per run, not once per board."""

from __future__ import annotations

import pytest

from hiredra import filter as funnel
from hiredra.sourcing import dedup as dd
from hiredra.sourcing.job import Job
from hiredra.sourcing.sink import ScoringSink
from hiredra.storage import db


@pytest.fixture()
def conn(pg_url):
    c = db.connect(pg_url)
    db.init_db(c)
    db.open_run(c, "R1", kind="scoring", config_name=None)
    yield c
    c.close()


def _row(key, **over):
    base = {"job_id": key, "url": f"u://{key}", "title": "Team Leader - Offline Merchants - Pune", "company": "Paytm",
            "source": "lever", "location": "Pune, India", "description": "Lead the merchant team."}  # fmt: skip
    return {**base, **over}


def _duplicate_of(conn, key):
    row = conn.execute(
        "SELECT c.job_key FROM jobs j LEFT JOIN jobs c ON c.id = j.duplicate_of WHERE j.job_key = %s", (key,)
    ).fetchone()
    return row[0]


# ── D6.3 ─────────────────────────────────────────────────────────────────────
def test_a_copy_of_a_stored_role_points_at_it_and_the_funnel_drops_it_first(conn):
    db.upsert_job(conn, _row("lever:pune"))
    db.upsert_job(conn, _row("lever:kolhapur", title="Team Leader - Offline Merchants - Kolhapur",
                             location="Kolhapur, India"))  # fmt: skip
    db.upsert_job(conn, _row("lever:other", description="Run the field sales team."))  # another role
    assert [_duplicate_of(conn, k) for k in ("lever:pune", "lever:kolhapur", "lever:other")] == [
        None, "lever:pune", None
    ]  # fmt: skip
    stats = funnel.run_funnel(conn, funnel.load_filter_config())
    assert stats.dropped_duplicate == 1
    job = db.get_job(conn, "lever:kolhapur")
    assert (job["status"], job["status_detail"].split(":")[0]) == ("dropped", "duplicate")


def test_a_closed_role_is_no_canonical_and_another_country_is_another_role(conn):
    db.upsert_job(conn, _row("lever:old"))
    conn.execute("UPDATE jobs SET closed_at = now() WHERE job_key = 'lever:old'")
    db.upsert_job(conn, _row("lever:new"))  # the role posted again after it closed: judged afresh
    db.upsert_job(conn, _row("lever:london", location="London, United Kingdom"))
    assert (_duplicate_of(conn, "lever:new"), _duplicate_of(conn, "lever:london")) == (None, None)


# ── D6.4 ─────────────────────────────────────────────────────────────────────
def _job(key, source, **over):
    return Job(**{**_row(key, source=source), **over})


def test_every_feed_is_an_aggregator_and_a_board_is_not():
    assert all(dd.is_aggregator_source(s) for s in ("jobspy-indeed", "himalayas", "themuse", "hackernews", "remoteok"))
    assert not any(dd.is_aggregator_source(s) for s in ("lever", "greenhouse", "workday", "apple"))


def test_the_companys_own_posting_wins_over_an_aggregator_copy_stored_first(conn):
    sink = ScoringSink(conn, "R1")
    sink.consume([_job("himalayas:paytm/x", "himalayas", url="https://himalayas.app/x")])
    sink.consume([_job("lever:pune", "lever")])  # the company's own copy, later: stored, the copy points at it
    assert _duplicate_of(conn, "himalayas:paytm/x") == "lever:pune"
    assert db.get_job(conn, "lever:pune") is not None
    sink.consume([_job("remoteok:9", "remoteok", url="https://remoteok.com/9")])  # an aggregator copy, after: seen
    assert db.get_job(conn, "remoteok:9") is None
    assert sink.stats().persisted_new == 2


def test_the_sink_reads_the_store_once_a_run(conn, monkeypatch):
    calls = []
    real = dd.load_seen_index
    monkeypatch.setattr("hiredra.sourcing.sink.load_seen_index", lambda c: (calls.append(1), real(c))[1])
    sink = ScoringSink(conn, "R1")
    sink.consume([_job("lever:a", "lever", title="Backend Engineer", url="u://a")])
    sink.consume([_job("jobspy-indeed:b", "jobspy-indeed", title="Backend Engineer", url="u://b")])
    assert calls == [1]  # one read for the whole run
    assert db.get_job(conn, "jobspy-indeed:b") is None  # matched against the job stored by the first board
