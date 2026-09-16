"""Tests for hiredra.liveness (build sub-slice 2b: the liveness recheck,
the 4th and last filter-funnel stage). Split out of tests/test_filter.py
alongside the check_liveness/LivenessStats module split -- disjoint fixtures
(network-fetch fakes, not text-gate config) and no shared helpers with the
text-gate tests there."""

from __future__ import annotations

import pytest

from hiredra import liveness
from hiredra.sourcing.http import HTTPError, HttpClient, HttpConfig
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import FetchContext
from hiredra.storage import connect, get_events, get_job, init_db, set_status, upsert_job


def _job(**overrides):
    job = {
        "job_id": overrides.pop("job_id", "j1"),
        "url": overrides.pop("url", "https://example.com/j1"),
        "title": "Backend Engineer",
        "company": "Acme",
        "description": "We need a backend engineer.",
        "source": "greenhouse",
    }
    job.update(overrides)
    return job


class _FakeLivenessProvider:
    """Mirrors tests/test_orchestrator.py's FakeProvider shape, plus the
    _board_url() hook check_liveness reuses to group jobs by board."""

    def __init__(self, name, board_url="https://fake/board", jobs=None, error=None):
        self.name = name
        self._board_url_value = board_url
        self._jobs = jobs if jobs is not None else []
        self._error = error
        self.fetch_calls = 0

    def detect(self, entry):
        return entry.provider == self.name

    def _board_url(self, entry):
        return self._board_url_value

    def fetch(self, entry, ctx):
        self.fetch_calls += 1
        if self._error is not None:
            raise self._error
        return list(self._jobs)


@pytest.fixture()
def ctx():
    c = FetchContext(http=HttpClient(HttpConfig()))
    yield c
    c.http.close()


def _filtered_job(conn, **overrides):
    """A job already sitting at status='filtered' -- what check_liveness reads."""
    job = _job(**overrides)
    upsert_job(conn, job)
    set_status(conn, job["job_id"], "filtered")
    return job["job_id"]


def test_liveness_keeps_a_job_still_in_the_fresh_listing(ctx):
    conn = connect(":memory:")
    init_db(conn)
    job_id = _filtered_job(conn, job_id="p1", url="https://fake.example/p1")
    fresh = [Job(job_id=job_id, url="https://fake.example/p1", title="x", company="Acme", source="greenhouse")]
    provider = _FakeLivenessProvider("greenhouse", jobs=fresh)

    stats = liveness.check_liveness(conn, ctx=ctx, registry=[provider])

    assert stats.checked == 1 and stats.boards_checked == 1
    assert stats.live == 1 and stats.gone == 0 and stats.ambiguous == 0
    assert get_job(conn, job_id)["status"] == "filtered"
    conn.close()


def test_liveness_drops_a_job_absent_from_the_fresh_listing(ctx):
    conn = connect(":memory:")
    init_db(conn)
    job_id = _filtered_job(conn, job_id="p1", url="https://fake.example/p1")
    provider = _FakeLivenessProvider("greenhouse", jobs=[])  # board is alive, this posting isn't listed anymore

    stats = liveness.check_liveness(conn, ctx=ctx, registry=[provider])

    assert stats.gone == 1 and stats.live == 0
    row = get_job(conn, job_id)
    assert row["status"] == "dropped" and row["status_detail"] == "liveness: gone"
    conn.close()


def test_liveness_drops_the_whole_board_on_a_404(ctx):
    conn = connect(":memory:")
    init_db(conn)
    j1 = _filtered_job(conn, job_id="p1", url="https://fake.example/p1")
    j2 = _filtered_job(conn, job_id="p2", url="https://fake.example/p2", company="Acme")
    provider = _FakeLivenessProvider("greenhouse", error=HTTPError("gone", status=404))

    stats = liveness.check_liveness(conn, ctx=ctx, registry=[provider])

    assert stats.boards_checked == 1  # both jobs grouped into one board, one fetch attempt
    assert stats.gone == 2
    assert get_job(conn, j1)["status_detail"] == "liveness: gone"
    assert get_job(conn, j2)["status_detail"] == "liveness: gone"
    conn.close()


def test_liveness_keeps_jobs_when_the_recheck_is_ambiguous(ctx):
    conn = connect(":memory:")
    init_db(conn)
    job_id = _filtered_job(conn, job_id="p1", url="https://fake.example/p1")
    provider = _FakeLivenessProvider("greenhouse", error=RuntimeError("timed out"))

    stats = liveness.check_liveness(conn, ctx=ctx, registry=[provider])

    assert stats.ambiguous == 1 and stats.gone == 0
    assert get_job(conn, job_id)["status"] == "filtered"  # unchanged -- inconclusive is never a drop
    conn.close()


def test_liveness_groups_multiple_jobs_at_the_same_board_into_one_fetch(ctx):
    conn = connect(":memory:")
    init_db(conn)
    _filtered_job(conn, job_id="p1", url="https://fake.example/p1", company="Acme")
    _filtered_job(conn, job_id="p2", url="https://fake.example/p2", company="Acme")
    _filtered_job(conn, job_id="p3", url="https://fake.example/p3", company="Acme")
    fresh = [
        Job(job_id="p1", url="https://fake.example/p1", title="x", company="Acme", source="greenhouse"),
        Job(job_id="p2", url="https://fake.example/p2", title="x", company="Acme", source="greenhouse"),
        Job(job_id="p3", url="https://fake.example/p3", title="x", company="Acme", source="greenhouse"),
    ]
    provider = _FakeLivenessProvider("greenhouse", jobs=fresh)

    stats = liveness.check_liveness(conn, ctx=ctx, registry=[provider])

    assert provider.fetch_calls == 1  # one board, one fetch, three jobs checked against it
    assert stats.boards_checked == 1 and stats.live == 3
    conn.close()


def test_liveness_skips_and_notes_jobs_with_no_board_mechanism(ctx):
    conn = connect(":memory:")
    init_db(conn)
    job_id = _filtered_job(conn, job_id="js1", url="https://www.linkedin.com/jobs/view/123", source="jobspy-linkedin")

    stats = liveness.check_liveness(conn, ctx=ctx, registry=[])  # no provider handles "jobspy-linkedin"

    assert stats.unchecked == 1 and stats.gone == 0
    row = get_job(conn, job_id)
    assert row["status"] == "filtered"  # kept, never fetched
    notes = [e for e in get_events(conn, job_id) if e["kind"] == "note"]
    assert len(notes) == 1 and notes[0]["detail"] == "liveness: unchecked"
    conn.close()


def test_liveness_never_touches_jobs_already_dropped_by_an_earlier_gate(ctx):
    conn = connect(":memory:")
    init_db(conn)
    upsert_job(conn, _job(job_id="d1", url="https://fake.example/d1"))
    set_status(conn, "d1", "dropped", detail="role_type: no allow term matched")
    provider = _FakeLivenessProvider("greenhouse", jobs=[])

    stats = liveness.check_liveness(conn, ctx=ctx, registry=[provider])

    assert stats.checked == 0 and provider.fetch_calls == 0
    assert get_job(conn, "d1")["status_detail"] == "role_type: no allow term matched"  # untouched
    conn.close()
