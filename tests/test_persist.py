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
def conn():
    c = db.connect(":memory:")
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
    jobs = GreenhouseProvider().fetch(
        CompanyEntry(name="Acme", extra={"slug": "acme"}), FetchContext(http=http)
    )
    result = persist_jobs(conn, jobs)

    assert result.inserted == len(jobs) == 2
    stored = db.get_job(conn, "greenhouse:12345")
    assert stored["source"] == "greenhouse"
    assert stored["status"] == "fetched"
    assert stored["title"] == "Backend Engineer"
    assert stored["is_remote"] is None
    assert Job.from_row(stored) == jobs[0]
