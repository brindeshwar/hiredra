"""Tests for the Ashby provider (build sub-slice 2a.5)."""

from __future__ import annotations

import httpx
import pytest

from hiredra.sourcing.http import Blocked, FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers.ashby import AshbyProvider

AB = AshbyProvider()


def _entry(**over) -> CompanyEntry:
    base = {"name": "Acme", "careers_url": "https://jobs.ashbyhq.com/acme"}
    base.update(over)
    return CompanyEntry(**base)


@pytest.mark.parametrize(
    "entry, expected",
    [
        (_entry(), True),
        (CompanyEntry(name="A", careers_url="https://jobs.lever.co/a"), False),
        (CompanyEntry(name="A", api="https://api.ashbyhq.com/posting-api/job-board/a"), True),
        (CompanyEntry(name="A", extra={"slug": "a"}), True),
        (_entry(provider="lever"), False),
        (CompanyEntry(name="A"), False),
    ],
)
def test_detect(entry, expected):
    assert AB.detect(entry) is expected


def test_bad_api_host_does_not_detect():
    assert AB.detect(CompanyEntry(name="A", api="https://evil.example.com/posting-api/job-board/a")) is False


def test_fetch_maps_jobs(http_factory, ashby_payload):
    http = http_factory(lambda req: httpx.Response(200, json=ashby_payload))
    jobs = AB.fetch(_entry(), FetchContext(http=http))

    assert [j.job_id for j in jobs] == ["ashby:job_001", "ashby:job_002"]  # malformed skipped
    j0 = jobs[0]
    assert j0.url == "https://jobs.ashbyhq.com/acme/job_001"
    assert j0.apply_url == "https://jobs.ashbyhq.com/acme/job_001/application"
    assert j0.source == "ashby"
    assert j0.description == "Backend work."
    assert j0.location == "New York · Remote (US) · Austin · USA"  # folded, " · "
    assert j0.date_posted == "2026-08-20"
    assert j0.job_type == "FullTime"
    assert j0.salary == "$180K - $220K"
    assert j0.is_remote is False  # workplaceType "Onsite" -> explicit False
    assert (j0.work_mode, j0.employment_type) == ("onsite", "full_time")
    assert (j0.salary_min, j0.salary_max, j0.salary_currency, j0.salary_period) == (180000, 220000, "USD", "year")
    assert (jobs[1].work_mode, jobs[1].salary_min) == ("remote", None)  # equity is not pay; none stated


def test_remote_from_workplace_type(http_factory, ashby_payload):
    http = http_factory(lambda req: httpx.Response(200, json=ashby_payload))
    jobs = AB.fetch(_entry(), FetchContext(http=http))
    assert jobs[1].is_remote is True  # workplaceType "Remote"


def test_longer_timeout_is_requested(http_factory, ashby_payload):
    seen = {}

    def handler(req):
        seen["timeout"] = req.extensions.get("timeout")
        return httpx.Response(200, json=ashby_payload)

    http = http_factory(handler)
    AB.fetch(_entry(), FetchContext(http=http))
    assert seen["timeout"]["read"] == 30.0


def test_includeCompensation_and_host(http_factory, ashby_payload):
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(200, json=ashby_payload)

    http = http_factory(handler)
    AB.fetch(_entry(), FetchContext(http=http))
    assert seen["url"] == ("https://api.ashbyhq.com/posting-api/job-board/acme?includeCompensation=true")


def test_empty_board(http_factory):
    http = http_factory(lambda req: httpx.Response(200, json={"jobs": []}))
    assert AB.fetch(_entry(), FetchContext(http=http)) == []


def test_non_https_api_refused(http_factory):
    http = http_factory(lambda req: httpx.Response(200, json={"jobs": []}))
    with pytest.raises(FetchError):
        AB.fetch(
            CompanyEntry(name="A", api="http://api.ashbyhq.com/posting-api/job-board/a"),
            FetchContext(http=http),
        )


def test_404_and_429(http_factory):
    with pytest.raises(HTTPError):
        AB.fetch(_entry(), FetchContext(http=http_factory(lambda req: httpx.Response(404))))
    with pytest.raises(Blocked):
        AB.fetch(_entry(), FetchContext(http=http_factory(lambda req: httpx.Response(429))))
