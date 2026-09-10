"""Tests for the Lever provider (build sub-slice 2a.5)."""

from __future__ import annotations

import httpx
import pytest

from hiredra.sourcing.http import Blocked, FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers.lever import LeverProvider

LV = LeverProvider()


def _entry(**over) -> CompanyEntry:
    base = {"name": "Acme", "careers_url": "https://jobs.lever.co/acme"}
    base.update(over)
    return CompanyEntry(**base)


@pytest.mark.parametrize(
    "entry, expected",
    [
        (_entry(), True),
        (CompanyEntry(name="A", careers_url="https://job-boards.greenhouse.io/a"), False),
        (CompanyEntry(name="A", api="https://api.lever.co/v0/postings/a"), True),
        (CompanyEntry(name="A", api="https://api.eu.lever.co/v0/postings/a"), True),
        (CompanyEntry(name="A", extra={"slug": "a"}), True),
        (_entry(provider="ashby"), False),
        (CompanyEntry(name="A"), False),
    ],
)
def test_detect(entry, expected):
    assert LV.detect(entry) is expected


def test_bad_api_host_does_not_detect():
    assert LV.detect(CompanyEntry(name="A", api="https://evil.example.com/v0/postings/a")) is False


def test_fetch_maps_jobs(http_factory, lever_payload):
    http = http_factory(lambda req: httpx.Response(200, json=lever_payload))
    jobs = LV.fetch(_entry(), FetchContext(http=http))

    assert [j.job_id for j in jobs] == ["lever:aaaa-1111", "lever:bbbb-2222"]  # malformed skipped
    j0 = jobs[0]
    assert j0.url == "https://jobs.lever.co/acme/aaaa-1111"
    assert j0.apply_url == "https://jobs.lever.co/acme/aaaa-1111/apply"
    assert j0.title == "Senior Backend Engineer"
    assert j0.source == "lever"
    assert j0.description == "Build backend systems."
    assert j0.location == "Berlin; Remote - EU"       # folded, deduped, "; "
    assert j0.date_posted == "2024-09-10"             # epoch-ms -> date
    assert j0.job_type == "Full-time"
    assert j0.is_remote is False                       # workplaceType "hybrid" -> explicit False


def test_remote_from_workplace_type(http_factory, lever_payload):
    http = http_factory(lambda req: httpx.Response(200, json=lever_payload))
    jobs = LV.fetch(_entry(), FetchContext(http=http))
    assert jobs[1].is_remote is True                   # workplaceType "remote"


def test_eu_host_is_derived(http_factory):
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(200, json=[])

    http = http_factory(handler)
    LV.fetch(CompanyEntry(name="A", careers_url="https://jobs.eu.lever.co/acme"), FetchContext(http=http))
    assert seen["url"] == "https://api.eu.lever.co/v0/postings/acme"


def test_empty_and_non_array(http_factory):
    http1 = http_factory(lambda req: httpx.Response(200, json=[]))
    assert LV.fetch(_entry(), FetchContext(http=http1)) == []
    http2 = http_factory(lambda req: httpx.Response(200, json={"not": "an array"}))
    assert LV.fetch(_entry(), FetchContext(http=http2)) == []


def test_non_https_api_refused(http_factory):
    http = http_factory(lambda req: httpx.Response(200, json=[]))
    with pytest.raises(FetchError):
        LV.fetch(CompanyEntry(name="A", api="http://api.lever.co/v0/postings/a"), FetchContext(http=http))


def test_404_and_429(http_factory):
    with pytest.raises(HTTPError):
        LV.fetch(_entry(), FetchContext(http=http_factory(lambda req: httpx.Response(404))))
    with pytest.raises(Blocked):
        LV.fetch(_entry(), FetchContext(http=http_factory(lambda req: httpx.Response(429))))
