"""Tests for the Greenhouse provider (build sub-slice 2a.4)."""

from __future__ import annotations

import httpx
import pytest

from hiredra.sourcing.http import HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers.greenhouse import GreenhouseProvider

GH = GreenhouseProvider()


def _entry(**over) -> CompanyEntry:
    base = {"name": "Acme", "careers_url": "https://job-boards.greenhouse.io/acme"}
    base.update(over)
    return CompanyEntry(**base)


# ── detect / board resolution ──────────────────────────────────────────────
@pytest.mark.parametrize(
    "entry, expected",
    [
        (_entry(), True),
        (CompanyEntry(name="A", careers_url="https://jobs.lever.co/a"), False),
        (CompanyEntry(name="A", api="https://boards-api.greenhouse.io/v1/boards/a/jobs"), True),
        (CompanyEntry(name="A", extra={"slug": "a"}), True),
        (_entry(provider="lever"), False),
        (CompanyEntry(name="A"), False),
    ],
)
def test_detect(entry, expected):
    assert GH.detect(entry) is expected


def test_bad_api_host_does_not_detect():
    assert GH.detect(CompanyEntry(name="A", api="https://evil.example.com/v1/boards/a/jobs")) is False


# ── fetch: mapping ─────────────────────────────────────────────────────────
def test_fetch_maps_jobs(http_factory, greenhouse_payload):
    http = http_factory(lambda req: httpx.Response(200, json=greenhouse_payload))
    jobs = GH.fetch(_entry(), FetchContext(http=http))

    assert [j.job_id for j in jobs] == ["greenhouse:12345", "greenhouse:67890"]  # malformed skipped

    first = jobs[0]
    assert first.url == "https://job-boards.greenhouse.io/acme/jobs/12345"
    assert first.title == "Backend Engineer"
    assert first.company == "Acme"
    assert first.source == "greenhouse"
    assert first.description == "Build APIs. 0-2 years experience."
    assert first.location == "San Francisco, CA"
    assert first.date_posted == "2026-08-15"       # from first_published, date only
    assert first.is_remote is None                 # no clear signal -> unknown, NOT False
    assert first.years_required is None            # never parsed here


def test_remote_inferred_only_on_clear_signal(http_factory, greenhouse_payload):
    http = http_factory(lambda req: httpx.Response(200, json=greenhouse_payload))
    jobs = GH.fetch(_entry(), FetchContext(http=http))
    assert jobs[1].is_remote is True               # "AI Engineer (Remote)" / "Remote - US"


def test_is_remote_is_never_false(http_factory):
    payload = {"jobs": [{
        "id": 1, "title": "Backend Engineer",
        "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/1",
        "location": {"name": "New York, NY"}, "content": "<p>x</p>",
        "updated_at": "2026-09-01T00:00:00Z",
    }]}
    http = http_factory(lambda req: httpx.Response(200, json=payload))
    assert GH.fetch(_entry(), FetchContext(http=http))[0].is_remote is None


def test_date_posted_falls_back_to_updated_at(http_factory, greenhouse_payload):
    http = http_factory(lambda req: httpx.Response(200, json=greenhouse_payload))
    jobs = GH.fetch(_entry(), FetchContext(http=http))
    assert jobs[1].date_posted == "2026-09-05"     # job 67890 has no first_published


def test_empty_board_returns_empty_list(http_factory):
    http = http_factory(lambda req: httpx.Response(200, json={"jobs": []}))
    assert GH.fetch(_entry(), FetchContext(http=http)) == []


def test_content_true_is_requested(http_factory, greenhouse_payload):
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(200, json=greenhouse_payload)

    http = http_factory(handler)
    GH.fetch(_entry(), FetchContext(http=http))
    assert "content=true" in seen["url"]
    assert seen["url"].startswith("https://boards-api.greenhouse.io/v1/boards/acme/jobs")


# ── fetch: SSRF + errors ──────────────────────────────────────────────────
def test_non_https_api_is_refused(http_factory):
    from hiredra.sourcing.http import FetchError

    http = http_factory(lambda req: httpx.Response(200, json={"jobs": []}))
    entry = CompanyEntry(name="A", api="http://boards-api.greenhouse.io/v1/boards/a/jobs")
    with pytest.raises(FetchError):
        GH.fetch(entry, FetchContext(http=http))


def test_404_board_propagates_as_httperror(http_factory):
    http = http_factory(lambda req: httpx.Response(404))
    with pytest.raises(HTTPError) as ei:
        GH.fetch(_entry(), FetchContext(http=http))
    assert ei.value.status == 404


def test_api_redirect_is_not_followed(http_factory):
    http = http_factory(
        lambda req: httpx.Response(302, headers={"location": "https://elsewhere.example.com/"})
    )
    with pytest.raises(HTTPError):
        GH.fetch(_entry(), FetchContext(http=http))
