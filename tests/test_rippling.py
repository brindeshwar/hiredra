"""Rippling provider (build step 3): cases ported from freehire's ``rippling_test.go`` (MIT), plus hydration and every
location kept. Faked transport, no network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import rippling

API = "https://ats.rippling.com/api/v1/board/acme"
LISTING = [
    {"uuid": "u-1", "name": "Backend Engineer", "url": "https://ats.rippling.com/acme/jobs/u-1",
     "workLocation": {"label": "Bengaluru, India"}},
    {"uuid": "u-1", "name": "Backend Engineer", "workLocation": {"label": "Pune, India"}},  # once per location
    {"uuid": "u-2", "name": "Data Intern", "workLocation": {"label": "Remote (India)"}},
]  # fmt: skip
DETAIL = {
    "description": {"company": "<p>We are Acme.</p>", "role": "<p>Build <b>APIs</b>.</p>"},
    "workLocations": ["Bengaluru, India", "Pune, India"],
    "employmentType": {"label": "SALARIED_FT", "id": "Salaried, full-time"},
    "createdOn": "2026-09-30T01:50:52.166000-07:00",
}


class _Http:
    def __init__(self, routes):
        self.routes = routes
        self.asked: list[str] = []

    def get_json(self, url, *, timeout=None):
        self.asked.append(url)
        body = self.routes.get(url)
        if body is None:
            raise HTTPError("not found", status=404, url=url)
        return body


def _fetch(http, seen=lambda job_id: False, board="acme"):
    entry = CompanyEntry(name="Acme", provider="rippling", extra={"slug": board})
    return rippling.RipplingProvider().fetch(entry, FetchContext(http=http, seen=seen))


def test_listing_then_each_new_posting_s_detail_role_text_only():
    http = _Http({f"{API}/jobs": LISTING, f"{API}/jobs/u-1": DETAIL})
    [backend] = _fetch(http)  # u-2's detail 404s: skipped, read again next run
    assert http.asked == [f"{API}/jobs", f"{API}/jobs/u-1", f"{API}/jobs/u-2"]
    assert (backend.job_id, backend.url, backend.title) == (
        "rippling:u-1", "https://ats.rippling.com/acme/jobs/u-1", "Backend Engineer"
    )  # fmt: skip
    assert backend.description == "Build APIs."  # the company boilerplate is left out
    assert (backend.location, backend.date_posted, backend.job_type) == (
        "Bengaluru, India; Pune, India", "2026-09-30", "fulltime"
    )  # fmt: skip


def test_a_stored_posting_is_re_sighted_from_the_listing():
    http = _Http({f"{API}/jobs": LISTING})
    jobs = _fetch(http, seen=lambda job_id: True)
    assert http.asked == [f"{API}/jobs"]
    assert [(j.job_id, j.location, j.is_remote, j.description) for j in jobs] == [
        ("rippling:u-1", "Bengaluru, India", None, None), ("rippling:u-2", "Remote (India)", True, None),
    ]  # fmt: skip
    assert jobs[1].url == "https://ats.rippling.com/acme/jobs/u-2"


@pytest.mark.parametrize(("label", "want"), [("Salaried, full-time", "fulltime"), ("Hourly, part-time", "parttime"),
                                             ("Intern", "intern"), ("Contractor", "contract"), (None, None)])  # fmt: skip
def test_job_type(label, want):
    detail = {**DETAIL, "employmentType": {"id": label}}
    [job] = _fetch(_Http({f"{API}/jobs": LISTING[:1], f"{API}/jobs/u-1": detail}))
    assert job.job_type == want


def test_a_detail_without_role_text_waits_and_a_bad_listing_fails():
    empty = {**DETAIL, "description": {"company": "<p>We are Acme.</p>", "role": ""}}
    assert _fetch(_Http({f"{API}/jobs": LISTING[:1], f"{API}/jobs/u-1": empty})) == []
    with pytest.raises(FetchError):
        _fetch(_Http({f"{API}/jobs": {"error": "x"}}))
    with pytest.raises(HTTPError):
        _fetch(_Http({}))


@pytest.mark.parametrize("bad", ["", "Acme Co", "acme/jobs", "-acme"])
def test_a_malformed_board_is_refused(bad):
    with pytest.raises(FetchError):
        _fetch(_Http({}), board=bad)
