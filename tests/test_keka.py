"""Keka provider (build step 3): cases ported from freehire's ``keka_test.go`` (MIT), plus every location kept and a
gone board. Faked transport, no network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import keka


class _Http:
    def __init__(self, body):
        self.body = body
        self.asked: list[str] = []

    def get_json(self, url, *, timeout=None):
        self.asked.append(url)
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


def _fetch(http, board="acme"):
    entry = CompanyEntry(name="Acme", provider="keka", extra={"slug": board})
    return keka.KekaProvider().fetch(entry, FetchContext(http=http))


def test_one_request_lists_the_board_with_its_text():
    http = _Http([
        {"id": 160039, "title": "Sr. Solution Architect", "description": "<div>Do the work.</div>",
         "jobLocations": [{"city": "Hyderabad", "state": "TG", "countryName": "India"},
                          {"city": "Dallas", "state": "TX", "countryName": "United States"}],
         "jobType": 2, "publishedOn": "2026-09-07T08:18:34.693Z"},
        {"id": 160038, "title": "HR Intern", "description": "<div>Assist the team.</div>", "jobLocations": [],
         "jobType": 1, "publishedOn": "2026-05-01T00:00:00Z"},
        {"id": 7, "title": "Remote Support Engineer", "jobType": 5},
    ])  # fmt: skip
    architect, intern, support = _fetch(http)
    assert http.asked == ["https://acme.keka.com/careers/api/jobs/default/active"]
    assert (architect.job_id, architect.title) == ("keka:acme:160039", "Sr. Solution Architect")
    assert architect.url == "https://acme.keka.com/careers/jobdetails/160039"
    assert architect.location == "Hyderabad, TG, India; Dallas, TX, United States"  # freehire keeps the first
    assert (architect.company, architect.description, architect.date_posted) == ("Acme", "Do the work.", "2026-09-07")
    assert architect.job_type == "fulltime"
    assert (intern.location, intern.job_type, intern.is_remote) == (None, "parttime", None)
    assert (support.job_type, support.is_remote, support.description) == (None, True, None)


@pytest.mark.parametrize(
    ("experience", "want"),
    [("5+", "Experience: 5+ years"), ("0-1", "Experience: 0-1 years"), ("4+ yrs", "Experience: 4+ yrs"),
     ("5 Years+", "Experience: 5 Years+"), ("  ", None), (None, None)],
)  # fmt: skip
def test_the_experience_field_opens_the_text(experience, want):
    [job] = _fetch(_Http([{"id": 1, "title": "SDE 2", "description": "<p>Build.</p>", "experience": experience}]))
    assert job.description == (f"{want}\n\nBuild." if want else "Build.")


def test_the_gate_reads_the_experience_field():
    from hiredra.filter import load_filter_config, passes_seniority_gate

    cfg = load_filter_config()
    [senior, fresher] = _fetch(_Http([
        {"id": 1, "title": "Software Development Engineer 2", "description": "<p>Build.</p>", "experience": "3"},
        {"id": 2, "title": "Software Engineer", "description": "<p>Build.</p>", "experience": "0-1"},
    ]))  # fmt: skip
    assert not passes_seniority_gate(senior.title, senior.description, cfg).keep
    assert passes_seniority_gate(fresher.title, fresher.description, cfg).keep


def test_a_posting_without_a_title_or_id_is_skipped():
    assert _fetch(_Http([{"id": 1}, {"title": "No id"}, {"id": True, "title": "Bool id"}, "x"])) == []


def test_a_listing_failure_fails_the_board():
    with pytest.raises(FetchError):
        _fetch(_Http(FetchError("timed out")))
    with pytest.raises(FetchError):
        _fetch(_Http({"error": "not a list"}))
    assert _fetch(_Http([])) == []


def test_a_board_that_does_not_exist_is_a_404():
    gone = HTTPError("redirect", status=302,
                     location="https://nosuch.keka.com/careers/Content/TenantNotFound.html")  # fmt: skip
    with pytest.raises(HTTPError) as err:
        _fetch(_Http(gone))
    assert err.value.status == 404
    other = HTTPError("redirect", status=302, location="https://acme.keka.com/login")
    with pytest.raises(HTTPError) as err:
        _fetch(_Http(other))
    assert err.value.status == 302


@pytest.mark.parametrize("bad", ["", "Acme Co", "acme.keka.com", "-acme"])
def test_a_malformed_board_is_refused(bad):
    with pytest.raises(FetchError):
        _fetch(_Http([]), board=bad)
    assert keka.KekaProvider().board_url(CompanyEntry(name="A", provider="keka", extra={"slug": bad})) is None
