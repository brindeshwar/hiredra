"""YC startup jobs (build step 3, wave 3; Hiredra's own adapter): the public listing pages' rows, each new posting's
page, the experience line, robots.txt. Made-up pages, faked transport, no network."""

from __future__ import annotations

import html
import json

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import ycjobs

BASE = "https://www.ycombinator.com"


def _row(pid, **over):
    return {"id": pid, "title": f"Engineer {pid}", "url": f"/companies/acme/jobs/x{pid}-engineer", "location": "IN",
            "type": "Full-time", "minExperience": "Any (new grads ok)", "companyName": "Acme", **over}  # fmt: skip


def _listing(rows):
    data = html.escape(json.dumps({"component": "WaasJobListingsPage", "props": {"jobPostings": rows}}), quote=True)
    return f'<html><div id="app" data-page="{data}"></div></html>'


def _job_page(text="<p>Build agents.</p>", remote=False):
    posting = {"@context": "https://schema.org", "@type": "JobPosting", "title": "x", "description": text,
               "datePosted": "2026-10-22T05:34:47Z", **({"jobLocationType": "TELECOMMUTE"} if remote else {})}  # fmt: skip
    return f'<script type="application/ld+json">{json.dumps(posting)}</script>'


class _Http:
    def __init__(self, pages, robots=None):
        self.pages, self.robots = pages, robots
        self.asked: list[str] = []

    def get_text(self, url, *, timeout=None):
        self.asked.append(url)
        if url.endswith("/robots.txt"):
            if self.robots is None:
                raise HTTPError("none", status=404, url=url)
            return self.robots
        if url not in self.pages:
            return _listing([])
        page = self.pages[url]
        if isinstance(page, Exception):
            raise page
        return page


def _fetch(http, board="ycombinator.com", seen=None):
    entry = CompanyEntry(name="YC", provider="ycjobs", extra={"slug": board})
    return ycjobs.YCJobsProvider().fetch(entry, FetchContext(http=http, seen=seen or (lambda key: False)))


def test_the_listing_pages_then_each_new_postings_page():
    http = _Http({
        f"{BASE}/jobs/location/india": _listing([_row(1), _row(2, title="", companyName="")]),
        f"{BASE}/jobs/location/remote": _listing([_row(1), _row(3, location="Remote", minExperience="3+ years",
                                                                type="Internship")]),
        f"{BASE}/companies/acme/jobs/x1-engineer": _job_page(),
        f"{BASE}/companies/acme/jobs/x3-engineer": _job_page(remote=True),
    })  # fmt: skip
    first, third = _fetch(http)
    assert [u for u in http.asked if "/jobs/location" in u or "/role/" in u] == [f"{BASE}/{p}" for p in ycjobs.PAGES]
    assert (first.job_id, first.url, first.title, first.company, first.source) == (
        "ycjobs:1", f"{BASE}/companies/acme/jobs/x1-engineer", "Engineer 1", "Acme", "ycjobs"
    )  # fmt: skip
    assert first.description == "Experience: Any (new grads ok)\n\nBuild agents."
    assert (first.location, first.date_posted, first.job_type, first.is_remote) == (
        "IN",
        "2026-10-22",
        "fulltime",
        None,
    )
    assert (third.description.split("\n")[0], third.job_type, third.is_remote) == (
        "Experience: 3+ years",
        "intern",
        True,
    )


def test_a_stored_posting_is_re_sighted_without_its_page():
    http = _Http({f"{BASE}/jobs/location/india": _listing([_row(1)])})
    [job] = _fetch(http, seen=lambda key: key == "ycjobs:1")
    assert job.description is None and not any("/companies/" in u for u in http.asked)


@pytest.mark.parametrize(
    "http",
    [
        _Http({f"{BASE}/jobs/location/india": "<html>no data</html>"}),
        _Http({f"{BASE}/jobs/location/remote": FetchError("down")}),
        _Http({}, robots="User-agent: *\nDisallow: /jobs/\n"),
    ],
)
def test_a_listing_page_that_cannot_be_read_or_is_refused_fails_the_board(http):
    with pytest.raises(FetchError):
        _fetch(http)


def test_the_board_is_the_site():
    with pytest.raises(FetchError):
        _fetch(_Http({}), board="workatastartup.com")
    provider = ycjobs.YCJobsProvider()
    entry = CompanyEntry(name="x", provider="ycjobs", extra={"slug": "ycombinator.com"})
    assert provider.board_url(entry) and not provider.detect(entry) and provider.many_companies
