"""Apple provider (build step 3): cases ported from freehire's ``apple_test.go`` (MIT), plus the location board read
from the search page and the responsibilities kept with the text. Made-up pages, faked transport, no network."""

from __future__ import annotations

import json

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import apple


def _row(pid, title="Engineer", place=("Bengaluru", "India"), **over):
    return {
        "positionId": pid, "postingTitle": title, "transformedPostingTitle": title.lower().replace(" ", "-"),
        "postDateInGMT": "2026-04-22T14:44:56.003755186Z", "homeOffice": False,
        "locations": [{"name": place[0], "countryName": place[1]}], **over,
    }  # fmt: skip


def _page(rows, total):
    data = {"loaderData": {"root": {}, "search": {"searchResults": rows, "totalRecords": total}}}
    script = f"window.__staticRouterHydrationData = JSON.parse({json.dumps(json.dumps(data))});"
    return f"<html><body><script nonce='x'>{script}</script></body></html>"


class _Http:
    def __init__(self, pages, details=None, robots=None):
        self.pages, self.details, self.robots = pages, details or {}, robots
        self.texts: list[str] = []
        self.jsons: list[str] = []

    def get_text(self, url, *, timeout=None):
        self.texts.append(url)
        if url.endswith("/robots.txt"):
            if self.robots is None:
                raise HTTPError("moved", status=301, url=url)
            return self.robots
        page = int(url.split("&page=")[1]) if "&page=" in url else 1
        return self.pages.get(page, _page([], 0))

    def get_json(self, url, *, timeout=None):
        self.jsons.append(url)
        pid = url.split("/jobDetails/")[1].split("?")[0]
        if pid not in self.details:
            raise FetchError("boom", url=url)
        return {"res": self.details[pid]}


def _fetch(http, board="india-INDC", seen=None):
    entry = CompanyEntry(name="Apple", provider="apple", extra={"slug": board})
    return apple.AppleProvider().fetch(entry, FetchContext(http=http, seen=seen or (lambda key: False)))


def test_a_location_is_paged_deduped_by_position_and_mapped():
    rows = [_row("1", "Software Engineer"), _row("1", "Software Engineer", place=("Hyderabad", "India"))]
    http = _Http(
        {1: _page(rows, 3), 2: _page([_row("2", "SRE", place=("India", "India"), homeOffice=True)], 3)},
        {"1": {"jobSummary": "Join the team.", "description": "Build it.", "responsibilities": "Own the service.",
               "minimumQualifications": "2+ years of Python.", "preferredQualifications": "Swift."},
         "2": {"description": "Keep it running.", "minimumQualifications": "", "preferredQualifications": ""}},
    )  # fmt: skip
    first, second = _fetch(http)
    assert http.texts == [
        "https://jobs.apple.com/robots.txt",
        "https://jobs.apple.com/en-us/search?location=india-INDC",
        "https://jobs.apple.com/en-us/search?location=india-INDC&page=2",
    ]
    assert http.jsons == [f"https://jobs.apple.com/api/v1/jobDetails/{i}?locale=en-us" for i in ("1", "2")]
    assert (first.job_id, first.url, first.title, first.company, first.source) == (
        "apple:1", "https://jobs.apple.com/en-us/details/1/software-engineer", "Software Engineer", "Apple", "apple"
    )  # fmt: skip
    assert first.description == (
        "Join the team.\n\nBuild it.\n\nResponsibilities\nOwn the service.\n\n"
        "Minimum qualifications\n2+ years of Python.\n\nPreferred qualifications\nSwift."
    )
    assert (first.location, first.date_posted, first.is_remote) == ("Bengaluru, India", "2026-04-22", None)
    assert (second.location, second.description, second.is_remote) == ("India", "Keep it running.", True)


def test_a_stored_posting_is_re_sighted_from_the_page_and_a_detail_without_text_is_skipped():
    http = _Http({1: _page([_row("1"), _row("2"), _row("3")], 3)}, {"2": {"description": " "}})
    jobs = _fetch(http, seen=lambda key: key == "apple:1")
    assert [(j.job_id, j.description) for j in jobs] == [("apple:1", None)]  # 2 has no text, 3's detail failed
    assert http.jsons == [f"https://jobs.apple.com/api/v1/jobDetails/{i}?locale=en-us" for i in ("2", "3")]


def test_two_cities_of_one_role_on_one_row_are_both_kept():
    row = _row(
        "1", locations=[{"name": "Bengaluru", "countryName": "India"}, {"name": "Hyderabad", "countryName": "India"}]
    )
    assert apple.location(row) == "Bengaluru, India; Hyderabad, India"


@pytest.mark.parametrize(
    "http",
    [
        _Http({1: _page([_row("1")], 9)}),  # an empty page short of the stated count: freehire stops, we fail
        _Http({1: "<html>no data</html>"}),
        _Http({1: _page([_row("1")], 1)}, robots="User-agent: *\nDisallow: /en-us/search\n"),
    ],
)
def test_a_walk_that_is_not_whole_or_not_allowed_fails(http):
    with pytest.raises(FetchError):
        _fetch(http)


@pytest.mark.parametrize("bad", ["", "India", "india", "INDC", "india-INDC&page=2"])
def test_the_board_is_one_search_location(bad):
    with pytest.raises(FetchError):
        _fetch(_Http({}), board=bad)
    provider = apple.AppleProvider()
    assert provider.board_url(CompanyEntry(name="Apple", provider="apple", extra={"slug": "india-INDC"})) == (
        "https://jobs.apple.com/en-us/search?location=india-INDC"
    )
    assert not provider.detect(CompanyEntry(name="Apple", provider="apple", extra={"slug": "india-INDC"}))
