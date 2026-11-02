"""Amazon provider (build step 3): cases ported from freehire's ``amazon_test.go`` (MIT), plus the country board and
the qualifications kept with the text. Faked transport, no network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import FetchError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import amazon


def _posting(i, **over):
    return {
        "id_icims": str(i), "title": f"SDE {i} ", "job_path": f"/en/jobs/{i}/sde",
        "normalized_location": "Bengaluru, Karnataka, IND", "posted_date": "November  1, 2026",
        "description": "<p>Build services.</p>", "basic_qualifications": "- 3+ years of experience<br/>- Java",
        "preferred_qualifications": "- Masters", "job_schedule_type": "full-time", **over,
    }  # fmt: skip


class _Http:
    def __init__(self, pages):
        self.pages = pages
        self.asked: list[str] = []

    def get_json(self, url, *, timeout=None):
        self.asked.append(url)
        offset = int(url.split("offset=")[1].split("&")[0])
        page = self.pages.get(offset, {"hits": 0, "jobs": []})
        if isinstance(page, Exception):
            raise page
        return page


def _fetch(http, board="IND"):
    entry = CompanyEntry(name="Amazon", provider="amazon", extra={"slug": board})
    return amazon.AmazonProvider().fetch(entry, FetchContext(http=http))


def test_a_country_is_read_page_by_page_with_the_qualifications_in_the_text():
    http = _Http({0: {"hits": 101, "jobs": [_posting(i) for i in range(100)]},
                  100: {"hits": 101, "jobs": [_posting(100, is_intern=True, posted_date="x")]}})  # fmt: skip
    jobs = _fetch(http)
    assert http.asked == [
        f"{amazon.SEARCH}?result_limit=100&offset=0&sort=recent&country=IND",
        f"{amazon.SEARCH}?result_limit=100&offset=100&sort=recent&country=IND",
    ]
    first, last = jobs[0], jobs[-1]
    assert len(jobs) == 101
    assert (first.job_id, first.url, first.title, first.company) == (
        "amazon:0", "https://www.amazon.jobs/en/jobs/0/sde", "SDE 0", "Amazon"
    )  # fmt: skip
    assert first.description == (
        "Build services.\n\nBasic qualifications\n- 3+ years of experience\n- Java\n\nPreferred qualifications\n- Masters"
    )
    assert (first.location, first.date_posted, first.job_type) == (
        "Bengaluru, Karnataka, IND",
        "2026-11-01",
        "fulltime",
    )
    assert (last.job_type, last.date_posted) == ("intern", None)


def test_an_empty_page_ends_the_walk_and_a_failure_fails_the_board():
    assert (
        _fetch(_Http({0: {"hits": 500, "jobs": [_posting(1)]}, 100: {"hits": 500, "jobs": []}}))[0].job_id == "amazon:1"
    )
    with pytest.raises(FetchError):
        _fetch(_Http({0: {"hits": 200, "jobs": [_posting(1)] * 100}, 100: FetchError("timed out")}))
    with pytest.raises(FetchError):
        _fetch(_Http({0: {"error": "x"}}))


def test_a_posting_without_a_title_id_or_path_is_skipped():
    page = {"hits": 3, "jobs": [_posting(1, title=""), _posting(2, id_icims=None), _posting(3, job_path="")]}
    assert _fetch(_Http({0: page})) == []


@pytest.mark.parametrize("bad", ["", "India", "IN", "IND,USA"])
def test_the_board_is_one_country_code(bad):
    with pytest.raises(FetchError):
        _fetch(_Http({}), board=bad)
    assert amazon.AmazonProvider().board_url(CompanyEntry(name="A", provider="amazon", extra={"slug": "ind"})) == (
        "https://www.amazon.jobs/en/search?country=IND"
    )


def test_the_search_cap_is_reported_incomplete():
    notes: list[str] = []
    http = _Http({0: {"hits": 10_000, "jobs": [_posting(1)]}, 100: {"hits": 10_000, "jobs": []}})
    entry = CompanyEntry(name="Amazon", provider="amazon", extra={"slug": "USA"})
    amazon.AmazonProvider().fetch(entry, FetchContext(http=http, incomplete=notes.append))
    assert notes == ["10000 hits: the search's cap"]
