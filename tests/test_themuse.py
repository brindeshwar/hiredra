"""The Muse (build step 3, wave 3): freehire's ``themuse_test.go`` (MIT) mapping cases on made-up pages, plus the India
board read by category and level, a query past the page limit read again in halves, and the hourly allowance. No
network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import FetchError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import themuse


def _posting(pid, **over):
    return {
        "id": pid, "name": f"Engineer {pid}", "contents": "<p>Build.</p>", "publication_date": "2026-10-30T12:00:00Z",
        "locations": [{"name": "Bangalore, India"}, {"name": "Flexible / Remote"}], "levels": [{"name": "Entry Level"}],
        "refs": {"landing_page": f"https://www.themuse.com/jobs/acme/engineer-{pid}"}, "company": {"name": "Acme"},
        **over,
    }  # fmt: skip


class _Http:
    """Answers a query by its (category, level, cities, page); every other query has no results."""

    def __init__(self, answers, fail_after=None):
        self.answers, self.fail_after = answers, fail_after
        self.asked: list[tuple] = []

    def get_json(self, url, *, params=None, timeout=None):
        assert url == themuse.API
        d = {}
        cities = []
        for k, v in params:
            if k == "location":
                cities.append(v)
            else:
                d[k] = v
        key = (d["category"], d["level"], len(cities), d["page"])
        self.asked.append(key)
        if self.fail_after is not None and len(self.asked) > self.fail_after:
            raise FetchError("429")
        return self.answers.get(key, {"results": [], "page_count": 0})


def _fetch(http, board="IN"):
    entry = CompanyEntry(name="The Muse", provider="themuse", extra={"slug": board})
    return themuse.TheMuseProvider().fetch(entry, FetchContext(http=http))


ALL = len(themuse.INDIA)


def test_a_posting_maps_with_every_place_and_the_internship_level():
    se = "Software Engineering"
    http = _Http({
        (se, "Internship", ALL, 1): {"results": [_posting(1, levels=[{"name": "Internship"}])], "page_count": 2},
        (se, "Internship", ALL, 2): {"results": [_posting(2, company={"name": ""})], "page_count": 2},
        (se, "Entry Level", ALL, 1): {"results": [_posting(1), _posting(3)], "page_count": 1},
    })  # fmt: skip
    first, third = _fetch(http)
    assert (first.job_id, first.url, first.title, first.company, first.source) == (
        "themuse:1", "https://www.themuse.com/jobs/acme/engineer-1", "Engineer 1", "Acme", "themuse"
    )  # fmt: skip
    assert (first.location, first.description, first.date_posted, first.job_type, first.is_remote) == (
        "Bangalore, India; Flexible / Remote", "Build.", "2026-10-30", "intern", True
    )  # fmt: skip
    assert third.job_type is None  # posting 2 has no company: skipped; posting 1 is read once
    assert len(http.asked) == len(themuse.CATEGORIES) * len(themuse.LEVELS) + 1  # one request per empty query


def test_a_query_past_the_page_limit_is_read_again_in_halves_of_its_cities():
    se, senior, half = "Software Engineering", "Senior Level", ALL // 2
    http = _Http({
        (se, senior, ALL, 1): {"results": [_posting(1)], "page_count": 120},
        (se, senior, half, 1): {"results": [_posting(2)], "page_count": 1},
        (se, senior, ALL - half, 1): {"results": [_posting(2), _posting(3)], "page_count": 1},
    })  # fmt: skip
    assert sorted(j.job_id for j in _fetch(http)) == ["themuse:2", "themuse:3"]  # the whole query is not read
    assert (se, senior, ALL, 2) not in http.asked


def test_a_later_failure_keeps_what_was_read_and_a_first_one_fails_the_board():
    se = "Software Engineering"
    http = _Http({(se, "Internship", ALL, 1): {"results": [_posting(1)], "page_count": 1}}, fail_after=3)
    assert [j.job_id for j in _fetch(http)] == ["themuse:1"]
    with pytest.raises(FetchError):
        _fetch(_Http({}, fail_after=0))


def test_the_board_is_india():
    with pytest.raises(FetchError):
        _fetch(_Http({}), board="US")
    provider = themuse.TheMuseProvider()
    entry = CompanyEntry(name="x", provider="themuse", extra={"slug": "in"})
    assert provider.board_url(entry) and not provider.detect(entry) and provider.many_companies
