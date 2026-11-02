"""IBM provider (build step 3): cases ported from freehire's ``ibm_test.go`` (MIT), on small made-up pages of the same
shape (not freehire's fixtures). Faked transport, no network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import FetchError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import ibm


class _Resp:
    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


def _page(first, n, total, relation="eq"):
    hits = [
        {"_source": {"field_text_01": i, "title": f"Job {i}", "body": f"Body {i}"}} for i in range(first, first + n)
    ]
    return {"hits": {"total": {"value": total, "relation": relation}, "hits": hits}}


class _Http:
    """Answers by the request's ``from``; an offset with no page answers an empty page stating ``empty_total``."""

    def __init__(self, pages, fail=(), empty_total=0):
        self.pages, self.fail, self.empty_total = pages, set(fail), empty_total
        self.bodies: list[dict] = []

    def request(self, method, url, *, json=None, timeout=None):
        assert (method, url) == ("POST", ibm.SEARCH)
        self.bodies.append(json)
        if json["from"] in self.fail:
            raise FetchError("boom")
        return _Resp(self.pages.get(json["from"], _page(0, 0, self.empty_total)))


def _fetch(http, board=ibm.BOARD):
    entry = CompanyEntry(name="IBM", provider="ibm", extra={"slug": board})
    return ibm.IBMProvider().fetch(entry, FetchContext(http=http))


def test_a_posting_maps_with_its_facets_cut_and_its_text_repaired():
    source = {
        "field_text_01": 136943, "title": "Internship  Cloud FullStack Developer", "dcdate": "2026-10-13",
        "body": "You’ll build things. Youâ€™ll learn. Skills: • Java • SQL Italy Software Engineering Hybrid Internship "
                "Multiple Cities IBM Italia S.p.A.",
        "field_keyword_05": "Italy", "field_keyword_08": "Software Engineering", "field_keyword_17": "Hybrid",
        "field_keyword_18": "Internship", "field_keyword_19": "Multiple Cities", "field_keyword_20": "IBM Italia S.p.A.",
    }  # fmt: skip
    pro = {"field_text_01": 136151, "title": "Developer", "body": "Do it.", "field_keyword_05": "Italy",
           "field_keyword_17": "", "field_keyword_18": "Professional", "field_keyword_19": "BARI, IT"}  # fmt: skip
    page = {"hits": {"total": {"value": 2, "relation": "eq"}, "hits": [{"_source": source}, {"_source": pro}]}}
    intern, dev = _fetch(_Http({0: page}))
    assert (intern.job_id, intern.title, intern.company) == (
        "ibm:136943",
        "Internship Cloud FullStack Developer",
        "IBM",
    )
    assert intern.url == "https://careers.ibm.com/careers/JobDetail?jobId=136943"
    assert intern.description == "You’ll build things. You’ll learn. Skills:\n• Java\n• SQL"
    assert (intern.location, intern.is_remote, intern.job_type, intern.date_posted) == ("Italy", False, "intern",
                                                                                        "2026-10-13")  # fmt: skip
    assert (dev.location, dev.is_remote, dev.job_type) == ("BARI, Italy", None, None)  # no stated mode stays unknown


def test_the_walk_sorts_by_a_unique_key_and_pages_to_the_stated_total():
    http = _Http({0: _page(1, 100, 150), 100: _page(101, 50, 150)})
    assert len(_fetch(http)) == 150
    assert [b["from"] for b in http.bodies] == [0, 100]
    assert http.bodies[0]["sort"] == [{"id": "asc"}] and "body" in http.bodies[0]["_source"]
    exact = _Http({0: _page(1, 100, 100)}, empty_total=100)
    assert len(_fetch(exact)) == 100  # an exact multiple of the page ends on an empty page


@pytest.mark.parametrize(
    "http",
    [
        _Http({}, fail={0}),
        _Http({0: _page(1, 100, 150)}, fail={100}),  # a later page failing fails the whole board
        _Http({0: _page(1, 100, 104), 100: _page(100, 4, 104)}),  # a re-served row: short of the total
        _Http({0: _page(1, 3, 3, relation="gte")}),  # not an exact count
    ],
)
def test_a_walk_that_is_not_whole_fails(http):
    with pytest.raises(FetchError):
        _fetch(http)


@pytest.mark.parametrize(
    ("city", "country", "want"),
    [("Bangalore, IN", "India", "Bangalore, India"), ("Multiple Cities", "United States", "United States"),
     ("", "Canada", "Canada"), ("Yorktown Heights, US", "", "Yorktown Heights, US"),
     ("Seoul, KR", "Korea, Republic of", "Seoul, KR")],
)  # fmt: skip
def test_location(city, country, want):
    assert ibm.location(city, country) == want


def test_only_its_own_board():
    with pytest.raises(FetchError):
        _fetch(_Http({}), board="ibm")
    assert ibm.IBMProvider().board_url(CompanyEntry(name="IBM", provider="ibm", extra={"slug": "x"})) is None
