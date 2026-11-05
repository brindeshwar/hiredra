"""Phenom provider (parity step): cases ported from freehire's ``phenom_test.go`` (MIT), plus the stated count, every
place with its country, the shared pace and robots.txt. Made-up answers, faked transport, no network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import phenom

HOST = "careers.example.com"


def _row(seq, **over):
    return {"jobSeqNo": seq, "title": f"Engineer {seq}", "locale": "en_US", "postedDate": "2026-09-08T00:00:00.000+0000",
            "multi_location": ["Pune, India, 411001"], "type": "Full time",
            "applyUrl": f"https://acme.wd1.myworkdayjobs.com/External/job/{seq}/apply", **over}  # fmt: skip


class _Resp:
    def __init__(self, data):
        self.data = data

    def json(self):
        if isinstance(self.data, Exception):
            raise self.data
        return self.data


class _Http:
    """Listing pages from ``rows`` (``total`` stated), details from ``details``; records every call."""

    def __init__(self, rows, details=None, total=None, robots=None):
        self.rows, self.details, self.total, self.robots = rows, details or {}, total, robots
        self.calls: list[dict] = []

    def get_text(self, url, *, timeout=None):
        if self.robots is None:
            raise HTTPError("none", status=404, url=url)
        return self.robots

    def request(self, method, url, *, json=None, timeout=None):
        assert (method, url) == ("POST", f"https://{HOST}/widgets")
        self.calls.append(json)
        if json["ddoKey"] == "refineSearch":
            page = self.rows[json["from"] : json["from"] + json["size"]]
            total = len(self.rows) if self.total is None else self.total
            return _Resp({"refineSearch": {"totalHits": total, "data": {"jobs": page}}})
        body = self.details.get(json["jobSeqNo"], "<p>Build it.</p>")
        return _Resp(body if isinstance(body, Exception) else {"jobDetail": {"data": {"job": {"description": body}}}})


def _fetch(http, seen=None, notes=None):
    entry = CompanyEntry(name="Acme", provider="phenom", extra={"slug": HOST})
    ctx = FetchContext(
        http=http, seen=seen or (lambda key: False), incomplete=(notes if notes is not None else []).append
    )
    return phenom.PhenomProvider(interval=0).fetch(entry, ctx)


def test_the_listing_pages_by_the_stated_count_then_each_new_jobs_detail():
    rows = [_row(f"S{i}") for i in range(phenom.PAGE + 1)]
    http = _Http(rows, details={"S3": ""})  # a detail without text: skipped, read again next run
    jobs = _fetch(http)
    assert len(jobs) == phenom.PAGE  # 101 listed, S3 skipped
    assert [c["from"] for c in http.calls if c["ddoKey"] == "refineSearch"] == [0, phenom.PAGE]
    j = jobs[0]
    assert (j.job_id, j.url, j.title, j.company, j.source) == (
        f"phenom:{HOST}:S0",
        f"https://{HOST}/us/en/job/S0",
        "Engineer S0",
        "Acme",
        "phenom",
    )
    assert (j.location, j.description, j.date_posted, j.job_type) == (
        "Pune, India",
        "Build it.",
        "2026-09-08",
        "fulltime",
    )
    assert j.apply_url == "https://acme.wd1.myworkdayjobs.com/External/job/S0/apply"


def test_a_stored_job_is_re_sighted_from_the_listing_with_no_detail():
    http = _Http([_row("OLD"), _row("NEW")])
    jobs = _fetch(http, seen=lambda key: key == f"phenom:{HOST}:OLD")
    assert {j.job_id: j.description for j in jobs} == {f"phenom:{HOST}:OLD": None, f"phenom:{HOST}:NEW": "Build it."}
    assert [c["jobSeqNo"] for c in http.calls if c["ddoKey"] == "jobDetail"] == ["NEW"]


def test_a_listing_short_of_its_stated_count_is_reported_incomplete():
    notes: list[str] = []
    _fetch(_Http([_row("A")], total=5), notes=notes)
    assert notes == [f"phenom {HOST}: listed 1 of the 5 jobs it states"]
    with pytest.raises(FetchError, match="listed none"):
        _fetch(_Http([], total=3))


@pytest.mark.parametrize(
    ("row", "want"),
    [
        (
            {"multi_location": ["Gurgaon, India, 122002", "Bengaluru, Karnataka, India"]},
            "Gurgaon, India; Bengaluru, Karnataka, India",
        ),  # fmt: skip
        ({"multi_location": ["London, United Kingdom", "London, United Kingdom"]}, "London, United Kingdom"),
        ({"cityStateCountry": "Austin, Texas, United States"}, "Austin, Texas, United States"),
        ({"cityState": "Gurgaon, Haryāna", "country": "India"}, "Gurgaon, Haryāna, India"),
        ({}, None),
    ],
)
def test_location_keeps_every_place_with_its_country(row, want):
    assert phenom.location(row) == want


@pytest.mark.parametrize(
    ("locale", "want"),
    [
        ("en_GLOBAL", f"https://{HOST}/global/en/job/X1"),
        ("en_us", f"https://{HOST}/us/en/job/X1"),
        ("", f"https://{HOST}/job/X1"),
    ],  # fmt: skip
)
def test_job_url(locale, want):
    assert phenom.job_url(HOST, {"jobSeqNo": "X1", "locale": locale}) == want


def test_a_detail_that_fails_skips_only_that_job():
    http = _Http([_row("A"), _row("B")], details={"A": ValueError("not json")})
    assert [j.job_id for j in _fetch(http)] == [f"phenom:{HOST}:B"]


def test_robots_txt_is_obeyed():
    with pytest.raises(FetchError, match="disallows /widgets"):
        _fetch(_Http([_row("A")], robots="User-agent: *\nDisallow: /widgets\n"))
    assert len(_fetch(_Http([_row("A")], robots="User-agent: *\nDisallow: */px-widgets\n"))) == 1


def test_every_request_waits_its_turn_on_one_pace():
    slept: list[float] = []
    clock = iter([0.0, 0.2, 0.3, 5.0])
    pace = phenom.Pace(1.0, sleep=slept.append)
    import hiredra.sourcing.providers._common as mod  # the pace lives with the shared helpers

    real = mod.time.monotonic
    mod.time.monotonic = lambda: next(clock)
    try:
        for _ in range(4):
            pace.wait()
    finally:
        mod.time.monotonic = real
    assert slept == pytest.approx([0.8, 1.7])  # t=0 goes; 0.2 waits to 1.0; 0.3 waits to 2.0; 5.0 goes at once


def test_only_a_host_is_a_board_and_none_is_detected():
    provider = phenom.PhenomProvider()
    assert provider.board_url(CompanyEntry(name="A", provider="phenom", extra={"slug": HOST})) == f"https://{HOST}/"
    assert provider.board_url(CompanyEntry(name="A", provider="phenom", extra={"slug": "not a host"})) is None
    assert not provider.detect(CompanyEntry(name="A", careers_url=f"https://{HOST}/"))
