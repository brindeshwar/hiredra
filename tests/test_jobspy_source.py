"""Tests for the JobSpy Path-3 source (build sub-slice 2a.6).

``scrape_jobs`` is always injected (a fake); nothing here hits the network.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from hiredra.sourcing.jobspy_source import (
    JobSpyConfig,
    JobSpySource,
    SiteQuery,
)
from hiredra.sourcing.persist import persist_jobs
from hiredra.sourcing.provider import FetchContext
from hiredra.storage import db

_ROW = {
    "id": "in-abc",
    "site": "indeed",
    "job_url": "https://in.indeed.com/viewjob?jk=abc",
    "job_url_direct": "https://boards.greenhouse.io/x/jobs/1",
    "title": "Backend Engineer",
    "company": "Acme",
    "location": "Bengaluru, India",
    "date_posted": "2026-09-01",
    "job_type": "fulltime",
    "min_amount": 1_500_000.0,
    "max_amount": 2_500_000.0,
    "currency": "INR",
    "interval": "yearly",
    "is_remote": False,
    "description": "Build APIs. Python, FastAPI.",
}
_ROW2 = {
    **_ROW,
    "id": "in-def",
    "job_url": "https://in.indeed.com/viewjob?jk=def",
    "title": "AI Engineer, Remote",
    "is_remote": True,
    "job_url_direct": None,
}


class FakeScrape:
    """Configurable stand-in for jobspy.scrape_jobs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.by_site: dict = {}  # site key -> DataFrame | Exception | callable | None

    def __call__(self, **kw):
        self.calls.append(kw)
        site = kw["site_name"][0]
        value = self.by_site.get(site)
        if isinstance(value, BaseException):
            raise value
        if callable(value):
            return value(**kw)
        if value is not None:
            return value
        return pd.DataFrame()

    def site_calls(self, site: str) -> list[dict]:
        return [c for c in self.calls if c["site_name"][0] == site]


def _source(fake, **cfg):
    slept: list[float] = []
    src = JobSpySource(JobSpyConfig(**cfg), sleep=slept.append, scrape=fake)
    return src, slept


@pytest.fixture()
def ctx(http_factory):
    return FetchContext(http=http_factory(lambda req: __import__("httpx").Response(200, json={})))


# ── mapping ────────────────────────────────────────────────────────────────
def test_maps_rows_to_jobs(ctx):
    fake = FakeScrape()
    fake.by_site["indeed"] = pd.DataFrame([_ROW, _ROW2])
    src, _ = _source(fake, sites=("indeed",), default_query=SiteQuery(search_terms=("backend engineer",)))
    result = src.fetch(ctx)

    assert [o.status for o in result.outcomes] == ["ok"]
    j0, j1 = result.jobs[0], result.jobs[1]
    assert j0.job_id == "jobspy-indeed:in-abc"
    assert j0.source == "jobspy-indeed"
    assert j0.url == "https://in.indeed.com/viewjob?jk=abc"
    assert j0.apply_url == "https://boards.greenhouse.io/x/jobs/1"
    assert j0.date_posted == "2026-09-01"
    assert j0.job_type == "fulltime"
    assert j0.salary == "INR 1,500,000-2,500,000 /yearly"
    assert j0.is_remote is None  # JobSpy False -> not trusted, no keyword
    assert j1.is_remote is True  # "AI Engineer, Remote" keyword + JobSpy True


def test_nan_fields_become_none(ctx):
    row = {
        **_ROW,
        "description": None,
        "min_amount": float("nan"),
        "max_amount": float("nan"),
        "job_url_direct": float("nan"),
    }
    fake = FakeScrape()
    fake.by_site["indeed"] = pd.DataFrame([row])
    src, _ = _source(fake, sites=("indeed",), default_query=SiteQuery(search_terms=("x",)))
    job = src.fetch(ctx).jobs[0]
    assert job.description is None and job.salary is None and job.apply_url is None


def test_malformed_row_skipped(ctx):
    fake = FakeScrape()
    fake.by_site["indeed"] = pd.DataFrame([{**_ROW, "company": None}])
    src, _ = _source(fake, sites=("indeed",), default_query=SiteQuery(search_terms=("x",)))
    result = src.fetch(ctx)
    assert result.jobs == [] and result.outcomes[0].status == "empty"


# ── LinkedIn-specific query ────────────────────────────────────────────────
def test_linkedin_uses_its_own_query(ctx):
    fake = FakeScrape()
    src, _ = _source(
        fake,
        sites=("indeed", "linkedin"),
        default_query=SiteQuery(search_terms=("broad one", "broad two")),
        linkedin_query=SiteQuery(search_terms=("tight",), job_type="fulltime", hours_old=168),
        linkedin_fetch_description=True,
    )
    src.fetch(ctx)

    indeed_terms = {c["search_term"] for c in fake.site_calls("indeed")}
    linkedin_calls = fake.site_calls("linkedin")
    assert indeed_terms == {"broad one", "broad two"}
    assert [c["search_term"] for c in linkedin_calls] == ["tight"]
    assert linkedin_calls[0]["job_type"] == "fulltime"
    assert linkedin_calls[0]["hours_old"] == 168
    assert linkedin_calls[0]["linkedin_fetch_description"] is True
    assert "linkedin_fetch_description" not in fake.site_calls("indeed")[0]


def test_linkedin_falls_back_to_default_when_no_linkedin_query(ctx):
    fake = FakeScrape()
    src, _ = _source(fake, sites=("linkedin",), linkedin_query=None, default_query=SiteQuery(search_terms=("shared",)))
    src.fetch(ctx)
    assert [c["search_term"] for c in fake.site_calls("linkedin")] == ["shared"]


# ── blocked / empty / error / cooldown ─────────────────────────────────────
def _blocks_via_log(logger_name: str, message: str):
    def _handler(**kw):
        logging.getLogger(logger_name).error(message)
        return pd.DataFrame()

    return _handler


def test_blocked_via_log_trips_cooldown(ctx):
    fake = FakeScrape()
    fake.by_site["linkedin"] = _blocks_via_log(
        "JobSpy:LinkedIn", "429 Response - Blocked by LinkedIn for too many requests"
    )
    src, _ = _source(fake, sites=("linkedin",), linkedin_query=SiteQuery(search_terms=("x",)))
    result = src.fetch(ctx)

    assert result.outcomes[0].status == "blocked"
    assert "jobspy-linkedin" in ctx.http.cooldowns.active()

    # second run in the same session skips LinkedIn without calling scrape
    fake.calls.clear()
    result2 = src.fetch(ctx)
    assert result2.outcomes[0].status == "cooldown"
    assert fake.calls == []


@pytest.mark.parametrize(
    "message",
    [
        "429 Response - Blocked by LinkedIn for too many requests",
        "LinkedIn response status code 403 - <html>...",
        'Naukri API response status code 406 - {"message":"recaptcha required","statusCode":406}',
        "Indeed responded with status code: 403 (submit GitHub issue ...)",
        "Glassdoor: Cloudflare challenge encountered",
    ],
)
def test_block_marker_regex_matches_real_jobspy_lines(ctx, message):
    from hiredra.sourcing.jobspy_source import _block_marker

    assert _block_marker(["Scraping page 1 / 1", message, "finished scraping"]) == message


def test_naukri_406_recaptcha_is_a_block_not_empty(ctx):
    fake = FakeScrape()
    fake.by_site["naukri"] = _blocks_via_log(
        "JobSpy:Naukri",
        'Naukri API response status code 406 - {"message":"recaptcha required","statusCode":406}',
    )
    src, _ = _source(fake, sites=("naukri",), default_query=SiteQuery(search_terms=("x",)), cooldown_long=21600.0)
    result = src.fetch(ctx)
    assert result.outcomes[0].status == "blocked"
    assert "jobspy-naukri" in ctx.http.cooldowns.active()  # long cooldown, not "empty"


def test_blocked_via_exception_uses_long_cooldown_for_glassdoor(ctx):
    fake = FakeScrape()
    fake.by_site["glassdoor"] = RuntimeError("403 Forbidden - access denied")
    src, _ = _source(
        fake,
        sites=("glassdoor",),
        default_query=SiteQuery(search_terms=("x",)),
        cooldown_long=21600.0,
        cooldown_short=1800.0,
    )
    result = src.fetch(ctx)

    assert result.outcomes[0].status == "blocked"
    until = ctx.http.cooldowns.active()["jobspy-glassdoor"]
    assert until == pytest.approx(ctx.http.cooldowns._now() + 21600.0, abs=5)


def test_empty_is_not_a_block(ctx):
    fake = FakeScrape()  # returns empty DataFrame, logs nothing
    src, _ = _source(fake, sites=("indeed",), default_query=SiteQuery(search_terms=("x", "y")))
    result = src.fetch(ctx)
    assert result.outcomes[0].status == "empty"
    assert ctx.http.cooldowns.active() == {}


def test_non_block_error_is_isolated(ctx):
    fake = FakeScrape()
    fake.by_site["indeed"] = ValueError("bad param")
    fake.by_site["google"] = pd.DataFrame([_ROW])
    src, _ = _source(fake, sites=("indeed", "google"), default_query=SiteQuery(search_terms=("x",)))
    result = src.fetch(ctx)

    statuses = {o.site: o.status for o in result.outcomes}
    assert statuses == {"indeed": "error", "google": "ok"}
    assert len(result.jobs) == 1
    assert ctx.http.cooldowns.active() == {}  # a non-block error never cools down


# ── google single-call + pacing + guards ───────────────────────────────────
def test_google_single_call_when_google_search_term_set(ctx):
    fake = FakeScrape()
    src, _ = _source(
        fake,
        sites=("google",),
        default_query=SiteQuery(
            search_terms=("a", "b", "c"),
            google_search_term="software engineer jobs india fresher",
        ),
    )
    src.fetch(ctx)
    calls = fake.site_calls("google")
    assert len(calls) == 1
    assert calls[0]["google_search_term"] == "software engineer jobs india fresher"


def test_between_sites_delay(ctx):
    fake = FakeScrape()
    src, slept = _source(
        fake,
        sites=("indeed", "linkedin", "google"),
        between_sites_delay=4.0,
        default_query=SiteQuery(search_terms=("x",)),
        linkedin_query=SiteQuery(search_terms=("x",)),
    )
    src.fetch(ctx)
    assert slept == [4.0, 4.0]  # one gap between each of the 3 sites


def test_empty_sites_raises(ctx):
    src, _ = _source(FakeScrape(), sites=())
    with pytest.raises(ValueError):
        src.fetch(ctx)


# ── persist round-trip ─────────────────────────────────────────────────────
def test_fetch_to_persist(ctx):
    fake = FakeScrape()
    fake.by_site["indeed"] = pd.DataFrame([_ROW, _ROW2])
    src, _ = _source(fake, sites=("indeed",), default_query=SiteQuery(search_terms=("x",)))
    result = src.fetch(ctx)

    conn = db.connect(":memory:")
    db.init_db(conn)
    persisted = persist_jobs(conn, result.jobs)
    assert persisted.inserted == 2
    stored = db.get_job(conn, "jobspy-indeed:in-abc")
    assert stored["source"] == "jobspy-indeed" and stored["status"] == "fetched"
    conn.close()
