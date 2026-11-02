"""Tests for the Lever provider (build sub-slice 2a.5)."""

from __future__ import annotations

import httpx
from urllib.parse import urlsplit
import pytest

from hiredra.sourcing.http import Blocked, FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers.lever import LeverProvider

LV = LeverProvider()


def _entry(**over) -> CompanyEntry:
    base = {"name": "Acme", "careers_url": "https://jobs.lever.co/acme"}
    base.update(over)
    return CompanyEntry(**base)


@pytest.mark.parametrize(
    "entry, expected",
    [
        (_entry(), True),
        (CompanyEntry(name="A", careers_url="https://job-boards.greenhouse.io/a"), False),
        (CompanyEntry(name="A", api="https://api.lever.co/v0/postings/a"), True),
        (CompanyEntry(name="A", api="https://api.eu.lever.co/v0/postings/a"), True),
        (CompanyEntry(name="A", extra={"slug": "a"}), True),
        (_entry(provider="ashby"), False),
        (CompanyEntry(name="A"), False),
    ],
)
def test_detect(entry, expected):
    assert LV.detect(entry) is expected


def test_bad_api_host_does_not_detect():
    assert LV.detect(CompanyEntry(name="A", api="https://evil.example.com/v0/postings/a")) is False


def test_fetch_maps_jobs(http_factory, lever_payload):
    http = http_factory(lambda req: httpx.Response(200, json=lever_payload))
    jobs = LV.fetch(_entry(), FetchContext(http=http))

    assert [j.job_id for j in jobs] == ["lever:aaaa-1111", "lever:bbbb-2222"]  # malformed skipped
    j0 = jobs[0]
    assert j0.url == "https://jobs.lever.co/acme/aaaa-1111"
    assert j0.apply_url == "https://jobs.lever.co/acme/aaaa-1111/apply"
    assert j0.title == "Senior Backend Engineer"
    assert j0.source == "lever"
    assert j0.description == "Build backend systems."
    assert j0.location == "Berlin; Remote - EU"  # folded, deduped, "; "
    assert j0.date_posted == "2024-09-10"  # epoch-ms -> date
    assert j0.job_type == "Full-time"
    assert j0.is_remote is False  # workplaceType "hybrid" -> explicit False
    assert (j0.work_mode, j0.employment_type) == ("hybrid", "full_time")
    assert (j0.salary_min, j0.salary_max, j0.salary_currency, j0.salary_period) == (8, 10, "INR", "year")
    assert (jobs[1].work_mode, jobs[1].salary_min) == ("remote", None)  # no salaryRange: none


def test_remote_from_workplace_type(http_factory, lever_payload):
    http = http_factory(lambda req: httpx.Response(200, json=lever_payload))
    jobs = LV.fetch(_entry(), FetchContext(http=http))
    assert jobs[1].is_remote is True  # workplaceType "remote"


def test_eu_host_is_derived(http_factory):
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        return httpx.Response(200, json=[])

    http = http_factory(handler)
    LV.fetch(CompanyEntry(name="A", careers_url="https://jobs.eu.lever.co/acme"), FetchContext(http=http))
    assert seen["url"] == "https://api.eu.lever.co/v0/postings/acme"


def test_empty_and_non_array(http_factory):
    http1 = http_factory(lambda req: httpx.Response(200, json=[]))
    assert LV.fetch(_entry(), FetchContext(http=http1)) == []
    http2 = http_factory(lambda req: httpx.Response(200, json={"not": "an array"}))
    assert LV.fetch(_entry(), FetchContext(http=http2)) == []


def test_non_https_api_refused(http_factory):
    http = http_factory(lambda req: httpx.Response(200, json=[]))
    with pytest.raises(FetchError):
        LV.fetch(CompanyEntry(name="A", api="http://api.lever.co/v0/postings/a"), FetchContext(http=http))


def test_a_board_missing_on_the_default_host_is_asked_on_the_eu_host(http_factory):
    asked = []

    def handler(req):
        asked.append(str(req.url))
        return httpx.Response(200, json=[]) if req.url.host == "api.eu.lever.co" else httpx.Response(404)

    assert LV.fetch(_entry(), FetchContext(http=http_factory(handler))) == []
    assert [urlsplit(u).hostname for u in asked] == ["api.lever.co", "api.eu.lever.co"]


def test_404_and_429(http_factory):
    with pytest.raises(HTTPError):
        LV.fetch(_entry(), FetchContext(http=http_factory(lambda req: httpx.Response(404))))
    with pytest.raises(Blocked):
        LV.fetch(_entry(), FetchContext(http=http_factory(lambda req: httpx.Response(429))))


# -- the body lives in description + lists + additional, not in descriptionPlain (2026-10-26) ------
_FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def test_live_posting_body_includes_the_lists(http_factory):
    """A real posting captured from api.lever.co (board tsmg, 2026-10-26): descriptionPlain is the
    intro only; the requirements live in ``lists``."""
    payload = __import__("json").loads((_FIXTURES / "lever_live_posting_with_lists.json").read_text(encoding="utf-8"))
    http = http_factory(lambda req: httpx.Response(200, json=payload))
    (job,) = LV.fetch(_entry(careers_url="https://jobs.lever.co/tsmg"), FetchContext(http=http))
    plain = payload[0]["descriptionPlain"]
    assert len(job.description) > 3 * len(plain)  # 384 chars of intro vs the whole body
    for heading in ("Key Responsibilities", "Requirements", "Schedule & Commitment"):
        assert heading in job.description
    assert "<" not in job.description  # plain text, no markup left


def test_body_order_is_description_then_lists_then_additional(http_factory):
    posting = {
        "id": "p-1",
        "text": "Engineer",
        "hostedUrl": "https://jobs.lever.co/acme/p-1",
        "description": "<div>Intro text.</div>",
        "lists": [
            {"text": "Requirements", "content": "<li>3+ years of Python</li><li>SQL</li>"},
            {"text": "", "content": "<li>an item without a heading</li>"},
            "not a dict",
        ],
        "additional": "<div>Closing note.</div>",
        "descriptionPlain": "Intro text.",
    }
    http = http_factory(lambda req: httpx.Response(200, json=[posting]))
    (job,) = LV.fetch(_entry(), FetchContext(http=http))
    text = job.description
    assert text.index("Intro text.") < text.index("Requirements") < text.index("3+ years of Python")
    assert text.index("an item without a heading") < text.index("Closing note.")


@pytest.mark.parametrize("html_fields", [{}, {"description": "", "lists": [], "additional": ""}])
def test_falls_back_to_description_plain_when_no_html(http_factory, html_fields):
    posting = {
        "id": "p-2",
        "text": "Engineer",
        "hostedUrl": "https://jobs.lever.co/acme/p-2",
        "descriptionPlain": "Only the plain text.",
        **html_fields,
    }
    http = http_factory(lambda req: httpx.Response(200, json=[posting]))
    (job,) = LV.fetch(_entry(), FetchContext(http=http))
    assert job.description == "Only the plain text."
