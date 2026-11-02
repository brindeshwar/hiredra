"""Hacker News "Who is hiring?" (build step 3, wave 3): cases ported from freehire's ``hackernews_test.go`` (MIT), on
small made-up threads, plus the permalink as the URL and the header read from the first line. No network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import FetchError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import hackernews as hn


class _Http:
    def __init__(self, pages):
        self.pages = pages
        self.asked: list[str] = []

    def get_json(self, url, *, timeout=None):
        self.asked.append(url)
        page = self.pages[url]
        if isinstance(page, Exception):
            raise page
        return page


_SEARCH = {"hits": [
    {"objectID": "3", "title": "Ask HN: Who wants to be hired? (October 2026)"},
    {"objectID": "2", "title": "Ask HN: Who is hiring? (October 2026)"},
    {"objectID": "1", "title": "Ask HN: Who is hiring? (September 2026)"},
    {"objectID": "0", "title": "Ask HN: Who is hiring? (August 2026)"},
]}  # fmt: skip


def _post(pid, text, created="2026-10-01T15:00:00Z"):
    return {"id": pid, "text": text, "created_at": created}


def _fetch(http, board="news.ycombinator.com"):
    entry = CompanyEntry(name=board, provider="hackernews", extra={"slug": board})
    return hn.HackerNewsProvider().fetch(entry, FetchContext(http=http))


def test_the_two_newest_hiring_threads_are_read_and_each_post_is_a_job():
    acme = ('Acme | Backend Engineer | Remote (India) | Full-time | $100k | '
            '<a href="https:&#x2F;&#x2F;acme.com&#x2F;jobs" rel="nofollow">https:&#x2F;&#x2F;acme.com&#x2F;jobs</a>'
            "<p>We build things.<p>Apply below.")  # fmt: skip
    http = _Http({
        hn.SEARCH: _SEARCH,
        hn.ITEM.format(2): {"children": [_post(10, acme), _post(11, None), _post(12, "No pipes here at all")]},
        hn.ITEM.format(1): {"children": [_post(10, acme), _post(13, "Beta | Data Scientist | Onsite | Berlin")]},
    })  # fmt: skip
    jobs = _fetch(http)
    assert http.asked == [hn.SEARCH, hn.ITEM.format(2), hn.ITEM.format(1)]  # never "who wants to be hired"
    acme_job, beta = jobs
    assert (acme_job.job_id, acme_job.url, acme_job.apply_url) == (
        "hackernews:10", "https://news.ycombinator.com/item?id=10", "https://acme.com/jobs"
    )  # fmt: skip
    assert (acme_job.company, acme_job.title, acme_job.location, acme_job.is_remote) == (
        "Acme", "Backend Engineer", "Remote (India)", True
    )  # fmt: skip
    assert acme_job.description.endswith("We build things.\n\nApply below.")
    assert (acme_job.date_posted, acme_job.source) == ("2026-10-01", "hackernews")
    assert (beta.location, beta.apply_url, beta.is_remote) == (
        "Onsite",
        None,
        None,
    )  # no link: none; a commitment skipped


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("Acme | Engineer | Full-time | $150k | Berlin", ("Acme", "Engineer", "Berlin")),  # commitment, salary skipped
        ("Acme | Engineer | NYC\nWe are a team of ten.", ("Acme", "Engineer", "NYC")),  # the header is the first line
        ("Senior Software Engineer, Frontend | Acme | Remote", None),  # a role where the employer goes
        ("Lead Bank | Engineer | Kansas City", ("Lead Bank", "Engineer", "Kansas City")),  # an employer, not a role
        ("https://acme.com | Engineer", None),  # a link names no employer
        ("Acme", None),
    ],
)
def test_the_header(text, want):
    got = hn.header(text)
    assert (got[:3] if got else None) == want


def test_no_hiring_thread_or_an_unreadable_one_fails_the_board():
    with pytest.raises(FetchError):
        _fetch(_Http({hn.SEARCH: {"hits": [{"objectID": "3", "title": "Ask HN: Who wants to be hired?"}]}}))
    with pytest.raises(FetchError):
        _fetch(_Http({hn.SEARCH: _SEARCH, hn.ITEM.format(2): {"children": []}, hn.ITEM.format(1): FetchError("x")}))


def test_the_board_is_the_site():
    with pytest.raises(FetchError):
        _fetch(_Http({}), board="hn")
    provider = hn.HackerNewsProvider()
    entry = CompanyEntry(name="x", provider="hackernews", extra={"slug": "news.ycombinator.com"})
    assert provider.board_url(entry) and not provider.detect(entry) and provider.many_companies
