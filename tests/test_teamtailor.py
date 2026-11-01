"""Teamtailor provider (build step 3): freehire's ``teamtailor_test.go`` (MIT) remote-status and id cases, on the
RSS feed our port reads instead of the job pages. Faked transport, no network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import teamtailor

HOST = "acme.teamtailor.com"


def _item(job_id, title="Backend Engineer", status="hybrid", places=(("Bengaluru", "India"),), body="<p>Build.</p>"):
    locs = "".join(f"<tt:location><tt:name>Office</tt:name><tt:city>{c}</tt:city><tt:country>{k}</tt:country>"
                   "</tt:location>" for c, k in places)  # fmt: skip
    escaped = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f"<item><title>{title}</title><description>{escaped}</description>"
            f"<pubDate>Wed, 28 Oct 2026 22:46:17 +0530</pubDate>"
            f"<link>https://{HOST}/jobs/{job_id}-slug</link><remoteStatus>{status}</remoteStatus>"
            f"<tt:locations>{locs}</tt:locations></item>")  # fmt: skip


def _feed(*items):
    return ('<?xml version="1.0"?><rss version="2.0" xmlns:tt="https://teamtailor.com/locations"><channel>'
            f"<title>Acme</title>{''.join(items)}</channel></rss>")  # fmt: skip


class _Http:
    def __init__(self, pages):
        self.pages = list(pages)
        self.asked: list[str] = []

    def get_text(self, url, *, timeout=None):
        self.asked.append(url)
        page = self.pages.pop(0)
        if isinstance(page, Exception):
            raise page
        return page


def _fetch(http, board=HOST):
    entry = CompanyEntry(name="Acme", provider="teamtailor", extra={"slug": board})
    return teamtailor.TeamtailorProvider().fetch(entry, FetchContext(http=http))


def test_the_feed_lists_the_board_with_its_text():
    http = _Http([_feed(
        _item(8632722, places=(("Bengaluru", "India"), ("London", "United Kingdom"))),
        _item(8509129, title="Remote Support Engineer", status="none", places=()),
    )])  # fmt: skip
    backend, support = _fetch(http)
    assert http.asked == [f"https://{HOST}/jobs.rss?offset=0&per_page=500"]
    assert (backend.job_id, backend.url) == (f"teamtailor:{HOST}:8632722", f"https://{HOST}/jobs/8632722-slug")
    assert (backend.title, backend.company, backend.description) == ("Backend Engineer", "Acme", "Build.")
    assert (backend.location, backend.date_posted) == ("Bengaluru, India; London, United Kingdom", "2026-10-28")
    assert backend.is_remote is False  # hybrid is a stated arrangement
    # "none" states nothing, and a "Remote ..." title is never read as the arrangement (freehire)
    assert (support.location, support.is_remote) == (None, None)


@pytest.mark.parametrize(("status", "want"), [("fully", True), ("hybrid", False), ("onsite", False),
                                              ("none", None), ("temporary", None), ("", None)])  # fmt: skip
def test_remote_status(status, want):
    [job] = _fetch(_Http([_feed(_item(1, status=status, places=()))]))
    assert job.is_remote is want


def test_a_location_says_remote_when_the_feed_does_not():
    [job] = _fetch(_Http([_feed(_item(1, status="none", places=(("Remote", "India"),)))]))
    assert (job.location, job.is_remote) == ("Remote, India", True)


def test_a_full_page_asks_for_the_next():
    first = _feed(*(_item(i) for i in range(1, 501)))
    http = _Http([first, _feed(_item(501), _item(1))])  # a posting moving between pages is kept once
    jobs = _fetch(http)
    assert len(jobs) == 501
    assert http.asked[1] == f"https://{HOST}/jobs.rss?offset=500&per_page=500"


def test_an_item_without_a_title_or_job_link_is_skipped_and_a_bad_feed_fails():
    bad_link = _item(1).replace(f"https://{HOST}/jobs/1-slug", f"https://{HOST}/about")
    assert _fetch(_Http([_feed(bad_link, _item(2, title=""))])) == []
    with pytest.raises(FetchError):
        _fetch(_Http(["<!DOCTYPE html><html></html>"]))
    with pytest.raises(HTTPError):
        _fetch(_Http([HTTPError("not found", status=404)]))


def test_a_board_that_never_ends_fails():
    full = _feed(*(_item(i) for i in range(1, 501)))
    with pytest.raises(FetchError):
        _fetch(_Http([full] * teamtailor._MAX_PAGES))


@pytest.mark.parametrize("bad", ["", "acme", "https://acme.teamtailor.com", "acme.teamtailor.com/jobs"])
def test_a_malformed_board_is_refused(bad):
    with pytest.raises(FetchError):
        _fetch(_Http([]), board=bad)
