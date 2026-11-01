"""Taleo Business Edition provider -- the career site's RSS feed (build step 3, wave 1).

Ported from freehire's ``internal/ingest/sources/taleobe.go`` (MIT, see THIRD_PARTY_NOTICES.md). TBE is Oracle's
small-business Taleo on ``<pod>.tbe.taleo.net``, a different product from Taleo Enterprise. The board is
``<ORG>:<cws>``, the org code and the career-site number every TBE link carries (``?org=COVESTIC2&cws=37``); the pod
is left out, since orgs move between pods and the platform's dispatcher redirects to wherever the org lives now. One
feed lists the whole career site with each posting's HTML inline: no paging, no detail request.

A gone board is a redirect, never a status: an unknown org lands on the sign-in page, a deactivated one on a static
error page, an unknown career site on ``rssErrMsg.jsp``. Those are read as a 404, so the board is retired.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlencode, urlsplit
from xml.etree import ElementTree

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import clean, infer_remote, rss_date, rss_items

_ORG = re.compile(r"^[A-Z0-9_]+$")
_CWS = re.compile(r"^[0-9]+$")
_NS = "{urn:TBERss}"
_TIMEOUT_S = 30.0


def parse_board(board: str) -> tuple[str, str]:
    org, _, cws = (board or "").strip().partition(":")
    if not _ORG.match(org) or not _CWS.match(cws):
        raise FetchError(f'taleobe: board {board!r} must be "<ORG>:<cws>"')
    return org, cws


def feed_url(board: str) -> str:
    """The board's RSS feed, asked through the dispatcher so it reaches the org's pod wherever it lives."""
    org, cws = parse_board(board)
    feed = "https://tbe.taleo.net/ats/servlet/Rss?" + urlencode({"org": org, "cws": cws})
    return "https://tbe.taleo.net/dispatcher/servlet/DispatcherServlet?" + urlencode(
        {"org": org, "act": "redirectOrg", "redirectUrl": feed}
    )


def board_gone(final_url: str) -> bool:
    path = urlsplit(final_url).path
    return path.endswith(("/dispatcher/login.jsp", "/rssErrMsg.jsp")) or path.startswith("/errorpages/")


def _on_tbe(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    return urlsplit(url).scheme == "https" and (host == "tbe.taleo.net" or host.endswith(".tbe.taleo.net"))


class TaleoBEProvider:
    name = "taleobe"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        try:
            return feed_url((entry.extra or {}).get("slug") or "")
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        board = (entry.extra or {}).get("slug") or ""
        url = feed_url(board)
        resp = ctx.http.request("GET", url, follow_redirects=True, timeout=_TIMEOUT_S)
        final = str(resp.url)
        if board_gone(final):  # checked first: a gone board's page also fails to parse, which says nothing more
            raise HTTPError(f"taleobe: board {board} is gone (redirected to {final})", status=404, url=url)
        if not _on_tbe(final):
            raise FetchError(f"taleobe: {url} left Taleo for {final}", url=url)
        org, _ = parse_board(board)
        return [j for j in (to_job(entry.name, org, item) for item in rss_items(resp.text, self.name)) if j is not None]


def to_job(company: str, org: str, item: ElementTree.Element) -> Job | None:
    def text(tag: str) -> str | None:
        return clean(item.findtext(tag))

    link = (text("link") or "").replace("http://", "https://", 1)
    rid = text(f"{_NS}reqId") or (parse_qs(urlsplit(link).query).get("rid") or [None])[0]
    title = text("title")
    if not rid or not title or not link:
        return None
    city_state = ", ".join(p for p in (text(f"{_NS}locationCity"), text(f"{_NS}locationState")) if p)
    location = text(f"{_NS}location") or city_state or None
    return Job(
        job_id=f"taleobe:{org}:{rid}",
        url=link,
        title=title,
        company=company,
        source="taleobe",
        description=html_to_text(text(f"{_NS}html-description") or text("description")) or None,
        location=location,
        date_posted=rss_date(text("pubDate")),
        is_remote=infer_remote(f"{location or ''} {title}"),
    )
