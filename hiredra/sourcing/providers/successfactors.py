"""SAP SuccessFactors provider -- the career site's job sitemap, then each new posting's page (build step 3, wave 1).

Ported from freehire's ``internal/ingest/sources/successfactors.go`` (MIT, see THIRD_PARTY_NOTICES.md). The board is
the career site's host (``jobs.tetrapak.com``, ``<tenant>.jobs2web.com``). ``/job_sitemap.xml`` names every posting and
nothing else; a new posting's title and text are the schema.org microdata on its server-rendered page, which
redirects once to a localized URL on the same host (a hydrating source, D3.4). These are HTML pages, so robots.txt is
obeyed (D4.2).

As in freehire, the location is left empty: the page states it only in a layout block whose label each employer
words (measured 2026-11-01, Tetra Pak "Job Location:"), and our location filter keeps an unknown location rather than
guess. freehire's multi-employer "hub" sites (a curated tenant map) are not ported.
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

from hiredra.sourcing._html import html_to_text, itemprop_html, meta_property
from hiredra.sourcing.http import FetchError, HttpConfig
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import (
    DETAIL_BUDGET,
    Robots,
    clean,
    hydrate,
    infer_remote,
    iso_date,
    sitemap_locs,
)

_HOST = re.compile(r"^[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+$")
#: The posting id: the leading digits of the link's last segment, a locale suffix ignored (``.../98012-en_GB/``).
_JOB_ID = re.compile(r"/(\d+)(?:-[^/]*)?/?$")
#: The title slug before the id (``/job/Utrecht-Services-Sales-Manager/1452939233/``).
_SLUG = re.compile(r"/job/([^/]+)/\d+")
_TIMEOUT_S = 30.0


def site(entry: CompanyEntry) -> str:
    host = ((entry.extra or {}).get("slug") or "").strip().lower()
    if not _HOST.match(host):
        raise FetchError(f"successfactors: board {host!r} is not a host")
    return host


def job_id(loc: str) -> str | None:
    m = _JOB_ID.search(urlsplit(loc).path)
    return m.group(1) if m else None


class SuccessFactorsProvider:
    name = "successfactors"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        try:
            return f"https://{site(entry)}/"
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        host = site(entry)

        def get(url: str, *, follow: bool = False) -> str:
            if urlsplit(url).scheme != "https" or urlsplit(url).hostname != host:
                raise FetchError(f"successfactors: refusing {url!r} off the board's site {host}", url=url)
            resp = ctx.http.request("GET", url, follow_redirects=follow, timeout=_TIMEOUT_S)
            if urlsplit(str(resp.url)).hostname != host:
                raise FetchError(f"successfactors: {url} left the site for {resp.url}", url=url)
            return resp.text

        def robots_txt(url: str) -> str | None:
            try:
                return get(url)
            except FetchError:
                return None

        config = getattr(ctx.http, "config", None)
        robots = Robots(robots_txt, getattr(config, "user_agent", HttpConfig().user_agent))
        sitemap = f"https://{host}/job_sitemap.xml"
        if not robots.allowed(sitemap):
            raise FetchError(f"successfactors: {host} robots.txt disallows its job sitemap", url=sitemap)
        pages = [(loc, lastmod) for loc, lastmod in sitemap_locs(get(sitemap))[0] if job_id(loc)]
        pages = list(dict(pages).items())  # one entry per link, the last lastmod

        def detail(page: tuple[str, str | None]) -> Job | None:
            loc, lastmod = page
            if not robots.allowed(loc):
                return None
            return to_job(entry.name, host, loc, lastmod, get(loc, follow=True))

        return hydrate(
            pages,
            key=lambda page: f"successfactors:{host}:{job_id(page[0])}",
            seen=ctx.seen,
            listing=lambda page: listing_job(entry.name, host, *page),
            detail=detail,
            budget=DETAIL_BUDGET,
            label=f"successfactors {host}",
        )


def listing_job(company: str, host: str, loc: str, lastmod: str | None) -> Job:
    """A stored posting re-sighted in the sitemap, which carries only its link: the title is the link's slug (the
    store keeps the one the page gave, D3.4)."""
    slug = _SLUG.search(loc)
    title = unquote(slug.group(1)).replace("-", " ").strip() if slug else ""
    return Job(
        job_id=f"successfactors:{host}:{job_id(loc)}",
        url=loc,
        title=title or f"SuccessFactors posting {job_id(loc)}",
        company=company,
        source="successfactors",
        date_posted=iso_date(lastmod),
    )


def to_job(company: str, host: str, loc: str, lastmod: str | None, page: str) -> Job | None:
    """A posting from its page: the ``title`` microdata (else ``og:title``) and the richest ``description`` element,
    as a page may wrap near-empty layout regions around the real body. ``None`` without a title (skipped, read again
    next run)."""
    titles = [html_to_text(t) for t in itemprop_html(page, "title")]
    title = clean(next((t for t in titles if t), None)) or meta_property(page, "og:title")
    if not title:
        return None
    bodies = [html_to_text(b) for b in itemprop_html(page, "description")]
    return Job(
        job_id=f"successfactors:{host}:{job_id(loc)}",
        url=loc,
        title=title,
        company=company,
        source="successfactors",
        description=max(bodies, key=len, default="") or None,
        date_posted=iso_date(lastmod),
        is_remote=infer_remote(title),
    )
