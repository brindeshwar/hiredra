"""iCIMS provider -- the career site's sitemap, then each new posting's page (build step 3, wave 1).

Ported from freehire's ``internal/ingest/sources/icims.go`` (MIT, see THIRD_PARTY_NOTICES.md). The board is the slug
of a classic ``careers-<slug>.icims.com`` host, or a whole host (``globalcareers-msci.icims.com``, or an employer's own
domain), freehire's two forms. The sitemap, an index followed one level, names the postings and nothing else; a new
posting's text is the schema.org JobPosting on its server-rendered ``?in_iframe=1`` page (a hydrating source, D3.4).

One change from freehire: the page URL follows the posting link's shape, not the board's. freehire reads every
whole-host board as a "careers-home" site and asks ``/careers-home/jobs/<id>``, which answers 404 on the
``<prefix>-<company>.icims.com`` hosts most of our apply links name (measured 2026-11-01, MSCI). These are HTML pages,
so robots.txt is obeyed (D4.2).
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

from hiredra.sourcing._html import html_to_text, ld_job_posting
from hiredra.sourcing.http import FetchError, HttpConfig
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import (
    DETAIL_BUDGET,
    clean,
    hydrate,
    infer_remote,
    iso_date,
    Robots,
    sitemap_locs,
    work_mode,
)

_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_HOST = re.compile(r"^[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+$")
#: The posting id: ``/jobs/<digits>`` ended by a slash, query, fragment or the end (not ``/jobs/search``).
_JOB_ID = re.compile(r"/jobs/(\d+)(?:[/?#]|$)")
#: A classic posting link, ``/jobs/<id>/<title-slug>/job``: its page is the link plus ``in_iframe=1``.
_CLASSIC = re.compile(r"/jobs/\d+/([^/?#]+)/job\b")
_TIMEOUT_S = 30.0


def host(board: str) -> str:
    """The site a board names: a bare slug is ``careers-<slug>.icims.com``, anything with a dot is the host."""
    board = board.strip().lower()
    if "." in board:
        if not _HOST.match(board):
            raise FetchError(f"icims: board {board!r} is not a host")
        return board
    if not _SLUG.match(board):
        raise FetchError(f"icims: board {board!r} is not a slug")
    return f"careers-{board}.icims.com"


def job_id(loc: str) -> str | None:
    m = _JOB_ID.search(loc)
    return m.group(1) if m else None


def page_url(site: str, loc: str) -> str:
    """The server-rendered page of a posting link: a classic link with ``in_iframe=1`` added; any other (a
    careers-home ``/jobs/<id>?lang=..``) at ``/careers-home/jobs/<id>?in_iframe=1``."""
    if _CLASSIC.search(loc):
        return loc + ("&" if "?" in loc else "?") + "in_iframe=1"
    return f"https://{site}/careers-home/jobs/{job_id(loc)}?in_iframe=1"


def _available(value: object) -> str | None:
    """A JobPosting address part, with iCIMS's ``UNAVAILABLE`` placeholder dropped."""
    text = clean(value)
    return None if text == "UNAVAILABLE" else text


class ICIMSProvider:
    name = "icims"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        try:
            return f"https://{host((entry.extra or {}).get('slug') or '')}/jobs/search"
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        site = host((entry.extra or {}).get("slug") or "")

        def get(url: str) -> str:
            if urlsplit(url).scheme != "https" or urlsplit(url).hostname != site:
                raise FetchError(f"icims: refusing {url!r} off the board's site {site}", url=url)
            return ctx.http.get_text(url, timeout=_TIMEOUT_S)

        def robots_txt(url: str) -> str | None:
            try:
                return get(url)
            except FetchError:
                return None

        config = getattr(ctx.http, "config", None)
        robots = Robots(robots_txt, getattr(config, "user_agent", HttpConfig().user_agent))
        sitemap = f"https://{site}/sitemap.xml"
        if not robots.allowed(sitemap):
            raise FetchError(f"icims: {site} robots.txt disallows its sitemap", url=sitemap)
        pages, subs = sitemap_locs(get(sitemap))
        for sub in subs:  # a sub-sitemap's postings are part of the board: one failing fails the board (freehire)
            pages += sitemap_locs(get(sub))[0]
        locs = [loc for loc in dict.fromkeys(loc for loc, _ in pages) if job_id(loc)]

        def detail(loc: str) -> Job | None:
            page = page_url(site, loc)
            return to_job(entry.name, site, loc, get(page)) if robots.allowed(page) else None

        return hydrate(
            locs,
            key=lambda loc: f"icims:{site}:{job_id(loc)}",
            seen=ctx.seen,
            listing=lambda loc: listing_job(entry.name, site, loc),
            detail=detail,
            budget=DETAIL_BUDGET,
            label=f"icims {site}",
        )


def listing_job(company: str, site: str, loc: str) -> Job:
    """A stored posting re-sighted in the sitemap, which carries only its link: the title is the link's slug (the
    store keeps the one the page gave, D3.4)."""
    slug = _CLASSIC.search(loc)
    title = unquote(slug.group(1)).replace("-", " ").strip() if slug else ""
    return Job(
        job_id=f"icims:{site}:{job_id(loc)}",
        url=loc,
        title=title or f"iCIMS posting {job_id(loc)}",
        company=company,
        source="icims",
    )


def to_job(company: str, site: str, loc: str, page: str) -> Job | None:
    """A posting from its page's JobPosting; ``None`` when the page carries none or no title (skipped, read again
    next run). The first ``jobLocation`` is the location, as freehire takes it; ``TELECOMMUTE`` means remote."""
    posting = ld_job_posting(page)
    title = clean((posting or {}).get("title"))
    if not posting or not title:
        return None
    places = posting.get("jobLocation")
    place = (places[0] if places else {}) if isinstance(places, list) else places
    address = (place or {}).get("address") if isinstance(place, dict) else None
    address = address if isinstance(address, dict) else {}
    parts = (_available(address.get(k)) for k in ("addressLocality", "addressRegion", "addressCountry"))
    location = ", ".join(p for p in parts if p) or None
    workplace = "remote" if posting.get("jobLocationType") == "TELECOMMUTE" else None
    return Job(
        job_id=f"icims:{site}:{job_id(loc)}",
        url=loc,
        title=title,
        company=company,
        source="icims",
        description=html_to_text(posting.get("description")) or None,
        location=location,
        date_posted=iso_date(posting.get("datePosted")),
        work_mode=work_mode(workplace),
        is_remote=infer_remote(f"{location or ''} {title}", workplace_type=workplace),
    )
