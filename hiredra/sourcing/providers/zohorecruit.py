"""Zoho Recruit provider -- the careers page's embedded opening list, then each new posting's page (build step 3,
wave 1).

Ported from freehire's ``internal/ingest/sources/zoho.go`` (MIT, see THIRD_PARTY_NOTICES.md). The board is the careers
host (``kanerika.zohorecruit.com``, ``elfonze.zohorecruit.in``: the domain follows the tenant's data region). The
``/jobs/Careers`` page carries every opening as an HTML-escaped JSON array in a hidden ``<input id="jobs">``; a new
posting's text is read from its own page, which embeds the record as a JS-escaped string (a hydrating source, D3.4).
These are HTML pages, so robots.txt is obeyed (D4.2).

Some tenants put a plain-text copy of the text in the list too; it is the page's text with the paragraphs run
together (measured 2026-11-01, Elfonze and Agilisium), so the page is still read, as freehire does. A tenant's other
portals list a subset of ``Careers`` (Teleradiology Solutions' ``TRS``: 11 of its 21). A host that is no tenant
answers 200 with a "does not exist" page, read as a 404 so the board is retired.
"""

from __future__ import annotations

import html
import json
import re
from datetime import date
from urllib.parse import urlsplit

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError, HTTPError, HttpConfig
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import DETAIL_BUDGET, Robots, clean, hydrate, infer_remote, iso_date

_HOST = re.compile(r"^[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+$")
_INPUT = re.compile(r"<input\b[^>]*>", re.I)
_JOBS_ID = re.compile(r"""\bid\s*=\s*["']jobs["']""", re.I)
_VALUE = re.compile(r"""\bvalue\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.I)
_NO_TENANT = re.compile(r"<title>\s*Page does not exist\s*</title>", re.I)
#: The record's ``Job_Description`` in the page's JS-escaped blob (quotes written ``\x22``): the value runs to the
#: next field, an escaped ``","`` followed by a field name.
_DESCRIPTION = re.compile(r"\\x22Job_Description\\x22:\\x22(.*?)\\x22,\\x22[A-Za-z_]", re.S)
#: The record's ``Date_Opened``, its hyphens escaped (``2024\-11\-11``).
_OPENED = re.compile(r"\\x22Date_Opened\\x22:\\x22([0-9\\-]+)\\x22")
_SIMPLE_ESCAPES = {"\\": "\\", "/": "/", "n": "\n", "t": "\t", "r": "", '"': '"'}
_JOB_TYPE = {
    "full time": "fulltime", "part time": "parttime", "contract": "contract", "temporary": "contract",
    "freelance": "contract", "internship": "intern",
}  # fmt: skip
_TIMEOUT_S = 30.0


def host(entry: CompanyEntry) -> str:
    board = ((entry.extra or {}).get("slug") or "").strip().lower()
    if not _HOST.match(board):
        raise FetchError(f"zohorecruit: board {board!r} is not a host")
    return board


def openings(page: str) -> list[dict] | None:
    """The ``#jobs`` array a careers page embeds, or ``None`` when the page has none."""
    for tag in _INPUT.findall(page):
        if _JOBS_ID.search(tag) and (m := _VALUE.search(tag)):
            raw = html.unescape(m.group(1) if m.group(1) is not None else m.group(2))
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as err:
                raise FetchError(f"zohorecruit: the opening list is not JSON: {err}") from err
            return [o for o in data if isinstance(o, dict)] if isinstance(data, list) else None
    return None


def js_unescape(value: str) -> str:
    """Decode the page's JS string escaping, one level per pass until nothing changes: a rich-text value is escaped
    twice (a closing tag arrives as ``<\\\\\\/p>``), so one pass would leave a stray backslash. ``\\xNN`` and
    ``\\uNNNN`` decode to their character, an unknown ``\\X`` to ``X`` (folding Zoho's ``\\-`` back to ``-``)."""
    for _ in range(8):  # the observed nesting is two levels: slack, not a bound that matters
        out: list[str] = []
        i = 0
        while i < len(value):
            c = value[i]
            if c != "\\" or i + 1 >= len(value):
                out.append(c)
                i += 1
                continue
            e = value[i + 1]
            width = {"x": 2, "u": 4}.get(e, 0)
            digits = value[i + 2 : i + 2 + width]
            if width and len(digits) == width and all(d in "0123456789abcdefABCDEF" for d in digits):
                out.append(chr(int(digits, 16)))
                i += 2 + width
            else:
                out.append(_SIMPLE_ESCAPES.get(e, e))
                i += 2
        decoded = "".join(out)
        if decoded == value:
            break
        value = decoded
    return value


def _opened(value: str | None) -> str | None:
    """A ``Date_Opened`` as an ISO date; a future one is no publication date (freehire's rule)."""
    day = iso_date(value)
    return day if day and day <= date.today().isoformat() else None


class ZohoRecruitProvider:
    name = "zohorecruit"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        try:
            return f"https://{host(entry)}/jobs/Careers"
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        site = host(entry)

        def get(url: str) -> str:
            if urlsplit(url).scheme != "https" or urlsplit(url).hostname != site:
                raise FetchError(f"zohorecruit: refusing {url!r} off the board's site {site}", url=url)
            return ctx.http.get_text(url, timeout=_TIMEOUT_S)

        def robots_txt(url: str) -> str | None:
            try:
                return get(url)
            except FetchError:
                return None

        config = getattr(ctx.http, "config", None)
        robots = Robots(robots_txt, getattr(config, "user_agent", HttpConfig().user_agent))
        listing = f"https://{site}/jobs/Careers"
        if not robots.allowed(listing):
            raise FetchError(f"zohorecruit: {site} robots.txt disallows its careers page", url=listing)
        page = get(listing)
        found = openings(page)
        if found is None:
            if _NO_TENANT.search(page):
                raise HTTPError(f"zohorecruit: no board {site}", status=404, url=listing)
            raise FetchError(f"zohorecruit: {listing} has no opening list", url=listing)
        published = [o for o in found if o.get("Publish") is True and clean(o.get("id"))]

        def detail(o: dict) -> Job | None:
            url = f"{listing}/{o['id']}"
            return to_job(entry.name, site, o, get(url)) if robots.allowed(url) else None

        return hydrate(
            published,
            key=lambda o: f"zohorecruit:{site}:{o['id']}",
            seen=ctx.seen,
            listing=lambda o: to_job(entry.name, site, o, None),
            detail=detail,
            budget=DETAIL_BUDGET,
            label=f"zohorecruit {site}",
        )


def to_job(company: str, site: str, opening: dict, page: str | None) -> Job | None:
    """A posting from its list entry and, when read, its page. ``None`` when the entry has no title, or the page
    carries no text (skipped, read again next run). The list alone (a stored posting re-sighted) gives no text."""
    title, oid = clean(opening.get("Posting_Title")), clean(opening.get("id"))
    if not title or not oid:
        return None
    description = posted = None
    if page is not None:
        m = _DESCRIPTION.search(page)
        description = html_to_text(js_unescape(m.group(1))) if m else ""
        if not description:
            return None
        opened = _OPENED.search(page)
        posted = _opened(js_unescape(opened.group(1))) if opened else None
    location = ", ".join(p for p in (clean(opening.get("City")), clean(opening.get("Country"))) if p) or None
    remote = opening.get("Remote_Job") is True
    return Job(
        job_id=f"zohorecruit:{site}:{oid}",
        url=f"https://{site}/jobs/Careers/{oid}",
        title=title,
        company=company,
        source="zohorecruit",
        description=description,
        location=location,
        date_posted=posted,
        job_type=_JOB_TYPE.get((clean(opening.get("Job_Type")) or "").lower()),
        work_mode="remote" if remote else None,
        is_remote=infer_remote(location or "", workplace_type="remote" if remote else None),
    )
