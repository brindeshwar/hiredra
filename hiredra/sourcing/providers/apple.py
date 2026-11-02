"""Apple provider -- jobs.apple.com's search page for one location, then each new posting's detail API (build step
3, wave 2).

Ported from freehire's ``internal/ingest/sources/apple.go`` (MIT, see THIRD_PARTY_NOTICES.md). One employer, so
freehire has no board; ours is a location, Apple's own search filter (``india-INDC``), as Amazon's board is a country.
freehire walks the global search API, 20 rows a page (6,311 rows, 316 pages, measured 2026-11-02), and fetches every
posting's detail; India is 186 rows, 10 pages. The API's location filter answers no rows for any body tried, so the
listing is the search page itself, which server-renders the same rows (``loaderData.search``) for its location. These
are HTML pages, so robots.txt is obeyed (D4.2; jobs.apple.com serves none). A row is one (position, location), so a
role posted in two cities is one posting, read once; the walk must reach the stated row count, or it fails.

One change from freehire: the text keeps the ``responsibilities`` section, which freehire drops. Apple's page shows
five sections (summary, description, responsibilities, minimum and preferred qualifications), and the
responsibilities are the role's own duties, where the summary is often a team boilerplate.
"""

from __future__ import annotations

import json
import re
from urllib.parse import quote, urlsplit

from hiredra.sourcing.http import FetchError, HttpConfig
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import DETAIL_BUDGET, Robots, clean, hydrate, infer_remote, iso_date

SITE = "jobs.apple.com"
_LOCATION = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-[A-Z0-9]+$")  # the search URL's ``location``: india-INDC
_HYDRATION = re.compile(r"window\.__staticRouterHydrationData\s*=\s*JSON\.parse\((\"(?:[^\"\\]|\\.)*\")\)", re.S)
_MAX_PAGES = 100  # 2,000 rows: far above any one location (India 186); a page past it means the walk never ends
_SECTIONS = (("jobSummary", None), ("description", None), ("responsibilities", "Responsibilities"),
             ("minimumQualifications", "Minimum qualifications"),
             ("preferredQualifications", "Preferred qualifications"))  # fmt: skip
_TIMEOUT_S = 30.0


def location_filter(entry: CompanyEntry) -> str:
    board = ((entry.extra or {}).get("slug") or "").strip()
    if not _LOCATION.match(board):
        raise FetchError(f"apple: board {board!r} is not a search location (india-INDC)")
    return board


def search_url(board: str, page: int = 1) -> str:
    return f"https://{SITE}/en-us/search?location={quote(board)}" + (f"&page={page}" if page > 1 else "")


def search_page(page: str) -> tuple[list[dict], int]:
    """The ``(rows, total rows)`` a search page server-renders, or a ``FetchError`` when it carries none."""
    m = _HYDRATION.search(page)
    try:
        data = json.loads(json.loads(m.group(1))) if m else None
    except json.JSONDecodeError as err:
        raise FetchError(f"apple: the search page's data is not JSON: {err}") from err
    search = ((data or {}).get("loaderData") or {}).get("search") if isinstance(data, dict) else None
    rows = search.get("searchResults") if isinstance(search, dict) else None
    if not isinstance(rows, list) or not isinstance(search.get("totalRecords"), int):
        raise FetchError("apple: the search page carries no results")
    return [r for r in rows if isinstance(r, dict)], search["totalRecords"]


def location(row: dict) -> str | None:
    """A row's places, "Bengaluru, India" (a city) or "India" (a country-wide role), joined with "; "."""
    places = []
    for place in row.get("locations") or []:
        if not isinstance(place, dict):
            continue
        name, country = clean(place.get("name")), clean(place.get("countryName"))
        text = ", ".join(dict.fromkeys(p for p in (name, country) if p))
        if text and text not in places:
            places.append(text)
    return "; ".join(places) or None


def description(detail: dict) -> str | None:
    """The detail's sections in the page's order, each a plain-text block; a heading names the qualification and
    responsibility sections, as the page does."""
    parts = []
    for field, heading in _SECTIONS:
        text = (detail.get(field) or "").strip() if isinstance(detail.get(field), str) else ""
        if text:
            parts.append(f"{heading}\n{text}" if heading else text)
    return "\n\n".join(parts) or None


class AppleProvider:
    name = "apple"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        try:
            return search_url(location_filter(entry))
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return False  # one employer: a board is named explicitly, never recognised from a link

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        board = location_filter(entry)

        def get(url: str) -> str:
            if urlsplit(url).scheme != "https" or urlsplit(url).hostname != SITE:
                raise FetchError(f"apple: refusing {url!r} off {SITE}", url=url)
            return ctx.http.get_text(url, timeout=_TIMEOUT_S)

        def robots_txt(url: str) -> str | None:
            try:
                return get(url)
            except FetchError:
                return None

        config = getattr(ctx.http, "config", None)
        robots = Robots(robots_txt, getattr(config, "user_agent", HttpConfig().user_agent))
        rows: list[dict] = []
        for page in range(1, _MAX_PAGES + 1):
            url = search_url(board, page)
            if not robots.allowed(url):
                raise FetchError(f"apple: robots.txt disallows {url}", url=url)
            got, total = search_page(get(url))
            rows += got
            if not got or len(rows) >= total:
                break
        else:
            raise FetchError(f"apple {board}: no end after {_MAX_PAGES} pages")
        if len(rows) != total:
            raise FetchError(f"apple {board}: read {len(rows)} rows of the {total} stated")
        postings: dict[str, dict] = {}
        for row in rows:  # one row per (position, location): the first row of a position names it
            if pid := clean(row.get("positionId")):
                postings.setdefault(pid, row)

        def detail(pid: str) -> Job | None:
            api = f"https://{SITE}/api/v1/jobDetails/{quote(pid)}?locale=en-us"
            data = ctx.http.get_json(api, timeout=_TIMEOUT_S)
            res = data.get("res") if isinstance(data, dict) else None
            text = description(res) if isinstance(res, dict) else None
            return to_job(entry.name, postings[pid], text) if text else None

        return hydrate(
            postings,
            key=lambda pid: f"apple:{pid}",
            seen=ctx.seen,
            listing=lambda pid: to_job(entry.name, postings[pid], None),
            detail=detail,
            budget=DETAIL_BUDGET,
            label=f"apple {board}",
        )


def to_job(company: str, row: dict, text: str | None) -> Job | None:
    pid, title = clean(row.get("positionId")), clean(row.get("postingTitle"))
    if not pid or not title:
        return None
    slug = clean(row.get("transformedPostingTitle")) or ""
    where = location(row)
    return Job(
        job_id=f"apple:{pid}",
        url=f"https://{SITE}/en-us/details/{pid}/{slug}".rstrip("/"),
        title=title,
        company=company,
        source="apple",
        description=text,
        location=where,
        date_posted=iso_date(row.get("postDateInGMT")),
        work_mode="remote" if row.get("homeOffice") is True else None,
        is_remote=infer_remote(where or "", workplace_type="remote" if row.get("homeOffice") is True else None),
    )
