"""Keka provider -- the careers portal's own job API, no key (build step 3, wave 1).

Ported from freehire's ``internal/ingest/sources/keka.go`` (MIT, see THIRD_PARTY_NOTICES.md). Keka is an Indian HR
suite; the board is the company's subdomain, ``<board>.keka.com/careers``. One request to the API the portal's own
page calls lists every open posting with its full text: no paging, no detail request.

Two changes from freehire: the company is the board's (our identity), so the portal's display-name lookup is not
asked; and a posting's every location is kept, joined, where freehire keeps the first (measured 2026-11-01:
10 of Anblicks' 90 postings name two). A board that does not exist redirects to a "tenant not found" page, read as a
404 so the board is retired.
"""

from __future__ import annotations

import re

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import clean, infer_remote, iso_date

_BOARD = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_TIMEOUT_S = 30.0
#: ``jobType`` as the portal's own script renders it: 2 is "Full Time", 1 "Part time"; anything else stays unknown.
_JOB_TYPE = {2: "fulltime", 1: "parttime"}


def _board(entry: CompanyEntry) -> str:
    board = ((entry.extra or {}).get("slug") or "").strip().lower()
    if not _BOARD.match(board):
        raise FetchError(f"keka: board {board!r} is not a subdomain")
    return board


class KekaProvider:
    name = "keka"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        try:
            return f"https://{_board(entry)}.keka.com/careers"
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        board = _board(entry)
        url = f"https://{board}.keka.com/careers/api/jobs/default/active"
        try:
            postings = ctx.http.get_json(url, timeout=_TIMEOUT_S)
        except HTTPError as err:
            if err.location and "TenantNotFound" in err.location:
                raise HTTPError(f"keka: no board {board}", status=404, url=url) from err
            raise
        if not isinstance(postings, list):  # the whole board in one array; anything else is an error, not empty
            raise FetchError(f"keka: {url} is not a list of postings", url=url)
        jobs = (to_job(entry.name, board, p) for p in postings if isinstance(p, dict))
        return [j for j in jobs if j is not None]


def to_job(company: str, board: str, posting: dict) -> Job | None:
    title, pid = clean(posting.get("title")), posting.get("id")
    if not title or not isinstance(pid, int) or isinstance(pid, bool):
        return None
    places = posting.get("jobLocations") if isinstance(posting.get("jobLocations"), list) else []
    named = (", ".join(p for p in (clean(pl.get(k)) for k in ("city", "state", "countryName")) if p) for pl in places
             if isinstance(pl, dict))  # fmt: skip
    location = "; ".join(dict.fromkeys(n for n in named if n)) or None
    return Job(
        job_id=f"keka:{board}:{pid}",
        url=f"https://{board}.keka.com/careers/jobdetails/{pid}",
        title=title,
        company=company,
        source="keka",
        description=html_to_text(posting.get("description")) or None,
        location=location,
        date_posted=iso_date(posting.get("publishedOn")),
        job_type=_JOB_TYPE.get(posting.get("jobType")),
        is_remote=infer_remote(f"{location or ''} {title}"),
    )
