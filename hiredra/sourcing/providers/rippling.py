"""Rippling provider -- the public ATS board API, no key (build step 3, wave 1).

Ported from freehire's ``internal/ingest/sources/rippling.go`` (MIT, see THIRD_PARTY_NOTICES.md). The board is the
slug of ``ats.rippling.com/<board>``. One request lists the board without text; a new posting's text is its detail
(a hydrating source, D3.4). The detail's ``description`` holds the company's boilerplate and the role apart; only
the role is kept, as in freehire.

One change from freehire: a new posting's location is every one its detail names, joined, where freehire keeps the
list's single label.
"""

from __future__ import annotations

import re

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import DETAIL_BUDGET, clean, hydrate, infer_remote, iso_date

API = "https://ats.rippling.com/api/v1/board"
_BOARD = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_TIMEOUT_S = 30.0


def _board(entry: CompanyEntry) -> str:
    board = ((entry.extra or {}).get("slug") or "").strip().lower()
    if not _BOARD.match(board):
        raise FetchError(f"rippling: board {board!r} is not a slug")
    return board


def _job_type(label: str | None) -> str | None:
    """The detail's ``employmentType`` label ("Salaried, full-time"), only where it plainly states the arrangement."""
    text = (label or "").lower()
    for word, kind in (("intern", "intern"), ("part-time", "parttime"), ("contract", "contract"),
                       ("temporary", "contract"), ("full-time", "fulltime")):  # fmt: skip
        if word in text:
            return kind
    return None


class RipplingProvider:
    name = "rippling"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        try:
            return f"https://ats.rippling.com/{_board(entry)}/jobs"
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        board = _board(entry)
        listing = ctx.http.get_json(f"{API}/{board}/jobs", timeout=_TIMEOUT_S)
        if not isinstance(listing, list):  # the whole board in one array; anything else is an error, not empty
            raise FetchError(f"rippling: {API}/{board}/jobs is not a list of postings")
        by_uuid: dict[str, dict] = {}
        for p in listing:
            if isinstance(p, dict) and clean(p.get("uuid")):
                by_uuid.setdefault(p["uuid"], p)  # the first entry of a posting listed once per location
        postings = list(by_uuid.values())

        def detail(p: dict) -> Job | None:
            d = ctx.http.get_json(f"{API}/{board}/jobs/{p['uuid']}", timeout=_TIMEOUT_S)
            return to_job(entry.name, board, p, d if isinstance(d, dict) else None)

        return hydrate(
            postings,
            key=lambda p: f"rippling:{p['uuid']}",
            seen=ctx.seen,
            listing=lambda p: to_job(entry.name, board, p, None),
            detail=detail,
            budget=DETAIL_BUDGET,
            label=f"rippling {board}",
        )


def to_job(company: str, board: str, posting: dict, detail: dict | None) -> Job | None:
    """A posting from its list entry and, when read, its detail. ``None`` without a title, or when the detail has no
    role text (skipped, read again next run)."""
    title, uuid = clean(posting.get("name")), clean(posting.get("uuid"))
    if not title or not uuid:
        return None
    work = posting.get("workLocation")
    location = clean(work.get("label")) if isinstance(work, dict) else None
    description = posted = job_type = None
    if detail is not None:
        text = detail.get("description")
        description = html_to_text(text.get("role")) if isinstance(text, dict) else ""
        if not description:
            return None
        places = [clean(p) for p in detail.get("workLocations") or [] if isinstance(p, str)]
        location = "; ".join(dict.fromkeys(p for p in places if p)) or location
        posted = iso_date(detail.get("createdOn"))
        kind = detail.get("employmentType")
        job_type = _job_type(kind.get("id") if isinstance(kind, dict) else None)
    return Job(
        job_id=f"rippling:{uuid}",
        url=clean(posting.get("url")) or f"https://ats.rippling.com/{board}/jobs/{uuid}",
        title=title,
        company=company,
        source="rippling",
        description=description,
        location=location,
        date_posted=posted,
        job_type=job_type,
        is_remote=infer_remote(location or ""),
    )
