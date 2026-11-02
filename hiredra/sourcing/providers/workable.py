"""Workable provider -- the public job-board widget API, no key (build step 3, wave 1).

Ported from freehire's ``internal/ingest/sources/workable.go`` (MIT, see THIRD_PARTY_NOTICES.md). The board is the
account, ``apply.workable.com/<account>``. One request lists the whole board, each posting's text inline
(``details=true``), so there is no detail request and no paging.
"""

from __future__ import annotations

import re

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import assert_allowed_url, clean, infer_remote, iso_date, work_mode

API = "https://apply.workable.com/api/v1/widget/accounts"
_ALLOWED_HOSTS = frozenset({"apply.workable.com"})
_BOARD = re.compile(r"^[A-Za-z0-9_-]+$")
_TIMEOUT_S = 30.0
#: The widget's ``type`` as our job types; "Other" and absent stay unknown.
_JOB_TYPE = {
    "Full-time": "fulltime", "Part-time": "parttime", "Contract": "contract", "Temporary": "contract",
    "Internship": "intern",
}  # fmt: skip


def _board(entry: CompanyEntry) -> str:
    board = ((entry.extra or {}).get("slug") or "").strip()
    if not _BOARD.match(board):
        raise FetchError(f"workable: board {board!r} is not an account")
    return board


class WorkableProvider:
    name = "workable"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        try:
            return f"https://apply.workable.com/{_board(entry)}/"
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        url = f"{API}/{_board(entry)}?details=true"
        assert_allowed_url(url, _ALLOWED_HOSTS, self.name)
        data = ctx.http.get_json(url, timeout=_TIMEOUT_S)
        postings = data.get("jobs") if isinstance(data, dict) else None
        if not isinstance(postings, list):  # an emptied board answers "jobs": []; no array at all is an error
            raise FetchError(f"workable: {url} has no jobs array", url=url)
        by_code: dict[str, list[dict]] = {}
        for p in postings:
            if isinstance(p, dict) and (code := clean(p.get("shortcode"))):
                by_code.setdefault(code, []).append(p)
        jobs = (to_job(entry.name, entries) for entries in by_code.values())
        return [j for j in jobs if j is not None]


def to_job(company: str, entries: list[dict]) -> Job | None:
    """One posting from its list entries: the widget lists a job once per location under one shortcode, so the
    locations are joined, where freehire keeps the first entry's alone (measured 2026-11-01: Intellectsoft's 15
    entries are 5 jobs, one "Bengaluru" + "Pune")."""
    posting = entries[0]
    title, code = clean(posting.get("title")), clean(posting.get("shortcode"))
    if not title or not code:
        return None
    places = (", ".join(p for p in (clean(e.get(k)) for k in ("city", "state", "country")) if p) for e in entries)
    location = "; ".join(dict.fromkeys(p for p in places if p)) or None
    workplace = "remote" if any(e.get("telecommuting") is True for e in entries) else None
    return Job(
        job_id=f"workable:{code}",
        url=clean(posting.get("url")) or f"https://apply.workable.com/j/{code}",
        title=title,
        company=company,
        source="workable",
        description=html_to_text(posting.get("description")) or None,
        location=location,
        date_posted=iso_date(posting.get("published_on")),
        job_type=_JOB_TYPE.get(clean(posting.get("type")) or ""),
        work_mode=work_mode(workplace),
        is_remote=infer_remote(f"{location or ''} {title}", workplace_type=workplace),
    )
