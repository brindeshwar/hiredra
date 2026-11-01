"""SmartRecruiters provider -- the public postings API, no key (build step 3, wave 1).

Ported from freehire's ``internal/ingest/sources/smartrecruiters.go`` (MIT, see THIRD_PARTY_NOTICES.md). The board
id is the company identifier, ``api.smartrecruiters.com/v1/companies/<board>/postings``. The list has no
description, so each new posting's detail is one more GET (a hydrating source, D3.4). Every board shares this one
host, which meters callers: requests go one at a time.
"""

from __future__ import annotations

import re

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import (
    DETAIL_BUDGET,
    assert_allowed_url,
    clean,
    hydrate,
    infer_remote,
    iso_date,
)

API = "https://api.smartrecruiters.com/v1/companies"
PAGE = 100
_ALLOWED_HOSTS = frozenset({"api.smartrecruiters.com"})
_BOARD = re.compile(r"^[A-Za-z0-9_.-]+$")
_TIMEOUT_S = 30.0
#: ``typeOfEmployment`` ids as our job types.
_JOB_TYPE = {
    "permanent": "fulltime", "full-time": "fulltime", "part-time": "parttime", "contract": "contract",
    "temporary": "contract", "internship": "intern", "traineeship": "intern",
}  # fmt: skip


def _board(entry: CompanyEntry) -> str:
    board = ((entry.extra or {}).get("slug") or "").strip()
    if not _BOARD.match(board):
        raise FetchError(f"smartrecruiters: board {board!r} is not a company identifier")
    return board


def description(detail: dict) -> str | None:
    """The role sections; the company section only when they are all empty (some tenants put the whole ad
    there)."""
    sections = ((detail.get("jobAd") or {}).get("sections") or {}) if isinstance(detail.get("jobAd"), dict) else {}

    def text(name: str) -> str:
        section = sections.get(name)
        return (section.get("text") or "") if isinstance(section, dict) else ""

    body = text("jobDescription") + text("qualifications") + text("additionalInformation")
    return html_to_text(body if body.strip() else text("companyDescription")) or None


class SmartRecruitersProvider:
    name = "smartrecruiters"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        try:
            return f"{API}/{_board(entry)}/postings"
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        board = _board(entry)

        def get(url: str) -> dict:
            assert_allowed_url(url, _ALLOWED_HOSTS, self.name)
            data = ctx.http.get_json(url, timeout=_TIMEOUT_S)
            return data if isinstance(data, dict) else {}

        postings: list[dict] = []
        while True:
            page = get(f"{API}/{board}/postings?limit={PAGE}&offset={len(postings)}")
            content = [p for p in page.get("content") or () if isinstance(p, dict) and p.get("id")]
            postings += content
            if not content or len(postings) >= int(page.get("totalFound") or 0):
                break

        return hydrate(
            postings,
            key=lambda p: f"smartrecruiters:{p['id']}",
            seen=ctx.seen,
            listing=lambda p: to_job(entry.name, p, {}),
            detail=lambda p: to_job(entry.name, p, get(f"{API}/{board}/postings/{p['id']}")),
            budget=DETAIL_BUDGET,
            label=f"smartrecruiters {board}",
        )


def to_job(company: str, posting: dict, detail: dict) -> Job | None:
    """A posting as a Job; ``detail`` ``{}`` for a stored one re-sighted from the list. A detail without a
    ``postingUrl`` is an empty answer (an interstitial, ``null``), not a posting: skipped, tried again next run."""
    title = clean(posting.get("name"))
    if not title or (detail and not clean(detail.get("postingUrl"))):
        return None
    loc = posting.get("location") if isinstance(posting.get("location"), dict) else {}
    location = ", ".join(p for p in (clean(loc.get(k)) for k in ("city", "region", "country")) if p) or None
    workplace = "remote" if loc.get("remote") else "hybrid" if loc.get("hybrid") else None
    employment = detail.get("typeOfEmployment") if isinstance(detail.get("typeOfEmployment"), dict) else {}
    return Job(
        job_id=f"smartrecruiters:{posting['id']}",
        url=clean(detail.get("postingUrl")) or f"https://jobs.smartrecruiters.com/{posting['id']}",
        title=title,
        company=company,
        source="smartrecruiters",
        description=description(detail) if detail else None,
        location=location,
        date_posted=iso_date(posting.get("releasedDate")),
        job_type=_JOB_TYPE.get(employment.get("id") or ""),
        is_remote=infer_remote(location or "", workplace_type=workplace),
    )
