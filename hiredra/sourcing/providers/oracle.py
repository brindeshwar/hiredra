"""Oracle Recruiting Cloud provider -- public "CandidateExperience" REST JSON, no key (build step 3, wave 1).

Ported from freehire's ``internal/ingest/sources/oracle.go`` (MIT, see THIRD_PARTY_NOTICES.md). The board id is the
public host and site number, ``fa-evmr-saasfaprod1.fa.ocs.oraclecloud.com/CX_1``. The requisition list has no
description, so each new requisition's detail is one more GET (a hydrating source, D3.4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape

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
    work_mode,
)

PAGE = 100
_HOST = re.compile(r"^[a-z0-9-]+(?:\.[a-z0-9-]+)*\.oraclecloud\.com$")
_SITE = re.compile(r"^[A-Za-z0-9_]+$")
_TAG = re.compile(r"<[^>]*>")
_TIMEOUT_S = 30.0
#: Oracle's workplace-type codes, as ``infer_remote``'s workplace types.
_WORKPLACE = {"ORA_REMOTE": "remote", "ORA_HYBRID": "hybrid", "ORA_ON_SITE": "onsite"}


@dataclass(frozen=True, slots=True)
class Board:
    host: str
    site: str

    @property
    def api(self) -> str:
        return f"https://{self.host}/hcmRestApi/resources/latest"


def parse_board(board: str) -> Board:
    host, _, site = board.strip().strip("/").partition("/")
    host = host.lower()
    if not _HOST.match(host) or not _SITE.match(site):
        raise FetchError(f"oracle: board {board!r} must be 'host.oraclecloud.com/SITE'")
    return Board(host, site)


def list_url(board: Board, offset: int) -> str:
    # The offset goes INSIDE the finder clause: Oracle ignores a top-level &offset= and returns page one forever.
    return (
        f"{board.api}/recruitingCEJobRequisitions?onlyData=true"
        "&expand=requisitionList.secondaryLocations,requisitionList.workLocation"
        f"&finder=findReqs;siteNumber={board.site},limit={PAGE},offset={offset},sortBy=POSTING_DATES_DESC"
    )


def detail_url(board: Board, req_id: str) -> str:
    return (
        f"{board.api}/recruitingCEJobRequisitionDetails?onlyData=true&expand=all"
        f"&finder=ById;Id=%22{req_id}%22,siteNumber={board.site}"
    )


def _visible(html: str) -> bool:
    return bool(unescape(_TAG.sub("", html)).strip())


def description(detail: dict) -> str | None:
    """The three External* sections; when they are empty markup ("<br>"), the corporate and organisation text
    instead (seen on Goldman Sachs' campus postings)."""
    external = "".join(detail.get(k) or "" for k in (
        "ExternalDescriptionStr", "ExternalResponsibilitiesStr", "ExternalQualificationsStr"))  # fmt: skip
    body = external if _visible(external) else (
        (detail.get("CorporateDescriptionStr") or "") + (detail.get("OrganizationDescriptionStr") or ""))  # fmt: skip
    return html_to_text(body) or None


def _job_type(schedule: object) -> str | None:
    return {"full time": "fulltime", "part time": "parttime"}.get((clean(schedule) or "").lower())


class OracleProvider:
    name = "oracle"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        slug = entry.extra.get("slug") if entry.extra else None
        try:
            return list_url(parse_board(slug), 0) if slug else None
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        board = parse_board((entry.extra or {}).get("slug") or "")
        allowed = frozenset({board.host})

        def get(url: str) -> dict:
            assert_allowed_url(url, allowed, self.name)
            data = ctx.http.get_json(url, timeout=_TIMEOUT_S)
            return data if isinstance(data, dict) else {}

        def detail(req: dict) -> Job | None:
            items = get(detail_url(board, req["Id"])).get("items") or []
            return to_job(entry.name, board, req, items[0] if items and isinstance(items[0], dict) else None)

        return hydrate(
            list_requisitions(get, board),
            key=lambda req: f"oracle:{board.host}:{req['Id']}",
            seen=ctx.seen,
            listing=lambda req: to_job(entry.name, board, req, {}),
            detail=detail,
            budget=DETAIL_BUDGET,
            label=f"oracle {board.host}/{board.site}",
        )


def list_requisitions(get, board: Board) -> list[dict]:  # noqa: ANN001
    """Page the requisition list until a page is empty or ``TotalJobsCount`` is reached. The list nests under
    ``items[0]``; the ``expand`` keeps ``requisitionList`` filled."""
    out: list[dict] = []
    while True:
        items = get(list_url(board, len(out))).get("items") or []
        page = items[0] if items and isinstance(items[0], dict) else {}
        reqs = [r for r in page.get("requisitionList") or () if isinstance(r, dict) and r.get("Id")]
        if not reqs:
            return out
        out += reqs
        if len(out) >= int(page.get("TotalJobsCount") or 0):
            return out


def to_job(company: str, board: Board, req: dict, detail: dict | None) -> Job | None:
    """A requisition as a Job; ``detail`` ``{}`` for a stored one re-sighted from the list, ``None`` for a detail
    the platform answered with nothing (skipped: still new, it is tried again next run)."""
    title = clean(req.get("Title"))
    if not title or detail is None:
        return None
    location = clean(req.get("PrimaryLocation"))
    return Job(
        job_id=f"oracle:{board.host}:{req['Id']}",
        url=f"https://{board.host}/hcmUI/CandidateExperience/en/sites/{board.site}/job/{req['Id']}",
        title=title,
        company=company,
        source="oracle",
        description=description(detail) if detail else None,
        location=location,
        date_posted=iso_date(req.get("PostedDate")),
        job_type=_job_type(detail.get("JobSchedule")),
        work_mode=work_mode(_WORKPLACE.get(req.get("WorkplaceTypeCode") or "")),
        is_remote=infer_remote(location or "", workplace_type=_WORKPLACE.get(req.get("WorkplaceTypeCode") or "")),
    )
