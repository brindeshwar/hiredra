"""Oracle Recruiting Cloud provider -- public "CandidateExperience" REST JSON, no key (build step 3, wave 1).

Ported from freehire's ``internal/ingest/sources/oracle.go`` (MIT, see THIRD_PARTY_NOTICES.md). The board id is the
public host and site number, ``fa-evmr-saasfaprod1.fa.ocs.oraclecloud.com/CX_1``. The requisition list has no
description, so each new requisition's detail is one more GET (a hydrating source, D3.4).
"""

from __future__ import annotations

from collections.abc import Callable

import re
from dataclasses import dataclass, replace
from html import unescape

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import (
    DETAIL_BUDGET,
    Listed,
    assert_allowed_url,
    clean,
    infer_remote,
    iso_date,
    page_ranges,
    read_board,
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
        return read_board(self, entry, ctx, budget=DETAIL_BUDGET, label=f"oracle {board.host}/{board.site}")

    def list_chunk(self, entry: CompanyEntry, ctx: FetchContext, chunk: dict | None) -> Listed:
        board = parse_board((entry.extra or {}).get("slug") or "")
        got = read_chunk(lambda url: _get(ctx, board, url), board, chunk)
        return replace(got, items={f"oracle:{board.host}:{r['Id']}": r for r in got.items.values()})

    def listed(self, entry: CompanyEntry, ctx: FetchContext, info: dict, items: dict) -> tuple:
        """A list that stopped serving before its total (a huge site) saw part of the board."""
        served, total = info.get("served", 0), info.get("total", 0)
        if served and served < total:
            ctx.incomplete(f"{served} of {total} requisitions served")
        return ()

    def job(self, entry: CompanyEntry, ctx: FetchContext, info: dict, item: dict, detail: bool) -> Job | None:
        board = parse_board((entry.extra or {}).get("slug") or "")
        if not detail:
            return to_job(entry.name, board, item, {})
        items = _get(ctx, board, detail_url(board, item["Id"])).get("items") or []
        return to_job(entry.name, board, item, items[0] if items and isinstance(items[0], dict) else None)


def _get(ctx: FetchContext, board: Board, url: str) -> dict:
    assert_allowed_url(url, frozenset({board.host}), "oracle")
    if ctx.pace is not None:
        ctx.pace()
    data = ctx.http.get_json(url, timeout=_TIMEOUT_S)
    return data if isinstance(data, dict) else {}


def _requisitions(page_body: dict) -> tuple[list[dict], int]:
    """A list page's requisitions and the total it states. The list nests under ``items[0]``; the ``expand`` keeps
    ``requisitionList`` filled."""
    items = page_body.get("items") or []
    page = items[0] if items and isinstance(items[0], dict) else {}
    reqs = [r for r in page.get("requisitionList") or () if isinstance(r, dict) and r.get("Id")]
    return reqs, int(page.get("TotalJobsCount") or 0)


def read_chunk(get: Callable[[str], dict], board: Board, chunk: dict | None) -> Listed:
    """One chunk of the requisition list, keyed by id: ``None`` is the first page, which states the total and plans the
    rest as page ranges (``{"offset", "end"}``); a range pages until its end or an empty page."""
    if chunk is None:
        reqs, total = _requisitions(get(list_url(board, 0)))
        more = tuple({"offset": a, "end": b} for a, b in page_ranges(len(reqs), total, PAGE)) if reqs else ()
        return Listed({r["Id"]: r for r in reqs}, more=more, served=len(reqs), info={"total": total})
    rows: list[dict] = []
    offset = chunk["offset"]
    while offset < chunk["end"]:
        reqs, _ = _requisitions(get(list_url(board, offset)))
        if not reqs:
            break
        rows += reqs
        offset += len(reqs)
    return Listed({r["Id"]: r for r in rows}, served=len(rows))


def list_requisitions(get, board: Board, short: Callable[[str], None] = lambda reason: None) -> list[dict]:  # noqa: ANN001
    """Every requisition, the chunks read one after another; a list that stops serving before its total (a huge site)
    is reported to ``short``: the crawl saw part of the board."""
    first = read_chunk(get, board, None)
    out, served = dict(first.items), first.served
    for chunk in first.more:
        got = read_chunk(get, board, chunk)
        served += got.served
        for key, req in got.items.items():
            out.setdefault(key, req)
    total = (first.info or {}).get("total", 0)
    if served and served < total:
        short(f"{served} of {total} requisitions served")
    return list(out.values())


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
