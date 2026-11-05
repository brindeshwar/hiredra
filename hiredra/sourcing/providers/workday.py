"""Workday provider -- the public "CXS" careers API, no key (build step 3, wave 1).

Ported from freehire's ``internal/ingest/sources/workday.go`` (MIT, see THIRD_PARTY_NOTICES.md). The board id is
the public host and site, ``acme.wd1.myworkdayjobs.com/Careers``, or on the shared host
``wd1.myworkdaysite.com/<tenant>/<site>``. The listing is POST-only, 20 per page, with no description, so each
posting's detail is one more GET (a hydrating source, D3.4).

A board past 2,000 postings reports ``total: 2000`` and loops back to page one beyond it, so it is listed slice by
slice through its own facets (job family, location, ...), the dimension whose largest value is smallest first,
up to three dimensions deep; a board that still can't be listed completely fails rather than return part of
itself as the whole. Requests go one at a time (D4.2 allows two per host; freehire uses 8).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError, HTTPError
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

log = logging.getLogger("hiredra.sourcing.workday")

PAGE = 20
CAP = 2000  # the total a capped board reports
MAX_FACET_DEPTH = 3
_HOST = re.compile(r"^(?:[a-z0-9-]+\.wd\d+\.myworkdayjobs\.com|wd\d+\.myworkdaysite\.com)$")
_TIMEOUT_S = 30.0


@dataclass(frozen=True, slots=True)
class Board:
    host: str
    tenant: str
    site: str
    public_path: str

    @property
    def api(self) -> str:
        return f"https://{self.host}/wday/cxs/{self.tenant}/{self.site}"


def parse_board(board: str) -> Board:
    """``host/site`` (the tenant is the host's first label) or ``host/tenant/site`` (shared host)."""
    host, _, rest = board.strip().strip("/").partition("/")
    host = host.lower()
    if not _HOST.match(host) or not rest:
        raise FetchError(f"workday: board {board!r} must be 'tenant.wdN.myworkdayjobs.com/site'")
    tenant, sep, site = rest.partition("/")
    if sep:
        if not tenant or not site:
            raise FetchError(f"workday: board {board!r} must be 'host/tenant/site'")
        return Board(host, tenant, site, f"recruiting/{tenant}/{site}")
    return Board(host, host.split(".", 1)[0], rest, rest)


def _splittable(facets: Iterable[dict]) -> list[tuple[str, list[tuple[str, int]]]]:
    """Facet dimensions a query can be scoped by. A wrapper whose values are facets (``locationMainGroup``) is
    replaced by them; a value without an id is dropped."""
    out: list[tuple[str, list[tuple[str, int]]]] = []

    def walk(param: str, values: Iterable[dict]) -> None:
        usable = []
        for v in values or ():
            if not isinstance(v, dict):
                continue
            if v.get("facetParameter"):
                walk(v["facetParameter"], v.get("values") or ())
            elif v.get("id"):
                usable.append((v["id"], int(v.get("count") or 0)))
        if usable:
            out.append((param, usable))

    for f in facets or ():
        if isinstance(f, dict):
            walk(f.get("facetParameter") or "", f.get("values") or ())
    return out


def pick_split(facets: Iterable[dict], used: set[str]) -> tuple[str, list[tuple[str, int]]] | None:
    """The unused dimension whose largest value is smallest, among those whose values add up to a capped slice
    (Workday's ``distance`` facet sums to 5 on a 4,390-posting board: splitting on it would lose most of it)."""
    best, best_max = None, -1
    for param, values in _splittable(facets):
        if param in used:
            continue
        counts = [c for _, c in values]
        if sum(counts) < CAP:
            continue
        if best is None or max(counts) < best_max:
            best, best_max = (param, values), max(counts)
    return best


class WorkdayProvider:
    name = "workday"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        slug = entry.extra.get("slug") if entry.extra else None
        try:
            return parse_board(slug).api if slug else None
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        board = parse_board((entry.extra or {}).get("slug") or "")
        allowed = frozenset({board.host})

        def post(body: dict) -> dict:
            url = f"{board.api}/jobs"
            assert_allowed_url(url, allowed, self.name)
            resp = ctx.http.request("POST", url, json=body, timeout=_TIMEOUT_S)
            data = resp.json()
            return data if isinstance(data, dict) else {}

        def get(path: str) -> dict:
            url = f"{board.api}{path}"
            assert_allowed_url(url, allowed, self.name)
            data = ctx.http.get_json(url, timeout=_TIMEOUT_S)
            return data if isinstance(data, dict) else {}

        return hydrate(
            list_postings(post),
            key=lambda p: job_id(board, p),
            seen=ctx.seen,
            listing=lambda p: to_job(entry.name, board, p, {}),
            detail=lambda p: to_job(entry.name, board, p, get(p["externalPath"])),
            budget=DETAIL_BUDGET,
            label=f"workday {board.tenant}/{board.site}",
        )


def _page(post: Callable[[dict], dict], facets: dict, offset: int) -> dict:
    return post({"appliedFacets": facets, "limit": PAGE, "offset": offset, "searchText": ""})


def _page_all(post: Callable[[dict], dict], facets: dict, first: dict) -> list[dict]:
    """Page a query to its end from an already fetched first page. Some boards report the real total only on the
    first page and 0 after, so the first non-zero total is kept."""
    out: list[dict] = []
    total, page = -1, first
    while postings := [p for p in page.get("jobPostings") or () if isinstance(p, dict) and p.get("externalPath")]:
        out += postings
        if total < 0 and page.get("total"):
            total = int(page["total"])
        if 0 <= total <= len(out):
            break
        page = _page(post, facets, len(out))
    return out


def list_postings(post: Callable[[dict], dict]) -> list[dict]:
    """Every posting on the board. A first page answered 422 is a tenant this host no longer serves -- a made-up
    tenant gets the same 422, while a missing site of a real tenant gets a 404 -- so the board is gone, as for a
    404 (crawl 1, 2026-11-05: 30 boards, among them Comcast, moved from wd5 to wd115, and Dell, moved to Oracle)."""
    try:
        first = _page(post, {}, 0)
    except HTTPError as err:
        if err.status != 422:
            raise
        raise HTTPError(f"{err} -- the tenant is not on this host (moved)", status=404, url=err.url) from err
    if first.get("total") != CAP:
        return _page_all(post, {}, first)
    return _split(post, {}, first, set(), 0)


def _split(post: Callable[[dict], dict], facets: dict, capped: dict, used: set[str], depth: int) -> list[dict]:
    dim = pick_split(capped.get("facets") or (), used)
    if dim is None or depth >= MAX_FACET_DEPTH:
        raise FetchError(f"workday: board capped at {CAP} and no facet left to split it completely")
    param, values = dim
    by_path: dict[str, dict] = {}
    for value_id, _ in values:
        scoped = {**facets, param: [value_id]}
        page = _page(post, scoped, 0)
        part = (
            _split(post, scoped, page, used | {param}, depth + 1)
            if page.get("total") == CAP
            else _page_all(post, scoped, page)
        )
        by_path.update((p["externalPath"], p) for p in part)
    return list(by_path.values())


def _employment_type(time_type: Any) -> str | None:
    return {"full time": "fulltime", "part time": "parttime"}.get((clean(time_type) or "").lower())


def job_id(board: Board, posting: dict) -> str:
    return f"workday:{board.tenant}:{posting['externalPath']}"


def to_job(company: str, board: Board, posting: dict, detail: dict) -> Job | None:
    """A posting as a Job: from its detail, else (a stored posting re-sighted) from the listing alone."""
    info = detail.get("jobPostingInfo") if isinstance(detail.get("jobPostingInfo"), dict) else {}
    title = clean(info.get("title")) or clean(posting.get("title"))
    if not title:
        return None
    path = posting["externalPath"]
    location = clean(info.get("location")) or clean(posting.get("locationsText"))
    remote_type = (clean(info.get("remoteType")) or "").lower()
    return Job(
        job_id=job_id(board, posting),
        url=clean(info.get("externalUrl")) or f"https://{board.host}/{board.public_path}{path}",
        title=title,
        company=company,
        source="workday",
        description=html_to_text(info.get("jobDescription")) or None,
        location=location,
        date_posted=iso_date(info.get("startDate")),
        job_type=_employment_type(info.get("timeType")),
        work_mode=work_mode(remote_type),
        is_remote=True if "remote" in remote_type else infer_remote(location or ""),
    )
