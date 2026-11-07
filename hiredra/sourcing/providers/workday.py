"""Workday provider -- the public "CXS" careers API, no key (build step 3, wave 1).

Ported from freehire's ``internal/ingest/sources/workday.go`` (MIT, see THIRD_PARTY_NOTICES.md). The board id is
the public host and site, ``acme.wd1.myworkdayjobs.com/Careers``, or on the shared host
``wd1.myworkdaysite.com/<tenant>/<site>``. The listing is POST-only, 20 per page, with no description, so each
posting's detail is one more GET (a hydrating source, D3.4).

A board past 2,000 postings reports ``total: 2000`` and loops back to page one beyond it, so it is listed slice by
slice through its own facets (job family, location, ...), the dimension whose largest value is smallest first,
up to three dimensions deep; a board that still can't be listed completely fails rather than return part of
itself as the whole. Read on the belt, a board is chunks of up to 50 pages, a slice's probe one chunk each; outside it,
the chunks go one after another (D4.2 allows two requests per host; freehire uses 8).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import Blocked, FetchError, HTTPError
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

log = logging.getLogger("hiredra.sourcing.workday")

PAGE = 20
CAP = 2000  # the total a capped board reports
MAX_FACET_DEPTH = 3
_HOST = re.compile(r"^(?:[a-z0-9-]+\.wd\d+\.myworkdayjobs\.com|wd\d+\.myworkdaysite\.com)$")
_TIMEOUT_S = 30.0
MAINTENANCE_PAUSE_S = 1800.0  # a tenant in maintenance is asked again after this long


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
        return read_board(self, entry, ctx, budget=DETAIL_BUDGET, label=f"workday {board.tenant}/{board.site}")

    def list_chunk(self, entry: CompanyEntry, ctx: FetchContext, chunk: dict | None) -> Listed:
        board = parse_board((entry.extra or {}).get("slug") or "")
        got = read_chunk(lambda body: _request(ctx, board, "POST", "/jobs", body), chunk)
        return replace(got, items={job_id(board, p): p for p in got.items.values()})

    def listed(self, entry: CompanyEntry, ctx: FetchContext, info: dict, items: dict) -> tuple:
        return ()  # a board that can't be listed completely already failed in its chunk

    def job(self, entry: CompanyEntry, ctx: FetchContext, info: dict, item: dict, detail: bool) -> Job | None:
        board = parse_board((entry.extra or {}).get("slug") or "")
        return to_job(entry.name, board, item, _request(ctx, board, "GET", item["externalPath"]) if detail else {})


def _request(ctx: FetchContext, board: Board, method: str, path: str, body: dict | None = None) -> dict:
    """One call to the board's API, on the platform's pace when the belt sets one."""
    url = f"{board.api}{path}"
    assert_allowed_url(url, frozenset({board.host}), "workday")
    if ctx.pace is not None:
        ctx.pace()
    try:
        if method == "POST":
            data = ctx.http.request("POST", url, json=body, timeout=_TIMEOUT_S).json()
        else:
            data = ctx.http.get_json(url, timeout=_TIMEOUT_S)
    except HTTPError as err:
        if not (300 <= err.status < 400 and "maintenance" in (err.location or "").lower()):
            raise
        # Workday's maintenance (weekly: Friday 23:00 to Saturday 03:00 Pacific, at most 4 h; a tenant's own now and
        # then) redirects every call to community.workday.com/maintenance-page. It passes: the tenant pauses and its
        # task goes back on the belt, never a board that "can't be done" (live check, Saturday 2026-11-07).
        cooldowns = getattr(ctx.http, "cooldowns", None)
        until = cooldowns.trip(board.host, MAINTENANCE_PAUSE_S, reason="Workday maintenance") if cooldowns else 0.0
        raise Blocked(f"{url}: Workday maintenance (redirected to {err.location})", cooldown_key=board.host,
                      until=until, url=url) from err  # fmt: skip
    return data if isinstance(data, dict) else {}


def _page(post: Callable[[dict], dict], facets: dict, offset: int) -> dict:
    return post({"appliedFacets": facets, "limit": PAGE, "offset": offset, "searchText": ""})


def _postings(page: dict) -> list[dict]:
    return [p for p in page.get("jobPostings") or () if isinstance(p, dict) and p.get("externalPath")]


def read_chunk(post: Callable[[dict], dict], chunk: dict | None) -> Listed:
    """One chunk of a board's listing, its postings keyed by path. ``None`` is the first page; it says how the rest is
    read: by page ranges (``{"facets", "offset", "end"}``, ``end`` ``None`` when the board states no total), or, for a
    board capped at 2,000, by facet slices, each read first by a probe of its first page (``{"facets", "used",
    "depth"}``). A first page answered 422 is a tenant this host no longer serves -- a made-up tenant gets the same 422,
    while a missing site of a real tenant gets a 404 -- so the board is gone, as for a 404 (crawl 1, 2026-11-05: 30
    boards, among them Comcast, moved from wd5 to wd115, and Dell, moved to Oracle)."""
    if chunk is None:
        try:
            first = _page(post, {}, 0)
        except HTTPError as err:
            if err.status != 422:
                raise
            raise HTTPError(f"{err} -- the tenant is not on this host (moved)", status=404, url=err.url) from err
        return _probed(first, {}, (), 0)
    if "offset" not in chunk:
        return _probed(_page(post, chunk["facets"], 0), chunk["facets"], tuple(chunk["used"]), chunk["depth"])
    rows: list[dict] = []
    offset, end = chunk["offset"], chunk["end"]
    while end is None or offset < end:
        postings = _postings(_page(post, chunk["facets"], offset))
        if not postings:
            break
        rows += postings
        offset += len(postings)
    return Listed({p["externalPath"]: p for p in rows}, served=len(rows))


def _probed(page: dict, facets: dict, used: tuple[str, ...], depth: int) -> Listed:
    """A slice's first page: its postings and the ranges still to read, or, when the slice is capped, a probe of each
    value of the facet dimension that splits it best. Some boards report the real total only on the first page and 0
    after, so the first page's total plans the ranges."""
    if page.get("total") == CAP:
        dim = pick_split(page.get("facets") or (), set(used))
        if dim is None or depth >= MAX_FACET_DEPTH:
            raise FetchError(f"workday: board capped at {CAP} and no facet left to split it completely")
        param, values = dim
        return Listed(more=tuple({"facets": {**facets, param: [value_id]}, "used": [*used, param], "depth": depth + 1}
                                 for value_id, _ in values))  # fmt: skip
    postings = _postings(page)
    total = int(page.get("total") or 0)
    if not postings or (total and len(postings) >= total):
        more: tuple = ()
    elif total:
        more = tuple({"facets": facets, "offset": a, "end": b} for a, b in page_ranges(len(postings), total, PAGE))
    else:
        more = ({"facets": facets, "offset": len(postings), "end": None},)
    return Listed({p["externalPath"]: p for p in postings}, more=more, served=len(postings))


def list_postings(post: Callable[[dict], dict]) -> list[dict]:
    """Every posting on the board, its chunks read one after another (a slice's own chunks before the next slice);
    a posting listed in two slices counts once."""
    out: dict[str, dict] = {}
    todo: list = [None]
    while todo:
        got = read_chunk(post, todo.pop(0))
        for path, posting in got.items.items():
            out.setdefault(path, posting)
        todo[:0] = got.more
    return list(out.values())


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
