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
from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import assert_allowed_url, clean, infer_remote, iso_date

log = logging.getLogger("hiredra.sourcing.workday")

PAGE = 20
CAP = 2000  # the total a capped board reports
MAX_FACET_DEPTH = 3
#: Detail requests per board per run (D3.4): a stored posting needs none, and a huge new board drains over runs
#: instead of eating one (ABB's 2,035 postings took 29 min one at a time).
DETAIL_BUDGET = 200
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

        jobs, budget, waiting = [], DETAIL_BUDGET, 0
        for p in list_postings(post):
            if ctx.seen(job_id(board, p)):
                jobs.append(to_job(entry.name, board, p, {}))  # stored: re-sighted from the listing, no request
                continue
            if budget <= 0:
                waiting += 1  # new, over this run's budget: fetched next run
                continue
            budget -= 1
            try:
                detail = get(p["externalPath"])
            except FetchError as err:
                log.info("workday: %s detail %s: %s", board.tenant, p["externalPath"], err)
                continue
            if (job := to_job(entry.name, board, p, detail)) is not None:
                jobs.append(job)
        if waiting:
            log.info("workday: %s/%s: %d new postings wait for the next run", board.tenant, board.site, waiting)
        return [j for j in jobs if j is not None]


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
    first = _page(post, {}, 0)
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
        is_remote=True if "remote" in remote_type else infer_remote(location or ""),
    )
