"""Phenom provider -- an employer's Phenom career site, through the site's own ``/widgets`` API (parity step, 2026-11-04).

Ported from freehire's ``internal/ingest/sources/phenom.go`` (MIT, see THIRD_PARTY_NOTICES.md). Phenom People hosts
career sites under each employer's own name (``careers.dhl.com``, ``careers.bcg.com``); the board is that host. The
site's own pages POST to ``/widgets``: ``refineSearch`` lists 100 jobs a page with the count of all
(``totalHits``), without text; ``jobDetail`` gives one job's text (a hydrating source, D3.4). robots.txt is obeyed
(the sites disallow ``*/apply``, ``*/px-widgets`` and the like, not ``/widgets``).

Most Phenom sites only front another ATS: of 155,399 jobs on 89 live sites (measured 2026-11-04), 70% apply through
Workday boards we already crawl first-hand and 16% through boards our adapters read. ``boards discover-phenom`` reads
each site's apply links and imports those boards instead; only a site whose jobs reach us no other way (DHL's Avature,
BCG's own system, SuccessFactors EU) is crawled here.

All Phenom sites share Phenom's infrastructure, and freehire measured 403s on 45 of ~95 of them under its crawl's
combined volume, while the same request alone succeeded minutes later. So, as freehire does, every request of this
provider waits its turn on ONE pace, whatever the site: the belt's (60% of the measured ~1 a second), else about
one a second.

Two changes from freehire: the place is every location the job lists, each with its country (``multi_location``:
"Gurgaon, India, 122002"), where freehire keeps the first ``cityState`` ("Gurgaon, Haryāna", no country); and the
employer's ATS link is kept as the apply link.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError, HttpConfig
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import (
    DETAIL_BUDGET,
    Listed,
    Pace,
    Robots,
    clean,
    infer_remote,
    iso_date,
    page_ranges,
    read_board,
    work_mode,
)
from hiredra.sourcing.providers.remoteboards import job_type

_HOST = re.compile(r"^[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+$")
PAGE = 100  # the site serves the size asked for, up to this
_MAX_PAGES = 500  # 50,000 jobs: a ceiling far past the largest site measured (TJX, 10,012)
_ZIP = re.compile(r"^[A-Z0-9]{0,4}[ -]?\d{3,}[A-Z0-9 -]*$")  # a trailing postal code ("122002", "SW1A 1AA" not)
_TIMEOUT_S = 30.0


def site(entry: CompanyEntry) -> str:
    host = ((entry.extra or {}).get("slug") or "").strip().lower()
    if not _HOST.match(host):
        raise FetchError(f"phenom: board {host!r} is not a host")
    return host


def post(ctx: FetchContext, host: str, body: dict, pace: Pace) -> dict:
    """One ``/widgets`` call; anything but a JSON object is an error."""
    pace.wait()
    resp = ctx.http.request("POST", f"https://{host}/widgets", json=body, timeout=_TIMEOUT_S)
    try:
        data = resp.json()
    except ValueError as err:
        raise FetchError(f"phenom: {host}/widgets did not answer JSON", url=f"https://{host}/widgets") from err
    if not isinstance(data, dict):
        raise FetchError(f"phenom: {host}/widgets answered no object", url=f"https://{host}/widgets")
    return data


def listing_page(ctx: FetchContext, host: str, start: int, pace: Pace) -> tuple[list[dict], int | None]:
    """One listing page's jobs and the site's stated count of all."""
    found = post(ctx, host, {"deviceType": "desktop", "ddoKey": "refineSearch", "jobs": True, "from": start,
                             "size": PAGE}, pace).get("refineSearch")  # fmt: skip
    if not isinstance(found, dict):
        raise FetchError(f"phenom: {host} answered no job search", url=f"https://{host}/widgets")
    jobs = (found.get("data") or {}).get("jobs") if isinstance(found.get("data"), dict) else None
    total = found.get("totalHits")
    return [j for j in jobs or [] if isinstance(j, dict) and clean(j.get("jobSeqNo"))], (
        total if isinstance(total, int) and not isinstance(total, bool) else None
    )


class PhenomProvider:
    name = "phenom"
    boards_at_once = 2  # one pace for all of Phenom: two boards keep it busy; more would only wait on it

    def __init__(self, interval: float = 1.0) -> None:
        self.pace = Pace(interval)  # outside the belt: one for every Phenom site (freehire's measured ~1 a second)

    def _pace(self, ctx: FetchContext):
        """The belt's pace for Phenom when it sets one, else this provider's own."""
        return SimpleNamespace(wait=ctx.pace) if ctx.pace is not None else self.pace

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        try:
            return f"https://{site(entry)}/"
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return False  # a vanity host names no platform: a board is imported by ``boards discover-phenom``

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        return read_board(self, entry, ctx, budget=DETAIL_BUDGET, label=f"phenom {site(entry)}")

    def list_chunk(self, entry: CompanyEntry, ctx: FetchContext, chunk: dict | None) -> Listed:
        """``None``: robots.txt, then the first page, which states the count and plans the rest as page ranges
        (``{"start", "end"}``); a site that states none is read until a short page, up to the ceiling."""
        host = site(entry)
        if chunk is None:

            def robots_txt(url: str) -> str | None:
                try:
                    return ctx.http.get_text(url, timeout=_TIMEOUT_S)
                except FetchError:
                    return None

            config = getattr(ctx.http, "config", None)
            if not Robots(robots_txt, getattr(config, "user_agent", HttpConfig().user_agent)).allowed(
                f"https://{host}/widgets"
            ):
                raise FetchError(f"phenom: {host} robots.txt disallows /widgets", url=f"https://{host}/widgets")
            jobs, total = listing_page(ctx, host, 0, self._pace(ctx))
            end = min(total, PAGE * _MAX_PAGES) if total is not None else PAGE * _MAX_PAGES
            more = tuple({"start": a, "end": b} for a, b in page_ranges(PAGE, end, PAGE)) if len(jobs) == PAGE else ()
            return Listed(_rows(host, jobs), more=more, served=len(jobs), info={"total": total})
        rows: list[dict] = []
        for start in range(chunk["start"], chunk["end"], PAGE):
            jobs, _ = listing_page(ctx, host, start, self._pace(ctx))
            rows += jobs
            if len(jobs) < PAGE:
                break
        return Listed(_rows(host, rows), served=len(rows))

    def listed(self, entry: CompanyEntry, ctx: FetchContext, info: dict, items: dict) -> tuple:
        host, total = site(entry), info.get("total")
        if not items and total:
            raise FetchError(f"phenom: {host} states {total} jobs and listed none", url=f"https://{host}/widgets")
        if total is not None and len(items) < total:
            ctx.incomplete(f"phenom {host}: listed {len(items)} of the {total} jobs it states")
        return ()

    def job(self, entry: CompanyEntry, ctx: FetchContext, info: dict, item: dict, detail: bool) -> Job | None:
        host = site(entry)
        if not detail:
            return to_job(entry.name, host, item, None)
        found = post(ctx, host, {"deviceType": "desktop", "ddoKey": "jobDetail", "pageName": "job-details",
                                 "jobSeqNo": clean(item["jobSeqNo"])}, self._pace(ctx)).get("jobDetail")  # fmt: skip
        data = ((found or {}).get("data") or {}).get("job") if isinstance(found, dict) else None
        return to_job(entry.name, host, item, data if isinstance(data, dict) else {})


def _rows(host: str, jobs: list[dict]) -> dict[str, dict]:
    """Listing rows by job id, the first sighting kept."""
    rows: dict[str, dict] = {}
    for j in jobs:
        rows.setdefault(f"phenom:{host}:{clean(j['jobSeqNo'])}", j)
    return rows


def job_url(host: str, row: dict) -> str:
    """The site's page for a job: ``/<country>/<lang>/job/<seq>`` from its ``en_US`` locale (freehire's
    ``phenomJobURL``), else ``/job/<seq>``."""
    seq = clean(row.get("jobSeqNo"))
    lang, _, country = (clean(row.get("locale")) or "").partition("_")
    return (
        f"https://{host}/{country.lower()}/{lang.lower()}/job/{seq}"
        if lang and country
        else f"https://{host}/job/{seq}"
    )


def location(row: dict) -> str | None:
    """Every place the job lists, each with its country, a trailing postal code dropped ("Gurgaon, India, 122002" ->
    "Gurgaon, India"); else the one place it names."""
    places = []
    for raw in row.get("multi_location") if isinstance(row.get("multi_location"), list) else []:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        while parts and _ZIP.match(parts[-1]) and len(parts) > 1:
            parts.pop()
        if parts:
            places.append(", ".join(parts))
    if places:
        return "; ".join(dict.fromkeys(places))
    return clean(row.get("cityStateCountry")) or ", ".join(
        p for p in (clean(row.get("cityState")), clean(row.get("country"))) if p
    ) or None  # fmt: skip


def to_job(company: str, host: str, row: dict, detail: dict | None) -> Job | None:
    """A job from its listing row and, when bought, its detail; ``None`` when a detail carries no text (skipped, read
    again next run) or neither names a title."""
    text = html_to_text((detail or {}).get("description")) if detail is not None else None
    if detail is not None and not text:
        return None
    title = clean(row.get("title")) or clean((detail or {}).get("title"))
    if not title:
        return None
    place = location(row)
    mode = work_mode(clean(row.get("workplaceType")))
    return Job(
        job_id=f"phenom:{host}:{clean(row.get('jobSeqNo'))}",
        url=job_url(host, row),
        title=title,
        company=company,
        source="phenom",
        description=text,
        location=place,
        date_posted=iso_date(row.get("postedDate")),
        job_type=job_type(row.get("type")),
        apply_url=clean(row.get("applyUrl")),
        is_remote=infer_remote(f"{place or ''} {title}", workplace_type=mode),
        work_mode=mode,
    )
