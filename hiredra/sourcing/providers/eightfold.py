"""Eightfold provider -- the career site's public JSON API, no key (build step 3, wave 1).

Ported from freehire's ``internal/ingest/sources/eightfold.go`` (MIT, see THIRD_PARTY_NOTICES.md). The board is
``<host>/<domain>`` (``insight.eightfold.ai/insight.com``): the host serves the API, the domain is the tenant key it
requires. A board given as the host alone has its domain read off the ``/careers`` page, which names it (measured
2026-11-01: PayPal's and NAB's links carry none). The list serves 10 positions a page without text; a new position's
text is its detail (a hydrating source, D3.4). robots.txt allows both APIs.

A tenant answers exactly one of two list generations, ``/api/pcsx/search`` or the legacy ``/api/apply/v2/jobs``, and
the other with a 403 ("PCSX is not enabled", "Not authorized for PCSX"). Our client reads a 403 as a block and
cools the key down, so each generation's list calls carry their own cooldown key: a refused generation parks only
itself, and the other is tried. A 403 on a detail is Eightfold's rate limit (freehire measured ~290 requests a
window) and cools the host, leaving the rest of the board's new positions for the next run.

Eightfold limits an IP across all its tenants, so, as freehire does, every request of this provider waits its turn
on ONE pace, whatever the tenant: the belt's (60% of the measured ~1 a second), else about one a second. The walk is
read in chunks of 50 pages, planned from the count the first page states, so a board as big as Starbucks' (21,754
positions) is read to its end, where a walk capped at 2,000 pages failed it (crawl 2, 2026-11-05). Its throttle can also answer 405 (crawl 1, 2026-11-05: 81
tenants read 2 at once, unpaced; 11 minutes in, every request after answered 405, and 64 boards failed): a 405 parks
every tenant at once, as a block, so the rest of the run's Eightfold boards wait instead of being refused one by one.

Two changes from freehire: every listed location is kept, joined, where freehire keeps the first; and a detail
that fails or carries no text is skipped and read next run (D3.4), where freehire stores the posting without text.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from dataclasses import replace

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import Blocked, FetchError, HTTPError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import (
    DETAIL_BUDGET,
    Listed,
    Pace,
    clean,
    infer_remote,
    iso_date,
    log,
    page_ranges,
    read_board,
    work_mode,
)

_HOST = re.compile(r"^[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+$")
#: The tenant domain as the careers page states it (``"domain": "paypal.com"``).
_PAGE_DOMAIN = re.compile(r"domain[^a-z0-9]{1,12}([a-z0-9][a-z0-9.-]*\.[a-z]{2,})")
_PAGE = 10  # the server's page, whatever ``num`` asks for
_MAX_POSITIONS = 100_000  # a stated count past this is no board's: a runaway guard, not a page cap
_TIMEOUT_S = 30.0
THROTTLE_KEY = "eightfold.ai"  # the one cooldown a 405 sets, for every tenant


def parse_board(board: str) -> tuple[str, str | None]:
    host, _, domain = (board or "").strip().lower().partition("/")
    if not _HOST.match(host) or (domain and not _HOST.match(domain)):
        raise FetchError(f'eightfold: board {board!r} must be "<host>" or "<host>/<domain>"')
    return host, domain or None


def _workplace(value: str | None) -> str | None:
    text = (value or "").lower()
    return "remote" if "remote" in text else "hybrid" if "hybrid" in text else "onsite" if "onsite" in text else None


class EightfoldProvider:
    name = "eightfold"

    def __init__(self, interval: float = 1.0) -> None:
        self.pace = Pace(interval)  # outside the belt: one for every Eightfold tenant (freehire's ~1 request a second)

    def _get(self, ctx: FetchContext, url: str, *, text: bool = False, cooldown_key: str | None = None):
        """One request on the shared pace (the belt's, else this provider's); a 405 (Eightfold's throttle) parks
        every tenant, raised as a block."""
        cooldowns = getattr(ctx.http, "cooldowns", None)
        if cooldowns is not None:
            cooldowns.check(THROTTLE_KEY, url=url)
        (ctx.pace or self.pace.wait)()
        try:
            if text:
                return ctx.http.get_text(url, timeout=_TIMEOUT_S)
            return ctx.http.get_json(url, timeout=_TIMEOUT_S, cooldown_key=cooldown_key)
        except HTTPError as err:
            if err.status != 405:
                raise
            seconds = getattr(getattr(ctx.http, "config", None), "cooldown_seconds", 900.0)
            until = cooldowns.trip(THROTTLE_KEY, seconds, reason="HTTP 405") if cooldowns is not None else 0.0
            log.warning("eightfold: %s answered 405 (throttled) -- every tenant cools down %.0fs", url, seconds)
            raise Blocked(f"{url}: HTTP 405 (Eightfold's throttle)", cooldown_key=THROTTLE_KEY, until=until,
                          url=url) from err  # fmt: skip

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        try:
            return f"https://{parse_board((entry.extra or {}).get('slug') or '')[0]}/careers"
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        host, _ = parse_board((entry.extra or {}).get("slug") or "")
        return read_board(self, entry, ctx, budget=DETAIL_BUDGET, label=f"eightfold {host}")

    def list_chunk(self, entry: CompanyEntry, ctx: FetchContext, chunk: dict | None) -> Listed:
        """``None``: the tenant's domain (read off its careers page when the board names none) and the first page of
        whichever list generation it answers, which states the count and plans the rest as ranges of the walk
        (``{"host", "domain", "gen", "path", "sort", "start", "end"}``). A failure past the first page fails the
        board: the walk must reach the count the server states, or an empty page."""
        if chunk is not None and chunk.get("first"):  # the date walk's start: it plans its own rest, if served at all
            got = self._walk(ctx, chunk)
            rest = page_ranges(chunk["end"], got.info["count"], _PAGE) if got.served and got.info else []
            return replace(got, more=tuple({**chunk, "start": a, "end": b, "first": False} for a, b in rest))
        if chunk is not None:
            return self._walk(ctx, chunk)
        host, domain = parse_board((entry.extra or {}).get("slug") or "")
        if domain is None:
            # unescaped first: most pages write it HTML-escaped, ``&#34;domain&#34;: &#34;worley.com&#34;`` (parity
            # step, 2026-11-03: Worley, Dolby, Arcadis named no domain until then)
            page = html.unescape(self._get(ctx, f"https://{host}/careers", text=True))
            found = Counter(_PAGE_DOMAIN.findall(page))
            if not found:
                raise FetchError(f"eightfold: {host}/careers names no tenant domain")
            domain = found.most_common(1)[0][0]
        for gen, path in (("pcsx", "api/pcsx/search"), ("v2", "api/apply/v2/jobs")):
            walk = {"host": host, "domain": domain, "gen": gen, "path": path, "sort": "relevance", "start": 0,
                    "end": _PAGE}  # fmt: skip
            first = self._walk(ctx, walk, refused_ok=True)
            if first is None:
                continue  # this tenant answers the other generation
            count = first.info["count"]
            if count > _MAX_POSITIONS:
                raise FetchError(f"eightfold: {host} states {count} positions, past the {_MAX_POSITIONS} ceiling")
            more = tuple({**walk, "start": a, "end": b} for a, b in page_ranges(first.served, count, _PAGE))
            info = {"host": host, "domain": domain, "gen": gen, "path": path, "count": count}
            return replace(first, more=more if first.served else (), info=info)
        raise FetchError(f"eightfold: {host} answers neither list generation for {domain}")

    def listed(self, entry: CompanyEntry, ctx: FetchContext, info: dict, items: dict) -> tuple:
        """The order can shift between pages, serving some positions twice and others never (measured 2026-11-01:
        PayPal's 317 by relevance gave 276 distinct). A walk that comes up short is followed by one by date, and the
        two are joined (314 of 317); what is still missing is logged and read by a later run."""
        count = info.get("count", 0)
        if len(items) >= count:
            return ()
        if not info.get("by_date"):
            info["by_date"] = True
            walk = {k: info[k] for k in ("host", "domain", "gen", "path")} | {"sort": "timestamp"}
            return ({**walk, "start": 0, "end": page_ranges(0, count, _PAGE)[0][1], "first": True},)
        log.info("eightfold %s: %d of %d positions served", info["host"], len(items), count)
        ctx.incomplete(f"{len(items)} of {count} positions served")
        return ()

    def job(self, entry: CompanyEntry, ctx: FetchContext, info: dict, item: dict, detail: bool) -> Job | None:
        host, domain = info["host"], info["domain"]
        if not detail:
            return to_job(entry.name, host, domain, item, None)
        d = self._get(ctx, f"https://{host}/api/apply/v2/jobs/{item['id']}?domain={domain}")
        return to_job(entry.name, host, domain, item, d if isinstance(d, dict) else None)

    def _walk(self, ctx: FetchContext, chunk: dict, *, refused_ok: bool = False) -> Listed | None:
        """One range of a walk, keyed by job id, with the count the server states in ``info``. When the walk's very
        first page is refused: ``None`` if that may be (``refused_ok``: a 403 says the tenant answers the other
        generation), nothing at all for the date walk's first chunk (the date order isn't served); a throttle is the
        platform's, and raises."""
        host, domain, gen, path, sort = (chunk[k] for k in ("host", "domain", "gen", "path", "sort"))
        positions: dict[str, dict] = {}
        start, served, count = chunk["start"], 0, 0
        while start < chunk["end"]:
            url = f"https://{host}/{path}?domain={domain}&query=&start={start}&num={_PAGE}&sort_by={sort}"
            try:
                data = self._get(ctx, url, cooldown_key=f"{host} {gen}")
            except (Blocked, HTTPError) as err:
                refused = err.cooldown_key != THROTTLE_KEY if isinstance(err, Blocked) else err.status == 403
                if start == 0 and refused and (refused_ok or chunk.get("first")):
                    return None if refused_ok else Listed()
                raise
            body = (data.get("data") if gen == "pcsx" else data) if isinstance(data, dict) else None
            page = body.get("positions") if isinstance(body, dict) else None
            if not isinstance(page, list):
                raise FetchError(f"eightfold: {url} has no positions list", url=url)
            for p in page:
                if isinstance(p, dict) and isinstance(p.get("id"), int):
                    positions.setdefault(f"eightfold:{domain}:{p['id']}", p)
            start += len(page)
            served += len(page)
            count = body.get("count") or 0
            if not page or start >= count:
                break
        return Listed(positions, served=served, info={"count": count})


def to_job(company: str, host: str, domain: str, position: dict, detail: dict | None) -> Job | None:
    """A position from its list entry and, when read, its detail. ``None`` without a title, or when the detail has
    no text (skipped, read again next run)."""
    title, pid = clean(position.get("name")), position.get("id")
    if not title or not isinstance(pid, int):
        return None
    description = None
    if detail is not None:
        description = html_to_text(detail.get("job_description"))
        if not description:
            return None
    places = [position.get("location"), *(position.get("locations") or [])]
    location = "; ".join(dict.fromkeys(p for p in (clean(x) for x in places) if p)) or None
    posted = position.get("postedTs") or position.get("t_create")
    workplace = _workplace(position.get("workLocationOption") or position.get("work_location_option"))
    url = clean((detail or {}).get("canonicalPositionUrl")) or clean(position.get("canonicalPositionUrl"))
    return Job(
        job_id=f"eightfold:{domain}:{pid}",
        url=url or f"https://{host}/careers/job/{pid}",
        title=title,
        company=company,
        source="eightfold",
        description=description,
        location=location,
        date_posted=iso_date(posted * 1000) if isinstance(posted, int) and posted > 0 else None,
        work_mode=work_mode(workplace),
        is_remote=infer_remote(location or "", workplace_type=workplace),
    )
