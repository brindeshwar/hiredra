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
on ONE pace of about one a second, whatever the tenant. Its throttle can also answer 405 (crawl 1, 2026-11-05: 81
tenants read 2 at once, unpaced; 11 minutes in, every request after answered 405, and 64 boards failed): a 405 parks
every tenant at once, as a block, so the rest of the run's Eightfold boards wait instead of being refused one by one.

Two changes from freehire: every listed location is kept, joined, where freehire keeps the first; and a detail
that fails or carries no text is skipped and read next run (D3.4), where freehire stores the posting without text.
"""

from __future__ import annotations

import html
import re
from collections import Counter

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import Blocked, FetchError, HTTPError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import (
    DETAIL_BUDGET,
    Pace,
    clean,
    hydrate,
    infer_remote,
    iso_date,
    log,
    work_mode,
)

_HOST = re.compile(r"^[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+$")
#: The tenant domain as the careers page states it (``"domain": "paypal.com"``).
_PAGE_DOMAIN = re.compile(r"domain[^a-z0-9]{1,12}([a-z0-9][a-z0-9.-]*\.[a-z]{2,})")
_PAGE = 10  # the server's page, whatever ``num`` asks for
_MAX_PAGES = 2000
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
        self.pace = Pace(interval)  # one for every Eightfold tenant in a run (freehire's ~1 request a second)

    def _get(self, ctx: FetchContext, url: str, *, text: bool = False, cooldown_key: str | None = None):
        """One request on the shared pace; a 405 (Eightfold's throttle) parks every tenant, raised as a block."""
        cooldowns = getattr(ctx.http, "cooldowns", None)
        if cooldowns is not None:
            cooldowns.check(THROTTLE_KEY, url=url)
        self.pace.wait()
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
        host, domain = parse_board((entry.extra or {}).get("slug") or "")
        if domain is None:
            # unescaped first: most pages write it HTML-escaped, ``&#34;domain&#34;: &#34;worley.com&#34;`` (parity
            # step, 2026-11-03: Worley, Dolby, Arcadis named no domain until then)
            page = html.unescape(self._get(ctx, f"https://{host}/careers", text=True))
            found = Counter(_PAGE_DOMAIN.findall(page))
            if not found:
                raise FetchError(f"eightfold: {host}/careers names no tenant domain")
            domain = found.most_common(1)[0][0]
        positions = self._list(ctx, host, domain)

        def detail(p: dict) -> Job | None:
            url = f"https://{host}/api/apply/v2/jobs/{p['id']}?domain={domain}"
            d = self._get(ctx, url)
            return to_job(entry.name, host, domain, p, d if isinstance(d, dict) else None)

        return hydrate(
            positions,
            key=lambda p: f"eightfold:{domain}:{p['id']}",
            seen=ctx.seen,
            listing=lambda p: to_job(entry.name, host, domain, p, None),
            detail=detail,
            budget=DETAIL_BUDGET,
            label=f"eightfold {host}",
        )

    def _list(self, ctx: FetchContext, host: str, domain: str) -> list[dict]:
        """Every position, from whichever list generation the tenant answers. A failure past the first page fails
        the board: the walk must reach the count the server states, or an empty page.

        The order can shift between pages, serving some positions twice and others never (measured 2026-11-01:
        PayPal's 317 by relevance gave 276 distinct). A walk that comes up short is followed by one by date, and the
        two are joined (314 of 317); what is still missing is logged and read by a later run."""
        for generation, path in (("pcsx", "api/pcsx/search"), ("v2", "api/apply/v2/jobs")):
            walked = self._walk(ctx, host, domain, generation, path, "relevance")
            if walked is None:
                continue  # this tenant answers the other generation
            positions, count = walked
            if len(positions) < count:
                more, _ = self._walk(ctx, host, domain, generation, path, "timestamp") or ({}, 0)
                positions = {**more, **positions}
                if len(positions) < count:
                    log.info("eightfold %s: %d of %d positions served", host, len(positions), count)
                    ctx.incomplete(f"{len(positions)} of {count} positions served")
            return list(positions.values())
        raise FetchError(f"eightfold: {host} answers neither list generation for {domain}")

    def _walk(
        self, ctx: FetchContext, host: str, domain: str, generation: str, path: str, sort: str
    ) -> tuple[dict[int, dict], int] | None:
        """One walk of a list generation in one order: ``(positions by id, the count the server states)``, or
        ``None`` when the first page is refused (a 403: the tenant answers the other generation)."""
        positions: dict[int, dict] = {}
        start = 0
        for _ in range(_MAX_PAGES):
            url = f"https://{host}/{path}?domain={domain}&query=&start={start}&num={_PAGE}&sort_by={sort}"
            try:
                data = self._get(ctx, url, cooldown_key=f"{host} {generation}")
            except (Blocked, HTTPError) as err:
                refused = err.cooldown_key != THROTTLE_KEY if isinstance(err, Blocked) else err.status == 403
                if start == 0 and refused:  # the tenant answers the other generation; a throttle is the platform's
                    return None
                raise
            body = (data.get("data") if generation == "pcsx" else data) if isinstance(data, dict) else None
            page = body.get("positions") if isinstance(body, dict) else None
            if not isinstance(page, list):
                raise FetchError(f"eightfold: {url} has no positions list", url=url)
            for p in page:
                if isinstance(p, dict) and isinstance(p.get("id"), int):
                    positions.setdefault(p["id"], p)
            start += len(page)
            count = body.get("count") or 0
            if not page or start >= count:
                return positions, count
        raise FetchError(f"eightfold: {host} has not ended within {_MAX_PAGES} pages")


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
