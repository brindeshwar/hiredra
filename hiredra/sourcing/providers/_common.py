"""Helpers shared by the ATS providers.

Kept here so the third-and-later provider doesn't re-implement URL guarding,
timestamp normalization, or the remote heuristic. Providers stay tiny.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar
from urllib.parse import quote, urlsplit
from xml.etree import ElementTree

from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job

log = logging.getLogger("hiredra.sourcing.providers")
BOM = "\ufeff"
T = TypeVar("T")


def assert_allowed_url(url: str, allowed_hosts: frozenset[str], provider: str) -> None:
    """HTTPS + host in ``allowed_hosts``, else raise ``FetchError`` (SSRF guard)."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise FetchError(f"{provider}: refusing non-HTTPS url {url!r}", url=url)
    if parts.hostname not in allowed_hosts:
        raise FetchError(
            f"{provider}: refusing untrusted host {parts.hostname!r} (allowed: {', '.join(sorted(allowed_hosts))})",
            url=url,
        )


def clean(value: Any) -> str | None:
    """``value.strip()``, or ``None`` if it isn't a non-empty string."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def iso_date(value: Any) -> str | None:
    """Normalize a source timestamp to ``YYYY-MM-DD``.

    Accepts epoch-milliseconds (Lever ``createdAt``), an ISO-8601 string
    (Ashby ``publishedAt``, Greenhouse ``first_published``), or a
    ``date``/``datetime``/``pandas.Timestamp`` (JobSpy ``date_posted``).
    ``None`` for anything absent or unparseable.
    """
    if value is None or value == "" or isinstance(value, bool):
        return None
    if str(value) in ("NaT", "nan", "NaN", "None"):  # pandas null sentinels
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        raw = value.strip()
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            try:
                return datetime.strptime(raw[:10], "%Y-%m-%d").date().isoformat()
            except ValueError:
                return None
    return None


_REMOTE_RE = re.compile(r"\bremote\b|\banywhere\b|\bwork from home\b|\bwfh\b", re.I)
_WORKPLACE_REMOTE = frozenset({"remote"})
_WORKPLACE_NOT_REMOTE = frozenset({"hybrid", "on-site", "onsite", "on site", "in-office", "in office"})


def infer_remote(text: str, *, workplace_type: str | None = None) -> bool | None:
    """``True`` / ``False`` only on a clear signal; ``None`` on absence.

    A structured ``workplace_type`` (Lever/Ashby) is authoritative when it is a
    recognized value -- it can yield ``True`` or ``False``, because an explicit
    "hybrid" / "on-site" is a *present* signal, not an absence. Otherwise a
    keyword match on ``text`` (location + title) yields only ``True``; a missing
    keyword stays ``None``, never ``False``.
    """
    if workplace_type:
        wt = workplace_type.strip().lower()
        if wt in _WORKPLACE_REMOTE:
            return True
        if wt in _WORKPLACE_NOT_REMOTE:
            return False
    return True if _REMOTE_RE.search(text) else None


#: Detail requests per board per run for a hydrating provider (D3.4): a stored posting needs none, and a huge new
#: board drains over runs instead of eating one (Workday's ABB, 2,035 postings, took 29 min one at a time).
DETAIL_BUDGET = 200


def hydrate(
    items: Iterable[T],
    *,
    key: Callable[[T], str],
    seen: Callable[[str], bool],
    listing: Callable[[T], Job | None],
    detail: Callable[[T], Job | None],
    budget: int = DETAIL_BUDGET,
    label: str = "",
) -> list[Job]:
    """The listing-then-detail crawl (D3.4). A posting already stored (``seen(key(item))``) is re-sighted from the
    listing with no request; a new one costs ``detail``, up to ``budget`` per run, the rest waiting for the next.
    A detail that fails is skipped: still new, it is tried again next run."""
    jobs: list[Job] = []
    waiting = 0
    for item in items:
        if seen(key(item)):
            job = listing(item)
        elif budget <= 0:
            waiting += 1
            continue
        else:
            budget -= 1
            try:
                job = detail(item)
            except FetchError as err:
                log.info("%s: detail %s: %s", label, key(item), err)
                continue
        if job is not None:
            jobs.append(job)
    if waiting:
        log.info("%s: %d new postings wait for the next run", label, waiting)
    return jobs


#: The robots.txt rule and URL spelling compared: every character a path or query may carry literally stays, the
#: rest is percent-encoded (UTF-8), so ``/café`` and ``/caf%C3%A9`` are one path.
_URL_SAFE = "/?=&;:@!$'()*+,-._~[]%"
_PERCENT = re.compile(r"%[0-9a-fA-F]{2}")
#: The product token of a user-agent: "Hiredra/0.1 (+https://...)" is ``hiredra``.
_TOKEN = re.compile(r"[A-Za-z_-]+")

Rule = tuple[str, bool]  # (path pattern, allow)


def _encode(value: str) -> str:
    return _PERCENT.sub(lambda m: m.group(0).upper(), quote(value, safe=_URL_SAFE))


def _token(agent: str) -> str:
    m = _TOKEN.match(agent.strip())
    return m.group(0).lower() if m else ""


def robots_rules(text: str, agent: str) -> list[Rule]:
    """The rules robots.txt sets for ``agent`` (RFC 9309): every group naming its product token, case-insensitively,
    combined; else every ``*`` group combined; else none. Consecutive ``User-agent`` lines share one group, a blank
    line ends nothing, and lines other than ``User-agent``/``Allow``/``Disallow`` are ignored. An empty rule path
    sets nothing (an empty ``Disallow:`` allows everything)."""
    token = _token(agent)
    groups: list[tuple[list[str], list[Rule]]] = []
    in_agents = False
    for raw in text.splitlines():
        key, sep, value = raw.split("#", 1)[0].partition(":")
        key, value = key.strip().lower(), value.strip()
        if not sep:
            continue
        if key == "user-agent":
            if not in_agents:
                groups.append(([], []))
                in_agents = True
            groups[-1][0].append(value.lower())
        elif key in ("allow", "disallow"):
            in_agents = False
            if groups and value:
                groups[-1][1].append((_encode(value), key == "allow"))
    for wanted in (lambda a: bool(token) and _token(a) == token, lambda a: a == "*"):
        chosen = [group for agents, group in groups if any(wanted(a) for a in agents)]
        if chosen:  # a group of ours with no rules allows everything: no fall back to ``*``
            return [rule for group in chosen for rule in group]
    return []


def _matches(pattern: str, target: str) -> bool:
    """``*`` matches any run of characters and a final ``$`` ends the match; anything else is a prefix match."""
    anchored = pattern.endswith("$")
    body = pattern[:-1] if anchored else pattern
    regex = ".*".join(re.escape(part) for part in body.split("*"))
    return re.match(regex + (r"\Z" if anchored else ""), target, re.S) is not None


def robots_allows(rules: list[Rule], url: str) -> bool:
    """RFC 9309 matching on the URL's path AND query: the longest matching pattern decides, ``Allow`` winning a tie;
    no match allows. ``/robots.txt`` is always allowed."""
    parts = urlsplit(url)
    target = _encode((parts.path or "/") + (f"?{parts.query}" if parts.query else ""))
    if target == "/robots.txt":
        return True
    best: Rule | None = None
    for pattern, allow in rules:
        if _matches(pattern, target) and (
            best is None or len(pattern) > len(best[0]) or (len(pattern) == len(best[0]) and allow)
        ):
            best = (pattern, allow)
    return best is None or best[1]


class Robots:
    """robots.txt per host, read once (decision D4.2: respected for every HTML page we read). A host whose
    robots.txt is missing or unreadable allows everything, as the robots convention says. Matched by RFC 9309
    (``robots_rules``, ``robots_allows``), not ``urllib.robotparser``: that reads ``Disallow: /?`` as ``Disallow: /``,
    compares the path without the query and knows no ``*`` or ``$``, so it both blocks pages a site allows and
    lets through pages a site forbids (found on Google's careers site, 2026-11-02)."""

    def __init__(self, fetch_text: Callable[[str], str | None], agent: str) -> None:
        self._fetch = fetch_text
        self._agent = agent
        self._cache: dict[str, list[Rule]] = {}

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        rules = self._cache.get(origin)
        if rules is None:
            rules = robots_rules(self._fetch(origin + "/robots.txt") or "", self._agent)
            self._cache[origin] = rules
        return robots_allows(rules, url)


def sitemap_locs(xml: str) -> tuple[list[tuple[str, str | None]], list[str]]:
    """A sitemap's ``(pages, sub-sitemap locs)``, each page a ``(loc, lastmod)``: a ``<urlset>`` has the first, a
    ``<sitemapindex>`` the second (freehire's ``sitemapDoc``). Anything else is an error, never an empty board."""
    try:
        root = ElementTree.fromstring(xml.strip().removeprefix(BOM))
    except ElementTree.ParseError as err:
        raise FetchError(f"sitemap is not XML: {err}") from err
    kind = _local(root.tag).lower()
    if kind not in ("urlset", "sitemapindex"):
        raise FetchError(f"sitemap root is <{kind}>")
    pages: list[tuple[str, str | None]] = []
    subs: list[str] = []
    for child in root:
        fields = {_local(c.tag): (c.text or "").strip() for c in child}
        if not fields.get("loc"):
            continue
        if _local(child.tag) == "url":
            pages.append((fields["loc"], fields.get("lastmod") or None))
        elif _local(child.tag) == "sitemap":
            subs.append(fields["loc"])
    return pages, subs


def _local(tag: str) -> str:
    """An XML tag without its namespace."""
    return tag.rsplit("}", 1)[-1]


def rss_items(feed: str, provider: str) -> list[ElementTree.Element]:
    """An RSS feed's ``<item>`` elements; a body that is not RSS is an error, never an empty board."""
    try:
        root = ElementTree.fromstring(feed.strip().removeprefix(BOM))
    except ElementTree.ParseError as err:
        raise FetchError(f"{provider}: the feed is not XML: {err}") from err
    if root.tag != "rss":
        raise FetchError(f"{provider}: the feed's root is <{root.tag}>")
    return root.findall("./channel/item")


def rss_date(value: str | None) -> str | None:
    """An RFC 822 ``pubDate`` as ``YYYY-MM-DD``."""
    try:
        return parsedate_to_datetime(value).date().isoformat() if value else None
    except (TypeError, ValueError):
        return None
