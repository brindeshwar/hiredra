"""Shared HTTP layer for the sourcing providers -- port of career-ops _http.mjs.

One client, held on the FetchContext, that every provider calls. It owns the
reliability machinery so providers stay tiny:

* retry + exponential-jitter backoff on transient failure (timeout, connection
  error, 500 / 502 / 504);
* cooldown-and-skip on a block (429 / 403 / 406 / 503, or a bot-challenge page):
  the offending host is parked for a while and every later request to it raises
  ``Blocked`` immediately -- never a retry storm (docs/sourcing.md sec. 5);
* a proxy knob, wired but OFF by default (docs/decisions.md #20).

A block is always an exception, never an empty result, so the caller can tell
"blocked" from "no jobs today" (docs/sourcing.md sec. 5).

Synchronous by design: parallelism across providers, if ever needed, is the
orchestrator's concern (sub-slice 8), not this layer's.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

log = logging.getLogger("hiredra.sourcing.http")

_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)

# "The board is blocking us" -- cool the host down, never retry.
_BLOCK_STATUSES = frozenset({403, 406, 429, 503})
# Transient server errors -- safe to retry a few times.
_TRANSIENT_STATUSES = frozenset({500, 502, 504})


@dataclass(frozen=True, slots=True)
class HttpConfig:
    """Tuning knobs. Defaults are starting points, tuned by experiment (sec. 4)."""

    timeout: float = 15.0             # per-request seconds
    attempts: int = 3                 # total tries (1 + 2 retries) on transient failure
    backoff_base: float = 1.0         # exponential-with-jitter: ~1s, 2s, 4s
    backoff_max: float = 30.0
    cooldown_seconds: float = 900.0   # host cooldown after a block (15 min)
    max_cooldown: float = 3600.0      # clamp for a hostile Retry-After
    user_agent: str = _DEFAULT_UA     # polite, not anti-bot evasion (decisions.md #21)
    proxy: str | None = None          # OFF by default (decisions.md #20)
    follow_redirects: bool = False    # SSRF-safe default; providers opt in per-request
    sniff_challenges: bool = True     # narrow body-pattern bot-challenge detection


# ── exceptions ──────────────────────────────────────────────────────────────
class FetchError(Exception):
    """Base for anything this layer raises. Orchestrator contract: log and skip."""

    def __init__(self, message: str, *, url: str | None = None) -> None:
        super().__init__(message)
        self.url = url


class HTTPError(FetchError):
    """A non-retryable HTTP response: 4xx, or a redirect we will not follow."""

    def __init__(self, message: str, *, status: int, url: str | None = None) -> None:
        super().__init__(message, url=url)
        self.status = status


class TransientError(FetchError):
    """Timeout / connection error / 5xx that survived every retry attempt."""


class Blocked(FetchError):
    """The board is blocking us (status code or challenge page).

    Also raised -- with no network call -- for a request to a host that is
    still in cooldown from an earlier block.
    """

    def __init__(
        self,
        message: str,
        *,
        cooldown_key: str,
        until: float,
        retry_after: float | None = None,
        url: str | None = None,
    ) -> None:
        super().__init__(message, url=url)
        self.cooldown_key = cooldown_key
        self.until = until
        self.retry_after = retry_after


# ── challenge detection ─────────────────────────────────────────────────────
# Deliberately narrow. Every pattern is HTML markup or a JS-loader string that
# CANNOT occur in a well-formed JSON / XML API response, and _looks_like_challenge
# additionally refuses to sniff anything whose content-type isn't HTML/plain.
# A working ATS API can therefore never trip this. A missed challenge is cheaper
# than a false "blocked" on a live board (ATS APIs don't serve challenge pages).
_CHALLENGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"<title>\s*just a moment\.\.\.\s*</title>", re.I),                # Cloudflare
    re.compile(r"attention required!\s*\|\s*cloudflare", re.I),
    re.compile(r"cf-browser-verification|_cf_chl_opt|/cdn-cgi/challenge-platform/", re.I),
    re.compile(r"checking your browser before accessing", re.I),
    re.compile(r"<title>\s*you are being rate limited\s*</title>", re.I),
    re.compile(r"geo\.captcha-delivery\.com|captcha-delivery\.com/captcha", re.I),  # DataDome
    re.compile(r"<title>[^<]*(?:access denied|bot verification)[^<]*</title>", re.I),
    re.compile(r'<script[^>]+src="[^"]*hcaptcha\.com/(?:1/)?api\.js', re.I),
    re.compile(r'<div[^>]+class="g-recaptcha"', re.I),
)
_SNIFF_BYTES = 8192


def _looks_like_challenge(resp: httpx.Response) -> bool:
    ctype = resp.headers.get("content-type", "").lower()
    if "json" in ctype or "xml" in ctype:
        return False  # structured API responses are NEVER sniffed
    if ctype and not ctype.startswith(("text/html", "text/plain", "application/xhtml")):
        return False
    body = resp.text[:_SNIFF_BYTES]
    return any(p.search(body) for p in _CHALLENGE_PATTERNS)


def _parse_retry_after(value: str | None) -> float | None:
    """Retry-After header -> seconds from now, or None. Never negative."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=_dt.timezone.utc)
    delta = (when - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
    return max(0.0, delta)


# ── cooldown registry ──────────────────────────────────────────────────────
class CooldownRegistry:
    """Tracks ``{cooldown_key: until}`` for the life of a run. Clock is injectable."""

    def __init__(self, now: Callable[[], float]) -> None:
        self._now = now
        self._until: dict[str, float] = {}

    def check(self, key: str, *, url: str | None = None) -> None:
        until = self._until.get(key)
        if until is None:
            return
        remaining = until - self._now()
        if remaining > 0:
            raise Blocked(
                f"{key} is in cooldown for another {remaining:.0f}s",
                cooldown_key=key,
                until=until,
                url=url,
            )
        del self._until[key]

    def trip(self, key: str, seconds: float) -> float:
        """Park ``key``. Extends an existing cooldown, never shortens it."""
        until = max(self._now() + seconds, self._until.get(key, 0.0))
        self._until[key] = until
        return until

    def clear(self, key: str | None = None) -> None:
        if key is None:
            self._until.clear()
        else:
            self._until.pop(key, None)

    def active(self) -> dict[str, float]:
        now = self._now()
        return {k: v for k, v in self._until.items() if v > now}


# ── the client ─────────────────────────────────────────────────────────────
class HttpClient:
    """The shared client. ``ctx.http.get_json(url)`` etc. from every provider."""

    def __init__(
        self,
        config: HttpConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], Any] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        import time

        self.config = config or HttpConfig()
        self._sleep = sleep if sleep is not None else time.sleep
        self.cooldowns = CooldownRegistry(now if now is not None else time.monotonic)

        kwargs: dict[str, Any] = {
            "timeout": self.config.timeout,
            "headers": {"User-Agent": self.config.user_agent},
            "follow_redirects": False,  # per-request opt-in only
        }
        if transport is not None:
            kwargs["transport"] = transport
        elif self.config.proxy:
            kwargs["proxy"] = self.config.proxy
        self._client = httpx.Client(**kwargs)

    # context manager ------------------------------------------------------
    def __enter__(self) -> HttpClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # public -------------------------------------------------------------
    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        json: Any = None,
        cooldown_key: str | None = None,
        follow_redirects: bool | None = None,
    ) -> httpx.Response:
        """Guarded + retried request. Raises a ``FetchError`` subclass on failure."""
        key = cooldown_key or urlsplit(url).netloc or url
        follow = (
            self.config.follow_redirects
            if follow_redirects is None
            else follow_redirects
        )

        self.cooldowns.check(key, url=url)

        def _attempt() -> httpx.Response:
            try:
                resp = self._client.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    json=json,
                    follow_redirects=follow,
                )
            except httpx.TransportError as err:
                raise TransientError(
                    f"{method} {url}: {type(err).__name__}", url=url
                ) from err
            except httpx.RequestError as err:
                raise FetchError(
                    f"{method} {url}: {type(err).__name__}", url=url
                ) from err
            return self._guard(resp, key=key, url=url, follow=follow)

        retrying = Retrying(
            stop=stop_after_attempt(self.config.attempts),
            wait=wait_exponential_jitter(
                initial=self.config.backoff_base, max=self.config.backoff_max
            ),
            retry=retry_if_exception_type(TransientError),
            sleep=self._sleep,
            reraise=True,
        )
        return retrying(_attempt)

    def get_json(self, url: str, **kw: Any) -> Any:
        return self._as_json(self.request("GET", url, **kw), url)

    def get_text(self, url: str, **kw: Any) -> str:
        return self.request("GET", url, **kw).text

    def post_json(self, url: str, *, json: Any, **kw: Any) -> Any:
        return self._as_json(self.request("POST", url, json=json, **kw), url)

    # internals ---------------------------------------------------------
    @staticmethod
    def _as_json(resp: httpx.Response, url: str) -> Any:
        try:
            return resp.json()
        except ValueError as err:
            raise FetchError(f"{url}: response is not valid JSON", url=url) from err

    def _cooldown_len(self, retry_after: float | None) -> float:
        if retry_after is not None:
            return max(
                self.config.backoff_base,
                min(retry_after, self.config.max_cooldown),
            )
        return self.config.cooldown_seconds

    def _guard(
        self, resp: httpx.Response, *, key: str, url: str, follow: bool
    ) -> httpx.Response:
        status = resp.status_code

        if status in _BLOCK_STATUSES:
            retry_after = _parse_retry_after(resp.headers.get("retry-after"))
            seconds = self._cooldown_len(retry_after)
            until = self.cooldowns.trip(key, seconds)
            log.warning(
                "http: %s blocked (HTTP %d) -- cooling down %.0fs", key, status, seconds
            )
            raise Blocked(
                f"{url}: HTTP {status} (blocked)",
                cooldown_key=key,
                until=until,
                retry_after=retry_after,
                url=url,
            )

        if not follow and 300 <= status < 400:
            loc = resp.headers.get("location", "?")
            raise HTTPError(
                f"{url}: unexpected redirect (HTTP {status}) to {loc}",
                status=status,
                url=url,
            )

        if status in _TRANSIENT_STATUSES:
            raise TransientError(f"{url}: HTTP {status} (transient)", url=url)

        if status >= 400:
            raise HTTPError(f"{url}: HTTP {status}", status=status, url=url)

        if self.config.sniff_challenges and _looks_like_challenge(resp):
            seconds = self._cooldown_len(None)
            until = self.cooldowns.trip(key, seconds)
            log.warning(
                "http: %s served a challenge page -- cooling down %.0fs", key, seconds
            )
            raise Blocked(
                f"{url}: challenge page (HTTP {status})",
                cooldown_key=key,
                until=until,
                url=url,
            )

        return resp
