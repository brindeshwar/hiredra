"""Google free-tier quota handling, shared by every stage that calls Google (plan unit M0.1).

Moved out of ``hiredra/graph/`` (embedder + usage) so extraction and the later agents use the
same code; the graph modules re-export these names, so nothing that imports them changed.

* Failure classes: ``classify_failure`` tells an expected stop (daily quota, transient trouble,
  an API refusal) from a bug. The helpers read Google's ``QuotaFailure`` / ``RetryInfo``
  details through LangChain's wrapper, which re-raises the SDK error as ``__cause__``.
* ``DailyUsage``: a small counter for a per-DAY request limit, so a run stops BEFORE
  Google's cap instead of after a rejected request. It is ADVISORY; Google is the source of
  truth. One ``api_usage`` row per metered model per day (Google meters the model, not the stage). Every
  ATTEMPT is counted before it is sent, failed ones too, so the count can only run ahead of
  Google's (it fell 50 behind on 2026-10-19 when only successes counted).
* A "PerDay" 429 label is a claim, not a fact: only a long (>= 10 min) or missing retryDelay
  confirms it. A short throttle mislabelled PerDay must not switch a model off for the day.

Rule for every caller (from spoin_bundle's measured gemini-saga): Google's free tier sheds above
about 3 requests IN FLIGHT per source IP, across all keys. Hiredra calls Google one request at a
time per process, and the run locks keep at most two such processes alive at once. Never add
parallel or async fan-out to a Google call.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx
from google.genai.errors import APIError

PACIFIC = ZoneInfo("America/Los_Angeles")
IST = ZoneInfo("Asia/Kolkata")


class HTTPProviderTransient(RuntimeError):
    """A non-Google provider (NVIDIA) failed in a way a retry can fix: 429, 5xx, a timeout or a
    dropped connection. ``retry_after`` carries the server's own Retry-After when it sent one."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class HTTPProviderRefused(RuntimeError):
    """A non-Google provider answered and said no (400 / 401 / 403 / 404 / 422): not retried."""


class ProviderError(RuntimeError):
    """Google answered with something we can't use (wrong count/size, zero vector, ...).
    The embedder calls it ``EmbeddingError``."""


class DailyQuotaExceeded(ProviderError):
    """A DAILY quota is exhausted. Unlike the per-minute window this will not
    clear in a retry or two, so it is raised at once and never retried: callers
    stop cleanly and resume from the DB-backed queue on a later day."""


class DailyBudgetReached(DailyQuotaExceeded):
    """OUR local daily budget is used up. Raised BEFORE any API call, so Google's
    real cap is never hit. A ``DailyQuotaExceeded`` subclass, so anything that
    already handles a daily stop handles this too."""


# ─────────────────────────────────────────────────────────────────────────────
# reading Google's errors
# ─────────────────────────────────────────────────────────────────────────────
def _api_errors(exc: BaseException):
    """Every SDK ``APIError`` in ``exc``'s cause/context chain (usually one)."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, APIError):
            yield exc
        exc = exc.__cause__ or exc.__context__


#: A retryDelay this long is not a per-minute window clearing.
_DAILY_RETRY_DELAY_S = 600.0


def _quota_ids(exc: BaseException) -> list[str]:
    """``QuotaFailure`` quota ids on the error, e.g.
    ``EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier``."""
    ids: list[str] = []
    for e in _api_errors(exc):
        details = e.details if isinstance(e.details, dict) else {}
        body = details.get("error", details)
        for item in (body.get("details") or []) if isinstance(body, dict) else []:
            if isinstance(item, dict) and str(item.get("@type", "")).endswith("QuotaFailure"):
                ids += [str(v.get("quotaId", "")) for v in item.get("violations") or [] if isinstance(v, dict)]
    return ids


def _quota_values(exc: BaseException) -> list[str]:
    """The ``quotaValue`` (the limit itself) of each violated quota, so a daily
    stop records WHAT the cap is, not just which quota it was."""
    values: list[str] = []
    for e in _api_errors(exc):
        details = e.details if isinstance(e.details, dict) else {}
        body = details.get("error", details)
        for item in (body.get("details") or []) if isinstance(body, dict) else []:
            if isinstance(item, dict) and str(item.get("@type", "")).endswith("QuotaFailure"):
                values += [str(v.get("quotaValue", "")) for v in item.get("violations") or [] if isinstance(v, dict)]
    return values


def _is_daily_quota(exc: BaseException) -> bool:
    """A 429 whose quota id says per-DAY, or whose retryDelay is far too long to
    be a per-minute window (>= 10 min)."""
    if not any(e.code == 429 for e in _api_errors(exc)):
        return False
    hint = _retry_after_seconds(exc)
    if any("perday" in q.lower() for q in _quota_ids(exc)):
        # The label alone is a claim, not a fact: Google has labelled short throttles "PerDay"
        # (spoin_bundle's gemini-saga: one such label switched off 136 of 204 key slots in a
        # minute). A per-day label with a SHORT retryDelay is treated as a per-minute throttle.
        return hint is None or hint >= _DAILY_RETRY_DELAY_S
    return hint is not None and hint >= _DAILY_RETRY_DELAY_S


#: Network-level failures (no HTTP status at all): a dropped connection, a DNS
#: hiccup, a timeout. Worth a retry, and not a bug in our code.
_NETWORK_ERRORS = (httpx.TransportError, ConnectionError, TimeoutError)


def _cause_chain(exc: BaseException):
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _is_network_error(exc: BaseException) -> bool:
    return any(isinstance(e, _NETWORK_ERRORS) for e in _cause_chain(exc))


def _is_transient(exc: BaseException) -> bool:
    """429 / 5xx anywhere in the cause chain, or a network-level failure -- worth
    a retry. Anything else (a bad request, our own ProviderError) is not, and
    neither is a daily quota, which a retry cannot fix."""
    if _is_daily_quota(exc):
        return False
    if isinstance(exc, HTTPProviderTransient):
        return True
    return _is_network_error(exc) or any(e.code == 429 or e.code >= 500 for e in _api_errors(exc))


def classify_failure(exc: BaseException) -> str | None:
    """What kind of failure is this? Used to tell an expected stop from a bug.

    "daily_quota"  Google's daily cap (or our local budget): wait for the reset.
    "transient"    429 / 5xx / network trouble that outlasted the retries: re-run later.
    "api_error"    Google answered and said no (bad key, bad request, no access).
    "bad_response" Google answered with something unusable (wrong count/size/zero vector).
    None           anything else: an unknown exception is most likely a BUG, so callers
                   must re-raise it rather than turn it into a status line.
    """
    if isinstance(exc, DailyQuotaExceeded) or _is_daily_quota(exc):
        return "daily_quota"
    if _is_transient(exc):
        return "transient"
    if any(True for _ in _api_errors(exc)) or isinstance(exc, HTTPProviderRefused):
        return "api_error"
    if isinstance(exc, ProviderError):
        return "bad_response"
    return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """The server's own ``RetryInfo.retryDelay`` (e.g. ``"57s"``), if the error
    carries one. Free-tier 429s do. A non-Google provider's Retry-After counts too."""
    if isinstance(exc, HTTPProviderTransient) and exc.retry_after is not None:
        return exc.retry_after
    for e in _api_errors(exc):
        details = e.details if isinstance(e.details, dict) else {}
        body = details.get("error", details)
        for item in (body.get("details") or []) if isinstance(body, dict) else []:
            if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
                delay = str(item.get("retryDelay", "")).rstrip("s")
                try:
                    return float(delay)
                except ValueError:
                    return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# the daily counter
# ─────────────────────────────────────────────────────────────────────────────
def pacific_day(now: datetime) -> str:
    """The Pacific-time calendar date (the day Google's daily quota runs on)."""
    return now.astimezone(PACIFIC).date().isoformat()


def next_reset(now: datetime) -> datetime:
    """The next midnight Pacific, as an aware datetime."""
    local = now.astimezone(PACIFIC)
    midnight = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), tzinfo=PACIFIC)
    return midnight


def reset_text(now: datetime) -> str:
    r = next_reset(now)
    return f"{r:%Y-%m-%d %H:%M} Pacific = {r.astimezone(IST):%Y-%m-%d %H:%M} IST"


class DailyUsage:
    """Requests sent today (Pacific day), against a budget: one ``api_usage`` row per metered model per day (Q5).

    The count lives in Postgres since plan unit DB3b, so two processes add to it atomically. ``provider`` and
    ``model`` name the meter (Google meters the model, not the stage)."""

    def __init__(
        self,
        budget: int,
        conn,
        provider: str,
        model: str,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.budget = budget
        self.conn = conn
        self.provider = provider
        self.model = model
        self._now = now

    # -- reading ---------------------------------------------------------------
    def _today(self) -> dict:
        """Today's record. No row for today means a fresh day."""
        day = pacific_day(self._now())
        row = self.conn.execute(
            "SELECT calls, exhausted_at, note FROM api_usage WHERE provider = %s AND model = %s AND day = %s",
            (self.provider, self.model, day),
        ).fetchone()
        if row is None:
            return {"day": day, "requests": 0, "exhausted": False, "note": None}
        return {"day": day, "requests": row[0], "exhausted": row[1] is not None, "note": row[2]}

    def used(self) -> int:
        return self._today()["requests"]

    def exhausted(self) -> bool:
        """Google itself already said the daily quota is gone today."""
        return self._today()["exhausted"]

    def remaining(self) -> int:
        rec = self._today()
        return 0 if rec["exhausted"] else max(0, self.budget - rec["requests"])

    def summary(self) -> str:
        rec = self._today()
        if rec["exhausted"]:
            return f"Google's daily quota was already reported exhausted today ({rec['day']} Pacific); resets {reset_text(self._now())}"
        return f"{rec['requests']} of {self.budget} requests used today ({rec['day']} Pacific); resets {reset_text(self._now())}"

    # -- writing ---------------------------------------------------------------
    def reserve(self, n: int) -> None:
        """Check that ``n`` more requests may be sent today. Sends nothing and
        counts nothing; raises before any API call if they may not."""
        rec = self._today()
        if rec["exhausted"]:
            raise DailyQuotaExceeded(
                f"Google's daily quota was already reported exhausted today; resets {reset_text(self._now())}"
            )
        if rec["requests"] + n > self.budget:
            raise DailyBudgetReached(
                f"today's request budget is used ({rec['requests']} of {self.budget}; "
                f"{n} more would pass it); resets {reset_text(self._now())}"
            )

    def record(self, n: int) -> None:
        """Count ``n`` requests (added in one statement: two processes never lose each other's count)."""
        self.conn.execute(
            "INSERT INTO api_usage (provider, model, day, calls) VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (provider, model, day) DO UPDATE SET calls = api_usage.calls + excluded.calls",
            (self.provider, self.model, pacific_day(self._now()), n),
        )

    def mark_exhausted(self, note: str | None = None) -> None:
        """Google reported the daily quota is gone: refuse further calls today."""
        self.conn.execute(
            "INSERT INTO api_usage (provider, model, day, exhausted_at, note) VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (provider, model, day) DO UPDATE SET exhausted_at = excluded.exhausted_at, "
            "note = COALESCE(excluded.note, api_usage.note)",
            (self.provider, self.model, pacific_day(self._now()), self._now(), note),
        )


def jittered_wait(retry_state, rng: Callable[[], float] = random.random) -> float:
    """The owner's retry rule, for every provider: wait 1, 2, 4, 8 s (doubling per attempt) plus a
    random jitter of up to the same amount, so two of our processes never retry in lockstep. When
    the server names its own retry time, wait that + 1 s, plus up to 1 s of jitter. Capped at 120 s.
    Use as tenacity's ``wait=``."""
    hint = _retry_after_seconds(retry_state.outcome.exception())
    if hint is not None:
        return min(hint + 1.0 + rng(), 120.0)
    base = float(2 ** (retry_state.attempt_number - 1))
    return min(base + rng() * base, 120.0)
