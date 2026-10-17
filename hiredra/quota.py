"""Google free-tier quota handling, shared by every stage that calls Google (plan unit M0.1).

Moved out of ``hiredra/graph/`` (embedder + usage) so extraction and the later agents use the
same code; the graph modules re-export these names, so nothing that imports them changed.

* Failure classes: ``classify_failure`` tells an expected stop (daily quota, transient trouble,
  an API refusal) from a bug. The helpers read Google's ``QuotaFailure`` / ``RetryInfo``
  details through LangChain's wrapper, which re-raises the SDK error as ``__cause__``.
* ``DailyUsage``: a small local counter for a per-DAY request limit, so a run stops BEFORE
  Google's cap instead of after a rejected request. It is ADVISORY; Google is the source of
  truth. One counter file per metered model (Google meters the model, not the stage).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
from google.genai.errors import APIError

PACIFIC = ZoneInfo("America/Los_Angeles")
IST = ZoneInfo("Asia/Kolkata")


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
    if any("perday" in q.lower() for q in _quota_ids(exc)):
        return True
    hint = _retry_after_seconds(exc)
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
    if any(True for _ in _api_errors(exc)):
        return "api_error"
    if isinstance(exc, ProviderError):
        return "bad_response"
    return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """The server's own ``RetryInfo.retryDelay`` (e.g. ``"57s"``), if the error
    carries one. Free-tier 429s do."""
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
    """Requests sent today (Pacific day), against a budget. One file per metered model."""

    def __init__(
        self,
        budget: int,
        path: Path | str,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.budget = budget
        self.path = Path(path)
        self._now = now

    # -- reading ---------------------------------------------------------------
    def _today(self) -> dict:
        """Today's record. A missing, unreadable or other-day file means a fresh day."""
        day = pacific_day(self._now())
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"day": day, "requests": 0, "exhausted": False}
        if not isinstance(data, dict) or data.get("day") != day:
            return {"day": day, "requests": 0, "exhausted": False}
        try:
            requests = max(0, int(data.get("requests") or 0))
        except (TypeError, ValueError):
            requests = 0  # a garbled count is treated as none: the counter is advisory, Google enforces the cap
        return {
            "day": day,
            "requests": requests,
            "exhausted": bool(data.get("exhausted")),
            "note": data.get("note"),
        }

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
    def _save(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written counter

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
        """Count ``n`` requests that Google accepted."""
        rec = self._today()
        rec["requests"] += n
        self._save(rec)

    def mark_exhausted(self, note: str | None = None) -> None:
        """Google reported the daily quota is gone: refuse further calls today."""
        rec = self._today()
        rec["exhausted"] = True
        rec["exhausted_at"] = self._now().isoformat(timespec="seconds")
        if note:
            rec["note"] = note
        self._save(rec)
