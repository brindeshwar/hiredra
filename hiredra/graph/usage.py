"""A small local counter for Google's per-DAY embedding limit (build 4.3 fix).

Why it exists: Google counts every input inside a batch as one request, and the
free tier allows 1,000 requests per day (resetting at midnight Pacific). The
first real index build ran into that cap with no warning. This counter lets a
run stop BEFORE the cap instead of after a rejected request, and remembers a cap
that Google has already reported, so a second run the same day refuses at once
without making any API call.

It is ADVISORY. Google is the source of truth: if this file is missing, wrong,
or from another day, the worst case is one rejected request, which the embedder
already handles. It lives in ``data/`` (git-ignored) as one tiny JSON file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from hiredra.graph.embedder import DailyQuotaExceeded

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USAGE_PATH = _REPO_ROOT / "data" / "embed_usage.json"

PACIFIC = ZoneInfo("America/Los_Angeles")
IST = ZoneInfo("Asia/Kolkata")


class DailyBudgetReached(DailyQuotaExceeded):
    """OUR local daily budget is used up. Raised BEFORE any API call, so Google's
    real cap is never hit. A ``DailyQuotaExceeded`` subclass, so anything that
    already handles a daily stop handles this too."""


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
    """Requests sent today (Pacific day), against a budget."""

    def __init__(
        self,
        budget: int,
        path: Path | str = DEFAULT_USAGE_PATH,
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
