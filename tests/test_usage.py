"""Tests for hiredra.graph.usage: the local daily request counter (build 4.3 fix).
No network, no key: time is a fake clock."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from hiredra.graph.embedder import DailyQuotaExceeded
from hiredra.graph.usage import DailyBudgetReached, DailyUsage, next_reset, pacific_day, reset_text


def _at(text: str):
    """A clock frozen at an ISO UTC time."""
    moment = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    return lambda: moment


class Clock:
    def __init__(self, text: str):
        self.now = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def set(self, text: str):
        self.now = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def _usage(tmp_path, budget=900, when="2026-09-21T14:00:00"):
    clock = Clock(when)
    return DailyUsage(budget, tmp_path / "usage.json", now=clock), clock


# ── counting ─────────────────────────────────────────────────────────────────
def test_a_fresh_day_has_the_whole_budget(tmp_path):
    u, _ = _usage(tmp_path)
    assert (u.used(), u.remaining(), u.exhausted()) == (0, 900, False)


def test_recorded_requests_accumulate_and_survive_a_new_process(tmp_path):
    u, clock = _usage(tmp_path)
    u.record(50)
    u.record(50)
    assert (u.used(), u.remaining()) == (100, 800)
    again = DailyUsage(900, tmp_path / "usage.json", now=clock)     # a later run / another process
    assert again.used() == 100


def test_reserve_allows_up_to_exactly_the_budget_and_counts_nothing(tmp_path):
    u, _ = _usage(tmp_path, budget=100)
    u.reserve(100)                                        # exactly the budget: allowed
    assert u.used() == 0                                  # reserving is only a check
    u.record(60)
    u.reserve(40)
    with pytest.raises(DailyBudgetReached, match="60 of 100"):
        u.reserve(41)


def test_the_budget_error_is_raised_before_any_call_and_is_a_daily_quota_error(tmp_path):
    u, _ = _usage(tmp_path, budget=10)
    u.record(10)
    with pytest.raises(DailyQuotaExceeded):               # subclass: existing daily-stop handling still applies
        u.reserve(1)


def test_the_message_says_when_it_resets_in_both_time_zones(tmp_path):
    u, _ = _usage(tmp_path, budget=1, when="2026-09-21T14:00:00")
    u.record(1)
    with pytest.raises(DailyBudgetReached, match=r"2026-09-22 00:00 Pacific = 2026-09-22 12:30 IST"):
        u.reserve(1)


# ── Google said the day is over ──────────────────────────────────────────────
def test_marking_the_day_exhausted_stops_everything_until_it_resets(tmp_path):
    u, clock = _usage(tmp_path)
    u.mark_exhausted("reported by Google")
    assert u.exhausted() and u.remaining() == 0
    with pytest.raises(DailyQuotaExceeded, match="already reported exhausted"):
        u.reserve(1)
    assert "already reported exhausted" in u.summary()
    clock.set("2026-09-22T07:00:00")                      # midnight Pacific has passed
    assert not u.exhausted() and u.remaining() == 900


def test_exhausted_keeps_the_count_and_the_note(tmp_path):
    u, _ = _usage(tmp_path)
    u.record(700)
    u.mark_exhausted("server said so")
    saved = json.loads((tmp_path / "usage.json").read_text(encoding="utf-8"))
    assert saved["requests"] == 700 and saved["exhausted"] is True and saved["note"] == "server said so"
    assert saved["day"] == "2026-09-21" and "exhausted_at" in saved


# ── the day is a PACIFIC day ─────────────────────────────────────────────────
def test_the_day_rolls_at_midnight_pacific_not_utc(tmp_path):
    u, clock = _usage(tmp_path, when="2026-09-22T06:59:00")     # 23:59 PDT on Sep 21
    u.record(400)
    assert u.used() == 400
    clock.set("2026-09-22T07:00:00")                            # 00:00 PDT on Sep 22
    assert u.used() == 0 and u.remaining() == 900


def test_daylight_saving_shifts_the_utc_time_of_the_reset(tmp_path):
    """Winter (PST, UTC-8): midnight Pacific is 08:00 UTC, not 07:00."""
    u, clock = _usage(tmp_path, when="2026-12-01T07:30:00")     # 23:30 PST on Nov 30
    u.record(5)
    clock.set("2026-12-01T07:59:00")
    assert u.used() == 5                                        # still Nov 30 in Pacific time
    clock.set("2026-12-01T08:00:00")
    assert u.used() == 0                                        # Dec 1 in Pacific time


def test_pacific_day_and_next_reset_helpers():
    now = datetime(2026, 9, 21, 14, 11, tzinfo=timezone.utc)
    assert pacific_day(now) == "2026-09-21"
    assert next_reset(now).isoformat() == "2026-09-22T00:00:00-07:00"
    assert reset_text(now) == "2026-09-22 00:00 Pacific = 2026-09-22 12:30 IST"


# ── robustness ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("content", ["", "not json", "[]", '{"day": "2026-09-21", "requests": "abc"}', '{"nonsense": 1}'])
def test_a_missing_or_corrupt_file_means_a_fresh_day_not_a_crash(tmp_path, content):
    (tmp_path / "usage.json").write_text(content, encoding="utf-8")
    u, _ = _usage(tmp_path)
    assert (u.used(), u.remaining(), u.exhausted()) == (0, 900, False)   # garbage is ignored, never a crash


def test_a_file_from_another_day_is_ignored(tmp_path):
    (tmp_path / "usage.json").write_text(
        json.dumps({"day": "2026-09-20", "requests": 900, "exhausted": True}), encoding="utf-8")
    u, _ = _usage(tmp_path)
    assert (u.used(), u.exhausted(), u.remaining()) == (0, False, 900)


def test_writes_are_atomic_and_leave_no_temp_file(tmp_path):
    u, _ = _usage(tmp_path)
    u.record(3)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["usage.json"]


def test_the_state_directory_is_created_if_missing(tmp_path):
    clock = Clock("2026-09-21T14:00:00")
    u = DailyUsage(900, tmp_path / "nested" / "deeper" / "usage.json", now=clock)
    u.record(1)
    assert u.used() == 1


def test_the_summary_reads_plainly(tmp_path):
    u, _ = _usage(tmp_path)
    u.record(123)
    assert u.summary().startswith("123 of 900 requests used today (2026-09-21 Pacific); resets 2026-09-22 00:00 Pacific")
