"""Persistent, thread-safe cooldowns (build sub-slice 2a.8a)."""

from __future__ import annotations

import threading

import pytest

from hiredra.sourcing.http import Blocked, CooldownRegistry
from hiredra.storage import db


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


# ── registry: lock, load, on_trip ─────────────────────────────────────────
def test_default_clock_is_wall_time():
    import time

    r = CooldownRegistry()
    until = r.trip("h", 100)
    assert abs(until - (time.time() + 100)) < 5


def test_on_trip_callback_fires_after_trip():
    calls = []
    r = CooldownRegistry(now=Clock())
    r.on_trip = lambda k, u, reason: calls.append((k, u, reason))
    until = r.trip("api.x.com", 50, reason="HTTP 429")
    assert calls == [("api.x.com", until, "HTTP 429")]


def test_load_merges_non_expired_and_never_shortens():
    clk = Clock()
    r = CooldownRegistry(now=clk)
    r.trip("a", 100)  # until ~= 1_000_100
    r.load({"a": clk.t + 10, "b": clk.t + 500, "c": clk.t - 1})  # 'a' shorter, 'c' expired
    active = r.active()
    assert set(active) == {"a", "b"}
    assert active["a"] == pytest.approx(clk.t + 100)  # not shortened by the load


def test_trip_extends_never_shortens_across_load():
    clk = Clock()
    r = CooldownRegistry(now=clk)
    r.load({"k": clk.t + 1000})
    r.trip("k", 10)  # would be +10, but +1000 stands
    assert r.active()["k"] == pytest.approx(clk.t + 1000)


def test_concurrent_trips_are_consistent():
    r = CooldownRegistry(now=Clock())
    barrier = threading.Barrier(8)

    def worker(i):
        barrier.wait()
        for _ in range(200):
            r.trip(f"host-{i % 3}", 60)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert set(r.active()) == {"host-0", "host-1", "host-2"}


# ── storage: load_cooldowns / save_cooldown ───────────────────────────────
@pytest.fixture()
def conn(pg_url):
    c = db.connect(pg_url)
    db.init_db(c)
    yield c
    c.close()


def test_save_and_load_round_trip(conn):
    import time

    future = time.time() + 3600
    db.save_cooldown(conn, "jobspy-naukri", future, "HTTP 406 recaptcha required")
    assert db.load_cooldowns(conn) == {"jobspy-naukri": pytest.approx(future)}


def test_load_cooldowns_filters_expired(conn):
    import time

    db.save_cooldown(conn, "old", time.time() - 10, "x")
    db.save_cooldown(conn, "live", time.time() + 100, "y")
    assert set(db.load_cooldowns(conn)) == {"live"}


def test_save_cooldown_extends_never_shortens(conn):
    import time

    now = time.time()
    db.save_cooldown(conn, "k", now + 1000, "first")
    db.save_cooldown(conn, "k", now + 10, "second")  # shorter -> ignored
    assert db.load_cooldowns(conn)["k"] == pytest.approx(now + 1000)
    row = conn.execute("SELECT reason FROM cooldowns WHERE key='k'").fetchone()
    assert row["reason"] == "second"  # metadata still updates


def test_registry_persists_via_on_trip_and_reloads(conn):
    """The wiring the orchestrator does: on_trip -> save, then load on next run."""

    r1 = CooldownRegistry()  # wall clock
    r1.on_trip = lambda k, u, reason: db.save_cooldown(conn, k, u, reason)
    r1.trip("api.ashbyhq.com", 900, reason="HTTP 429")

    r2 = CooldownRegistry()  # a fresh "process"
    r2.load(db.load_cooldowns(conn))
    with pytest.raises(Blocked):
        r2.check("api.ashbyhq.com")
