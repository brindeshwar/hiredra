"""The conveyor belt's rules (owner's plan, 2026-11-07): the first allowed task, a platform's share, a host's share,
a task's time, and the layout. Pure bookkeeping: no threads, no network, a fake clock."""

from __future__ import annotations

from hiredra.sourcing.belt import OTHER, Belt, BeltConfig, Platform, Task, lay_out


def _belt(at_once=None, caps=None):
    caps = caps or {}
    return Belt(at_once or {}, lambda key: caps.get(key, 2))


def test_a_free_worker_takes_the_first_task_on_the_belt():
    belt = _belt()
    belt.put_all([Task("board", OTHER, "b", 2.0, payload="second"), Task("board", OTHER, "a", 1.0, payload="first")])
    assert belt.take().payload == "first" and belt.take().payload == "second" and belt.take() is None


def test_a_platform_never_has_more_than_its_share_in_flight_and_the_next_allowed_task_goes_instead():
    belt = _belt(at_once={"workday": 2})
    belt.put_all([Task("list", "workday", f"w{i}", float(i), payload=f"w{i}") for i in range(4)])
    belt.put(Task("board", OTHER, "g", 10.0, payload="greenhouse"))
    taken = [belt.take().payload for _ in range(3)]
    assert taken == ["w0", "w1", "greenhouse"]  # Workday's share is full: the next allowed task, not a wait
    assert belt.take() is None
    belt.done(Task("list", "workday", "w0", 0.0))  # one Workday task finishes
    assert belt.take().payload == "w2"


def test_a_host_never_has_more_than_its_slots_in_flight():
    belt = _belt(caps={"api.greenhouse": 4})
    belt.put_all([Task("list", OTHER, "acme", float(i), payload=f"a{i}") for i in range(3)])
    belt.put(Task("list", OTHER, "beta", 5.0, payload="b"))
    belt.put_all([Task("board", OTHER, "api.greenhouse", 6.0 + i, payload=f"g{i}") for i in range(5)])
    taken = [t.payload for t in iter(belt.take, None)]
    assert taken == ["a0", "a1", "b", "g0", "g1", "g2", "g3"]  # 2 on a tenant host, 4 on the Greenhouse API


def test_a_task_put_back_waits_for_its_time():
    belt = _belt()
    belt.put(Task("list", OTHER, "h", 0.0, payload="later", not_before=100.0), now=0.0)
    belt.put(Task("list", OTHER, "h", 5.0, payload="now"), now=0.0)
    assert belt.take(now=0.0).payload == "now" and belt.take(now=0.0) is None
    assert belt.next_ready_in(now=40.0) == 60.0
    assert belt.take(now=100.0).payload == "later"  # back in its place when its time comes
    assert len(belt) == 0 and belt.next_ready_in(now=100.0) is None


def test_the_layout_puts_each_platforms_longest_board_first_and_every_platform_starts_at_once():
    estimates = [
        ("workday", 60.0, "wd-small"), ("workday", 3600.0, "accenture"), ("eightfold", 900.0, "ef"),
        (OTHER, 5.0, "gh1"), (OTHER, 5.0, "gh2"),
    ]  # fmt: skip
    order = [item for _, item in lay_out(estimates, {"workday": 8, "eightfold": 1}, 16)]
    assert order.index("accenture") < order.index("wd-small")
    assert {item for key, item in lay_out(estimates, {"workday": 8, "eightfold": 1}, 16) if key == 0.0} == {
        "accenture", "ef", "gh1"
    }  # fmt: skip  # each line's first task starts at minute 0


def test_a_platforms_rate_is_its_fill_of_the_measured_limit():
    rules = BeltConfig(fill=0.6, platforms={"workday": Platform(limit=10.0, at_once=8)})
    assert rules.per_second("workday") == 6.0 and rules.per_second("lever") is None


def test_a_platform_below_its_share_is_served_before_tasks_placed_earlier_on_the_belt():
    # crawl 3 (2026-11-07): Accenture's estimate put Workday's next board 18 minutes down the belt, and other boards,
    # placed earlier, took the free workers -- Workday ran 4 wide of its 8 (a board runs only 2 chunks at once)
    belt = _belt(at_once={"workday": 4})
    belt.put_all([Task("list", "workday", "accenture", 0.0, payload="acc") for _ in range(10)])
    belt.put(Task("list", "workday", "circlek", 1100.0, payload="ck"))
    belt.put_all([Task("board", OTHER, f"o{i}", 10.0 + i, payload="other") for i in range(10)])
    taken = [belt.take().payload for _ in range(5)]
    assert taken == ["acc", "acc", "ck", "other", "other"]  # Workday's free places first, then the rest


def test_a_full_host_with_hundreds_of_tasks_never_hides_the_boards_behind_it():
    # crawl 3 (2026-11-07): Accenture's and Circle K's chunks filled a look-ahead of 512, and Workday ran 2 boards
    belt = _belt(at_once={"workday": 8})
    belt.put_all([Task("list", "workday", "accenture", 0.0, payload="acc") for _ in range(400)])
    belt.put_all([Task("list", "workday", "circlek", 1.0, payload="ck") for _ in range(400)])
    belt.put_all([Task("list", "workday", f"wd{i}", 2.0 + i, payload=f"wd{i}") for i in range(10)])
    taken = [belt.take().payload for _ in range(8)]
    assert taken == ["acc", "acc", "ck", "ck", "wd0", "wd1", "wd2", "wd3"]  # Workday's 8, past the two full hosts
    assert belt.take() is None and len(belt) == 802


def test_a_host_set_aside_as_full_comes_back_when_its_task_is_done():
    belt = _belt()
    first = [Task("list", OTHER, "h", float(i), payload=f"h{i}") for i in range(3)]
    belt.put_all(first)
    belt.put(Task("list", OTHER, "z", 9.0, payload="z"))
    a, b, z = belt.take(), belt.take(), belt.take()
    assert [a.payload, b.payload, z.payload] == ["h0", "h1", "z"] and belt.take() is None  # h is full: set aside
    belt.done(a)
    assert belt.take().payload == "h2" and len(belt) == 0
