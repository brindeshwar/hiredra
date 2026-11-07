"""The conveyor belt (the owner's plan, 2026-11-07): the run's tasks laid out before it starts, taken by the workers.

A task is a whole small board, one chunk of a big board's listing, or one chunk of its job details. The belt holds
them in order; a worker that comes free takes the FIRST task the rules allow:

* **a platform's share** -- at most ``at_once`` tasks of a platform held to a rate (``belt.platforms`` in the run
  config: Workday, Eightfold, Phenom) are in flight at once; inside them, every request waits on the platform's one
  pace, at ``fill`` (60%) of its measured limit, so the limit is never reached;
* **a host's share** -- at most the HTTP client's slots for the task's host (2, or a ``host_limits`` entry: 4 on the
  Greenhouse, Lever and Ashby APIs);
* **its time** -- a task put back after a refusal waits until its ``not_before`` (the server's Retry-After, or the
  host's cooldown).

So a worker never waits on a task the rules forbid: it takes the next one that is allowed, a platform below its
share first (it is what the run's length hangs on), the rest after. The order is laid out by
the orchestrator (``lay_out``): each platform's boards longest first, the platforms interleaved so each runs steadily
from minute 0; a board's later chunks take its place on the belt, so a started board finishes before new ones start.

Pure bookkeeping, no threads and no network: the orchestrator owns the workers and calls ``take``/``done``/``put``.
"""

from __future__ import annotations

import bisect
import heapq
import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

#: How far down one platform's line ``take`` looks for a task whose host has room, before giving that line up for
#: the moment (a big board's chunks sit together; its host allows two at once).
_LOOK_AHEAD = 512
#: The line of every task with no platform rule.
OTHER = ""


@dataclass(frozen=True, slots=True)
class Platform:
    """A platform held to a rate: ``limit`` is the request rate its refusals start at (measured), ``at_once`` how many
    of its tasks may be in flight together (enough to keep its pace full: the rate times a request's time)."""

    limit: float
    at_once: int


@dataclass(frozen=True, slots=True)
class BeltConfig:
    """``belt:`` in the run config. ``fill``: the share of each platform's limit used (the owner's 50-60%: 60%).
    ``tries``: how many times a refused task is put back on the belt before its board fails for the run.
    ``watchdog_minutes``: a warning when no task has finished for this long."""

    fill: float = 0.6
    platforms: Mapping[str, Platform] = field(default_factory=dict, hash=False)
    tries: int = 4
    watchdog_minutes: float = 10.0

    def per_second(self, platform: str) -> float | None:
        p = self.platforms.get(platform)
        return self.fill * p.limit if p else None


@dataclass(eq=False)
class Task:
    """One unit on the belt. ``line`` is its platform (``OTHER`` for none), ``host`` the key of the host slots it uses
    (``None``: none), ``key`` its place on the belt (smaller first)."""

    kind: str  # "board" (a whole board) | "list" (a listing chunk) | "detail" (a chunk of job details) | "jobspy"
    line: str
    host: str | None
    key: float
    payload: Any = None
    board: Any = None  # the orchestrator's state for the board the task belongs to
    tries: int = 0
    not_before: float = 0.0


class Belt:
    """The tasks, in belt order, one line per platform; what is in flight; the rules."""

    def __init__(self, at_once: Mapping[str, int], host_cap: Callable[[str], int]) -> None:
        self._at_once = dict(at_once)
        self._host_cap = host_cap
        self._lines: dict[str, list[tuple[float, int, Task]]] = defaultdict(list)
        self._waiting: list[tuple[float, int, Task]] = []  # put back, by not_before
        self._seq = itertools.count()
        self.in_line: dict[str, int] = defaultdict(int)  # in flight, per line
        self.in_host: dict[str, int] = defaultdict(int)  # in flight, per host key

    def __len__(self) -> int:
        return sum(len(line) for line in self._lines.values()) + len(self._waiting)

    def put(self, task: Task, now: float = 0.0) -> None:
        """On the belt at its place; or set aside until its ``not_before`` when that is still to come."""
        item = (task.key, next(self._seq), task)
        if task.not_before > now:
            heapq.heappush(self._waiting, (task.not_before, item[1], task))
        else:
            bisect.insort(self._lines[task.line], item)

    def put_all(self, tasks: Iterable[Task], now: float = 0.0) -> None:
        for task in tasks:
            self.put(task, now)

    def take(self, now: float = 0.0) -> Task | None:
        """The first task the rules allow, now marked in flight; ``None`` when none is allowed at the moment."""
        while self._waiting and self._waiting[0][0] <= now:
            _, _, task = heapq.heappop(self._waiting)
            bisect.insort(self._lines[task.line], (task.key, next(self._seq), task))
        # A platform held to a rate is what the run's length hangs on, so a platform below its share is served
        # first; the other lines fill the workers left. Placing tasks by key alone left Workday half-empty: its next
        # board's key waits behind Accenture's estimate, though a board runs only 2 chunks at once (crawl 3, 2026-11-07).
        best = self._first(lambda name: name in self._at_once) or self._first(lambda name: name not in self._at_once)
        if best is None:
            return None
        name, i = best
        _, _, task = self._lines[name].pop(i)
        self.in_line[task.line] += 1
        if task.host is not None:
            self.in_host[task.host] += 1
        return task

    def _first(self, wanted: Callable[[str], bool]) -> tuple[str, int] | None:
        """Among the wanted lines with room, the allowed task with the smallest key: ``(line, index)``."""
        best: tuple[str, int] | None = None
        best_key: tuple[float, int] | None = None
        for name, line in self._lines.items():
            if not line or not wanted(name) or self.in_line[name] >= self._at_once.get(name, 1 << 30):
                continue
            for i in range(min(len(line), _LOOK_AHEAD)):
                key, seq, task = line[i]
                if task.host is None or self.in_host[task.host] < self._host_cap(task.host):
                    if best_key is None or (key, seq) < best_key:
                        best, best_key = (name, i), (key, seq)
                    break
        return best

    def done(self, task: Task) -> None:
        """A task taken earlier has finished (or failed): its platform's and host's places are free."""
        self.in_line[task.line] -= 1
        if task.host is not None:
            self.in_host[task.host] -= 1

    def next_ready_in(self, now: float) -> float | None:
        """Seconds until the next set-aside task may run, ``None`` when none is set aside."""
        return max(0.0, self._waiting[0][0] - now) if self._waiting else None


def lay_out(
    estimates: Iterable[tuple[str, float, Any]], width: Mapping[str, int], default_width: int
) -> list[tuple[float, Any]]:
    """The belt's order for the run's first tasks: ``(key, item)`` for each ``(line, est_seconds, item)``. Inside a
    line, longest first; each task's key is when it would start if its line ran steadily at its width from minute 0,
    so the lines interleave and every one starts at once (a short board fills in beside a long one)."""
    by_line: dict[str, list[tuple[float, Any]]] = defaultdict(list)
    for line, seconds, item in estimates:
        by_line[line].append((seconds, item))
    out: list[tuple[float, Any]] = []
    for line, items in by_line.items():
        items.sort(key=lambda x: -x[0])
        w = max(1, width.get(line, default_width))
        t = 0.0
        for seconds, item in items:
            out.append((t / w, item))
            t += max(seconds, 0.0)
    out.sort(key=lambda x: x[0])
    return out
