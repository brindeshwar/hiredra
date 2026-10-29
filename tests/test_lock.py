"""The run lock (hiredra/lock.py): a Postgres advisory lock held on a connection of its own (D1.10, plan unit DB3b).
Needs the pg_url fixture (conftest.py). The holder tests use a real second process, as the file lock's did."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hiredra.graph.lock import RunLock
from hiredra.lock import LockHeld
from hiredra.lock import RunLock as SharedLock

REPO = Path(__file__).resolve().parents[1]

# a child process that takes the lock, announces it, then holds it until killed
_HOLDER = """
import sys, time
sys.path.insert(0, {repo!r})
from hiredra.graph.lock import RunLock
import os
with RunLock(what="holder-process", url={url!r}):
    print("LOCKED", os.getpid(), flush=True)
    time.sleep(120)
"""


class _Holder:
    """A child process that really holds the lock. `pid` is the pid INSIDE the child (the one the lock
    note records): on Windows the venv's python.exe is a launcher, so Popen.pid is a different process."""

    def __init__(self, url):
        code = _HOLDER.format(repo=str(REPO), url=url)
        self.proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        line = self.proc.stdout.readline()  # blocks until the child really holds the lock
        assert line.startswith("LOCKED "), f"holder did not start: {line!r}"
        self.pid = int(line.split()[1])

    def kill(self):
        """Hard-kill the process that holds the lock (a crash), plus its launcher."""
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(self.pid), "/F", "/T"], capture_output=True)
        else:
            os.kill(self.pid, signal.SIGKILL)
        self.proc.kill()
        self.proc.wait(timeout=30)


def test_the_lock_can_be_taken_and_released_and_retaken(pg_url):
    with RunLock(what="sync", url=pg_url):
        pass
    with RunLock(what="sync", url=pg_url):
        pass


def test_a_second_holder_in_the_same_process_is_refused_with_who_holds_it(pg_url):
    with RunLock(what="sync", url=pg_url):
        with pytest.raises(LockHeld) as info:
            with RunLock(what="consistency", url=pg_url):
                pytest.fail("must not get the lock twice")
    msg = str(info.value)
    assert f"pid {os.getpid()}" in msg and "running 'sync'" in msg and "graph maintenance" in msg


def test_the_lock_is_released_when_the_body_raises(pg_url):
    with pytest.raises(RuntimeError):
        with RunLock(url=pg_url):
            raise RuntimeError("boom")
    with RunLock(url=pg_url):  # not stuck
        pass


def test_locks_are_independent_by_name(pg_url):
    with RunLock(what="sync", url=pg_url):  # the graph lock
        with SharedLock("llm", what="extraction", kind="flash-lite", url=pg_url):
            pass  # extraction never waits for graph maintenance


def test_a_long_what_still_names_the_holder(pg_url):
    with RunLock(what="graph maintenance: sync " + "x" * 80, url=pg_url):
        with pytest.raises(LockHeld, match=r"pid \d+, running 'graph maintenance: sync x+', since 20\d\d-"):
            with RunLock(url=pg_url):
                pass


def test_a_real_second_process_holding_the_lock_blocks_us(pg_url):
    holder = _Holder(pg_url)
    try:
        with pytest.raises(LockHeld) as info:
            with RunLock(what="sync", url=pg_url):
                pytest.fail("must not get the lock while another process holds it")
        assert f"pid {holder.pid}" in str(info.value) and "holder-process" in str(info.value)
    finally:
        holder.kill()


def test_the_lock_is_released_when_the_holder_is_killed(pg_url):
    """No stale locks: killing the holder (a crash) ends its session, and the server frees the lock."""
    holder = _Holder(pg_url)
    with pytest.raises(LockHeld):
        with RunLock(url=pg_url):
            pass
    holder.kill()
    deadline = time.time() + 15
    while True:
        try:
            with RunLock(what="after-crash", url=pg_url):
                break
        except LockHeld:
            assert time.time() < deadline, "the server never released the killed holder's lock"
            time.sleep(0.2)
