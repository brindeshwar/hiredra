"""Tests for hiredra.graph.lock: the run lock (build 4.3 hardening).

Includes a REAL second process, because a lock that only works inside one
process proves nothing about two concurrent `sync` runs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from hiredra.graph.lock import LockHeld, RunLock

# a child process that takes the lock, announces it, then holds it until killed
_HOLDER = """
import sys, time
sys.path.insert(0, {repo!r})
from hiredra.graph.lock import RunLock
import os
with RunLock({path!r}, what="holder-process"):
    print("LOCKED", os.getpid(), flush=True)
    time.sleep(120)
"""


class _Holder:
    """A child process that really holds the lock. `pid` is the pid INSIDE the child (the one the lock
    note records): on Windows the venv's python.exe is a launcher, so Popen.pid is a different process."""

    def __init__(self, path):
        code = _HOLDER.format(repo="D:/Projects/hiredra", path=str(path))
        self.proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        line = self.proc.stdout.readline()          # blocks until the child really holds the lock
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


def _spawn_holder(path):
    return _Holder(path)


def _read_note(path):
    """Read the file the way a competing process can: skipping the locked byte 0."""
    with open(path, "rb") as fh:
        fh.seek(1)
        return fh.read()


def test_the_lock_can_be_taken_and_released_and_retaken(tmp_path):
    p = tmp_path / "g.lock"
    with RunLock(p, what="sync"):
        pass
    with RunLock(p, what="sync"):
        pass


def test_a_second_holder_in_the_same_process_is_refused_with_who_holds_it(tmp_path):
    p = tmp_path / "g.lock"
    with RunLock(p, what="sync"):
        with pytest.raises(LockHeld) as info:
            with RunLock(p, what="consistency"):
                pytest.fail("the second run must not get the lock")
    msg = str(info.value)
    assert "another graph maintenance run is in progress" in msg and "nothing was done" in msg
    assert f"pid {os.getpid()}" in msg and "running 'sync'" in msg


def test_the_lock_is_released_when_the_body_raises(tmp_path):
    p = tmp_path / "g.lock"
    with pytest.raises(ZeroDivisionError):
        with RunLock(p):
            1 / 0
    with RunLock(p):                                   # not stuck
        pass


def test_the_lock_directory_is_created_if_missing(tmp_path):
    with RunLock(tmp_path / "nested" / "deeper" / "g.lock"):
        assert (tmp_path / "nested" / "deeper" / "g.lock").exists()


def test_the_note_is_cleared_on_release_so_nothing_stale_is_left_behind(tmp_path):
    p = tmp_path / "g.lock"
    with RunLock(p, what="sync"):
        assert b'"what": "sync"' in _read_note(p)      # the note is there while held
    assert p.read_bytes() == b"L" and _read_note(p) == b""   # only the sentinel byte remains


def test_a_real_second_process_holding_the_lock_blocks_us(tmp_path):
    p = tmp_path / "g.lock"
    holder = _spawn_holder(p)
    try:
        with pytest.raises(LockHeld) as info:
            with RunLock(p, what="sync"):
                pytest.fail("must not get the lock while another process holds it")
        assert f"pid {holder.pid}" in str(info.value) and "holder-process" in str(info.value)
    finally:
        holder.kill()


def test_the_lock_is_released_by_the_os_when_the_holder_is_killed(tmp_path):
    """No stale locks: killing the holder (a crash) frees the lock with no clean-up step."""
    p = tmp_path / "g.lock"
    holder = _spawn_holder(p)
    with pytest.raises(LockHeld):
        with RunLock(p):
            pass
    holder.kill()
    deadline = time.time() + 15
    while True:
        try:
            with RunLock(p, what="after-crash"):
                break
        except LockHeld:
            assert time.time() < deadline, "the OS never released the killed holder's lock"
            time.sleep(0.2)
