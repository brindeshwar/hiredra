"""A run lock so two quota-spending runs can't overlap (moved from hiredra/graph/lock.py, M0.1).

Why: the local daily budget counter is a plain file, so two runs started at once could both
pass the "is there budget left?" check and double-spend it. Graph maintenance holds
``data/graph.lock``; every process that spends flash-lite quota holds ``data/llm.lock``.

How: an OPERATING-SYSTEM file lock (`msvcrt.locking` on Windows, `fcntl.flock`
elsewhere), not a "pid file". The OS drops the lock the moment the holding
process ends -- even if it crashes or is killed -- so a lock can never go stale
and there is no clean-up step to get wrong. The lock file itself is left on disk;
that is normal.

File layout: byte 0 is the locked byte; from byte 1 on is a small JSON note (who
holds it, since when) so a refused run can say what is in the way. On Windows a
locked byte range can't be read, which is why the note lives outside it.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
    import msvcrt

    def _try_lock(fh) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(fh) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
else:  # pragma: no cover - exercised on POSIX only
    import fcntl

    def _try_lock(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fh) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class LockHeld(RuntimeError):
    """Another run holds the lock."""


def _read_holder(path: Path) -> str:
    """Who holds the lock, from the note after the locked byte (best effort)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(1)
            note = json.loads(fh.read().decode("utf-8") or "{}")
        return f"pid {note.get('pid', '?')}, running '{note.get('what', '?')}', since {note.get('since', '?')}"
    except (OSError, ValueError):
        return "another process (details unavailable)"


class RunLock:
    """``with RunLock(path, what="sync"): ...``  Raises ``LockHeld`` if taken.

    ``kind`` names the family of runs that share this lock, for the refusal message.
    """

    def __init__(self, path: Path | str, what: str = "run", kind: str = "LLM"):
        self.path = Path(path)
        self.what = what
        self.kind = kind
        self._fh = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            fh.seek(0, os.SEEK_END)
            if fh.tell() == 0:
                fh.write(b"L")  # byte 0: the byte we lock
                fh.flush()
            _try_lock(fh)
        except OSError:
            fh.close()
            raise LockHeld(
                f"another {self.kind} run is in progress ({_read_holder(self.path)}); nothing was done"
            ) from None
        # we hold it: leave a note after the locked byte (append mode writes at EOF, so cut back to 1 byte first)
        fh.truncate(1)
        note = {
            "pid": os.getpid(),
            "what": self.what,
            "since": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        fh.write(json.dumps(note).encode("utf-8"))
        fh.flush()
        self._fh = fh
        return self

    def __exit__(self, *exc) -> None:
        fh, self._fh = self._fh, None
        if fh is None:
            return
        try:
            fh.truncate(1)  # clear the note; the file itself stays
            fh.flush()
            _unlock(fh)
        finally:
            fh.close()
