"""A run lock so two graph maintenance commands can't overlap (build 4.3 hardening).

Why: `sync` and `consistency` spend Google's daily request budget, and the local
budget counter is a plain file, so two runs started at once could both pass the
"is there budget left?" check and double-spend it. `merges --apply` and `harvest`
write the same tables `sync` writes vectors into, so they must not overlap it either.

How: an OPERATING-SYSTEM file lock (`msvcrt.locking` on Windows, `fcntl.flock`
elsewhere), not a "pid file". The OS drops the lock the moment the holding
process ends -- even if it crashes or is killed -- so a lock can never go stale
and there is no clean-up step to get wrong. The lock file itself is left on disk;
that is normal.

The lock code lives in ``hiredra/lock.py`` (moved there in M0.1 so extraction shares it);
this module keeps the graph lock file and its "graph maintenance" wording.

File layout: byte 0 is the locked byte; from byte 1 on is a small JSON note (who
holds it, since when) so a refused run can say what is in the way. On Windows a
locked byte range can't be read, which is why the note lives outside it.
"""

from __future__ import annotations

from pathlib import Path

from hiredra import lock as _lock
from hiredra.lock import LockHeld  # noqa: F401 -- re-exported: graph code and tests import it from here

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK_PATH = _REPO_ROOT / "data" / "graph.lock"


class RunLock(_lock.RunLock):
    """The shared ``hiredra.lock.RunLock`` on the graph's lock file, with the graph wording."""

    def __init__(self, path: Path | str = DEFAULT_LOCK_PATH, what: str = "graph maintenance"):
        super().__init__(path, what=what, kind="graph maintenance")
