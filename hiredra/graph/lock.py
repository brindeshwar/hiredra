"""A run lock so two graph maintenance commands can't overlap (build 4.3 hardening).

Why: `merges --apply`, `harvest`, `overlay` and `sync` write the skill tables, so they must not overlap; scoring
holds the lock too (the freeze rule: the graph must not change mid-run).

The lock code lives in ``hiredra/lock.py`` (moved there in M0.1 so extraction shares it; a Postgres advisory lock
since DB3b); this module keeps the graph lock's name and its "graph maintenance" wording.
"""

from __future__ import annotations

from hiredra import lock as _lock

LOCK_NAME = "graph"


class RunLock(_lock.RunLock):
    """The shared ``hiredra.lock.RunLock`` on the graph lock, with the graph wording."""

    def __init__(self, what: str = "graph maintenance", url: str | None = None):
        super().__init__(LOCK_NAME, what=what, kind="graph maintenance", url=url)
