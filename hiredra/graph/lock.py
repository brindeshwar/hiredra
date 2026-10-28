"""A run lock so two graph maintenance commands can't overlap (build 4.3 hardening).

Why: `sync` and `consistency` spend Google's daily request budget, so two runs started at once could both pass the
"is there budget left?" check and double-spend it. `merges --apply` and `harvest` write the same tables `sync`
writes vectors into, so they must not overlap it either; scoring holds it too (the freeze rule).

The lock code lives in ``hiredra/lock.py`` (moved there in M0.1 so extraction shares it; a Postgres advisory lock
since DB3b); this module keeps the graph lock's name and its "graph maintenance" wording.
"""

from __future__ import annotations

from hiredra import lock as _lock
from hiredra.lock import LockHeld  # noqa: F401 -- re-exported: graph code and tests import it from here

LOCK_NAME = "graph"


class RunLock(_lock.RunLock):
    """The shared ``hiredra.lock.RunLock`` on the graph lock, with the graph wording."""

    def __init__(self, what: str = "graph maintenance", url: str | None = None):
        super().__init__(LOCK_NAME, what=what, kind="graph maintenance", url=url)
