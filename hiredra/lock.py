"""A run lock so two quota-spending runs can't overlap (moved from hiredra/graph/lock.py, M0.1; a Postgres
advisory lock since plan unit DB3b, decision D1.10).

Why: two runs started at once could both pass the "is there budget left?" check and double-spend the daily
quota. Graph maintenance holds the ``graph`` lock (the freeze rule); every process that spends flash-lite quota
holds the ``llm`` lock.

How: ``pg_try_advisory_lock`` on a connection of the lock's own, opened to the run's database. The server drops
the lock the moment that connection ends -- the process finishing, crashing or being killed -- so a lock can
never go stale and there is no clean-up step to get wrong (what the OS file lock gave before). The holder's
note (what, pid, since) is the connection's ``application_name``, so a refused run can say what is in the way.
Locks are per database: a test database never blocks the real one.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import psycopg

from hiredra import pg


class LockHeld(RuntimeError):
    """Another run holds the lock."""


def _holder(conn: psycopg.Connection, key: str) -> str:
    """Who holds the lock, from the holding connection's application_name (best effort)."""
    row = conn.execute(
        "SELECT a.application_name FROM pg_locks l JOIN pg_stat_activity a ON a.pid = l.pid "
        "WHERE l.locktype = 'advisory' AND l.granted AND l.objid::bigint = (hashtext(%s)::bigint & 4294967295) "
        "AND a.pid <> pg_backend_pid() LIMIT 1",
        (key,),
    ).fetchone()
    parts = (row[0] if row else "").split("|")
    if len(parts) != 4 or parts[0] != key:
        return "another process (details unavailable)"
    return f"pid {parts[1]}, running '{parts[2]}', since {parts[3]}"


class RunLock:
    """``with RunLock("graph", what="sync", url=args.db): ...``  Raises ``LockHeld`` if taken.

    ``kind`` names the family of runs that share this lock, for the refusal message. ``url`` is the run's
    database (default: ``HIREDRA_DATABASE_URL``).
    """

    def __init__(self, name: str, what: str = "run", kind: str = "LLM", url: str | None = None):
        self.key = f"hiredra:{name}"
        self.what = what
        self.kind = kind
        self.url = url
        self._conn: psycopg.Connection | None = None

    def __enter__(self) -> RunLock:
        # key|pid|what|since: Postgres keeps 63 bytes of application_name, so the note is compact and `what` gives way
        head = f"{self.key}|{os.getpid()}|"
        tail = f"|{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}"
        what = self.what.replace("|", "/")[: 63 - len(head) - len(tail)]
        conn = psycopg.connect(self.url or pg.database_url(), autocommit=True, application_name=head + what + tail)
        if not conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (self.key,)).fetchone()[0]:
            holder = _holder(conn, self.key)
            conn.close()
            raise LockHeld(f"another {self.kind} run is in progress ({holder}); nothing was done")
        self._conn = conn
        return self

    def __exit__(self, *exc) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()  # ending the session releases the lock
