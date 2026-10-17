"""``RunStore`` -- the orchestrator's bookkeeping side of the operational DB.

Distinct from the ``Sink`` (which owns *job* persistence, and for a backfill
points at the graph store, not the jobs DB). The ``RunStore`` owns:

* the ``runs`` row lifecycle (``begin`` / ``checkpoint`` / ``finish``), and the
  fresh-start stale sweep;
* cross-run cooldown durability (``load_cooldowns`` / ``save_cooldown``).

``run()`` imports no ``sqlite3`` -- it drives a ``RunStore`` and a ``Sink``,
both injected. Tests use ``FakeRunStore``.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Protocol

from hiredra.sourcing.sink import SinkStats
from hiredra.storage import db


@dataclass(frozen=True, slots=True)
class RunHandle:
    run_id: str
    started_at: str = ""
    companies_done: set[str] = field(default_factory=set)
    jobspy_done: bool = False
    resumed: bool = False


class RunStore(Protocol):
    def begin(self, kind: str, config_name: str | None, *, resume: bool) -> RunHandle: ...
    def checkpoint(self, run_id: str, *, companies_done: set[str], jobspy_done: bool) -> None: ...
    def finish(self, run_id: str, *, status: str, stats: SinkStats, report_json: str) -> None: ...
    def load_cooldowns(self) -> dict[str, float]: ...
    def save_cooldown(self, key: str, until: float, reason: str | None) -> None: ...


class SqliteRunStore:
    """Backed by the operational DB (``data/hiredra.db``).

    Does NOT own the connection -- the CLI opens it once and closes it once in a
    ``finally``. ``ScoringSink`` shares this same connection.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def begin(self, kind: str, config_name: str | None, *, resume: bool) -> RunHandle:
        if resume:
            row = db.unfinished_run(self._conn, kind, config_name)
            if row is not None:
                progress = json.loads(row["detail"] or "{}")
                return RunHandle(
                    run_id=row["run_id"],
                    started_at=row["started_at"],
                    companies_done=set(progress.get("companies_done", [])),
                    jobspy_done=bool(progress.get("jobspy_done", False)),
                    resumed=True,
                )
        else:
            db.sweep_stale_runs(self._conn, kind, config_name)
        run_id = db.new_run_id()
        started_at = db.open_run(self._conn, run_id, kind=kind, config_name=config_name)
        return RunHandle(run_id=run_id, started_at=started_at)

    def checkpoint(self, run_id: str, *, companies_done: set[str], jobspy_done: bool) -> None:
        db.checkpoint_run(
            self._conn,
            run_id,
            detail=json.dumps({"companies_done": sorted(companies_done), "jobspy_done": jobspy_done}),
        )

    def finish(self, run_id: str, *, status: str, stats: SinkStats, report_json: str) -> None:
        db.finish_run(
            self._conn,
            run_id,
            status=status,
            fetched=stats.fetched,
            collapsed_in_run=stats.collapsed_in_run,
            matched_in_store=stats.matched_in_store,
            persisted_new=stats.persisted_new,
            touched=stats.touched,
            detail=report_json,
        )

    def load_cooldowns(self) -> dict[str, float]:
        return db.load_cooldowns(self._conn)

    def save_cooldown(self, key: str, until: float, reason: str | None) -> None:
        # See db.save_cooldown's note: same-thread only until fetch is threaded.
        db.save_cooldown(self._conn, key, until, reason)


class FakeRunStore:
    """In-memory ``RunStore`` for orchestrator unit tests -- no DB."""

    def __init__(
        self,
        *,
        resume_handle: RunHandle | None = None,
        cooldowns: dict[str, float] | None = None,
    ) -> None:
        self._resume_handle = resume_handle
        self._cooldowns = dict(cooldowns or {})
        self.swept = 0
        self.begun: list[tuple[str, str | None, bool]] = []
        self.checkpoints: list[dict] = []
        self.finished: list[dict] = []
        self.saved_cooldowns: list[tuple[str, float, str | None]] = []

    def begin(self, kind, config_name, *, resume):
        self.begun.append((kind, config_name, resume))
        if resume and self._resume_handle is not None:
            return self._resume_handle
        if not resume:
            self.swept += 1
        return RunHandle(run_id="TESTRUN-0000")

    def checkpoint(self, run_id, *, companies_done, jobspy_done):
        self.checkpoints.append({"run_id": run_id, "companies_done": set(companies_done), "jobspy_done": jobspy_done})

    def finish(self, run_id, *, status, stats, report_json):
        self.finished.append({"run_id": run_id, "status": status, "stats": stats})

    def load_cooldowns(self):
        return dict(self._cooldowns)

    def save_cooldown(self, key, until, reason):
        self.saved_cooldowns.append((key, until, reason))
