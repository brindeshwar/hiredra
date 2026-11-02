"""``RunStore`` -- the orchestrator's bookkeeping side of the operational DB.

Distinct from the ``Sink`` (which owns *job* persistence). The ``RunStore`` owns:

* the ``runs`` row lifecycle (``begin`` / ``checkpoint`` / ``finish``), and the
  fresh-start stale sweep;
* cross-run cooldown durability (``load_cooldowns`` / ``save_cooldown``);
* each board's result and lifecycle (``board_result``, build step 5: D5.3-D5.5).

``run()`` imports no database driver -- it drives a ``RunStore`` and a ``Sink``,
both injected. Tests use ``FakeRunStore``.
"""

from __future__ import annotations

import json
import psycopg
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


@dataclass(frozen=True, slots=True)
class BoardResult:
    """What one board's crawl did (a ``board_crawls`` row, D5.4). ``complete``: the crawl succeeded and saw the board's
    whole listing, the one case in which its unlisted jobs may be closed (D5.3)."""

    board_id: int
    status: str  # ok | blocked | gone | error | unroutable
    jobs: int = 0
    new: int = 0
    seen: int = 0
    complete: bool = False
    detail: str | None = None
    seconds: float | None = None


class RunStore(Protocol):
    def begin(self, kind: str, config_name: str | None, *, resume: bool) -> RunHandle: ...
    def checkpoint(self, run_id: str, *, companies_done: set[str], jobspy_done: bool) -> None: ...
    def finish(self, run_id: str, *, status: str, stats: SinkStats, report_json: str) -> None: ...
    def load_cooldowns(self) -> dict[str, float]: ...
    def save_cooldown(self, key: str, until: float, reason: str | None) -> None: ...
    def board_result(self, run_id: str, result: BoardResult) -> None: ...


class PgRunStore:
    """Backed by the operational database (Postgres).

    Does NOT own the connection -- the CLI opens it once and closes it once in a
    ``finally``. ``ScoringSink`` shares this same connection.
    """

    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def begin(self, kind: str, config_name: str | None, *, resume: bool) -> RunHandle:
        if resume:
            row = db.unfinished_run(self._conn, kind, config_name)
            if row is not None:
                progress = row["detail"] or {}
                return RunHandle(
                    run_id=row["run_id"],
                    started_at=row["started_at"].isoformat(
                        timespec="seconds"
                    ),  # timestamptz -> the ISO text runs carry
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
        # Always on the main thread: the orchestrator queues a worker's trip and writes it here (D4.1).
        db.save_cooldown(self._conn, key, until, reason)

    def board_result(self, run_id: str, result: BoardResult) -> None:
        db.record_board_run(self._conn, run_id, result)


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
        self.board_results: list[BoardResult] = []

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

    def board_result(self, run_id, result):
        self.board_results.append(result)
