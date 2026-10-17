"""Sinks -- what the orchestrator does with each batch of fetched jobs.

The orchestrator core is sink-agnostic: it calls ``sink.consume(jobs)`` per
company (and once for the JobSpy batch) and ``sink.stats()`` at the end.

* ``ScoringSink`` -- holds the SQLite connection: dedup against the store, persist
  the new jobs (stamped with ``run_id``), ``touch`` the re-sighted ones.
* ``BackfillSink`` -- holds nothing DB-ish; intra-run ``collapse`` only, then
  hands jobs to skill extraction and discards them. Skill extraction is wired at
  graph-build (step 4); for now it just accumulates the collapsed jobs so a
  backfill dry-run works.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hiredra.sourcing.dedup import collapse, dedup
from hiredra.sourcing.job import Job
from hiredra.sourcing.persist import persist_jobs
from hiredra.storage import db


@dataclass(frozen=True, slots=True)
class SinkStats:
    fetched: int = 0  # raw Job count handed in
    collapsed_in_run: int = 0
    matched_in_store: int = 0
    persisted_new: int = 0
    touched: int = 0


@runtime_checkable
class Sink(Protocol):
    def consume(self, jobs: list[Job]) -> None: ...
    def stats(self) -> SinkStats: ...


class ScoringSink:
    """Persisting sink. One SQLite connection, owned by the caller (never closed here)."""

    def __init__(self, conn: sqlite3.Connection, run_id: str) -> None:
        self._conn = conn
        self._run_id = run_id
        self._fetched = 0
        self._collapsed = 0
        self._matched = 0
        self._new = 0
        self._touched = 0

    def consume(self, jobs: list[Job]) -> None:
        self._fetched += len(jobs)
        result = dedup(self._conn, jobs)
        self._collapsed += result.collapsed_in_run
        self._matched += result.matched_in_store
        pr = persist_jobs(self._conn, result.new, run_id=self._run_id)
        self._new += pr.inserted
        for job_id in result.seen:
            db.touch_job(self._conn, job_id, run_id=self._run_id)
        self._touched += len(result.seen)

    def stats(self) -> SinkStats:
        return SinkStats(
            fetched=self._fetched,
            collapsed_in_run=self._collapsed,
            matched_in_store=self._matched,
            persisted_new=self._new,
            touched=self._touched,
        )


class BackfillSink:
    """Non-persisting sink. Real skill extraction is wired at graph-build (step 4)."""

    def __init__(self) -> None:
        self.jobs: list[Job] = []
        self._fetched = 0
        self._collapsed = 0

    def consume(self, jobs: list[Job]) -> None:
        self._fetched += len(jobs)
        kept, collapsed = collapse(list(jobs))
        self._collapsed += collapsed
        self.jobs.extend(kept)
        # TODO(step 4): hand `kept` to skill extraction, then discard.

    def stats(self) -> SinkStats:
        return SinkStats(fetched=self._fetched, collapsed_in_run=self._collapsed)
