"""Sinks -- what the orchestrator does with each batch of fetched jobs.

The orchestrator core is sink-agnostic: it calls ``sink.consume(jobs)`` per
company (and once for the JobSpy batch) and ``sink.stats()`` at the end. A
company batch also carries its provenance: the board slug and the list tag.

* ``ScoringSink`` -- holds the database connection: dedup against the store, persist
  the new jobs (stamped with ``run_id``), ``touch`` the re-sighted ones, and fill
  ``company_slug`` / ``list_tag`` on both where still NULL (v5).

The ``Sink`` protocol stays although ``ScoringSink`` is the only real sink: the orchestrator's unit tests drive it
with a fake one.
"""

from __future__ import annotations

import psycopg
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from hiredra.sourcing.dedup import load_seen_index, SeenIndex, dedup
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
    def consume(
        self,
        jobs: list[Job],
        *,
        company_slug: str | None = None,
        list_tag: str | None = None,
        provenance: str = "orphan",
    ) -> None: ...
    def stats(self) -> SinkStats: ...


class ScoringSink:
    """Persisting sink. One database connection, owned by the caller (never closed here)."""

    def __init__(self, conn: psycopg.Connection, run_id: str) -> None:
        self._conn = conn
        self._run_id = run_id
        self._fetched = 0
        self._collapsed = 0
        self._matched = 0
        self._new = 0
        self._touched = 0
        self._index: SeenIndex | None = None

    def consume(
        self,
        jobs: list[Job],
        *,
        company_slug: str | None = None,
        list_tag: str | None = None,
        provenance: str = "orphan",
    ) -> None:
        self._fetched += len(jobs)
        if self._index is None:  # the store read once per run, not once per board (6.4 s on 118k jobs)
            self._index = load_seen_index(self._conn)
        result = dedup(self._conn, jobs, index=self._index)
        self._collapsed += result.collapsed_in_run
        self._matched += result.matched_in_store
        pr = persist_jobs(self._conn, result.new, run_id=self._run_id)
        self._new += pr.inserted
        for job in result.new:
            self._index.add(job.job_id, job.url, job.apply_url, job.company, job.title, job.source, job.location)
        for job, aggregator_copy in result.supersedes:  # D6.4: the company's own copy is the role's canonical one
            db.mark_duplicate(self._conn, aggregator_copy, of=job.job_id)
        db.touch_jobs(self._conn, list(result.seen), run_id=self._run_id)  # the board's re-sightings, batched
        self._touched += len(result.seen)
        db.fill_provenance(
            self._conn,
            [j.job_id for j in result.new] + list(result.seen),
            company_slug=company_slug,
            list_tag=list_tag,
            provenance=provenance,
            sources={j.source for j in jobs},  # the board's own jobs: a matched copy from elsewhere stays put
        )

    def stats(self) -> SinkStats:
        return SinkStats(
            fetched=self._fetched,
            collapsed_in_run=self._collapsed,
            matched_in_store=self._matched,
            persisted_new=self._new,
            touched=self._touched,
        )
