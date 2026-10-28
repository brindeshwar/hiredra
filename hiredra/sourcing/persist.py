"""Write fetched jobs to the store.

Thin now: dedup (build sub-slice 7) runs before this, and the orchestrator
(sub-slice 8) calls it. See docs/sourcing.md sec. 9-10. (The one-off
``backfill_slugs`` for pre-v5 SQLite rows went with the move to Postgres: every
imported ATS job already sits on its board.)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from hiredra.sourcing.job import Job
from hiredra.storage import db


@dataclass(frozen=True, slots=True)
class PersistResult:
    inserted: int  # job_ids not seen before
    reseen: int  # job_ids already stored -> re-sighted (last_seen_at bumped, nothing regressed)

    @property
    def total(self) -> int:
        return self.inserted + self.reseen


def persist_jobs(conn, jobs: Iterable[Job], *, run_id: str | None = None) -> PersistResult:
    """Upsert each job (stamping new rows with ``run_id``). New vs re-sighted."""
    inserted = reseen = 0
    for job in jobs:
        if db.job_exists(conn, job.job_id):
            reseen += 1
        else:
            inserted += 1
        db.upsert_job(conn, job.to_dict(), first_seen_run=run_id)
    return PersistResult(inserted=inserted, reseen=reseen)
