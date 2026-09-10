"""Write fetched jobs to SQLite.

Thin now: dedup (build sub-slice 7) runs before this, and the orchestrator
(sub-slice 8) calls it. See docs/sourcing.md sec. 9-10.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass

from hiredra.sourcing.job import Job
from hiredra.storage import db


@dataclass(frozen=True, slots=True)
class PersistResult:
    inserted: int  # job_ids not seen before
    reseen: int    # job_ids already stored -> re-sighted (last_seen_at bumped, nothing regressed)

    @property
    def total(self) -> int:
        return self.inserted + self.reseen


def persist_jobs(conn: sqlite3.Connection, jobs: Iterable[Job]) -> PersistResult:
    """Upsert each job. Returns how many were new vs re-sighted."""
    inserted = reseen = 0
    for job in jobs:
        if db.job_exists(conn, job.job_id):
            reseen += 1
        else:
            inserted += 1
        db.upsert_job(conn, job.to_dict())
    return PersistResult(inserted=inserted, reseen=reseen)
