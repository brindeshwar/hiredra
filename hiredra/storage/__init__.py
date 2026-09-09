"""Hiredra storage layer — SQLite persistence for jobs, status, and outcomes.

See ``hiredra/storage/schema.sql`` for the schema and ``docs/sourcing.md`` §10.
"""

from hiredra.storage.db import (
    DEFAULT_DB_PATH,
    OUTCOMES,
    SCHEMA_VERSION,
    STATUSES,
    add_note,
    connect,
    get_events,
    get_job,
    init_db,
    iter_jobs,
    job_exists,
    record_outcome,
    set_status,
    status_counts,
    upsert_job,
)

__all__ = [
    "DEFAULT_DB_PATH",
    "OUTCOMES",
    "SCHEMA_VERSION",
    "STATUSES",
    "add_note",
    "connect",
    "get_events",
    "get_job",
    "init_db",
    "iter_jobs",
    "job_exists",
    "record_outcome",
    "set_status",
    "status_counts",
    "upsert_job",
]
