"""Write fetched jobs to SQLite.

Thin now: dedup (build sub-slice 7) runs before this, and the orchestrator
(sub-slice 8) calls it. See docs/sourcing.md sec. 9-10. Also the one-off
``backfill_slugs`` that gives pre-v5 rows their provenance (SL1).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, Provider, company_slug
from hiredra.sourcing.registry import REGISTRY, route
from hiredra.storage import db


@dataclass(frozen=True, slots=True)
class PersistResult:
    inserted: int  # job_ids not seen before
    reseen: int  # job_ids already stored -> re-sighted (last_seen_at bumped, nothing regressed)

    @property
    def total(self) -> int:
        return self.inserted + self.reseen


def persist_jobs(conn: sqlite3.Connection, jobs: Iterable[Job], *, run_id: str | None = None) -> PersistResult:
    """Upsert each job (stamping new rows with ``run_id``). New vs re-sighted."""
    inserted = reseen = 0
    for job in jobs:
        if db.job_exists(conn, job.job_id):
            reseen += 1
        else:
            inserted += 1
        db.upsert_job(conn, job.to_dict(), first_seen_run=run_id)
    return PersistResult(inserted=inserted, reseen=reseen)


@dataclass
class SlugBackfillStats:
    rows: int = 0  # ATS rows still missing a slug or a tag
    from_url: int = 0  # slug read from the job URL by the provider's own resolver
    from_list: int = 0  # slug from the company list, matched by provider + exact name
    from_roster: int = 0  # the company name IS a slug on the open-jobs roster (the old random list)
    unresolved: int = 0  # slug left NULL; liveness falls back to the URL
    tagged: int = 0  # list_tag set from the list, matched by provider + slug


def backfill_slugs(
    conn: sqlite3.Connection,
    entries: Iterable[CompanyEntry],
    roster: set[tuple[str, str]] = frozenset(),
    registry: Sequence[Provider] = REGISTRY,
) -> SlugBackfillStats:
    """Fill ``company_slug`` / ``list_tag`` on stored ATS rows that predate v5 (SL1).

    Slug, first rule that works: the job URL through ``company_slug`` (the provider's own
    resolver); then the company list, when exactly one board has that provider + name; then
    the open-jobs roster, because the old random list used the roster slug as the company
    name. Tag: the list's ``extra.tag`` for that provider + slug. Fills only NULLs.
    """
    providers = {p.name: p for p in registry}
    by_name: dict[tuple[str, str], set[str]] = {}
    tags: dict[tuple[str, str], str] = {}
    for entry in entries:
        provider = route(entry, registry)
        slug = company_slug(provider, entry) if provider else None
        if not slug:
            continue
        by_name.setdefault((provider.name, entry.name), set()).add(slug)
        tag = entry.extra.get("tag") if entry.extra else None
        if tag:
            tags[(provider.name, slug.lower())] = str(tag)

    stats = SlugBackfillStats()
    rows = conn.execute(
        f"SELECT job_id, source, url, company, company_slug FROM jobs "
        f"WHERE source IN ({', '.join('?' for _ in providers)}) AND (company_slug IS NULL OR list_tag IS NULL)",
        list(providers),
    ).fetchall()
    for job_id, source, url, company, slug in rows:
        stats.rows += 1
        if not slug:
            slug = company_slug(providers[source], CompanyEntry(name=company, careers_url=url, provider=source))
            if slug:
                stats.from_url += 1
            elif len(by_name.get((source, company), ())) == 1:
                (slug,) = by_name[(source, company)]
                stats.from_list += 1
            elif (source, company.lower()) in roster:
                slug = company
                stats.from_roster += 1
            else:
                stats.unresolved += 1
                continue
        tag = tags.get((source, slug.lower()))
        stats.tagged += tag is not None
        db.fill_provenance(conn, [job_id], company_slug=slug, list_tag=tag)
    return stats
