"""One-off repair (2026-10-26): Lever descriptions were stored from ``descriptionPlain``, which holds
the intro only, so every Lever job's requirements were missing (docs/freehire-merge/04-verification.md).

The storage layer never overwrites a stored job's content on a re-fetch (a flaky fetch must not corrupt a
good row), so this script does the repair explicitly:

1. Re-fetch every Lever board already in the database with the fixed adapter (4 boards at a time, through
   the normal polite HttpClient).
2. Where the new body differs from the stored one, update ``description`` and add a note to the job.
3. Re-run the funnel's three text stages (location, seniority, role type) on each repaired job and apply
   any change, in either direction:
   - now dropped  -> status ``dropped`` with the new reason (whatever stage it was at);
   - now kept, was dropped -> back to ``filtered`` (it was just seen live on its board);
   - now kept, was extracted/scored -> back to ``filtered``, so extraction re-reads the full body.
4. Jobs no longer on their board (closed since) can't be repaired; they are counted and left as they are.

Back up first: the script copies data/hiredra.db to data/hiredra.db.pre-lever-repair.bak.
    PYTHONPATH=. uv run python scripts/repair_lever_descriptions.py
"""

from __future__ import annotations

import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from hiredra.filter import load_filter_config, passes_location, passes_role_type, passes_seniority_gate
from hiredra.sourcing.http import FetchError, HttpClient, HttpConfig
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers.lever import LeverProvider
from hiredra.storage import add_note, connect, set_status
from hiredra.storage.db import set_years_required

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "hiredra.db"
WORKERS = 4


def boards(conn) -> list[tuple[str, str, str]]:
    """(slug, company name, careers url) for every Lever board in the database."""
    rows = conn.execute(
        "SELECT company_slug, MIN(company), MIN(url) FROM jobs WHERE source = 'lever' AND company_slug IS NOT NULL "
        "GROUP BY company_slug"
    ).fetchall()
    out = []
    for slug, company, url in rows:
        host = urlsplit(url).hostname or "jobs.lever.co"
        host = host if host.endswith("lever.co") else "jobs.lever.co"
        out.append((slug, company, f"https://{host}/{slug}"))
    return out


def main() -> None:
    shutil.copy2(DB, DB.with_name("hiredra.db.pre-lever-repair.bak"))
    conn = connect(DB)
    cfg = load_filter_config()
    provider = LeverProvider()
    todo = boards(conn)
    print(f"{len(todo)} Lever boards to re-fetch", flush=True)

    fetched: dict[str, str] = {}
    failed_boards = 0
    with HttpClient(HttpConfig()) as http:
        ctx = FetchContext(http=http)

        def fetch(board):
            slug, company, careers = board
            try:
                return provider.fetch(CompanyEntry(name=company, careers_url=careers, provider="lever"), ctx)
            except FetchError:
                return None

        with ThreadPoolExecutor(WORKERS) as pool:
            for i, jobs in enumerate(pool.map(fetch, todo), 1):
                if jobs is None:
                    failed_boards += 1
                else:
                    for j in jobs:
                        if j.description:
                            fetched[j.job_id] = j.description
                if i % 100 == 0:
                    print(f"  {i}/{len(todo)} boards", flush=True)

    stats = Counter()
    transitions = Counter()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = conn.execute(
        "SELECT job_id, title, location, description, status FROM jobs WHERE source = 'lever'"
    ).fetchall()
    for job_id, title, location, old_desc, status in rows:
        new_desc = fetched.get(job_id)
        if new_desc is None:
            stats["not on its board any more"] += 1
            continue
        if new_desc == (old_desc or ""):
            stats["unchanged"] += 1
            continue
        stats["repaired"] += 1
        with conn:
            conn.execute("UPDATE jobs SET description = ?, updated_at = ? WHERE job_id = ?", (new_desc, now, job_id))
        add_note(conn, job_id, f"repaired: Lever full body ({len(old_desc or '')} -> {len(new_desc)} chars)")

        loc = passes_location(title, location, new_desc, cfg)
        reason = None if loc.keep else f"location: {loc.reason}"
        if reason is None:
            sen = passes_seniority_gate(title, new_desc, cfg)
            set_years_required(conn, job_id, sen.years)
            reason = None if sen.keep else f"seniority: {sen.reason}"
        if reason is None:
            role = passes_role_type(title, cfg)
            reason = None if role.keep else f"role_type: {role.reason}"

        if reason is not None and status != "dropped":
            set_status(conn, job_id, "dropped", detail=f"{reason} (after the Lever full-body repair)")
            transitions[f"{status} -> dropped"] += 1
        elif reason is None and status == "dropped":
            set_status(conn, job_id, "filtered", detail="kept after the Lever full-body repair")
            transitions["dropped -> filtered"] += 1
        elif reason is None and status in ("extracted", "scored"):
            set_status(conn, job_id, "filtered", detail="re-extract: description repaired (Lever full body)")
            transitions[f"{status} -> filtered (re-extract)"] += 1
        elif reason is not None and status == "dropped":
            transitions["dropped, still dropped"] += 1
    conn.close()
    print(f"boards failed to fetch: {failed_boards}")
    print("jobs:", dict(stats))
    print("status changes:", dict(transitions))


if __name__ == "__main__":
    main()
