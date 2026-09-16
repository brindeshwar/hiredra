"""Sub-slice 2b: liveness recheck (build step 2b, docs/scoring.md sec 1).

The fourth and last 2b filter stage. Runs as a separate pass, after
``hiredra.filter.run_funnel``, over status='filtered' -- not folded into
run_funnel's per-job loop, since it needs to see the whole survivor set to
group jobs by board before fetching (one fetch per company, not one per
job). It's also the only 2b stage with a network cost, hence last and
separate: run_funnel itself stays pure CPU.

ATS-sourced jobs are grouped by board (via each provider's own
``_board_url`` -- the same resolver already used to fetch, so an EU Lever
host or any other provider-specific quirk is handled for free, not
re-derived here) and re-fetched once per group; a job absent from the
fresh listing, or a whole board that 404s, is dropped (``status='dropped'``,
``detail='liveness: gone'`` -- reusing the orchestrator's existing 'gone'
vocabulary for a confirmed removal, no new ``jobs.status`` value). JobSpy-
sourced jobs have no board to re-list -- deliberately NOT fetched directly
(that would mean scraping Indeed/LinkedIn postings one by one, reopening a
boundary this project has held since Path 3 was built); they're kept and
flagged ``add_note(..., "liveness: unchecked")`` instead, same convention
the location stage uses for a kept-by-default job. Anything ambiguous (a
block, a timeout, a non-404 error) also keeps the job unchanged -- an
inconclusive recheck is never evidence of death.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from hiredra.sourcing.http import HTTPError, HttpClient, HttpConfig
from hiredra.sourcing.provider import CompanyEntry, FetchContext, Provider
from hiredra.sourcing.registry import REGISTRY
from hiredra.storage import add_note, iter_jobs, set_status


@dataclass
class LivenessStats:
    checked: int = 0         # every status='filtered' job examined
    boards_checked: int = 0  # distinct company boards re-fetched (ATS jobs only)
    live: int = 0            # confirmed still in the board's current listing
    gone: int = 0            # dropped: absent from a fresh listing, or the whole board 404'd
    unchecked: int = 0       # no board mechanism for this job (JobSpy, or an unresolvable board) -- kept, noted
    ambiguous: int = 0       # board recheck inconclusive (blocked/timeout/other error) -- kept, unchanged


def check_liveness(
    conn, ctx: FetchContext | None = None, registry: Sequence[Provider] = REGISTRY
) -> LivenessStats:
    """Re-verify every 'filtered' job is still live, one fetch per board."""
    stats = LivenessStats()
    owns_ctx = ctx is None
    if ctx is None:
        ctx = FetchContext(http=HttpClient(HttpConfig()))

    provider_by_name = {p.name: p for p in registry}
    try:
        jobs = list(iter_jobs(conn, status="filtered"))

        # board key -> (provider, a representative entry to re-fetch with, jobs in this group)
        groups: dict[str, tuple[Provider, CompanyEntry, list[dict]]] = {}
        for job in jobs:
            stats.checked += 1
            provider = provider_by_name.get(job["source"])
            board_url = None
            if provider is not None and job["url"]:
                entry = CompanyEntry(name=job["company"], careers_url=job["url"], provider=provider.name)
                board_url = provider._board_url(entry)  # reuse: the same resolver fetch() already used
            if board_url is None:
                add_note(conn, job["job_id"], "liveness: unchecked")
                stats.unchecked += 1
                continue
            key = f"{provider.name}:{board_url}"
            if key not in groups:
                groups[key] = (provider, entry, [])
            groups[key][2].append(job)

        for provider, entry, group_jobs in groups.values():
            stats.boards_checked += 1
            try:
                fresh_jobs = provider.fetch(entry, ctx)
            except HTTPError as err:
                if err.status == 404:  # the whole board is gone -- every survivor from it is too
                    for job in group_jobs:
                        set_status(conn, job["job_id"], "dropped", detail="liveness: gone")
                        stats.gone += 1
                else:
                    stats.ambiguous += len(group_jobs)
            except Exception:  # noqa: BLE001 -- Blocked/other FetchError/anything unexpected: ambiguous, never a drop signal
                stats.ambiguous += len(group_jobs)
            else:
                fresh_ids = {j.job_id for j in fresh_jobs}
                for job in group_jobs:
                    if job["job_id"] in fresh_ids:
                        stats.live += 1
                    else:
                        set_status(conn, job["job_id"], "dropped", detail="liveness: gone")
                        stats.gone += 1
    finally:
        if owns_ctx:
            ctx.http.close()

    return stats
