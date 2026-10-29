"""The orchestrator -- the shared entry point for a scoring run and the graph
backfill (docs/sourcing.md sec. 9). Same code, differs only by ``RunConfig`` +
``Sink`` + ``RunStore``.

``run()`` imports no database driver -- both DB-touching concerns (job persistence,
run/cooldown metadata) are injected as a ``Sink`` and a ``RunStore``. Unit tests
drive it with fakes and no real database.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from hiredra.sourcing.config import RunConfig, list_provenance, load_companies
from hiredra.sourcing.http import Blocked, FetchError, HTTPError, HttpClient
from hiredra.sourcing.jobspy_source import JobSpySource, SiteOutcome
from hiredra.sourcing.provider import CompanyEntry, FetchContext, Provider, company_slug
from hiredra.sourcing.registry import REGISTRY, route
from hiredra.sourcing.runstore import RunStore
from hiredra.sourcing.sink import Sink, SinkStats

log = logging.getLogger("hiredra.sourcing.orchestrator")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class CompanyOutcome:
    name: str
    status: str  # "ok" | "unroutable" | "blocked" | "gone" | "error" | "skipped"
    provider: str | None
    jobs: int
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class RunReport:
    run_id: str
    kind: str
    config_name: str
    started_at: str
    finished_at: str
    status: str  # "ok" | "partial" | "failed"
    resumed: bool
    companies: tuple[CompanyOutcome, ...]
    jobspy: tuple[SiteOutcome, ...]
    stats: SinkStats
    unroutable: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "run_id": self.run_id,
                "kind": self.kind,
                "config_name": self.config_name,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "status": self.status,
                "resumed": self.resumed,
                "companies": [asdict(c) for c in self.companies],
                "jobspy": [asdict(o) for o in self.jobspy],
                "stats": asdict(self.stats),
                "unroutable": list(self.unroutable),
            },
            indent=2,
        )


# The only statuses that push a run to 'partial'. NOT 'empty', NOT 'cooldown',
# NOT 'unroutable', NOT 'gone' -- see docs/sourcing.md sec. 9 / the design-chat
# reasoning: those are normal/expected outcomes, not failures worth flagging on
# every run. 'gone' (a definitive 404) is a stale-roster artifact, not a
# problem -- common on a large generated company list built from a days-old
# slug snapshot; it shouldn't read as "this run degraded" any more than
# 'unroutable' does.
_PARTIAL_TRIGGERS = frozenset({"blocked", "error"})


def progress_key(entry: CompanyEntry) -> str:
    """The run-progress key for one company-list entry: name + provider + board.

    Not the name alone: a generated list can hold two DIFFERENT boards under one display name
    (e.g. the same company on greenhouse and ashby), and keying by name skipped the second one
    as "already fetched this run" (found in M0.3: 11 boards)."""
    board = (entry.extra.get("slug") if entry.extra else None) or entry.careers_url or entry.api or ""
    return f"{entry.name}|{entry.provider or ''}|{board}"


def _roll_up(companies: Sequence[CompanyOutcome], jobspy: Sequence[SiteOutcome]) -> str:
    if any(c.status in _PARTIAL_TRIGGERS for c in companies):
        return "partial"
    if any(o.status in _PARTIAL_TRIGGERS for o in jobspy):
        return "partial"
    return "ok"


def run(
    config: RunConfig,
    *,
    sink_factory: Callable[[str], Sink],
    run_store: RunStore,
    ctx: FetchContext | None = None,
    registry: Sequence[Provider] = REGISTRY,
    jobspy_source: JobSpySource | None = None,
    resume: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> RunReport:
    """Fetch every configured company + the JobSpy batch, handing each batch to
    ``sink.consume()``. Fail-soft per company and per JobSpy site (sec. 5/9) --
    only a config error or a genuine bug raises out of this function.
    """
    rng = rng or random.Random()
    owns_ctx = ctx is None
    if ctx is None:
        ctx = FetchContext(http=HttpClient(config.http))

    try:
        # Cross-process cooldown durability (decisions.md #27).
        ctx.http.cooldowns.load(run_store.load_cooldowns())
        ctx.http.cooldowns.on_trip = run_store.save_cooldown

        handle = run_store.begin(config.kind, config.config_name, resume=resume)
        sink = sink_factory(handle.run_id)
        companies_done: set[str] = set(handle.companies_done)
        jobspy_done = handle.jobspy_done

        companies: list[CompanyEntry] = load_companies(config.companies_path) if config.companies_path else []
        if config.path2_enabled:
            log.warning(
                "orchestrator: path2 is enabled in config but Path-2 discovery "
                "is not implemented yet -- the slot is a no-op"
            )
            # companies += discover(config.path2, ctx)  # Path-2 slot

        outcomes: list[CompanyOutcome] = []
        unroutable: list[str] = []
        attempted = 0
        for entry in companies:
            key = progress_key(entry)
            if key in companies_done:
                outcomes.append(CompanyOutcome(entry.name, "skipped", None, 0, "already fetched this run"))
                continue
            if attempted:
                sleep(config.pacing.between_companies + rng.random() * config.pacing.jitter)
            attempted += 1

            provider = route(entry, registry)
            if provider is None:
                outcomes.append(CompanyOutcome(entry.name, "unroutable", None, 0))
                unroutable.append(entry.name)
            else:
                try:
                    jobs_c = provider.fetch(entry, ctx)
                except Blocked as err:
                    outcomes.append(CompanyOutcome(entry.name, "blocked", provider.name, 0, str(err)))
                except HTTPError as err:
                    # A definitive 404 means the board is gone/renamed -- common with a
                    # generated company list built from a days-old slug snapshot. Benign,
                    # like 'unroutable': never a PARTIAL trigger. Any other HTTP status
                    # (403 without a block marker, 5xx that isn't retried here, etc.) is
                    # still a real 'error'.
                    status = "gone" if err.status == 404 else "error"
                    outcomes.append(CompanyOutcome(entry.name, status, provider.name, 0, str(err)))
                except FetchError as err:
                    outcomes.append(CompanyOutcome(entry.name, "error", provider.name, 0, str(err)))
                except Exception as err:  # noqa: BLE001 -- a provider bug must not kill the run
                    log.exception("orchestrator: %s.fetch(%r) raised unexpectedly", provider.name, entry.name)
                    outcomes.append(
                        CompanyOutcome(
                            entry.name,
                            "error",
                            provider.name,
                            0,
                            f"{type(err).__name__}: {err}",
                        )
                    )
                else:
                    tag = entry.extra.get("tag") if entry.extra else None
                    sink.consume(
                        jobs_c,
                        company_slug=company_slug(provider, entry),
                        list_tag=str(tag) if tag else None,
                        provenance=list_provenance(config.companies_path),
                    )
                    outcomes.append(CompanyOutcome(entry.name, "ok", provider.name, len(jobs_c)))

            companies_done.add(key)
            run_store.checkpoint(handle.run_id, companies_done=companies_done, jobspy_done=jobspy_done)

        if unroutable and companies and len(unroutable) == len(companies):
            log.warning(
                "orchestrator: ALL %d companies unroutable -- check %s",
                len(companies),
                config.companies_path,
            )

        jobspy_outcomes: tuple[SiteOutcome, ...] = ()
        if not jobspy_done:
            source = jobspy_source or JobSpySource(config.jobspy)
            try:
                result = source.fetch(ctx)
            except Exception as err:  # noqa: BLE001 -- a JobSpy bug must not lose the company results
                log.exception("orchestrator: jobspy_source.fetch() raised")
                jobspy_outcomes = (SiteOutcome("jobspy", "error", 0, f"{type(err).__name__}: {err}"),)
                # jobspy_done stays False -> a --resume retries the whole batch
            else:
                sink.consume(result.jobs)
                jobspy_outcomes = tuple(result.outcomes)
                jobspy_done = True
                run_store.checkpoint(handle.run_id, companies_done=companies_done, jobspy_done=jobspy_done)

        status = _roll_up(outcomes, jobspy_outcomes)
        stats = sink.stats()
        report = RunReport(
            run_id=handle.run_id,
            kind=config.kind,
            config_name=config.config_name,
            started_at=handle.started_at,
            finished_at=_now_iso(),
            status=status,
            resumed=handle.resumed,
            companies=tuple(outcomes),
            jobspy=jobspy_outcomes,
            stats=stats,
            unroutable=tuple(unroutable),
        )
        run_store.finish(handle.run_id, status=status, stats=stats, report_json=report.to_json())
        return report

    except Exception:
        # A bug or config error, not a per-company/per-site failure (those are
        # already handled above and never raise). Best-effort: try to mark the
        # run failed if we got far enough to have a handle.
        try:
            run_store.finish(
                handle.run_id,  # type: ignore[possibly-undefined]
                status="failed",
                stats=SinkStats(),
                report_json=json.dumps({"error": "orchestrator crashed"}),
            )
        except Exception:  # noqa: BLE001 -- best-effort only, never mask the real error
            pass
        raise
    finally:
        if owns_ctx:
            ctx.http.close()
