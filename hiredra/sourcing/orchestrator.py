"""The orchestrator -- the entry point for a sourcing run (docs/sourcing.md sec. 9), configured by a
``RunConfig`` and given a ``Sink`` and a ``RunStore``.

``run()`` imports no database driver -- both DB-touching concerns (job persistence,
run/cooldown metadata) are injected as a ``Sink`` and a ``RunStore``. Unit tests
drive it with fakes and no real database.
"""

from __future__ import annotations

import json
import logging
import queue
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone

from hiredra.sourcing.config import RunConfig
from hiredra.sourcing.http import Blocked, FetchError, HTTPError, HttpClient
from hiredra.sourcing.jobspy_source import JobSpySource, SiteOutcome
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext, Provider, company_slug
from hiredra.sourcing.registry import REGISTRY, route
from hiredra.sourcing.runstore import BoardResult, RunStore
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
    companies: Sequence[CompanyEntry] = (),
    sink_factory: Callable[[str], Sink],
    run_store: RunStore,
    ctx: FetchContext | None = None,
    registry: Sequence[Provider] = REGISTRY,
    jobspy_source: JobSpySource | None = None,
    resume: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    seen: Callable[[str], bool] | None = None,
    eligible: Callable[[list[Job]], int] | None = None,
) -> RunReport:
    """Fetch every company in ``companies`` (the caller picks them: the CLI reads the ``boards`` table, D2.1) +
    the JobSpy batch, handing each batch to ``sink.consume()``. Fail-soft per company and per JobSpy site (sec. 5/9) --
    only a config error or a genuine bug raises out of this function. ``seen`` (is a job id already stored?) goes
    into the fetch context for hydrating providers (D3.4); ``eligible`` counts a board's India/remote-eligible
    postings for its cadence (D2.3), injected so the orchestrator stays filter-free.
    """
    rng = rng or random.Random()
    owns_ctx = ctx is None
    if ctx is None:
        ctx = FetchContext(http=HttpClient(config.http), **({"seen": seen} if seen else {}))

    try:
        # Cross-process cooldown durability (decisions.md #27).
        ctx.http.cooldowns.load(run_store.load_cooldowns())
        ctx.http.cooldowns.on_trip = run_store.save_cooldown

        handle = run_store.begin(config.kind, config.config_name, resume=resume)
        sink = sink_factory(handle.run_id)
        companies_done: set[str] = set(handle.companies_done)
        jobspy_done = handle.jobspy_done

        if config.path2_enabled:
            log.warning(
                "orchestrator: path2 is enabled in config but Path-2 discovery "
                "is not implemented yet -- the slot is a no-op"
            )
            # companies += discover(config.path2, ctx)  # Path-2 slot

        def fetch(entry: CompanyEntry) -> tuple:
            """One board's fetch, safe on a worker thread: no database, no sink. ``(provider, jobs, None, notes,
            seconds)`` on success, ``(provider, None, outcome, notes, seconds)`` otherwise; ``notes`` are the
            provider's reasons the listing was incomplete (D5.3), from a context of the board's own."""
            notes: list[str] = []
            started = time.monotonic()
            return (*_fetch(entry, replace(ctx, incomplete=notes.append)), notes, time.monotonic() - started)

        def _fetch(
            entry: CompanyEntry, ctx: FetchContext
        ) -> tuple[Provider | None, list[Job] | None, CompanyOutcome | None]:
            provider = route(entry, registry)
            if provider is None:
                return None, None, CompanyOutcome(entry.name, "unroutable", None, 0)
            try:
                return provider, provider.fetch(entry, ctx), None
            except Blocked as err:
                return provider, None, CompanyOutcome(entry.name, "blocked", provider.name, 0, str(err))
            except HTTPError as err:
                # A definitive 404 means the board is gone/renamed -- common with a
                # generated company list built from a days-old slug snapshot. Benign,
                # like 'unroutable': never a PARTIAL trigger. Any other HTTP status
                # (403 without a block marker, 5xx that isn't retried here, etc.) is
                # still a real 'error'.
                status = "gone" if err.status == 404 else "error"
                return provider, None, CompanyOutcome(entry.name, status, provider.name, 0, str(err))
            except FetchError as err:
                return provider, None, CompanyOutcome(entry.name, "error", provider.name, 0, str(err))
            except Exception as err:  # noqa: BLE001 -- a provider bug must not kill the run
                log.exception("orchestrator: %s.fetch(%r) raised unexpectedly", provider.name, entry.name)
                outcome = CompanyOutcome(entry.name, "error", provider.name, 0, f"{type(err).__name__}: {err}")
                return provider, None, outcome

        by_index: dict[int, CompanyOutcome] = {}
        unroutable: list[str] = []
        trips: queue.SimpleQueue = queue.SimpleQueue()

        def record(i: int, entry: CompanyEntry, result: tuple) -> None:
            """On the main thread: the board's jobs to the sink, its outcome and lifecycle (D5.3-D5.5), the
            checkpoint (per-company persist)."""
            provider, jobs_c, outcome, notes, seconds = result
            before = sink.stats()
            n_eligible = eligible(jobs_c) if eligible is not None and outcome is None else 0
            if outcome is None:
                tag = entry.extra.get("tag") if entry.extra else None
                sink.consume(
                    jobs_c,
                    company_slug=company_slug(provider, entry),
                    list_tag=str(tag) if tag else None,
                    provenance=str((entry.extra or {}).get("provenance") or "orphan"),
                )
                outcome = CompanyOutcome(entry.name, "ok", provider.name, len(jobs_c))
            elif outcome.status == "unroutable":
                unroutable.append(entry.name)
            by_index[i] = outcome
            if (board_id := (entry.extra or {}).get("board_id")) is not None:
                after = sink.stats()
                whole = outcome.status == "ok" and getattr(provider, "full_listing", True) and not notes
                if outcome.status == "ok" and not whole:
                    notes = notes or ["the provider's listing is a window, not the whole board"]
                run_store.board_result(
                    handle.run_id,
                    BoardResult(
                        board_id=int(board_id), status=outcome.status, jobs=outcome.jobs,
                        new=after.persisted_new - before.persisted_new, seen=after.touched - before.touched,
                        complete=whole, eligible=n_eligible, detail=outcome.detail or ("; ".join(notes) or None),
                        seconds=seconds,
                    ),
                )  # fmt: skip
            while not trips.empty():  # cooldowns tripped on worker threads, written here (one connection)
                run_store.save_cooldown(*trips.get())
            companies_done.add(progress_key(entry))
            run_store.checkpoint(handle.run_id, companies_done=companies_done, jobspy_done=jobspy_done)

        todo: list[tuple[int, CompanyEntry]] = []
        for i, entry in enumerate(companies):
            if progress_key(entry) in companies_done:
                by_index[i] = CompanyOutcome(entry.name, "skipped", None, 0, "already fetched this run")
            else:
                todo.append((i, entry))

        workers = max(1, config.pacing.workers)
        if workers == 1:
            for n, (i, entry) in enumerate(todo):
                if n:
                    sleep(config.pacing.between_companies + rng.random() * config.pacing.jitter)
                record(i, entry, fetch(entry))
        else:
            # D4.1: a pool of threads fetches; this thread alone writes (sink, checkpoints, cooldowns). The HTTP
            # client's per-host slots (D4.2) pace a host, so there is no pause between companies.
            ctx.http.cooldowns.on_trip = lambda key, until, reason: trips.put((key, until, reason))
            pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fetch")
            try:
                futures = {pool.submit(fetch, entry): (i, entry) for i, entry in todo}
                for future in as_completed(futures):
                    i, entry = futures[future]
                    record(i, entry, future.result())
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
                ctx.http.cooldowns.on_trip = run_store.save_cooldown
                while not trips.empty():
                    run_store.save_cooldown(*trips.get())
        outcomes = [by_index[i] for i in sorted(by_index)]  # the list's order, whatever order they finished in

        if unroutable and companies and len(unroutable) == len(companies):
            log.warning(
                "orchestrator: ALL %d companies unroutable -- check the board list",
                len(companies),
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
