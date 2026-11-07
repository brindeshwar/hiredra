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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor
from concurrent.futures import wait as wait_any
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from hiredra.sourcing.belt import OTHER, Belt, Task, lay_out
from hiredra.sourcing.config import RunConfig
from hiredra.sourcing.http import Blocked, FetchError, HTTPError, HttpClient
from hiredra.sourcing.jobspy_source import JobSpySource, SiteOutcome
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext, Provider, company_slug
from hiredra.sourcing.providers import _common
from hiredra.sourcing.providers._common import CHUNK, Pace
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


#: A board's estimated time when it has no last crawl to go by (seconds): its place on the belt only.
_UNKNOWN_SECONDS = 30.0


@dataclass(eq=False)
class _Board:
    """One board read on the belt: its tasks' shared state, on the main thread only."""

    i: int
    entry: CompanyEntry
    provider: Any
    ctx: FetchContext
    key: float
    line: str
    host: str | None
    notes: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)
    items: dict = field(default_factory=dict)
    served: int = 0
    open: int = 0  # its tasks on the belt or in flight
    jobs: list = field(default_factory=list)
    seconds: float = 0.0  # the work time of its tasks, summed
    failed: CompanyOutcome | None = None


def _outcome_of(err: BaseException, entry: CompanyEntry, provider: Provider) -> CompanyOutcome:
    """A board's failure as its outcome: blocked, gone (a definitive 404 -- common with a generated company list built
    from a days-old slug snapshot; benign, like 'unroutable', never a PARTIAL trigger) or error (any other HTTP
    status, a fetch error, or a provider bug, which must not kill the run)."""
    if isinstance(err, Blocked):
        return CompanyOutcome(entry.name, "blocked", provider.name, 0, str(err))
    if isinstance(err, HTTPError):
        return CompanyOutcome(entry.name, "gone" if err.status == 404 else "error", provider.name, 0, str(err))
    if isinstance(err, FetchError):
        return CompanyOutcome(entry.name, "error", provider.name, 0, str(err))
    log.error("orchestrator: %s on %r raised unexpectedly", provider.name, entry.name, exc_info=err)
    return CompanyOutcome(entry.name, "error", provider.name, 0, f"{type(err).__name__}: {err}")


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
    seen_many: Callable[[list[str]], set[str]] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> RunReport:
    """Fetch every company in ``companies`` (the caller picks them: the CLI reads the ``boards`` table, D2.1) +
    the JobSpy batch, handing each batch to ``sink.consume()``. Fail-soft per company and per JobSpy site (sec. 5/9) --
    only a config error or a genuine bug raises out of this function. ``seen`` (is a job id already stored?) goes
    into the fetch context for hydrating providers (D3.4); ``eligible`` counts a board's India/remote-eligible
    postings for its cadence (D2.3), injected so the orchestrator stays filter-free. ``seen_many`` answers ``seen`` for
    a whole listing at once (the belt asks it once a chunked board's listing is read); left out, ``seen`` is asked per
    job.

    With more than one worker the run is the conveyor belt (``belt.py``): every due board's first task laid out before
    the start, JobSpy among them at minute 0, big boards read as chunks, each platform held to its share.
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
            except Exception as err:  # noqa: BLE001 -- a provider bug must not kill the run
                return provider, None, _outcome_of(err, entry, provider)

        by_index: dict[int, CompanyOutcome] = {}
        unroutable: list[str] = []
        jobspy_outcomes: tuple[SiteOutcome, ...] = ()
        jobspy_ran = False  # the belt ran JobSpy among the boards
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
            # D4.1: worker threads fetch; this thread alone writes (sink, checkpoints, cooldowns).
            ctx.http.cooldowns.on_trip = lambda key, until, reason: trips.put((key, until, reason))
            try:
                jobspy_result = _run_belt(
                    config, todo, ctx, registry, workers, record,
                    None if jobspy_done else (jobspy_source or JobSpySource(config.jobspy)),
                    seen, seen_many, sleep, clock,
                )  # fmt: skip
            finally:
                ctx.http.cooldowns.on_trip = run_store.save_cooldown
                while not trips.empty():
                    run_store.save_cooldown(*trips.get())
            if jobspy_result is not None:
                jobspy_ran = True
                if isinstance(jobspy_result, BaseException):
                    log.error("orchestrator: jobspy_source.fetch() raised", exc_info=jobspy_result)
                    jobspy_outcomes = (
                        SiteOutcome("jobspy", "error", 0, f"{type(jobspy_result).__name__}: {jobspy_result}"),
                    )
                else:
                    sink.consume(jobspy_result.jobs)
                    jobspy_outcomes = tuple(jobspy_result.outcomes)
                    jobspy_done = True
                    run_store.checkpoint(handle.run_id, companies_done=companies_done, jobspy_done=jobspy_done)
        outcomes = [by_index[i] for i in sorted(by_index)]  # the list's order, whatever order they finished in

        if unroutable and companies and len(unroutable) == len(companies):
            log.warning(
                "orchestrator: ALL %d companies unroutable -- check the board list",
                len(companies),
            )

        if not jobspy_done and not jobspy_ran:
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


def _host_slots(http: Any, entry: CompanyEntry, provider: Provider | None) -> tuple[str | None, Callable[[str], int]]:
    """The key of the host slots a board's requests take, and how to read a key's room: the HTTP client's own
    (``slot_key``/``slot_cap``), so the belt never hands out a task whose host is full."""
    url = provider.board_url(entry) if provider is not None else None
    host = urlsplit(url).hostname if url else None
    key_of = getattr(http, "slot_key", None)
    cap = getattr(http, "slot_cap", None) or (lambda key: 2)
    if host is None:
        return None, cap
    return (key_of(host) if key_of else host), cap


def _run_belt(
    config: RunConfig,
    todo: list[tuple[int, CompanyEntry]],
    ctx: FetchContext,
    registry: Sequence[Provider],
    workers: int,
    record: Callable[[int, CompanyEntry, tuple], None],
    jobspy: JobSpySource | None,
    seen: Callable[[str], bool] | None,
    seen_many: Callable[[list[str]], set[str]] | None,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> Any:
    """Run the boards (and JobSpy, when given) on the conveyor belt; each board is ``record``ed on this thread as it
    completes. Returns JobSpy's result (or the exception it raised), ``None`` when it was not run."""
    rules = config.belt
    paces = {name: Pace(1.0 / rules.per_second(name)) for name in rules.platforms}
    host_cap = getattr(ctx.http, "slot_cap", None) or (lambda key: 2)
    belt = Belt({name: p.at_once for name, p in rules.platforms.items()}, host_cap)

    boards: list[_Board] = []
    for i, entry in todo:
        provider = route(entry, registry)
        line = provider.name if provider is not None and provider.name in rules.platforms else OTHER
        host, _ = _host_slots(ctx.http, entry, provider)
        board = _Board(i, entry, provider, ctx, 0.0, line, host)
        board.ctx = replace(ctx, incomplete=board.notes.append, pace=paces[line].wait if line in paces else None)
        boards.append(board)
    width = {name: p.at_once for name, p in rules.platforms.items()}
    estimates = [(b.line, float((b.entry.extra or {}).get("last_seconds") or _UNKNOWN_SECONDS), b) for b in boards]
    for key, board in lay_out(estimates, width, workers):
        board.key = key
        kind = "list" if board.provider is not None and hasattr(board.provider, "list_chunk") else "board"
        _put(belt, Task(kind, board.line, board.host, key, board=board))
    if jobspy is not None:
        belt.put(Task("jobspy", OTHER, None, -1.0, payload=jobspy))  # at minute 0, beside the boards

    def stored(keys: list[str]) -> set[str]:
        if seen_many is not None:
            return seen_many(keys)
        ask = seen or ctx.seen
        return {k for k in keys if ask(k)}

    def work(task: Task) -> tuple[str, Any, float]:
        """On a worker: run one task. ``("ok", value, seconds)`` or ``("err", exception, seconds)``."""
        started = time.monotonic()
        try:
            b = task.board
            if task.kind == "jobspy":
                value = task.payload.fetch(ctx)
            elif task.kind == "board":
                if b.provider is None:
                    value = None
                else:
                    value = b.provider.fetch(b.entry, b.ctx)
            elif task.kind == "list":
                value = b.provider.list_chunk(b.entry, b.ctx, task.payload)
            else:
                value = _details(b, task.payload)
            return "ok", value, time.monotonic() - started
        except Exception as err:  # noqa: BLE001 -- a task's failure is its board's outcome, never the run's
            return "err", err, time.monotonic() - started

    def finish(b: _Board) -> None:
        if b.failed is not None:
            record(b.i, b.entry, (b.provider, None, b.failed, b.notes, b.seconds))
        else:
            record(b.i, b.entry, (b.provider, b.jobs, None, b.notes, b.seconds))

    def fail(b: _Board, err: BaseException) -> None:
        if b.failed is None:
            b.failed = _outcome_of(err, b.entry, b.provider)

    def listing_read(b: _Board) -> None:
        """Every listing chunk is in: a second pass, or the details of the new postings, or the board is done."""
        b.info["served"] = b.served
        try:
            more = list(b.provider.listed(b.entry, b.ctx, b.info, b.items))
        except Exception as err:  # noqa: BLE001
            fail(b, err)
            finish(b)
            return
        if more:
            for chunk in more:
                _put(belt, Task("list", b.line, b.host, b.key, payload=chunk, board=b))
            return
        keys = list(b.items)
        known = stored(keys)
        new = []
        for k in keys:
            if k in known:
                job = b.provider.job(b.entry, b.ctx, b.info, b.items[k], False)
                if job is not None:
                    b.jobs.append(job)
            else:
                new.append(b.items[k])
        budget = _common.DETAIL_BUDGET
        if len(new) > budget:
            log.info("%s %s: %d new postings wait for the next run", b.provider.name, b.entry.name, len(new) - budget)
            new = new[:budget]
        for at in range(0, len(new), CHUNK):
            _put(belt, Task("detail", b.line, b.host, b.key, payload=new[at : at + CHUNK], board=b))
        if not new:
            finish(b)

    def handle(task: Task, status: str, value: Any, seconds: float) -> Any:
        b = task.board
        if task.kind == "jobspy":
            return value
        b.seconds += seconds
        b.open -= 1
        if task.kind == "board":
            if status == "err":
                result = (b.provider, None, _outcome_of(value, b.entry, b.provider), b.notes, b.seconds)
            elif b.provider is None:
                result = (None, None, CompanyOutcome(b.entry.name, "unroutable", None, 0), b.notes, b.seconds)
            else:
                result = (b.provider, value, None, b.notes, b.seconds)
            record(b.i, b.entry, result)
            return None
        if status == "err":
            fail(b, value)
        elif task.kind == "list":
            if task.payload is None:
                b.info = dict(value.info or {})
            for k, item in value.items.items():
                b.items.setdefault(k, item)
            b.served += value.served
            if b.failed is None:
                for chunk in value.more:
                    _put(belt, Task("list", b.line, b.host, b.key, payload=chunk, board=b))
        else:
            b.jobs.extend(value)
        if b.open == 0:
            if b.failed is not None:
                finish(b)
            elif task.kind == "list":
                listing_read(b)
            else:
                finish(b)
        return None

    jobspy_result: Any = None
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fetch")
    running: dict[Any, Task] = {}
    try:
        while True:
            now = clock()
            while len(running) < workers:
                task = belt.take(now)
                if task is None:
                    break
                if task.board is not None and task.board.failed is not None:  # its board already failed
                    belt.done(task)
                    task.board.open -= 1
                    if task.board.open == 0:
                        finish(task.board)
                    continue
                running[pool.submit(work, task)] = task
            if not running:
                if not len(belt):
                    break
                wait_s = belt.next_ready_in(now)
                if wait_s is None:  # tasks no rule will ever allow: a config error, not a reason to spin
                    log.error("orchestrator: %d tasks left that no rule allows -- check belt.platforms", len(belt))
                    break
                sleep(wait_s)
                continue
            done, _ = wait_any(running, timeout=belt.next_ready_in(now), return_when=FIRST_COMPLETED)
            for future in done:
                task = running.pop(future)
                belt.done(task)
                status, value, seconds = future.result()
                got = handle(task, status, value, seconds)
                if task.kind == "jobspy":
                    jobspy_result = got if status == "ok" else value
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return jobspy_result


def _put(belt: Belt, task: Task) -> None:
    if task.board is not None:
        task.board.open += 1
    belt.put(task)


def _details(b: _Board, items: list) -> list[Job]:
    """On a worker: a chunk of a board's new postings, each with its detail. A detail that fails is skipped: still
    new, it is read next run (D3.4)."""
    jobs = []
    for item in items:
        try:
            job = b.provider.job(b.entry, b.ctx, b.info, item, True)
        except FetchError as err:
            log.info("%s %s: a detail failed: %s", b.provider.name, b.entry.name, err)
            continue
        if job is not None:
            jobs.append(job)
    return jobs
