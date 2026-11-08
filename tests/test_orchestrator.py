"""Tests for the orchestrator (build sub-slice 2a.8b). All fake -- no real DB,
no real network. ``run()``'s own connection/network creation is exercised only
via the injected ``HttpClient``/``RunStore``.
"""

from __future__ import annotations

import random
import threading
import time

import pytest

from hiredra.sourcing.belt import BeltConfig, Platform
from hiredra.sourcing.config import Pacing, RunConfig, load_companies
from hiredra.sourcing.http import Blocked, FetchError, HTTPError, HttpClient, HttpConfig
from hiredra.sourcing.job import Job
from hiredra.sourcing.jobspy_source import JobSpyConfig, JobSpyResult, SiteOutcome, SiteQuery
from hiredra.sourcing.orchestrator import run
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import Listed
from hiredra.sourcing.runstore import FakeRunStore, RunHandle
from hiredra.sourcing.sink import SinkStats


def _config(**over):
    base = dict(
        kind="scoring",
        config_name="test.yml",
        boards=(),
        jobspy=JobSpyConfig(sites=(), default_query=SiteQuery(search_terms=("x",))),
        http=HttpConfig(),
        pacing=Pacing(between_companies=0.0, jitter=0.0),
    )
    base.update(over)
    return RunConfig(**base)


def _companies_yaml(tmp_path, entries):
    lines = ["companies:"]
    for name, provider in entries:
        lines.append(f"  - name: {name}")
        lines.append(f"    provider: {provider}")
    p = tmp_path / "companies.yml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def _job(**over):
    base = dict(url="u://x", title="Backend Engineer", company="C", source="fake")
    base.update(over)
    return Job(**base)


class FakeProvider:
    def __init__(self, name, jobs=None, error=None):
        self.name = name
        self._jobs = jobs if jobs is not None else []
        self._error = error
        self.calls: list[str] = []

    def detect(self, entry):
        return entry.provider == self.name

    def board_url(self, entry):  # greenhouse-shaped, so company_slug() reads the name back out
        return f"https://fake/v1/boards/{entry.name.lower()}/jobs"

    def fetch(self, entry, ctx):
        self.calls.append(entry.name)
        if self._error is not None:
            raise self._error
        return list(self._jobs)


class FakeJobSpySource:
    def __init__(self, result=None, raise_=None):
        self._result = result if result is not None else JobSpyResult(jobs=[], outcomes=())
        self._raise = raise_
        self.calls = 0

    def fetch(self, ctx):
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return self._result


class RecordingSink:
    def __init__(self):
        self.batches: list[list[Job]] = []
        self.provenance: list[tuple[str | None, str | None]] = []

    def consume(self, jobs, *, company_slug=None, list_tag=None, provenance="orphan"):
        self.batches.append(list(jobs))
        self.provenance.append((company_slug, list_tag))
        self.list_provenance = provenance

    def stats(self):
        n = sum(len(b) for b in self.batches)
        return SinkStats(fetched=n, persisted_new=n)


@pytest.fixture()
def ctx():
    c = FetchContext(http=HttpClient(HttpConfig()))
    yield c
    c.http.close()


# ── happy path ─────────────────────────────────────────────────────────────
def test_happy_path(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("Acme", "p1"), ("Beta", "p2")])
    p1 = FakeProvider("p1", jobs=[_job(job_id="p1:1", company="Acme")])
    p2 = FakeProvider("p2", jobs=[_job(job_id="p2:1", company="Beta", title="Data Engineer")])
    js = FakeJobSpySource(
        JobSpyResult(
            jobs=[_job(job_id="jobspy-x:1", source="jobspy-x", company="X")],
            outcomes=(SiteOutcome("indeed", "ok", 1),),
        )
    )
    store = FakeRunStore()
    sinks: list[RecordingSink] = []

    def factory(run_id):
        s = RecordingSink()
        sinks.append(s)
        return s

    report = run(
        _config(),
        companies=load_companies(companies),
        sink_factory=factory,
        run_store=store,
        ctx=ctx,
        registry=(p1, p2),
        jobspy_source=js,
        sleep=lambda s: None,
    )

    assert report.status == "ok"
    assert [c.status for c in report.companies] == ["ok", "ok"]
    assert p1.calls == ["Acme"] and p2.calls == ["Beta"]
    assert js.calls == 1
    assert store.begun == [("scoring", "test.yml", False)]
    assert len(store.finished) == 1 and store.finished[0]["status"] == "ok"
    assert len(store.checkpoints) == 3  # 2 companies + 1 jobspy
    assert store.checkpoints[-1]["jobspy_done"] is True
    assert len(sinks[0].batches) == 3  # p1, p2, jobspy


# ── status roll-up ───────────────────────────────────────────────────────
def test_fetcherror_makes_partial_other_companies_still_run(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("Acme", "p1"), ("Beta", "p2")])
    p1 = FakeProvider("p1", error=FetchError("boom"))
    p2 = FakeProvider("p2", jobs=[_job(job_id="p2:1")])
    report = run(
        _config(),
        companies=load_companies(companies),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(p1, p2),
        jobspy_source=FakeJobSpySource(),
        sleep=lambda s: None,
    )
    assert [c.status for c in report.companies] == ["error", "ok"]
    assert report.status == "partial"


def test_blocked_makes_partial(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("Acme", "p1")])
    p1 = FakeProvider("p1", error=Blocked("blocked", cooldown_key="p1host", until=0.0))
    report = run(
        _config(),
        companies=load_companies(companies),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(p1,),
        jobspy_source=FakeJobSpySource(),
        sleep=lambda s: None,
    )
    assert report.companies[0].status == "blocked"
    assert report.status == "partial"


def test_provider_bug_is_caught_as_error_not_a_crash(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("Acme", "p1")])
    p1 = FakeProvider("p1", error=RuntimeError("provider bug"))
    report = run(
        _config(),
        companies=load_companies(companies),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(p1,),
        jobspy_source=FakeJobSpySource(),
        sleep=lambda s: None,
    )
    assert report.companies[0].status == "error"
    assert report.status == "partial"


def test_unroutable_stays_ok_but_is_reported(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("Ghost", "nobody")])
    report = run(
        _config(),
        companies=load_companies(companies),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(),
        jobspy_source=FakeJobSpySource(),
        sleep=lambda s: None,
    )
    assert report.companies[0].status == "unroutable"
    assert report.unroutable == ("Ghost",)
    assert report.status == "ok"


def test_404_is_gone_not_error_and_stays_ok(tmp_path, ctx):
    # A definitive 404 (board renamed/removed -- common on a large generated
    # company list built from a days-old slug snapshot) is benign, like
    # 'unroutable', not a PARTIAL trigger.
    companies = _companies_yaml(tmp_path, [("Acme", "p1"), ("Beta", "p2")])
    p1 = FakeProvider("p1", error=HTTPError("not found", status=404))
    p2 = FakeProvider("p2", jobs=[_job(job_id="p2:1")])
    report = run(
        _config(),
        companies=load_companies(companies),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(p1, p2),
        jobspy_source=FakeJobSpySource(),
        sleep=lambda s: None,
    )
    assert [c.status for c in report.companies] == ["gone", "ok"]
    assert report.status == "ok"


def test_non_404_http_error_still_makes_partial(tmp_path, ctx):
    # Only a 404 gets the benign treatment -- other HTTP errors (a 500 that
    # wasn't retried away, a malformed-response 400, etc.) still surface loudly.
    companies = _companies_yaml(tmp_path, [("Acme", "p1")])
    p1 = FakeProvider("p1", error=HTTPError("server error", status=500))
    report = run(
        _config(),
        companies=load_companies(companies),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(p1,),
        jobspy_source=FakeJobSpySource(),
        sleep=lambda s: None,
    )
    assert report.companies[0].status == "error"
    assert report.status == "partial"


def test_empty_jobspy_site_is_ok_not_partial(ctx):
    js = FakeJobSpySource(JobSpyResult(jobs=[], outcomes=(SiteOutcome("google", "empty", 0),)))
    report = run(
        _config(),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(),
        jobspy_source=js,
        sleep=lambda s: None,
    )
    assert report.status == "ok"


def test_jobspy_blocked_makes_partial(ctx):
    js = FakeJobSpySource(JobSpyResult(jobs=[], outcomes=(SiteOutcome("naukri", "blocked", 0, "406"),)))
    report = run(
        _config(),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(),
        jobspy_source=js,
        sleep=lambda s: None,
    )
    assert report.status == "partial"


def test_jobspy_exception_does_not_lose_company_results(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("Acme", "p1")])
    p1 = FakeProvider("p1", jobs=[_job(job_id="p1:1")])
    js = FakeJobSpySource(raise_=RuntimeError("jobspy bug"))
    store = FakeRunStore()
    report = run(
        _config(),
        companies=load_companies(companies),
        sink_factory=lambda r: RecordingSink(),
        run_store=store,
        ctx=ctx,
        registry=(p1,),
        jobspy_source=js,
        sleep=lambda s: None,
    )
    assert report.companies[0].status == "ok"
    assert report.jobspy[0].status == "error"
    assert report.status == "partial"
    assert store.checkpoints[-1]["jobspy_done"] is False  # so --resume retries the batch


# ── pacing ─────────────────────────────────────────────────────────────────
def test_pacing_sleeps_between_companies_not_before_the_first(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("A", "p"), ("B", "p"), ("C", "p")])
    p = FakeProvider("p", jobs=[])
    slept: list[float] = []
    cfg = _config(pacing=Pacing(between_companies=2.0, jitter=0.0))
    run(
        cfg,
        companies=load_companies(companies),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(p,),
        jobspy_source=FakeJobSpySource(),
        sleep=slept.append,
        rng=random.Random(0),
    )
    assert slept == [2.0, 2.0]


# ── a pool of workers (D4.1) ────────────────────────────────────────────────
class _SlowProvider(FakeProvider):
    """Records how many fetches run at once and which thread fetches; one board's fetch trips a cooldown."""

    def __init__(self, name, **kw):
        super().__init__(name, **kw)
        self.lock = threading.Lock()
        self.running = self.peak = 0
        self.threads: set[str] = set()

    def fetch(self, entry, ctx):
        with self.lock:
            self.running += 1
            self.peak = max(self.peak, self.running)
            self.threads.add(threading.current_thread().name)
        time.sleep(0.05)
        if entry.name == "B":
            ctx.http.cooldowns.trip("b.example", 60, reason="HTTP 429")
        with self.lock:
            self.running -= 1
        return super().fetch(entry, ctx)


def test_workers_fetch_at_once_and_one_writer_thread_writes(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [(n, "p") for n in "ABCDEF"])
    p = _SlowProvider("p", jobs=[_job(job_id="p:1")])
    sink = RecordingSink()
    sink_threads: list[str] = []
    real_consume = sink.consume
    sink.consume = lambda jobs, **kw: (sink_threads.append(threading.current_thread().name), real_consume(jobs, **kw))
    store = FakeRunStore()
    slept: list[float] = []
    report = run(
        _config(pacing=Pacing(between_companies=2.0, jitter=0.0, workers=4)),
        companies=load_companies(companies),
        sink_factory=lambda r: sink,
        run_store=store,
        ctx=ctx,
        registry=(p,),
        jobspy_source=FakeJobSpySource(),
        sleep=slept.append,
    )
    assert p.peak > 1 and all(t.startswith("fetch") for t in p.threads)  # boards fetched at once, on workers
    assert sink_threads == ["writer"] * 7  # every board's write and JobSpy's batch on the one writer, never on a worker
    assert len(sink_threads) == 7  # six boards and the JobSpy batch
    assert [c.name for c in report.companies] == list("ABCDEF")  # the list's order, not the finishing order
    assert slept == []  # the per-host slots pace, not a pause between companies
    assert [k for k, _, _ in store.saved_cooldowns] == ["b.example"]  # tripped on a worker, written by the writer
    assert ctx.http.cooldowns.on_trip == store.save_cooldown
    assert len(store.checkpoints[-1]["companies_done"]) == 6 and store.checkpoints[-1]["jobspy_done"]
    assert len(store.checkpoints) < 7  # saved every 30 s and at the end, not after every board


def test_a_platform_keeps_to_its_share_and_never_holds_the_belt(tmp_path, ctx):
    # crawl 2 (2026-11-05): Eightfold's boards, listed together, held all 16 workers waiting on its one pace. On the
    # belt a platform has its share (at_once); the other boards run beside it.
    gate = threading.Event()

    class Paced(_SlowProvider):
        def __init__(self, name, **kw):
            super().__init__(name, **kw)
            self.released: list[bool] = []

        def board_url(self, entry):
            return f"https://{entry.name.lower()}.paced.example/"

        def fetch(self, entry, ctx):
            self.released.append(gate.wait(timeout=5))  # waits until the other provider's boards are done
            return super().fetch(entry, ctx)

    class Fast(FakeProvider):
        done = 0

        def board_url(self, entry):
            return f"https://{entry.name.lower()}.fast.example/"

        def fetch(self, entry, ctx):
            jobs = super().fetch(entry, ctx)
            Fast.done += 1
            if Fast.done == 3:
                gate.set()
            return jobs

    companies = _companies_yaml(
        tmp_path, [("A", "paced"), ("B", "paced"), ("C", "paced"), ("X", "fast"), ("Y", "fast"), ("Z", "fast")]
    )
    paced = Paced("paced", jobs=[])
    report = run(
        _config(pacing=Pacing(workers=2), belt=BeltConfig(platforms={"paced": Platform(limit=1000.0, at_once=1)})),
        companies=load_companies(companies),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(paced, Fast("fast", jobs=[])),
        jobspy_source=FakeJobSpySource(),
    )
    assert paced.released == [True, True, True]  # X, Y and Z were fetched while the paced board waited
    assert paced.peak == 1  # its share: one at a time
    assert [c.status for c in report.companies] == ["ok"] * 6


class _Chunked:
    """A chunked provider: a board of ``n`` postings, ``page`` to a chunk, each detail one call."""

    name = "chunky"

    def __init__(self, n=7, page=3, fail_chunk=None, error=None, fail_times=1 << 30, detail_error=None):
        self.n, self.page, self.fail_chunk, self.fail_times = n, page, fail_chunk, fail_times
        self.error = error or (lambda: Blocked("429", cooldown_key="k", until=0.0))
        self.detail_error = detail_error  # (item index, times): that detail is refused this many times
        self.chunks: list = []
        self.details: list[str] = []
        self.threads: set[str] = set()
        self.lock = threading.Lock()

    def detect(self, entry):
        return entry.provider == self.name

    def board_url(self, entry):
        return f"https://{entry.name.lower()}.chunky.example/"

    def fetch(self, entry, ctx):  # pragma: no cover -- the belt reads it by chunks
        raise AssertionError("a chunked board is read by chunks on the belt")

    def list_chunk(self, entry, ctx, chunk):
        with self.lock:
            self.chunks.append(chunk)
            self.threads.add(threading.current_thread().name)
        if self.fail_chunk is not None and chunk == self.fail_chunk and self.fail_times > 0:
            self.fail_times -= 1
            raise self.error()
        start = chunk or 0
        items = {f"chunky:{entry.name}:{i}": {"i": i} for i in range(start, min(start + self.page, self.n))}
        more = tuple(range(self.page, self.n, self.page)) if chunk is None else ()
        return Listed(items, more=more, served=len(items), info={"total": self.n} if chunk is None else None)

    def listed(self, entry, ctx, info, items):
        if info["served"] < info["total"]:
            ctx.incomplete("short")
        return ()

    def job(self, entry, ctx, info, item, detail):
        if detail:
            with self.lock:
                self.details.append(f"{entry.name}:{item['i']}")
                if self.detail_error and self.detail_error[0] == item["i"] and self.detail_error[1] > 0:
                    self.detail_error = (item["i"], self.detail_error[1] - 1)
                    raise Blocked("429", cooldown_key="k", until=0.0)
        return _job(job_id=f"chunky:{entry.name}:{item['i']}", description="text" if detail else None)


def test_a_chunked_board_is_read_chunk_by_chunk_and_only_its_new_postings_cost_a_detail(tmp_path, ctx):
    provider = _Chunked(n=7, page=3)
    asked: list[list[str]] = []
    store = FakeRunStore()
    sink = RecordingSink()

    def seen_many(keys):
        asked.append(sorted(keys))
        return {k for k in keys if k.endswith((":0", ":1"))}  # two already stored

    report = run(
        _config(pacing=Pacing(workers=3)),
        companies=[CompanyEntry(name="A", provider="chunky", extra={"board_id": 1})],
        sink_factory=lambda r: sink,
        run_store=store,
        ctx=ctx,
        registry=(provider,),
        jobspy_source=FakeJobSpySource(),
        seen_many=seen_many,
    )
    assert sorted(provider.chunks, key=lambda c: c or 0) == [None, 3, 6]  # the first chunk, then the two it planned
    assert len(asked) == 1 and len(asked[0]) == 7  # stored? asked once, for the whole listing
    assert sorted(provider.details) == [f"A:{i}" for i in range(2, 7)]  # a detail for each new posting only
    assert all(t.startswith("fetch") for t in provider.threads)
    [result] = store.board_results
    assert (result.status, result.jobs, result.complete) == ("ok", 7, True)
    assert report.companies[0].status == "ok" and max(len(b) for b in sink.batches) == 7  # JobSpy's empty batch too


class _Clock:
    """A fake monotonic clock that the belt's sleep moves on."""

    def __init__(self):
        self.t = 0.0
        self.slept: list[float] = []

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


def _belt_run(provider, ctx, *, workers=2, tries=4, store=None, clock=None, **kw):
    clock = clock or _Clock()
    return run(
        _config(pacing=Pacing(workers=workers), belt=BeltConfig(tries=tries)),
        companies=[CompanyEntry(name="A", provider=provider.name, extra={"board_id": 1})],
        sink_factory=lambda r: RecordingSink(),
        run_store=store or FakeRunStore(),
        ctx=ctx,
        registry=(provider,),
        jobspy_source=FakeJobSpySource(),
        sleep=clock.sleep,
        clock=clock,
        **kw,
    )


def test_an_error_that_cant_be_done_fails_its_board_and_its_other_chunks_are_dropped(ctx):
    provider = _Chunked(n=30, page=3, fail_chunk=3, error=lambda: HTTPError("400", status=400))
    store = FakeRunStore()
    report = _belt_run(provider, ctx, store=store)
    assert report.companies[0].status == "error" and store.board_results[0].status == "error"
    assert provider.chunks.count(3) == 1 and provider.details == []  # never put back; nothing read past it


def test_a_refused_chunk_goes_back_on_the_belt_and_its_board_completes_later_in_the_run(ctx):
    provider = _Chunked(n=7, page=3, fail_chunk=3, fail_times=2)  # refused twice, then served
    store = FakeRunStore()
    report = _belt_run(provider, ctx, store=store)
    assert provider.chunks.count(3) == 3  # tried, put back, tried, put back, served
    assert report.companies[0].status == "ok" and store.board_results[0].jobs == 7


def test_a_chunk_still_refused_after_its_tries_fails_its_board_for_the_run(ctx):
    provider = _Chunked(n=7, page=3, fail_chunk=3)  # refused every time
    report = _belt_run(provider, ctx, tries=3)
    assert provider.chunks.count(3) == 3 and report.companies[0].status == "blocked"


def test_a_refused_detail_puts_the_rest_of_its_chunk_back_and_keeps_what_it_read(ctx):
    provider = _Chunked(n=4, page=4, detail_error=(2, 1))  # the third posting's detail is refused once
    store = FakeRunStore()
    report = _belt_run(provider, ctx, store=store)
    assert provider.details == ["A:0", "A:1", "A:2", "A:2", "A:3"]  # 0 and 1 kept; 2 and 3 read when put back
    assert report.companies[0].status == "ok" and store.board_results[0].jobs == 4


def test_can_wait_is_only_for_errors_that_dont_mean_it_cant_be_done():
    from hiredra.sourcing.http import TransientError
    from hiredra.sourcing.orchestrator import can_wait

    assert can_wait(Blocked("429", cooldown_key="k", until=0.0)) and can_wait(TransientError("timeout"))
    assert can_wait(HTTPError("408", status=408))
    assert not any(can_wait(HTTPError(str(s), status=s)) for s in (400, 401, 404, 410))
    assert not can_wait(FetchError("no positions list")) and not can_wait(ValueError("a bug"))


def test_the_watchdog_warns_when_no_task_finishes_for_too_long(ctx, caplog):
    class Slow(FakeProvider):
        def fetch(self, entry, ctx):
            time.sleep(0.4)
            return []

    caplog.set_level("WARNING", logger="hiredra.sourcing.orchestrator")
    run(
        _config(pacing=Pacing(workers=2), belt=BeltConfig(watchdog_minutes=0.1 / 60)),  # 0.1 s
        companies=[CompanyEntry(name="A", provider="slow")],
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(Slow("slow"),),
        jobspy_source=FakeJobSpySource(),
    )
    assert any("watchdog" in r.message and "board A" in r.message for r in caplog.records)


def test_ctrl_c_finishes_the_tasks_in_flight_saves_their_boards_and_leaves_the_run_to_resume(tmp_path, ctx):
    started = threading.Event()

    class P(FakeProvider):
        def board_url(self, entry):
            return f"https://{entry.name.lower()}.p.example/"

        def fetch(self, entry, ctx):
            started.set()
            time.sleep(0.2)
            return [_job(job_id=f"p:{entry.name}")]

    class Interrupting:
        fired = False

        def __call__(self):
            # a real Ctrl-C reaches only the main thread; the writer reads the clock too
            if started.is_set() and not Interrupting.fired and threading.current_thread() is threading.main_thread():
                Interrupting.fired = True
                raise KeyboardInterrupt
            return time.monotonic()

    store = FakeRunStore()
    companies = _companies_yaml(tmp_path, [(n, "p") for n in "ABCDEFGH"])
    entries = [
        CompanyEntry(name=e.name, provider=e.provider, extra={**e.extra, "board_id": i + 1})
        for i, e in enumerate(load_companies(companies))
    ]
    with pytest.raises(KeyboardInterrupt):
        run(
            _config(pacing=Pacing(workers=2)),
            companies=entries,
            sink_factory=lambda r: RecordingSink(),
            run_store=store,
            ctx=ctx,
            registry=(P("p"),),
            jobspy_source=FakeJobSpySource(),
            clock=Interrupting(),
        )
    saved = len(store.board_results)
    assert 1 <= saved < 8  # the boards in flight finished and were saved; no new one started
    assert store.finished == []  # the run stays unfinished: --resume goes on from its checkpoint
    assert len(store.checkpoints) == saved


def test_jobspy_runs_at_minute_zero_beside_the_boards(tmp_path, ctx):
    order: list[str] = []

    class Spy(FakeJobSpySource):
        def fetch(self, ctx):
            order.append("jobspy")
            return super().fetch(ctx)

    class P(FakeProvider):
        def fetch(self, entry, ctx):
            order.append(entry.name)
            return super().fetch(entry, ctx)

    companies = _companies_yaml(tmp_path, [(n, "p") for n in "ABC"])
    report = run(
        _config(pacing=Pacing(workers=1 + 1)),
        companies=load_companies(companies),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(P("p"),),
        jobspy_source=Spy(),
    )
    assert order[0] == "jobspy" or "jobspy" in order[:2]  # taken first, beside the first board
    assert report.status == "ok" and len(report.companies) == 3


def test_a_worker_failure_is_that_boards_outcome(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("A", "ok"), ("B", "bad")])
    report = run(
        _config(pacing=Pacing(workers=2)),
        companies=load_companies(companies),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(FakeProvider("ok", jobs=[_job()]), FakeProvider("bad", error=FetchError("x"))),
        jobspy_source=FakeJobSpySource(),
    )
    assert [(c.name, c.status) for c in report.companies] == [("A", "ok"), ("B", "error")]
    assert report.status == "partial"


# ── resume ────────────────────────────────────────────────────────────────
def test_resume_skips_done_companies(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("A", "p"), ("B", "p")])
    p = FakeProvider("p", jobs=[_job(job_id="p:new")])
    handle = RunHandle(run_id="R1", companies_done={"A|p|"}, resumed=True)
    store = FakeRunStore(resume_handle=handle)
    report = run(
        _config(),
        companies=load_companies(companies),
        sink_factory=lambda r: RecordingSink(),
        run_store=store,
        ctx=ctx,
        registry=(p,),
        jobspy_source=FakeJobSpySource(),
        resume=True,
        sleep=lambda s: None,
    )
    statuses = {c.name: c.status for c in report.companies}
    assert statuses == {"A": "skipped", "B": "ok"}
    assert p.calls == ["B"]
    assert report.resumed is True


def test_two_boards_with_one_name_are_both_fetched(tmp_path, ctx):
    """M0.3 bug: progress keyed by name skipped the second board as "already fetched this run"."""
    p = tmp_path / "companies.yml"
    p.write_text(
        "companies:\n"
        "  - {name: Paradigm, provider: gh, slug: paradigm}\n"
        "  - {name: Paradigm, provider: ab, slug: Paradigm}\n"
        "  - {name: Kayak, provider: ab, slug: KAYAK}\n"
        "  - {name: Kayak, provider: ab, slug: kayak}\n",
        encoding="utf-8",
    )
    gh, ab = FakeProvider("gh", jobs=[_job(job_id="gh:1")]), FakeProvider("ab", jobs=[_job(job_id="ab:1")])
    store = FakeRunStore()
    report = run(
        _config(),
        companies=load_companies(p),
        sink_factory=lambda r: RecordingSink(),
        run_store=store,
        ctx=ctx,
        registry=(gh, ab),
        jobspy_source=FakeJobSpySource(),
        sleep=lambda s: None,
    )
    assert [c.status for c in report.companies] == ["ok", "ok", "ok", "ok"]
    assert gh.calls == ["Paradigm"] and ab.calls == ["Paradigm", "Kayak", "Kayak"]
    assert store.checkpoints[-1]["companies_done"] == {
        "Paradigm|gh|paradigm",
        "Paradigm|ab|Paradigm",
        "Kayak|ab|KAYAK",
        "Kayak|ab|kayak",
    }


def test_resume_skips_jobspy_when_already_done(ctx):
    handle = RunHandle(run_id="R1", jobspy_done=True, resumed=True)
    store = FakeRunStore(resume_handle=handle)
    js = FakeJobSpySource()
    report = run(
        _config(),
        sink_factory=lambda r: RecordingSink(),
        run_store=store,
        ctx=ctx,
        registry=(),
        jobspy_source=js,
        resume=True,
        sleep=lambda s: None,
    )
    assert js.calls == 0
    assert report.jobspy == ()


def test_no_unfinished_run_on_resume_starts_fresh(ctx):
    store = FakeRunStore(resume_handle=None)
    report = run(
        _config(),
        sink_factory=lambda r: RecordingSink(),
        run_store=store,
        ctx=ctx,
        registry=(),
        jobspy_source=FakeJobSpySource(),
        resume=True,
        sleep=lambda s: None,
    )
    assert report.resumed is False


# ── cooldowns wiring ────────────────────────────────────────────────────────
def test_cooldowns_loaded_and_on_trip_wired(ctx):
    store = FakeRunStore(cooldowns={"h": time.time() + 1000})
    run(
        _config(),
        sink_factory=lambda r: RecordingSink(),
        run_store=store,
        ctx=ctx,
        registry=(),
        jobspy_source=FakeJobSpySource(),
        sleep=lambda s: None,
    )
    assert "h" in ctx.http.cooldowns.active()
    assert ctx.http.cooldowns.on_trip == store.save_cooldown


# ── ctx ownership ────────────────────────────────────────────────────────
def test_run_closes_the_ctx_it_creates(monkeypatch):
    import hiredra.sourcing.orchestrator as orch_mod
    from hiredra.sourcing.http import HttpClient as RealHttpClient

    created: list[RealHttpClient] = []

    class TrackingClient(RealHttpClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            created.append(self)

    monkeypatch.setattr(orch_mod, "HttpClient", TrackingClient)

    run(
        _config(),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        jobspy_source=FakeJobSpySource(),
        sleep=lambda s: None,
    )  # no ctx passed

    assert len(created) == 1
    assert created[0]._client.is_closed


def test_run_does_not_close_an_injected_ctx(ctx):
    run(
        _config(),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(),
        jobspy_source=FakeJobSpySource(),
        sleep=lambda s: None,
    )
    assert ctx.http._client.is_closed is False


# ── unexpected crash ────────────────────────────────────────────────────────
def test_unexpected_crash_marks_run_failed_and_reraises(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("A", "p")])
    p = FakeProvider("p", jobs=[_job(job_id="p:1")])

    class BadSink:
        def consume(self, jobs, **_):
            raise RuntimeError("sink bug")

        def stats(self):
            return SinkStats()

    store = FakeRunStore()
    with pytest.raises(RuntimeError, match="sink bug"):
        run(
            _config(),
            companies=load_companies(companies),
            sink_factory=lambda r: BadSink(),
            run_store=store,
            ctx=ctx,
            registry=(p,),
            jobspy_source=FakeJobSpySource(),
            sleep=lambda s: None,
        )
    assert store.finished[-1]["status"] == "failed"


# ── RunReport JSON ──────────────────────────────────────────────────────────
def test_report_to_json_round_trips(tmp_path, ctx):
    import json

    companies = _companies_yaml(tmp_path, [("A", "p")])
    p = FakeProvider("p", jobs=[_job(job_id="p:1")])
    report = run(
        _config(),
        companies=load_companies(companies),
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(p,),
        jobspy_source=FakeJobSpySource(),
        sleep=lambda s: None,
    )
    parsed = json.loads(report.to_json())
    assert parsed["status"] == "ok"
    assert parsed["companies"][0]["name"] == "A"


def test_a_failed_write_stops_the_belt_and_the_run_fails_loudly(tmp_path, ctx):
    class Broken(RecordingSink):
        def consume(self, jobs, **kw):
            raise RuntimeError("the database went away")

    store = FakeRunStore()
    companies = _companies_yaml(tmp_path, [(n, "p") for n in "ABCD"])
    with pytest.raises(RuntimeError, match="went away"):
        run(
            _config(pacing=Pacing(workers=2)),
            companies=load_companies(companies),
            sink_factory=lambda r: Broken(),
            run_store=store,
            ctx=ctx,
            registry=(FakeProvider("p", jobs=[_job()]),),
            jobspy_source=FakeJobSpySource(),
        )
    assert store.finished and store.finished[-1]["status"] == "failed"


def test_the_belt_logs_a_status_line_of_what_is_in_flight_and_left(ctx, caplog):
    class Slow(FakeProvider):
        def fetch(self, entry, ctx):
            time.sleep(0.3)
            return []

    caplog.set_level("INFO", logger="hiredra.sourcing.status")
    run(
        _config(pacing=Pacing(workers=2), belt=BeltConfig(status_minutes=0.1 / 60)),  # every 0.1 s
        companies=[CompanyEntry(name=n, provider="slow", extra={"board_id": i}) for i, n in enumerate("AB")],
        sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(),
        ctx=ctx,
        registry=(Slow("slow"),),
        jobspy_source=FakeJobSpySource(),
    )
    lines = [r.getMessage() for r in caplog.records if r.name == "hiredra.sourcing.status"]
    assert lines and "boards done" in lines[0] and "of 2" in lines[0] and "in flight" in lines[0]
    assert "saves waiting" in lines[0]


def test_a_wait_for_put_back_tasks_keeps_the_status_line_and_is_no_stall_to_the_watchdog(ctx, caplog):
    # live check, 2026-11-07: T-Mobile waited 30 min out Workday's maintenance; the log fell silent for 20 min and
    # the watchdog then cried "no task has finished for 19 min" over a planned wait
    provider = _Chunked(n=7, page=3, fail_chunk=3, fail_times=1)  # refused once: put back for a while
    clock = _Clock()
    caplog.set_level("INFO", logger="hiredra.sourcing.status")  # the watchdog's warnings are caught as well
    store = FakeRunStore()
    provider.error = lambda: Blocked("429", cooldown_key="k", until=time.time() + 3600)  # an hour's cooldown
    report = run(
        _config(pacing=Pacing(workers=2), belt=BeltConfig(watchdog_minutes=10, status_minutes=10)),
        companies=[CompanyEntry(name="A", provider="chunky", extra={"board_id": 1})],
        sink_factory=lambda r: RecordingSink(),
        run_store=store,
        ctx=ctx,
        registry=(provider,),
        jobspy_source=FakeJobSpySource(),
        sleep=clock.sleep,
        clock=clock,
    )
    assert report.companies[0].status == "ok"  # waited out the hour, then read
    assert max(clock.slept) <= 600.0 + 1e-6  # slept in steps of at most the status interval
    assert len([r for r in caplog.records if r.name == "hiredra.sourcing.status"]) >= 5
    assert not any("watchdog" in r.getMessage() for r in caplog.records)


def test_jobspy_is_saved_the_moment_it_finishes_so_a_later_stop_never_searches_again(tmp_path, ctx):
    # crawl 3 (2026-11-07): JobSpy's jobs were saved only at the run's end, so the stop for a fix lost them and
    # --resume searched again
    class Slow(FakeProvider):
        def fetch(self, entry, ctx):
            time.sleep(0.3)
            return []

    class Spy(FakeJobSpySource):
        pass

    store = FakeRunStore()
    sink = RecordingSink()
    seen_at: list[int] = []
    real_consume = sink.consume

    def consume(jobs, **kw):
        if jobs and jobs[0].source == "jobspy-indeed":
            seen_at.append(len(store.board_results))  # how many boards were saved when JobSpy's jobs were
        real_consume(jobs, **kw)

    sink.consume = consume
    companies = _companies_yaml(tmp_path, [(n, "p") for n in "ABCD"])
    entries = [CompanyEntry(name=e.name, provider=e.provider, extra={**e.extra, "board_id": i + 1})
               for i, e in enumerate(load_companies(companies))]  # fmt: skip
    run(
        _config(pacing=Pacing(workers=2)),
        companies=entries,
        sink_factory=lambda r: sink,
        run_store=store,
        ctx=ctx,
        registry=(Slow("p"),),
        jobspy_source=Spy(result=JobSpyResult(jobs=[_job(source="jobspy-indeed", job_id="js:1")], outcomes=())),
    )
    assert seen_at and seen_at[0] < 4  # saved before the last board, not after the run
    assert any(cp["jobspy_done"] for cp in store.checkpoints[:-1])  # and the progress said so then
