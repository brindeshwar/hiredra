"""Tests for the orchestrator (build sub-slice 2a.8b). All fake -- no real DB,
no real network. ``run()``'s own connection/network creation is exercised only
via the injected ``HttpClient``/``RunStore``.
"""

from __future__ import annotations

import random
import time

import pytest

from hiredra.sourcing.config import Pacing, RunConfig
from hiredra.sourcing.http import Blocked, FetchError, HTTPError, HttpClient, HttpConfig
from hiredra.sourcing.job import Job
from hiredra.sourcing.jobspy_source import JobSpyConfig, JobSpyResult, SiteOutcome, SiteQuery
from hiredra.sourcing.orchestrator import run
from hiredra.sourcing.provider import FetchContext
from hiredra.sourcing.runstore import FakeRunStore, RunHandle
from hiredra.sourcing.sink import SinkStats


def _config(**over):
    base = dict(
        kind="scoring",
        config_name="test.yml",
        companies_path=None,
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

    def consume(self, jobs):
        self.batches.append(list(jobs))

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
    js = FakeJobSpySource(JobSpyResult(
        jobs=[_job(job_id="jobspy-x:1", source="jobspy-x", company="X")],
        outcomes=(SiteOutcome("indeed", "ok", 1),),
    ))
    store = FakeRunStore()
    sinks: list[RecordingSink] = []

    def factory(run_id):
        s = RecordingSink()
        sinks.append(s)
        return s

    report = run(
        _config(companies_path=companies), sink_factory=factory, run_store=store,
        ctx=ctx, registry=(p1, p2), jobspy_source=js, sleep=lambda s: None,
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
        _config(companies_path=companies), sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(), ctx=ctx, registry=(p1, p2),
        jobspy_source=FakeJobSpySource(), sleep=lambda s: None,
    )
    assert [c.status for c in report.companies] == ["error", "ok"]
    assert report.status == "partial"


def test_blocked_makes_partial(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("Acme", "p1")])
    p1 = FakeProvider("p1", error=Blocked("blocked", cooldown_key="p1host", until=0.0))
    report = run(
        _config(companies_path=companies), sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(), ctx=ctx, registry=(p1,),
        jobspy_source=FakeJobSpySource(), sleep=lambda s: None,
    )
    assert report.companies[0].status == "blocked"
    assert report.status == "partial"


def test_provider_bug_is_caught_as_error_not_a_crash(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("Acme", "p1")])
    p1 = FakeProvider("p1", error=RuntimeError("provider bug"))
    report = run(
        _config(companies_path=companies), sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(), ctx=ctx, registry=(p1,),
        jobspy_source=FakeJobSpySource(), sleep=lambda s: None,
    )
    assert report.companies[0].status == "error"
    assert report.status == "partial"


def test_unroutable_stays_ok_but_is_reported(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("Ghost", "nobody")])
    report = run(
        _config(companies_path=companies), sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(), ctx=ctx, registry=(), jobspy_source=FakeJobSpySource(),
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
        _config(companies_path=companies), sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(), ctx=ctx, registry=(p1, p2),
        jobspy_source=FakeJobSpySource(), sleep=lambda s: None,
    )
    assert [c.status for c in report.companies] == ["gone", "ok"]
    assert report.status == "ok"


def test_non_404_http_error_still_makes_partial(tmp_path, ctx):
    # Only a 404 gets the benign treatment -- other HTTP errors (a 500 that
    # wasn't retried away, a malformed-response 400, etc.) still surface loudly.
    companies = _companies_yaml(tmp_path, [("Acme", "p1")])
    p1 = FakeProvider("p1", error=HTTPError("server error", status=500))
    report = run(
        _config(companies_path=companies), sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(), ctx=ctx, registry=(p1,),
        jobspy_source=FakeJobSpySource(), sleep=lambda s: None,
    )
    assert report.companies[0].status == "error"
    assert report.status == "partial"


def test_empty_jobspy_site_is_ok_not_partial(ctx):
    js = FakeJobSpySource(JobSpyResult(jobs=[], outcomes=(SiteOutcome("google", "empty", 0),)))
    report = run(
        _config(), sink_factory=lambda r: RecordingSink(), run_store=FakeRunStore(),
        ctx=ctx, registry=(), jobspy_source=js, sleep=lambda s: None,
    )
    assert report.status == "ok"


def test_jobspy_blocked_makes_partial(ctx):
    js = FakeJobSpySource(JobSpyResult(jobs=[], outcomes=(SiteOutcome("naukri", "blocked", 0, "406"),)))
    report = run(
        _config(), sink_factory=lambda r: RecordingSink(), run_store=FakeRunStore(),
        ctx=ctx, registry=(), jobspy_source=js, sleep=lambda s: None,
    )
    assert report.status == "partial"


def test_jobspy_exception_does_not_lose_company_results(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("Acme", "p1")])
    p1 = FakeProvider("p1", jobs=[_job(job_id="p1:1")])
    js = FakeJobSpySource(raise_=RuntimeError("jobspy bug"))
    store = FakeRunStore()
    report = run(
        _config(companies_path=companies), sink_factory=lambda r: RecordingSink(),
        run_store=store, ctx=ctx, registry=(p1,), jobspy_source=js, sleep=lambda s: None,
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
    cfg = _config(companies_path=companies, pacing=Pacing(between_companies=2.0, jitter=0.0))
    run(
        cfg, sink_factory=lambda r: RecordingSink(), run_store=FakeRunStore(), ctx=ctx,
        registry=(p,), jobspy_source=FakeJobSpySource(), sleep=slept.append,
        rng=random.Random(0),
    )
    assert slept == [2.0, 2.0]


# ── resume ────────────────────────────────────────────────────────────────
def test_resume_skips_done_companies(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("A", "p"), ("B", "p")])
    p = FakeProvider("p", jobs=[_job(job_id="p:new")])
    handle = RunHandle(run_id="R1", companies_done={"A"}, resumed=True)
    store = FakeRunStore(resume_handle=handle)
    report = run(
        _config(companies_path=companies), sink_factory=lambda r: RecordingSink(),
        run_store=store, ctx=ctx, registry=(p,), jobspy_source=FakeJobSpySource(),
        resume=True, sleep=lambda s: None,
    )
    statuses = {c.name: c.status for c in report.companies}
    assert statuses == {"A": "skipped", "B": "ok"}
    assert p.calls == ["B"]
    assert report.resumed is True


def test_resume_skips_jobspy_when_already_done(ctx):
    handle = RunHandle(run_id="R1", jobspy_done=True, resumed=True)
    store = FakeRunStore(resume_handle=handle)
    js = FakeJobSpySource()
    report = run(
        _config(), sink_factory=lambda r: RecordingSink(), run_store=store, ctx=ctx,
        registry=(), jobspy_source=js, resume=True, sleep=lambda s: None,
    )
    assert js.calls == 0
    assert report.jobspy == ()


def test_no_unfinished_run_on_resume_starts_fresh(ctx):
    store = FakeRunStore(resume_handle=None)
    report = run(
        _config(), sink_factory=lambda r: RecordingSink(), run_store=store, ctx=ctx,
        registry=(), jobspy_source=FakeJobSpySource(), resume=True, sleep=lambda s: None,
    )
    assert report.resumed is False


# ── cooldowns wiring ────────────────────────────────────────────────────────
def test_cooldowns_loaded_and_on_trip_wired(ctx):
    store = FakeRunStore(cooldowns={"h": time.time() + 1000})
    run(
        _config(), sink_factory=lambda r: RecordingSink(), run_store=store, ctx=ctx,
        registry=(), jobspy_source=FakeJobSpySource(), sleep=lambda s: None,
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
        _config(), sink_factory=lambda r: RecordingSink(), run_store=FakeRunStore(),
        jobspy_source=FakeJobSpySource(), sleep=lambda s: None,
    )  # no ctx passed

    assert len(created) == 1
    assert created[0]._client.is_closed


def test_run_does_not_close_an_injected_ctx(ctx):
    run(
        _config(), sink_factory=lambda r: RecordingSink(), run_store=FakeRunStore(),
        ctx=ctx, registry=(), jobspy_source=FakeJobSpySource(), sleep=lambda s: None,
    )
    assert ctx.http._client.is_closed is False


# ── unexpected crash ────────────────────────────────────────────────────────
def test_unexpected_crash_marks_run_failed_and_reraises(tmp_path, ctx):
    companies = _companies_yaml(tmp_path, [("A", "p")])
    p = FakeProvider("p", jobs=[_job(job_id="p:1")])

    class BadSink:
        def consume(self, jobs):
            raise RuntimeError("sink bug")

        def stats(self):
            return SinkStats()

    store = FakeRunStore()
    with pytest.raises(RuntimeError, match="sink bug"):
        run(
            _config(companies_path=companies), sink_factory=lambda r: BadSink(),
            run_store=store, ctx=ctx, registry=(p,), jobspy_source=FakeJobSpySource(),
            sleep=lambda s: None,
        )
    assert store.finished[-1]["status"] == "failed"


# ── RunReport JSON ──────────────────────────────────────────────────────────
def test_report_to_json_round_trips(tmp_path, ctx):
    import json

    companies = _companies_yaml(tmp_path, [("A", "p")])
    p = FakeProvider("p", jobs=[_job(job_id="p:1")])
    report = run(
        _config(companies_path=companies), sink_factory=lambda r: RecordingSink(),
        run_store=FakeRunStore(), ctx=ctx, registry=(p,), jobspy_source=FakeJobSpySource(),
        sleep=lambda s: None,
    )
    parsed = json.loads(report.to_json())
    assert parsed["status"] == "ok"
    assert parsed["companies"][0]["name"] == "A"
