"""Job and board lifecycle (build step 5, decisions D5.3-D5.5): a board's result row, pending -> active, a 404 retires
the board, unlisted jobs closed only after a crawl that saw the whole listing and only past the grace, a job listed
again reopened. The orchestrator side: what makes a crawl complete. Postgres for the store; fakes elsewhere."""

from __future__ import annotations

import pytest

from hiredra.sourcing.config import Pacing, RunConfig
from hiredra.sourcing.http import HttpClient, HttpConfig, HTTPError
from hiredra.sourcing.job import Job
from hiredra.sourcing.jobspy_source import JobSpyConfig, JobSpyResult, SiteQuery
from hiredra.sourcing.orchestrator import run
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.runstore import BoardResult, FakeRunStore
from hiredra.sourcing.sink import SinkStats
from hiredra.storage import db


# ── the store ───────────────────────────────────────────────────────────────
@pytest.fixture()
def conn(pg_url):
    c = db.connect(pg_url)
    db.init_db(c)
    db.open_run(c, "R1", kind="scoring", config_name=None)
    yield c
    c.close()


def _board(conn, status="pending"):
    return conn.execute(
        "INSERT INTO boards (provider, board, status, provenance) VALUES ('lever', 'acme', %s, 'curated') RETURNING id",
        (status,),
    ).fetchone()[0]


def _job(conn, key, board_id, seen_hours_ago=0):
    db.upsert_job(conn, {"job_id": key, "url": f"u://{key}", "title": "Engineer", "company": "Acme", "source": "lever"})
    conn.execute(
        "UPDATE jobs SET board_id = %s, last_seen_at = now() - make_interval(hours => %s) WHERE job_key = %s",
        (board_id, seen_hours_ago, key),
    )


def _state(conn, key):
    return conn.execute("SELECT closed_at IS NOT NULL FROM jobs WHERE job_key = %s", (key,)).fetchone()[0]


def test_a_clean_crawl_makes_a_pending_board_active_and_records_its_row(conn):
    board = _board(conn)
    db.record_board_run(conn, "R1", BoardResult(board_id=board, status="ok", jobs=3, new=2, seen=1, complete=True,
                                                seconds=1.5))  # fmt: skip
    assert conn.execute("SELECT status FROM boards WHERE id = %s", (board,)).fetchone()[0] == "active"
    row = conn.execute("SELECT status, fetched, new, closed, duration_ms, error FROM board_crawls").fetchone()
    assert tuple(row) == ("ok", 3, 2, 0, 1500, None)


def test_a_board_that_answers_404_is_retired_with_its_reason(conn):
    board = _board(conn, status="active")
    db.record_board_run(conn, "R1", BoardResult(board_id=board, status="gone", detail="HTTP 404"))
    status, reason = conn.execute("SELECT status, status_reason FROM boards WHERE id = %s", (board,)).fetchone()
    assert status == "retired" and reason.startswith("gone: the board answered 404")
    assert conn.execute("SELECT status FROM board_crawls").fetchone()[0] == "gone"


def test_only_a_complete_crawl_closes_and_only_jobs_past_the_grace(conn):
    board = _board(conn, status="active")
    _job(conn, "lever:listed", board)
    _job(conn, "lever:gone", board, seen_hours_ago=72)
    _job(conn, "lever:recent", board, seen_hours_ago=12)  # missed once: within the 48 h grace
    window = BoardResult(board_id=board, status="ok", complete=False, detail="a window")
    assert db.record_board_run(conn, "R1", window) == 0
    assert conn.execute("SELECT status, error FROM board_crawls").fetchone() == ("partial", "a window")
    assert db.record_board_run(conn, "R1", BoardResult(board_id=board, status="error", complete=True)) == 0
    assert conn.execute("SELECT status FROM board_crawls").fetchone()[0] == "failed"  # one row a run: the last
    assert not _state(conn, "lever:gone")  # a window or a failed crawl closes nothing

    assert db.record_board_run(conn, "R1", BoardResult(board_id=board, status="ok", complete=True)) == 1
    assert (_state(conn, "lever:listed"), _state(conn, "lever:gone"), _state(conn, "lever:recent")) == (
        False, True, False
    )  # fmt: skip
    assert conn.execute("SELECT status, closed FROM board_crawls").fetchone() == ("ok", 1)
    events = conn.execute(
        "SELECT detail FROM job_events e JOIN jobs j ON j.id = e.job_id WHERE j.job_key = 'lever:gone' AND kind = 'note'"
    ).fetchall()
    assert events == [("closed: unlisted 48 hours after a complete crawl",)]

    db.touch_job(conn, "lever:gone")  # listed again: open again
    assert not _state(conn, "lever:gone")
    assert (
        "reopened"
        in conn.execute(
            "SELECT detail FROM job_events e JOIN jobs j ON j.id = e.job_id WHERE j.job_key = 'lever:gone'"
            " ORDER BY e.id DESC LIMIT 1"
        ).fetchone()[0]
    )


# ── the orchestrator: what makes a crawl complete ───────────────────────────
class _Provider:
    def __init__(self, name, *, full=True, note=None, error=None):
        self.name, self.full_listing, self.note, self.error = name, full, note, error

    def detect(self, entry):
        return entry.provider == self.name

    def board_url(self, entry):
        return None

    def fetch(self, entry, ctx):
        if self.error:
            raise self.error
        if self.note:
            ctx.incomplete(self.note)
        return [
            Job(job_id=f"{self.name}:1", url=f"u://{self.name}", title="Engineer", company="Acme", source=self.name)
        ]


class _Sink:
    def __init__(self):
        self.n = 0

    def consume(self, jobs, **kw):
        self.n += len(jobs)

    def stats(self):
        return SinkStats(fetched=self.n, persisted_new=self.n)


class _NoJobSpy:
    def fetch(self, ctx):
        return JobSpyResult(jobs=[], outcomes=())


@pytest.mark.parametrize("workers", [1, 3])
def test_each_board_reports_whether_its_crawl_saw_the_whole_listing(workers):
    providers = (_Provider("whole"), _Provider("window", full=False), _Provider("capped", note="10000 hits: the cap"),
                 _Provider("gone", error=HTTPError("404", status=404)))  # fmt: skip
    entries = [CompanyEntry(name=p.name, provider=p.name, extra={"board_id": i}) for i, p in enumerate(providers)]
    entries.append(CompanyEntry(name="listed only", provider="whole"))  # no board row: nothing to record
    store = FakeRunStore()
    config = RunConfig(kind="scoring", config_name="t.yml", boards=(), http=HttpConfig(),
                       jobspy=JobSpyConfig(sites=(), default_query=SiteQuery(search_terms=("x",))),
                       pacing=Pacing(between_companies=0.0, jitter=0.0, workers=workers))  # fmt: skip
    ctx = FetchContext(http=HttpClient(HttpConfig()))
    try:
        run(config, companies=entries, sink_factory=lambda r: _Sink(), run_store=store, ctx=ctx, registry=providers,
            jobspy_source=_NoJobSpy(), sleep=lambda s: None)  # fmt: skip
    finally:
        ctx.http.close()
    got = {r.board_id: (r.status, r.complete, r.jobs, r.new, r.detail) for r in store.board_results}
    assert got == {
        0: ("ok", True, 1, 1, None),
        1: ("ok", False, 1, 1, "the provider's listing is a window, not the whole board"),
        2: ("ok", False, 1, 1, "10000 hits: the cap"),
        3: ("gone", False, 0, 0, "404"),
    }
    assert all(r.seconds is not None and r.seconds >= 0 for r in store.board_results)
