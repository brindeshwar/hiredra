"""RunStore (build sub-slice 2a.8a)."""

from __future__ import annotations

import json

import pytest

from hiredra.sourcing.runstore import RunHandle, SqliteRunStore
from hiredra.sourcing.sink import SinkStats
from hiredra.storage import db


@pytest.fixture()
def conn():
    c = db.connect(":memory:")
    db.init_db(c)
    yield c
    c.close()


def test_fresh_begin_opens_a_running_row(conn):
    store = SqliteRunStore(conn)
    handle = store.begin("scoring", "scoring.yml", resume=False)
    assert not handle.resumed and handle.companies_done == set()
    row = conn.execute("SELECT * FROM runs WHERE run_id=?", (handle.run_id,)).fetchone()
    assert row["status"] == "running" and row["kind"] == "scoring"


def test_fresh_begin_sweeps_stale_same_config_only(conn):
    # a stale scoring run + a stale backfill run
    db.open_run(conn, "OLD-scoring", kind="scoring", config_name="scoring.yml")
    db.open_run(conn, "OLD-backfill", kind="backfill", config_name="backfill.yml")

    SqliteRunStore(conn).begin("scoring", "scoring.yml", resume=False)

    assert conn.execute("SELECT status FROM runs WHERE run_id='OLD-scoring'").fetchone()[0] == "failed"
    assert conn.execute("SELECT status FROM runs WHERE run_id='OLD-backfill'").fetchone()[0] == "running"


def test_resume_reuses_unfinished_run_and_does_not_sweep(conn):
    store = SqliteRunStore(conn)
    handle = store.begin("scoring", "scoring.yml", resume=False)
    store.checkpoint(handle.run_id, companies_done={"Stripe", "Ramp"}, jobspy_done=False)
    # a second, older stale run that a sweep WOULD kill
    db.open_run(conn, "OLDER", kind="scoring", config_name="scoring.yml")

    resumed = store.begin("scoring", "scoring.yml", resume=True)
    assert resumed.run_id == handle.run_id
    assert resumed.resumed and resumed.companies_done == {"Stripe", "Ramp"}
    # not swept
    assert conn.execute("SELECT status FROM runs WHERE run_id='OLDER'").fetchone()[0] == "running"


def test_resume_with_no_unfinished_run_starts_fresh(conn):
    handle = SqliteRunStore(conn).begin("scoring", "scoring.yml", resume=True)
    assert not handle.resumed
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_checkpoint_writes_progress_json(conn):
    store = SqliteRunStore(conn)
    h = store.begin("scoring", "scoring.yml", resume=False)
    store.checkpoint(h.run_id, companies_done={"B", "A"}, jobspy_done=True)
    detail = json.loads(conn.execute("SELECT detail FROM runs WHERE run_id=?", (h.run_id,)).fetchone()[0])
    assert detail == {"companies_done": ["A", "B"], "jobspy_done": True}


def test_finish_records_stats_and_status(conn):
    store = SqliteRunStore(conn)
    h = store.begin("scoring", "scoring.yml", resume=False)
    store.finish(
        h.run_id,
        status="partial",
        stats=SinkStats(fetched=50, persisted_new=12, touched=38, collapsed_in_run=3),
        report_json='{"ok": true}',
    )
    row = conn.execute("SELECT * FROM runs WHERE run_id=?", (h.run_id,)).fetchone()
    assert row["status"] == "partial" and row["finished_at"] is not None
    assert row["fetched"] == 50 and row["persisted_new"] == 12 and row["touched"] == 38
    assert row["detail"] == '{"ok": true}'


def test_finish_rejects_running(conn):
    store = SqliteRunStore(conn)
    h = store.begin("scoring", "scoring.yml", resume=False)
    with pytest.raises(ValueError):
        store.finish(h.run_id, status="running", stats=SinkStats(), report_json="{}")


def test_cooldown_bridge(conn):
    import time

    store = SqliteRunStore(conn)
    store.save_cooldown("h", time.time() + 500, "HTTP 429")
    assert set(store.load_cooldowns()) == {"h"}
