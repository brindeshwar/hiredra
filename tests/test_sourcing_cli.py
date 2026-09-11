"""Tests for `python -m hiredra.sourcing` (build sub-slice 2a.8b).

JobSpy is always faked here -- these tests never touch the network.
"""

from __future__ import annotations

import pytest

from hiredra.sourcing.jobspy_source import JobSpyResult
from hiredra.storage import db


class _FakeJobSpySource:
    def __init__(self, config):
        pass

    def fetch(self, ctx):
        return JobSpyResult(jobs=[], outcomes=())


def test_run_then_runs_subcommands(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hiredra.sourcing.orchestrator.JobSpySource", _FakeJobSpySource)

    cfg_path = tmp_path / "scoring.yml"
    cfg_path.write_text("kind: scoring\njobspy: {sites: []}\n", encoding="utf-8")
    db_path = tmp_path / "test.db"

    from hiredra.sourcing.__main__ import main

    rc = main(["run", "--config", str(cfg_path), "--db", str(db_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "-> OK" in out
    assert "(none configured)" in out  # no companies_path in this config

    rc2 = main(["runs", "--db", str(db_path)])
    out2 = capsys.readouterr().out
    assert rc2 == 0
    assert "scoring" in out2 and "ok" in out2

    conn = db.connect(db_path)
    try:
        rows = db.list_runs(conn)
        assert len(rows) == 1 and rows[0]["status"] == "ok"
    finally:
        conn.close()


def test_print_report_handles_non_ascii_content(capsys):
    """Job/company/detail strings can carry arbitrary Unicode (e.g. a company
    name, or a block reason quoting a non-English error page)."""
    from hiredra.sourcing.orchestrator import CompanyOutcome, RunReport
    from hiredra.sourcing.sink import SinkStats
    from hiredra.sourcing.__main__ import _print_report

    report = RunReport(
        run_id="R1", kind="scoring", config_name="s.yml",
        started_at="t0", finished_at="t1", status="partial", resumed=False,
        companies=(
            CompanyOutcome("Café Corp™", "error", "greenhouse", 0, "blocked — café ⚠"),
        ),
        jobspy=(), stats=SinkStats(), unroutable=("Naïve Co",),
    )
    _print_report(report)  # must not raise
    out = capsys.readouterr().out
    assert "R1" in out and "unroutable" in out
    unroutable_line = next(line for line in out.splitlines() if "unroutable --" in line)
    assert "[!]" in unroutable_line and "⚠" not in unroutable_line  # our marker is ASCII


def test_stdout_reconfigure_is_what_prevents_the_crash():
    """Directly exercises the mechanism __main__ applies at import
    (``stream.reconfigure(errors="replace")``), independent of import-order/
    module-caching: a strict cp1252 stream raises on U+26A0; reconfigured with
    errors='replace' it doesn't. This is the actual bug that surfaced on the
    real end-to-end run (Windows console, default codepage)."""
    import io

    strict = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    with pytest.raises(UnicodeEncodeError):
        strict.write("⚠ blocked")

    lenient = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    lenient.reconfigure(errors="replace")
    lenient.write("⚠ blocked")  # must not raise now
    lenient.flush()


def test_resume_flag_reuses_the_run(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hiredra.sourcing.orchestrator.JobSpySource", _FakeJobSpySource)

    cfg_path = tmp_path / "scoring.yml"
    cfg_path.write_text("kind: scoring\njobspy: {sites: []}\n", encoding="utf-8")
    db_path = tmp_path / "test.db"

    from hiredra.sourcing.__main__ import main

    conn = db.connect(db_path)
    db.init_db(conn)
    db.open_run(conn, "PRIOR-0000", kind="scoring", config_name="scoring.yml")
    db.checkpoint_run(conn, "PRIOR-0000", detail='{"companies_done": [], "jobspy_done": false}')
    conn.close()

    rc = main(["run", "--config", str(cfg_path), "--db", str(db_path), "--resume"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PRIOR-0000" in out and "RESUMED" in out

    conn = db.connect(db_path)
    try:
        rows = db.list_runs(conn)
        assert len(rows) == 1  # resumed, not a second row
        assert rows[0]["run_id"] == "PRIOR-0000"
    finally:
        conn.close()
