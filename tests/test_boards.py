"""The boards table as the crawl source (build slice 2.1, decision D2.1): which boards a run picks, how a
company list is imported, and a CLI run that crawls the table. JobSpy and the providers are faked; no network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.jobspy_source import JobSpyResult
from hiredra.storage import db


@pytest.fixture()
def conn(pg_url):
    c = db.connect(pg_url)
    db.init_db(c)
    yield c
    c.close()


def _board(conn, provider, board, provenance, *, status="active", name=None):
    company_id = db._company_id(conn, name) if name else None
    conn.execute(
        "INSERT INTO boards (provider, board, company_id, status, provenance) VALUES (%s, %s, %s, %s, %s)",
        (provider, board, company_id, status, provenance),
    )


def _write(tmp_path, body, name="companies.yml"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# ── load_boards ─────────────────────────────────────────────────────────────
def test_load_boards_picks_by_provenance_and_skips_retired(conn):
    _board(conn, "lever", "zeta", "curated", name="Zeta Corp")
    _board(conn, "ashby", "ramp", "curated", status="pending", name="Ramp")
    _board(conn, "greenhouse", "acme", "openjobs")
    _board(conn, "greenhouse", "orph", "orphan")
    _board(conn, "greenhouse", "gone", "curated", status="retired")
    _board(conn, "greenhouse", "nope", "curated", status="rejected")

    curated = db.load_boards(conn, ("curated",))
    assert [(r["provider"], r["board"], r["name"]) for r in curated] == [
        ("ashby", "ramp", "Ramp"),
        ("lever", "zeta", "Zeta Corp"),
    ]  # provider order, pending and active only
    assert [r["board"] for r in db.load_boards(conn, ("all",))] == ["ramp", "acme", "orph", "zeta"]
    assert [r["board"] for r in db.load_boards(conn, ("openjobs", "orphan"))] == ["acme", "orph"]
    assert db.load_boards(conn, ()) == []


# ── import_board / boards import ────────────────────────────────────────────
def test_import_board_outcomes(conn):
    kw = dict(provider="ashby", board="ramp", name="Ramp", provenance="curated", source_name="companies.yml")
    assert db.import_board(conn, **kw) == "new"
    row = conn.execute("SELECT status, provenance, company_id FROM boards WHERE board = 'ramp'").fetchone()
    assert row["status"] == "pending" and row["provenance"] == "curated" and row["company_id"] is not None

    assert db.import_board(conn, **{**kw, "provenance": "yc", "list_tag": "confirmed"}) == "known"
    row = conn.execute("SELECT provenance, list_tag FROM boards WHERE board = 'ramp'").fetchone()
    assert (row["provenance"], row["list_tag"]) == ("curated", "confirmed")  # first provenance wins; blanks fill

    assert db.import_board(conn, **kw, enabled=False) == "retired"
    assert db.load_boards(conn, ("curated",)) == []
    assert db.import_board(conn, **kw) == "re-enabled"
    assert [r["board"] for r in db.load_boards(conn, ("curated",))] == ["ramp"]

    assert db.import_board(conn, **{**kw, "board": "never"}, enabled=False) == "skipped"


def test_a_known_board_keeps_its_company_and_mints_none_for_another_spelling(conn):
    _board(conn, "greenhouse", "1910genetics", "openjobs", name="1910 Genetics")
    kw = dict(provider="greenhouse", board="1910genetics", provenance="ats-inventory", source_name="greenhouse.csv")
    assert db.import_board(conn, name="1910", **kw) == "known"
    assert conn.execute("SELECT name FROM companies").fetchall() == [("1910 Genetics",)]

    conn.execute("UPDATE boards SET company_id = NULL")  # a board with no company yet does get one
    assert db.import_board(conn, name="1910", **kw) == "known"
    assert conn.execute(
        "SELECT c.name FROM boards b JOIN companies c ON c.id = b.company_id"
    ).fetchone()[0] == "1910"  # fmt: skip


def test_a_checked_discovery_records_its_spelling_as_an_alias(conn):
    _board(conn, "greenhouse", "razorpaysoftwareprivatelimited", "openjobs", name="Razorpaysoftwareprivatelimited")
    kw = dict(provider="greenhouse", board="razorpaysoftwareprivatelimited", provenance="atsdetect", source_name="yc")
    assert db.import_board(conn, name="Razorpay", alias=True, **kw) == "known"
    razorpay = conn.execute("SELECT company_id FROM boards").fetchone()[0]
    assert db._company_id(conn, "Razorpay") == razorpay
    assert conn.execute("SELECT count(*) FROM companies").fetchone()[0] == 1

    db._company_id(conn, "Weekday")  # a spelling that is already a company is a merge, not an alias
    assert db.record_alias(conn, "Weekday", razorpay) is False
    assert conn.execute("SELECT alias FROM company_slug_aliases").fetchall() == [("razorpay",)]


def test_a_board_retired_for_another_reason_is_not_re_enabled(conn):
    _board(conn, "ashby", "ramp", "curated", status="retired")
    conn.execute("UPDATE boards SET status_reason = 'board removed by the company'")
    kw = dict(provider="ashby", board="ramp", name="Ramp", provenance="curated", source_name="companies.yml")
    assert db.import_board(conn, **kw) == "known"
    assert conn.execute("SELECT status FROM boards").fetchone()[0] == "retired"


def test_cli_boards_import(tmp_path, capsys, pg_url):
    from hiredra.sourcing.__main__ import main

    path = _write(
        tmp_path,
        "companies:\n"
        "  - {name: Ramp, slug: ramp, provider: ashby}\n"
        "  - {name: Leverdemo, careers_url: 'https://jobs.lever.co/leverdemo'}\n"
        "  - {name: Old, slug: old, provider: ashby, enabled: false}\n",
    )
    assert main(["boards", "import", str(path), "--db", pg_url]) == 0
    assert "3 entries -> new 2, skipped 1" in capsys.readouterr().out
    conn = db.connect(pg_url)
    try:
        rows = db.load_boards(conn, ("curated",))
        assert [(r["provider"], r["board"], r["name"]) for r in rows] == [
            ("ashby", "ramp", "Ramp"),
            ("lever", "leverdemo", "Leverdemo"),
        ]
    finally:
        conn.close()

    assert main(["boards", "import", str(path), "--db", pg_url]) == 0  # idempotent
    assert "new" not in capsys.readouterr().out.split("->")[1]

    bad = _write(tmp_path, "companies:\n  - {name: Mystery, careers_url: 'https://example.com/jobs'}\n", "bad.yml")
    assert main(["boards", "import", str(bad), "--db", pg_url]) == 1
    assert "no provider or board slug for 1: Mystery" in capsys.readouterr().out


# ── a run crawls the table ──────────────────────────────────────────────────
class _NoJobSpy:
    def __init__(self, config):
        pass

    def fetch(self, ctx):
        return JobSpyResult(jobs=[], outcomes=())


class _Recorder:
    name = "ashby"

    def __init__(self):
        self.entries = []

    def detect(self, entry):
        return entry.provider == self.name

    def board_url(self, entry):
        return None

    def fetch(self, entry, ctx):
        self.entries.append(entry)
        return []


def test_cli_run_crawls_the_boards_the_config_names(tmp_path, monkeypatch, capsys, pg_url):
    from hiredra.sourcing import __main__ as cli

    main = cli.main
    conn = db.connect(pg_url)
    try:
        db.init_db(conn)
        _board(conn, "ashby", "ramp", "curated", name="Ramp")
        _board(conn, "ashby", "acme", "openjobs", name="Acme")
        _board(conn, "ashby", "gone", "curated", status="retired")
    finally:
        conn.close()
    recorder = _Recorder()
    real_run = cli.run_orchestrator
    monkeypatch.setattr("hiredra.sourcing.orchestrator.JobSpySource", _NoJobSpy)
    monkeypatch.setattr(cli, "run_orchestrator", lambda config, **kw: real_run(config, registry=(recorder,), **kw))

    curated = _write(tmp_path, "kind: scoring\nboards: [curated]\njobspy: {sites: []}\n", "scoring.yml")
    assert main(["run", "--config", str(curated), "--db", pg_url]) == 0
    assert [(e.name, e.extra["slug"], e.extra["provenance"]) for e in recorder.entries] == [("Ramp", "ramp", "curated")]

    recorder.entries.clear()
    wide = _write(tmp_path, "kind: scoring\nboards: all\njobspy: {sites: []}\n", "scoring.wide.yml")
    assert main(["run", "--config", str(wide), "--db", pg_url]) == 0
    assert [e.name for e in recorder.entries] == ["Acme", "Ramp"]
    assert "-> OK" in capsys.readouterr().out


# ── public ATS-company lists (slice 2.3) ────────────────────────────────────
def test_load_inventory(tmp_path):
    from hiredra.sourcing.config import ConfigError, load_inventory

    p = _write(
        tmp_path,
        "name,slug,url\n10x Team,10xteam,https://jobs.ashbyhq.com/10xteam\n,noname,https://x\nNo Slug,,https://y\n",
        "ashby.csv",
    )
    [entry] = load_inventory(p, "ashby")
    assert (entry.name, entry.provider, entry.careers_url) == ("10x Team", "ashby", "https://jobs.ashbyhq.com/10xteam")
    assert entry.extra == {"slug": "10xteam", "provenance": "ats-inventory"}
    with pytest.raises(ConfigError):
        load_inventory(_write(tmp_path, "name,url\nA,https://a\n", "legacy.csv"), "ashby")


def test_cli_imports_an_inventory_csv(tmp_path, capsys, pg_url):
    from hiredra.sourcing.__main__ import main

    conn = db.connect(pg_url)
    try:
        db.init_db(conn)
        _board(conn, "lever", "100ms", "openjobs", name="100MS")
    finally:
        conn.close()
    path = _write(
        tmp_path,
        "name,slug,url\n100MS,100ms,https://jobs.lever.co/100ms\n15Five,15five,https://jobs.lever.co/15five\n",
        "lever.csv",
    )
    assert main(["boards", "import", str(path), "--db", pg_url]) == 0
    assert "2 entries -> known 1, new 1" in capsys.readouterr().out
    conn = db.connect(pg_url)
    try:
        rows = conn.execute("SELECT board, provenance, status FROM boards ORDER BY board").fetchall()
        assert [tuple(r) for r in rows] == [("100ms", "openjobs", "active"), ("15five", "ats-inventory", "pending")]
    finally:
        conn.close()

    other = _write(tmp_path, "name,slug,url\nAcme,acme,https://jobs.smartrecruiters.com/Acme\n", "smartrecruiters.csv")
    assert main(["boards", "import", str(other), "--db", pg_url]) == 1  # no SmartRecruiters adapter yet (build step 3)
    assert "no provider or board slug for 1: Acme" in capsys.readouterr().out
