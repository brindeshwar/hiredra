"""SL1: schema v5 (company_slug, list_tag, extraction_version), the public board_url,
provenance written by the sink, liveness rebuilding a board from the stored slug, and the
backfill for rows stored before v5."""

from __future__ import annotations

import pytest

from hiredra import liveness
from hiredra.sourcing.__main__ import main as sourcing_main
from hiredra.sourcing.job import Job
from hiredra.sourcing.orchestrator import run
from hiredra.sourcing.persist import backfill_slugs
from hiredra.sourcing.provider import CompanyEntry, company_slug
from hiredra.sourcing.providers.ashby import AshbyProvider
from hiredra.sourcing.providers.greenhouse import GreenhouseProvider
from hiredra.sourcing.providers.lever import LeverProvider
from hiredra.sourcing.sink import ScoringSink
from hiredra.storage import db
from tests.test_liveness import _FakeLivenessProvider, _filtered_job
from tests.test_migration_v4 import _make_v3_db
from tests.test_orchestrator import FakeJobSpySource, FakeProvider, FakeRunStore, RecordingSink, _config

_V5_COLS = {"company_slug", "list_tag", "extraction_version"}
GH, LEVER, ASHBY = GreenhouseProvider(), LeverProvider(), AshbyProvider()


def _cols(conn, table="jobs") -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


@pytest.fixture()
def conn():
    c = db.connect(":memory:")
    db.init_db(c)
    yield c
    c.close()


def _store(conn, job_id, *, source="greenhouse", url=None, company="Acme"):
    db.upsert_job(
        conn,
        {
            "job_id": job_id,
            "url": url or f"https://x/{job_id}",
            "title": "Engineer",
            "company": company,
            "source": source,
        },
    )


# ── migration v4 -> v5 ─────────────────────────────────────────────────────
def _make_v4_db(path):
    conn = _make_v3_db(path)
    db._migrate_v4(conn)
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    return conn


def test_v4_to_v5_adds_columns_and_keeps_every_row(tmp_path):
    path = tmp_path / "v4.db"
    _make_v4_db(path).close()

    conn = db.connect(path)
    db.init_db(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION == 5
    assert _V5_COLS <= _cols(conn)
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 3
    row = db.get_job(conn, "j1")
    assert row["status"] == "extracted" and row["company_slug"] is None and row["list_tag"] is None
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    db.init_db(conn)  # a re-run is a no-op
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
    conn.close()


def test_migrated_jobs_table_matches_a_fresh_one(tmp_path):
    _make_v4_db(tmp_path / "v4.db").close()
    migrated = db.connect(tmp_path / "v4.db")
    db.init_db(migrated)
    fresh = db.connect(tmp_path / "fresh.db")
    db.init_db(fresh)
    assert _cols(migrated) == _cols(fresh)
    migrated.close()
    fresh.close()


# ── company_slug / board_url ───────────────────────────────────────────────
@pytest.mark.parametrize(
    ("provider", "url", "slug"),
    [
        (GH, "https://job-boards.greenhouse.io/acme/jobs/123", "acme"),
        (GH, "https://boards.greenhouse.io/acme", "acme"),
        (LEVER, "https://jobs.lever.co/acme/1f2e-uuid", "acme"),
        (LEVER, "https://jobs.eu.lever.co/acme/1f2e-uuid", "acme"),
        (ASHBY, "https://jobs.ashbyhq.com/acme/1f2e-uuid", "acme"),
        (GH, "https://acme.com/careers?gh_jid=123", None),  # job on the company's own domain
    ],
)
def test_company_slug_reads_the_board_url(provider, url, slug):
    assert company_slug(provider, CompanyEntry(name="Acme", careers_url=url, provider=provider.name)) == slug


def test_a_board_url_ending_at_the_marker_has_no_slug():
    class Bare:
        name = "p"

        def board_url(self, entry):
            return "https://x/v1/boards"

    assert company_slug(Bare(), CompanyEntry(name="Acme")) is None  # type: ignore[arg-type]


def test_list_slug_wins_over_the_url():
    entry = CompanyEntry(
        name="Acme", careers_url="https://acme.com/careers", provider="greenhouse", extra={"slug": "acme"}
    )
    assert company_slug(GH, entry) == "acme"
    assert GH.board_url(entry) == "https://boards-api.greenhouse.io/v1/boards/acme/jobs"


def test_eu_lever_host_survives_a_stored_slug():
    entry = CompanyEntry(
        name="Acme", careers_url="https://jobs.eu.lever.co/acme/x", provider="lever", extra={"slug": "acme"}
    )
    assert LEVER.board_url(entry) == "https://api.eu.lever.co/v0/postings/acme"


# ── fill_provenance + the sink ─────────────────────────────────────────────
def test_fill_provenance_fills_nulls_and_never_overwrites(conn):
    _store(conn, "a")
    db.fill_provenance(conn, ["a"], company_slug="acme", list_tag=None)
    db.fill_provenance(conn, ["a"], company_slug="other", list_tag="unknown")
    row = db.get_job(conn, "a")
    assert (row["company_slug"], row["list_tag"]) == ("acme", "unknown")
    db.fill_provenance(conn, ["a"], company_slug=None, list_tag="confirmed")
    assert db.get_job(conn, "a")["list_tag"] == "unknown"  # the first tag stays


def test_scoring_sink_writes_provenance_on_new_and_reseen_rows(conn):
    _store(conn, "greenhouse:old", url="u://old")  # stored before v5: no provenance yet
    db.fill_provenance(conn, [], company_slug="x", list_tag="y")  # empty list: harmless
    sink = ScoringSink(conn, run_id="R1")
    jobs = [
        Job(job_id="greenhouse:new", url="u://new", title="Data Engineer", company="Acme", source="greenhouse"),
        Job(job_id="greenhouse:old", url="u://old", title="Engineer", company="Acme", source="greenhouse"),
    ]
    sink.consume(jobs, company_slug="acme", list_tag="confirmed")
    for job_id in ("greenhouse:new", "greenhouse:old"):
        row = db.get_job(conn, job_id)
        assert (row["company_slug"], row["list_tag"]) == ("acme", "confirmed")

    sink.consume(jobs)  # a JobSpy-style batch with no provenance changes nothing
    assert db.get_job(conn, "greenhouse:new")["company_slug"] == "acme"


def test_orchestrator_passes_slug_and_tag_per_company(tmp_path):
    companies = tmp_path / "companies.yml"
    companies.write_text(
        "companies:\n"
        "  - name: Acme\n    provider: p\n    slug: acme-co\n    extra: {tag: confirmed}\n"
        "  - name: Beta\n    provider: p\n",
        encoding="utf-8",
    )
    p = FakeProvider("p", jobs=[Job(job_id="p:1", url="u://1", title="Engineer", company="C", source="p")])
    sinks: list[RecordingSink] = []

    def factory(run_id):
        sinks.append(RecordingSink())
        return sinks[-1]

    run(
        _config(companies_path=str(companies)),
        sink_factory=factory,
        run_store=FakeRunStore(),
        ctx=None,
        registry=(p,),
        jobspy_source=FakeJobSpySource(),
        sleep=lambda s: None,
    )
    # Acme: the list's slug and tag. Beta: slug read from board_url, no tag. JobSpy: nothing.
    assert sinks[0].provenance == [("acme-co", "confirmed"), ("beta", None), (None, None)]


# ── liveness ───────────────────────────────────────────────────────────────
def test_liveness_hands_the_stored_slug_to_board_url(conn):
    with_slug = _filtered_job(conn, job_id="a", url="https://acme.com/careers?gh_jid=1")
    db.fill_provenance(conn, [with_slug], company_slug="acme", list_tag=None)
    _filtered_job(conn, job_id="b", url="https://job-boards.greenhouse.io/beta/jobs/2", company="Beta")
    provider = _FakeLivenessProvider("greenhouse")

    liveness.check_liveness(conn, ctx=object(), registry=[provider])  # type: ignore[arg-type]

    by_url = {e.careers_url: e for e in provider.entries}
    assert by_url["https://acme.com/careers?gh_jid=1"].extra == {"slug": "acme"}
    assert by_url["https://job-boards.greenhouse.io/beta/jobs/2"].extra == {}


def test_real_greenhouse_resolves_a_company_domain_job_only_with_the_slug():
    url = "https://acme.com/careers?gh_jid=1"
    assert GH.board_url(CompanyEntry(name="Acme", careers_url=url, provider="greenhouse")) is None
    assert GH.board_url(CompanyEntry(name="Acme", careers_url=url, provider="greenhouse", extra={"slug": "acme"}))


# ── backfill_slugs ─────────────────────────────────────────────────────────
def test_backfill_uses_url_then_list_then_roster(conn):
    _store(conn, "g1", url="https://job-boards.greenhouse.io/acme/jobs/1", company="Acme")
    _store(conn, "g2", url="https://stripe.com/jobs/x?gh_jid=2", company="Stripe")
    _store(conn, "g3", url="https://carvana.com/careers?gh_jid=3", company="carvana")
    _store(conn, "g4", url="https://nowhere.example/job/4", company="Mystery")
    _store(conn, "j1", source="jobspy-indeed", url="https://indeed.example/1", company="Acme")
    _store(conn, "g5", url="https://job-boards.greenhouse.io/kept/jobs/5", company="Kept")
    db.fill_provenance(conn, ["g5"], company_slug="kept", list_tag="unknown")
    entries = [
        CompanyEntry(name="Acme", provider="greenhouse", extra={"slug": "acme", "tag": "confirmed"}),
        CompanyEntry(name="Stripe", provider="greenhouse", extra={"slug": "stripe"}),
        CompanyEntry(name="Kept", provider="greenhouse", extra={"slug": "kept", "tag": "confirmed"}),
    ]

    stats = backfill_slugs(conn, entries, roster={("greenhouse", "carvana")})

    def prov(job_id):
        row = db.get_job(conn, job_id)
        return row["company_slug"], row["list_tag"]

    assert prov("g1") == ("acme", "confirmed")
    assert prov("g2") == ("stripe", None)
    assert prov("g3") == ("carvana", None)
    assert prov("g4") == (None, None)
    assert prov("j1") == (None, None)  # JobSpy rows are left alone
    assert prov("g5") == ("kept", "unknown")  # a stored tag is never overwritten
    assert (stats.rows, stats.from_url, stats.from_list, stats.from_roster, stats.unresolved, stats.tagged) == (
        4,
        1,
        1,
        1,
        1,
        1,
    )
    assert backfill_slugs(conn, entries, roster={("greenhouse", "carvana")}).rows == 3  # only the tagless ones


def test_backfill_list_match_must_be_unambiguous(conn):
    _store(conn, "g1", url="https://acme.com/careers?gh_jid=1", company="Acme")
    entries = [
        CompanyEntry(name="Acme", provider="greenhouse", extra={"slug": "acme"}),
        CompanyEntry(name="Acme", provider="greenhouse", extra={"slug": "acme-eu"}),
    ]
    assert backfill_slugs(conn, entries).unresolved == 1
    assert db.get_job(conn, "g1")["company_slug"] is None


def test_backfill_cli_runs_without_lists_or_roster(tmp_path, capsys):
    path = tmp_path / "h.db"
    c = db.connect(path)
    db.init_db(c)
    _store(c, "g1", url="https://job-boards.greenhouse.io/acme/jobs/1")
    c.close()

    code = sourcing_main(
        [
            "backfill-slugs",
            "--db",
            str(path),
            "--lists",
            str(tmp_path / "none.yml"),
            "--roster",
            str(tmp_path / "none.json"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "url=1" in out and "ATS rows with a slug: 1/1 (100.0%)" in out
