"""SL1: the public board_url, provenance written by the sink (a job's board, which holds the slug and the
list tag since the move to Postgres), and liveness rebuilding a board from the stored slug."""

from __future__ import annotations

import pytest

from hiredra import liveness
from hiredra.sourcing.config import load_companies
from hiredra.sourcing.job import Job
from hiredra.sourcing.orchestrator import run
from hiredra.sourcing.provider import CompanyEntry, company_slug
from hiredra.sourcing.providers.ashby import AshbyProvider
from hiredra.sourcing.providers.greenhouse import GreenhouseProvider
from hiredra.sourcing.providers.lever import LeverProvider
from hiredra.sourcing.sink import ScoringSink
from hiredra.storage import db
from tests.test_liveness import _FakeLivenessProvider, _filtered_job
from tests.test_orchestrator import FakeJobSpySource, FakeProvider, FakeRunStore, RecordingSink, _config

GH, LEVER, ASHBY = GreenhouseProvider(), LeverProvider(), AshbyProvider()


@pytest.fixture()
def conn(pg_url):
    c = db.connect(pg_url)
    db.init_db(c)
    db.open_run(c, "R1", kind="scoring", config_name=None)  # jobs.first_seen_run references runs
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
    _store(conn, "greenhouse:a")
    db.fill_provenance(conn, ["greenhouse:a"], company_slug="acme", list_tag=None, provenance="curated")
    db.fill_provenance(conn, ["greenhouse:a"], company_slug="other", list_tag="unknown")
    row = db.get_job(conn, "greenhouse:a")
    assert (row["company_slug"], row["list_tag"]) == ("acme", None)  # the job keeps its first board
    db.fill_provenance(conn, ["greenhouse:a"], company_slug=None, list_tag="confirmed")  # no slug: nothing to file
    assert db.get_job(conn, "greenhouse:a")["list_tag"] is None
    db.fill_provenance(conn, ["greenhouse:a"], company_slug="ACME", list_tag="confirmed")  # same board, any case
    db.fill_provenance(conn, ["greenhouse:a"], company_slug="acme", list_tag="unknown")
    assert db.get_job(conn, "greenhouse:a")["list_tag"] == "confirmed"  # a board's first tag stays
    assert conn.execute("SELECT provenance FROM boards WHERE board = 'acme'").fetchone()[0] == "curated"


def test_fill_provenance_gives_jobspy_jobs_no_board(conn):
    _store(conn, "jobspy-indeed:in-1", source="jobspy-indeed")
    db.fill_provenance(conn, ["jobspy-indeed:in-1"], company_slug="acme", list_tag="confirmed")
    assert db.get_job(conn, "jobspy-indeed:in-1")["company_slug"] is None
    assert conn.execute("SELECT COUNT(*) FROM boards").fetchone()[0] == 0


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
        _config(),
        companies=load_companies(companies),
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
    with_slug = _filtered_job(conn, job_id="greenhouse:a", url="https://acme.com/careers?gh_jid=1")
    db.fill_provenance(conn, [with_slug], company_slug="acme", list_tag=None)
    _filtered_job(conn, job_id="greenhouse:b", url="https://job-boards.greenhouse.io/beta/jobs/2", company="Beta")
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
