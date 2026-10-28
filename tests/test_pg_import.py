"""Tests for the SQLite -> Postgres import (hiredra/pg_import.py, plan unit DB2). They need Postgres, so they skip
without HIREDRA_TEST_DATABASE_URL (the db_url fixture, conftest.py). The SQLite side is a small file built with the
real storage layer, carrying each edge case the real database has."""

from __future__ import annotations

import json

import pytest

psycopg = pytest.importorskip("psycopg")
from hiredra import pg, pg_import  # noqa: E402
from hiredra.storage import (
    add_note,
    connect,
    fill_provenance,
    finish_run,
    init_db,
    open_run,
    set_extraction,
    upsert_job,
)  # noqa: E402

LISTED = {
    ("ashby", "elevenlabs"): {"board": "ElevenLabs", "name": "ElevenLabs", "tag": "confirmed", "careers_url": None,
                              "provenance": "openjobs"},
    ("lever", "quietco"): {"board": "quietco", "name": "QuietCo", "tag": "unknown", "careers_url": None,
                           "provenance": "curated"},
}  # fmt: skip
USAGE = [("gemini", "gemini-3.5-flash-lite", "2026-10-27", 158), ("nvidia", "nemotron", "2026-10-23", 2)]
SKILLS = [
    {"name": "python", "requirement": "required"},
    {"name": "go", "requirement": "required"},
    {"name": "python", "requirement": "good_to_have"},  # the same skill twice in one extraction
]


@pytest.fixture()
def sqlite_path(tmp_path):
    path = tmp_path / "hiredra.db"
    c = connect(path)
    init_db(c)
    open_run(c, "20261017T163834Z-cdc4", kind="scoring", config_name="scoring.wide.yml")
    finish_run(c, "20261017T163834Z-cdc4", status="ok", touched=3, detail=json.dumps({"companies": 2}))

    def job(jid, company, slug=None, tag=None, location="Pune, India"):
        upsert_job(c, {"job_id": jid, "url": f"https://x/{jid}", "title": "Backend Engineer, Pune", "company": company,
                       "source": jid.split(":")[0], "location": location, "description": "<p>Build APIs.</p>"},
                   first_seen_run="20261017T163834Z-cdc4")  # fmt: skip
        if slug or tag:
            fill_provenance(c, [jid], company_slug=slug, list_tag=tag)

    job("ashby:a1", "ElevenLabs", "ElevenLabs", "confirmed")
    job("ashby:a2", "ElevenLabs", "elevenlabs", "confirmed")  # same board, other case
    job("ashby:a3", "ElevenLabs", "elevenlabs")  # no tag of its own: the board's comes from the list
    job("greenhouse:7", "Acme Inc.", "acme", "unknown", location="London, UK")  # a board on no list: orphan
    job("greenhouse:8", "Acme", "acme", "unknown")  # same role as :7, but in India
    job("jobspy-indeed:in-1", "Acme", "acme", "confirmed")  # JobSpy with a slug: gets no board
    set_extraction(c, "greenhouse:7", skills=json.dumps(SKILLS), domain_bucket="backend", legitimacy="ok",
                   extraction_confidence="high", extraction_version="v1")  # fmt: skip
    set_extraction(c, "ashby:a1", skills="[]", domain_bucket=None, legitimacy=None, extraction_confidence="low",
                   extraction_version="v1")  # fmt: skip
    c.execute(
        "UPDATE jobs SET status = 'scored', score = -0.527244, score_features = ?, review_reason = ?, is_remote = 1, "
        "date_posted = '2026-10-01', years_required = 0, scored_at = '2026-10-25T09:43:32+00:00' WHERE job_id = 'greenhouse:7'",
        (
            json.dumps({"coverage": 0.25, "nearest": {"go": ["golang", 0.912345]}}),
            json.dumps(["suspect", "few_skills"]),
        ),
    )
    c.execute("UPDATE jobs SET date_posted = 'last week' WHERE job_id = 'ashby:a3'")
    c.executemany(
        "INSERT INTO skill_vocab (name, first_seen_at, seen_count, mine, level, source) VALUES (?, ?, ?, ?, ?, ?)",
        [("python", "2026-09-10T07:16:25+00:00", 3, 1, "production", "profile"),
         ("c++", "2026-09-11T07:16:25+00:00", 1, 0, None, "market"),
         ("entity framework", "2026-09-12T07:16:25+00:00", 1, 0, None, "jd")],
    )  # fmt: skip
    c.execute("INSERT INTO skill_alias VALUES ('py', 'python', '2026-09-10T07:16:25+00:00')")
    add_note(c, "greenhouse:7", "a note")
    c.commit()
    c.close()
    return path


@pytest.fixture()
def imported(sqlite_path, db_url):
    lite = pg_import.open_sqlite(sqlite_path)
    with pg.connect(db_url) as conn:
        pg.migrate(conn)
        facts = pg_import.import_all(lite, conn, LISTED, USAGE)
        yield lite, conn, facts
    lite.close()


def test_import_verifies_clean_and_reports_each_deliberate_difference(imported):
    lite, conn, facts = imported
    checks = pg_import.verify(lite, conn, USAGE)
    assert [c for c in checks if not c[1]] == []
    assert len(checks) >= 16
    assert facts["repeated_skills_merged"] == 1
    assert facts["case_merged_boards"] == ["ashby:ElevenLabs / elevenlabs"]
    assert facts["list_tag_moved"] == 1  # ashby:a3 had none; its board's is the list's
    assert facts["jobspy_slugs_dropped"] == 1
    assert facts["unparseable_dates"] == 1


def test_boards_and_companies(imported):
    _, conn, _ = imported
    boards = {
        r[0]: r[1:] for r in conn.execute("SELECT provider || ':' || board, status, provenance, list_tag FROM boards")
    }
    assert boards == {
        "ashby:ElevenLabs": ("active", "openjobs", "confirmed"),  # the list's spelling
        "lever:quietco": ("pending", "curated", "unknown"),  # listed, no jobs yet
        "greenhouse:acme": ("active", "orphan", "unknown"),
    }
    rows = dict(conn.execute("SELECT j.job_key, c.slug FROM jobs j JOIN companies c ON c.id = j.company_id"))
    assert rows["greenhouse:7"] == rows["greenhouse:8"] == rows["jobspy-indeed:in-1"] == "acme"  # "Acme Inc." too
    assert conn.execute("SELECT board_id FROM jobs WHERE job_key = 'jobspy-indeed:in-1'").fetchone() == (None,)


def test_values_cross_exactly(imported):
    _, conn, _ = imported
    row = conn.execute(
        "SELECT score, is_remote, date_posted::text, review_reason, score_features, countries FROM jobs WHERE job_key = 'greenhouse:7'"
    ).fetchone()
    assert row[0] == -0.527244  # double precision: no rounding
    assert row[1:4] == (True, "2026-10-01", ["suspect", "few_skills"])
    assert row[4]["nearest"]["go"] == ["golang", 0.912345]
    assert row[5] == ["GB"]
    skills = conn.execute(
        "SELECT s.skill, s.requirement, s.position FROM job_skills s JOIN jobs j ON j.id = s.job_id "
        "WHERE j.job_key = 'greenhouse:7' ORDER BY s.position"
    ).fetchall()
    assert skills == [("python", "required", 0), ("go", "required", 1)]
    events = conn.execute(
        "SELECT e.kind FROM job_events e JOIN jobs j ON j.id = e.job_id WHERE j.job_key = 'greenhouse:7' ORDER BY e.id"
    ).fetchall()
    assert [k for (k,) in events] == ["seen", "note"]


def test_fingerprints_cluster_copies_but_never_across_countries(imported):
    _, conn, _ = imported
    fp = dict(conn.execute("SELECT job_key, role_fingerprint FROM jobs"))
    assert fp["ashby:a1"] == fp["ashby:a2"] == fp["ashby:a3"]
    assert fp["greenhouse:7"] != fp["greenhouse:8"]  # same company, title and text; London vs Pune (Q3)


def test_descriptions_attach_by_exact_name_only(imported):
    _, conn, _ = imported
    d = dict(conn.execute("SELECT name, description FROM skill_vocab"))
    assert d["python"] and d["c++"]  # freehire's slug "cpp", found by its label "C++"
    assert d["entity framework"] is None  # freehire aliases it to dotnet; a relative's description is not used
    assert conn.execute("SELECT COUNT(vector) FROM skill_vocab").fetchone() == (0,)


@pytest.mark.parametrize(
    "corrupt, caught_by",
    [
        ("UPDATE jobs SET title = 'Changed' WHERE job_key = 'ashby:a2'", "6-job random sample"),
        ("UPDATE jobs SET score = -0.527 WHERE job_key = 'greenhouse:7'", "6-job random sample"),
        ("DELETE FROM job_skills WHERE skill = 'go'", "rows: job_skills"),
        ("UPDATE job_skills SET requirement = 'good_to_have' WHERE skill = 'go'", "6-job random sample"),
        ("DELETE FROM job_events WHERE kind = 'note'", "rows: job_events"),
        (
            "UPDATE job_events SET detail = 'edited' WHERE kind = 'note'",
            "6-job random sample",
        ),  # only the sample sees it
        ("UPDATE jobs SET board_id = NULL WHERE job_key = 'ashby:a1'", "every ATS job is on its board"),
        ("UPDATE skill_vocab SET mine = false WHERE name = 'python'", "skill_vocab, every row"),
    ],
)
def test_verify_catches_each_kind_of_loss(imported, corrupt, caught_by):
    lite, conn, _ = imported
    conn.execute(corrupt)
    failed = [name for name, ok, _ in pg_import.verify(lite, conn, USAGE) if not ok]
    assert any(name.startswith(caught_by) for name in failed), failed


def test_company_names_take_the_most_used_spelling(imported):
    _, conn, _ = imported
    assert conn.execute("SELECT name FROM companies WHERE slug = 'acme'").fetchone() == ("Acme",)  # 2 jobs vs 1


def test_listed_boards_curated_wins_and_a_repeat_keeps_the_stronger_tag(tmp_path, monkeypatch):
    wide, curated = tmp_path / "wide.yml", tmp_path / "curated.yml"
    wide.write_text(
        "companies:\n"
        "  - {name: Lambda, slug: Lambda, provider: ashby, extra: {tag: confirmed}}\n"
        "  - {name: Lambda, slug: lambda, provider: ashby, extra: {tag: unknown}}\n"
        "  - {name: Kayak, slug: kayak, provider: ashby, extra: {tag: unknown}}\n"
        "  - {name: Kayak, slug: KAYAK, provider: ashby, extra: {tag: confirmed}}\n"
        "  - {name: Stripe, slug: stripe, provider: greenhouse, extra: {tag: unknown}}\n",
        encoding="utf-8",
    )
    curated.write_text("companies:\n  - {name: Stripe, slug: stripe, provider: greenhouse}\n", encoding="utf-8")
    monkeypatch.setattr(pg_import, "COMPANY_LISTS", (("openjobs", wide), ("curated", curated)))
    got = {k: (v["board"], v["tag"], v["provenance"]) for k, v in pg_import.listed_boards().items()}
    assert got == {
        ("ashby", "lambda"): ("Lambda", "confirmed", "openjobs"),  # the later, weaker repeat does not override
        ("ashby", "kayak"): ("KAYAK", "confirmed", "openjobs"),  # a stronger repeat does
        ("greenhouse", "stripe"): ("stripe", None, "curated"),  # the curated list wins outright
    }


def test_main_refuses_a_second_import_without_reset(sqlite_path, db_url, monkeypatch, tmp_path):
    monkeypatch.setenv("HIREDRA_DATABASE_URL", db_url)
    monkeypatch.setattr(pg_import, "listed_boards", lambda: LISTED)
    monkeypatch.setattr(pg_import, "usage_rows", lambda: USAGE)
    monkeypatch.setattr(pg_import, "DATA", tmp_path)
    assert pg_import.main(["--sqlite", str(sqlite_path)]) == 1  # migrations pending
    with pg.connect(db_url) as conn:
        pg.migrate(conn)
    assert pg_import.main(["--sqlite", str(sqlite_path)]) == 0
    assert pg_import.main(["--sqlite", str(sqlite_path)]) == 1  # already holds jobs
    assert pg_import.main(["--sqlite", str(sqlite_path), "--reset"]) == 0
    assert pg_import.main(["--sqlite", str(sqlite_path), "--verify"]) == 0
    assert len(list((tmp_path / "logs").glob("pg-import-*.md"))) >= 1
