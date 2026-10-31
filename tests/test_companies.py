"""One company per employer (build slice 2.2, decisions D2.4/D6.5): the merge plan, its write, the curated list,
and the write path that keeps a merged spelling from coming back as a new company."""

from __future__ import annotations

import pytest

from hiredra.sourcing import companies as cm
from hiredra.storage import db


def _row(id_, slug, name, jobs=0, boards=0):
    return {"id": id_, "slug": slug, "name": name, "jobs": jobs, "boards": boards}


# ── the plan (pure) ─────────────────────────────────────────────────────────
def test_fold_group_keeps_the_busiest_row_and_the_readable_spelling():
    rows = [_row(1, "together-ai", "Together AI", jobs=1), _row(2, "togetherai", "Togetherai", jobs=71, boards=1)]
    renames, groups = cm.plan_merges(rows)
    assert renames == []
    assert groups == [
        cm.MergeGroup(
            canon_id=2, slug="together-ai", name="Together AI", retired=((1, "together-ai"),),
            old_slug="togetherai", reason="fold", jobs=72,
        )
    ]  # fmt: skip


def test_a_lone_company_whose_key_rule_changed_is_renamed():
    renames, groups = cm.plan_merges([_row(5, "antino-labs-private", "ANTINO LABS PRIVATE LIMITED", jobs=3)])
    assert renames == [cm.Rename(5, "antino-labs-private", "antino-labs")] and groups == []


def test_a_rule_change_that_lands_on_an_existing_company_merges():
    rows = [_row(1, "acme-pvt", "Acme Pvt Ltd", jobs=2), _row(2, "acme", "Acme", jobs=9, boards=1)]
    renames, groups = cm.plan_merges(rows)
    assert renames == [] and [(g.canon_id, g.slug, g.retired) for g in groups] == [(2, "acme", ((1, "acme-pvt"),))]


def test_curated_groups_merge_what_no_rule_reaches_and_unknown_names_are_ignored():
    rows = [_row(1, "razorpay", "Razorpay", jobs=5), _row(2, "razorpay-software", "Razorpay Software", jobs=8)]
    assert cm.plan_merges(rows) == ([], [])
    _, groups = cm.plan_merges(rows, [["razorpay", "razorpay-software", "never-seen"]])
    assert [(g.canon_id, g.slug, g.reason) for g in groups] == [(2, "razorpay-software", "curated")]


def test_distinct_employers_stay_apart():
    rows = [_row(1, "capital", "Capital", jobs=3), _row(2, "capital-one", "Capital One", jobs=50)]
    assert cm.plan_merges(rows) == ([], [])
    assert cm.suggestions(rows) == [("Capital", "Capital One", 53)]  # a hint for the owner, never a merge


def test_load_curated(tmp_path):
    assert cm.load_curated(tmp_path / "missing.yml") == []
    p = tmp_path / "m.yml"
    p.write_text("merges:\n  - [Razorpay, Razorpay Software Pvt Ltd]\n", encoding="utf-8")
    assert cm.load_curated(p) == [["razorpay", "razorpay-software"]]
    p.write_text("merges:\n  - [Lonely]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cm.load_curated(p)


# ── the database ────────────────────────────────────────────────────────────
@pytest.fixture()
def conn(pg_url):
    c = db.connect(pg_url)
    db.init_db(c)
    db.open_run(c, "R1", kind="scoring", config_name=None)
    yield c
    c.close()


def _job(conn, key, company, source=None):
    source = source or key.split(":")[0]
    db.upsert_job(conn, {"job_id": key, "url": f"u://{key}", "title": "Engineer", "company": company, "source": source})


def _company_of(conn, key):
    return conn.execute(
        "SELECT c.slug FROM jobs j JOIN companies c ON c.id = j.company_id WHERE j.job_key = %s", (key,)
    ).fetchone()[0]


def test_a_new_spelling_finds_the_company_by_its_squashed_key_and_is_remembered(conn):
    _job(conn, "greenhouse:1", "Together AI")
    _job(conn, "jobspy-indeed:2", "TogetherAI", source="jobspy-indeed")
    assert _company_of(conn, "jobspy-indeed:2") == "together-ai"
    assert conn.execute("SELECT alias FROM company_slug_aliases").fetchall() == [("togetherai",)]
    assert conn.execute("SELECT count(*) FROM companies").fetchone()[0] == 1


def test_apply_merges_moves_everything_and_old_spellings_keep_landing_on_the_canon(conn):
    # two companies as the store holds them today (made before the squashed lookup existed)
    conn.execute(
        "INSERT INTO companies (slug, name) VALUES ('together-ai', 'Together AI'), ('togetherai', 'Togetherai')"
    )
    ids = dict(conn.execute("SELECT slug, id FROM companies").fetchall())
    conn.execute(
        "INSERT INTO boards (provider, board, company_id, provenance) VALUES ('ashby', 'togetherai', %s, 'openjobs')",
        (ids["togetherai"],),
    )
    _job(conn, "ashby:1", "Togetherai")
    conn.execute("UPDATE jobs SET company_id = %s", (ids["togetherai"],))
    _job(conn, "jobspy-indeed:2", "Together AI", source="jobspy-indeed")
    conn.execute("INSERT INTO companies (slug, name) VALUES ('acme-pvt', 'Acme Pvt Ltd')")

    renames, groups = cm.plan_merges(cm.company_rows(conn))
    cm.apply_merges(conn, renames, groups)

    assert conn.execute("SELECT slug, name FROM companies ORDER BY slug").fetchall() == [
        ("acme", "Acme Pvt Ltd"),
        ("together-ai", "Together AI"),
    ]
    canon = ids["togetherai"]  # the row with the board and the job stays
    assert {r[0] for r in conn.execute("SELECT company_id FROM jobs")} == {canon}
    assert conn.execute("SELECT company_id FROM boards").fetchone()[0] == canon
    aliases = dict(conn.execute("SELECT alias, company_id FROM company_slug_aliases").fetchall())
    assert aliases["togetherai"] == canon and "acme-pvt" in aliases and "together-ai" not in aliases

    _job(conn, "lever:3", "Togetherai")  # the old spelling, crawled again
    assert _company_of(conn, "lever:3") == "together-ai"
    assert cm.plan_merges(cm.company_rows(conn)) == ([], [])  # nothing left to do


def test_cli_dry_run_writes_nothing(conn, pg_url, capsys, tmp_path):
    from hiredra.sourcing.__main__ import main

    conn.execute("INSERT INTO companies (slug, name) VALUES ('levelai', 'Levelai'), ('level-ai', 'Level AI')")
    args = ["companies", "merge", "--db", pg_url, "--curated", str(tmp_path / "none.yml")]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "merge groups: 1" in out and "levelai + level-ai -> level-ai" in out and "DRY RUN" in out
    assert conn.execute("SELECT count(*) FROM companies").fetchone()[0] == 2
    assert main([*args, "--apply"]) == 0
    assert conn.execute("SELECT slug FROM companies").fetchall() == [("level-ai",)]
