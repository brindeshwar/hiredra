"""Tests for hiredra.graph.clean: the skill-name rules applied once to the stored vocabulary."""

from __future__ import annotations

from hiredra.graph import clean
from hiredra.graph.__main__ import main as graph_main
from hiredra.storage import (
    connect,
    get_job,
    get_skill_vocab,
    init_db,
    resolve_alias,
    set_extraction,
    set_status,
    touch_skill_vocab,
    upsert_job,
)
from hiredra.vocab import Resolver


def _job(conn, n: int, skills: list[tuple[str, str]]) -> str:
    job_id = f"greenhouse:{n}"
    upsert_job(conn, {"job_id": job_id, "url": f"https://x/{n}", "title": "Engineer", "company": "Acme",
                      "source": "greenhouse", "description": "text"})  # fmt: skip
    set_extraction(
        conn, job_id, skills=[{"name": s, "requirement": r} for s, r in skills], domain_bucket="backend",
        legitimacy="ok", extraction_confidence="high",
    )  # fmt: skip
    set_status(conn, job_id, "extracted")
    return job_id


def _setup(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    for name in ("javascript", "javascript", "java script", "docker", "kubernetes", "docker & kubernetes",
                 "mcp", "model context protocol (mcp)", "llamainデックス", "python", "python3"):  # fmt: skip
        touch_skill_vocab(conn, name)
    conn.execute("UPDATE skill_vocab SET mine = true, source = 'profile' WHERE name IN ('mcp', 'python')")
    a = _job(
        conn, 1, [("java script", "required"), ("javascript", "good_to_have"), ("docker & kubernetes", "required")]
    )
    b = _job(
        conn,
        2,
        [("llamainデックス", "required"), ("model context protocol (mcp)", "required"), ("python3", "required")],
    )
    return conn, a, b


def test_plan_lists_every_kind_and_changes_nothing(pg_url):
    conn, a, _ = _setup(pg_url)
    changes = {c.name: c for c in clean.plan(conn, Resolver(conn))}
    assert changes["java script"].kind == "merge" and changes["java script"].targets == ("javascript",)
    assert changes["python3"].targets == ("python",)
    assert changes["model context protocol (mcp)"].targets == ("mcp",)
    assert changes["docker & kubernetes"].kind == "split"
    assert changes["docker & kubernetes"].targets == ("docker", "kubernetes")
    assert changes["llamainデックス"].kind == "drop"
    assert "mcp" not in changes and "python" not in changes  # the owner's skills are never touched
    assert get_skill_vocab(conn, "java script") is not None  # a plan writes nothing
    conn.close()


def test_apply_rewrites_jobs_and_folds_the_nodes(pg_url):
    conn, a, b = _setup(pg_url)
    r = Resolver(conn)
    counts = clean.apply(conn, r, clean.plan(conn, r))
    assert counts["jobs_rewritten"] == 2
    assert [(s["name"], s["requirement"]) for s in get_job(conn, a)["skills"]] == [
        ("javascript", "required"),
        ("docker", "required"),
        ("kubernetes", "required"),
    ]
    assert [s["name"] for s in get_job(conn, b)["skills"]] == ["mcp", "python"]
    assert resolve_alias(conn, "java script") == "javascript"
    assert get_skill_vocab(conn, "docker & kubernetes") is None
    assert get_skill_vocab(conn, "llamainデックス") is None
    assert get_skill_vocab(conn, "javascript")["seen_count"] == 3  # the merged node's sightings moved over
    assert clean.plan(conn, Resolver(conn)) == []  # a second pass finds nothing left
    conn.close()


def test_cli_applies_only_the_reviewed_plan(pg_url, tmp_path):
    conn, _, _ = _setup(pg_url)
    plan_dir = str(tmp_path)
    assert graph_main(["--db", pg_url, "clean", "--apply", "--plan-dir", plan_dir]) == 1  # no dry run yet
    assert graph_main(["--db", pg_url, "clean", "--plan-dir", plan_dir]) == 0
    assert (tmp_path / "plan.md").exists() and get_skill_vocab(conn, "java script") is not None
    touch_skill_vocab(conn, "node js")
    touch_skill_vocab(conn, "node.js")
    assert graph_main(["--db", pg_url, "clean", "--apply", "--plan-dir", plan_dir]) == 1  # the database moved
    assert graph_main(["--db", pg_url, "clean", "--plan-dir", plan_dir]) == 0
    assert graph_main(["--db", pg_url, "clean", "--apply", "--plan-dir", plan_dir]) == 0
    assert get_skill_vocab(conn, "java script") is None and (tmp_path / "plan.applied.json").exists()
    conn.close()
