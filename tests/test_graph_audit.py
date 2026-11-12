"""Tests for hiredra.graph.audit: the health check at the end of a pipeline run."""

from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np

from hiredra.graph import audit, clean
from hiredra.graph.__main__ import main as graph_main
from hiredra.graph.index import save_vectors
from hiredra.storage import add_alias, connect, init_db, touch_skill_vocab
from hiredra.vocab import Resolver
from tests.test_graph_clean import _job
from tests.test_sync import _cfg, _unit

CFG = dataclasses.replace(_cfg(), inputs="bare")


def _world(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    for n in ("python", "fastapi", "java script", "javascript"):
        touch_skill_vocab(conn, n)
        save_vectors(conn, [n], np.asarray([_unit(n)]), CFG.embed_tag)
    conn.execute("UPDATE skill_vocab SET mine = true, source = 'profile' WHERE name = 'python'")
    return conn


def _by_name(a):
    return {c.name: c.value for c in a.checks}


def test_each_check_names_its_problem_and_its_fix(pg_url):
    conn = _world(pg_url)
    touch_skill_vocab(conn, "brand new")  # not embedded
    add_alias(conn, "py", "python")
    _job(conn, 1, [("fastapi", "required"), ("py", "required"), ("ghost", "good_to_have")])
    a = audit.run_audit(conn, CFG, clean_plan=clean.plan(conn, Resolver(conn)))
    got = _by_name(a)
    assert got["skills not embedded from their current input"] == 1
    assert got["skills with no description yet"] == 5
    assert got["extracted jobs waiting for skill descriptions"] == 1
    assert got["names graph clean would still change"] == 1  # java script -> javascript
    assert got["job rows on an old spelling (an alias)"] == 1
    assert got["job rows on a skill that does not exist"] == 1
    assert {c.fix for c in a.failed} >= {"python -m hiredra.graph sync", "python -m hiredra.graph.describe"}
    conn.close()


def test_a_healthy_graph_passes_and_matching_is_counted(pg_url, tmp_path):
    conn = _world(pg_url)
    conn.execute("UPDATE skill_vocab SET description_source = 'declined'")
    r = Resolver(conn)
    clean.apply(conn, r, clean.plan(conn, r))
    conn.execute(
        "INSERT INTO jobs (source, external_id, url, title, company, status, score_features) VALUES "
        "('greenhouse', 's1', 'u://s1', 't', 'c', 'scored', %s::jsonb)",
        ('{"skills": [{"skill": "fastapi", "nearest": "python", "fit": 1.0}, '
         '{"skill": "python", "nearest": "python", "fit": 1.0}, {"skill": "cobol", "nearest": "python", "fit": 0.0}, '
         '{"skill": "flask", "nearest": "python", "fit": 0.7}]}',),
    )  # fmt: skip
    a = audit.run_audit(conn, CFG, clean_plan=[])
    assert a.failed == []
    assert a.credit == {"missing": 1, "partial": 1, "full": 2}
    assert a.siblings == [("fastapi", "python", 1)]
    assert [n for n, _ in a.lonely] == ["python"]  # random unit vectors: nothing near it
    text = audit.write_report(a, tmp_path, date(2026, 11, 12)).read_text(encoding="utf-8")
    assert "ok: skills with no description yet | 0" in text and "fastapi <- python (1)" in text
    conn.close()


def test_cli_strict_exits_1_on_a_failed_check(pg_url, tmp_path):
    conn = _world(pg_url)
    conn.close()
    assert graph_main(["--db", pg_url, "audit", "--report-dir", str(tmp_path)]) == 0
    assert graph_main(["--db", pg_url, "audit", "--strict", "--report-dir", str(tmp_path)]) == 1
    assert (tmp_path / f"{date.today()}.md").exists()
