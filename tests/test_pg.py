"""Postgres tests (plan unit DB1; D1.12). They need a real server, so they skip unless
HIREDRA_TEST_DATABASE_URL points at one the tests may create and drop scratch databases on (CI's Linux job sets it;
locally: the docker compose service, see .env.example)."""

from __future__ import annotations

import pytest

psycopg = pytest.importorskip("psycopg")
from hiredra import pg  # noqa: E402

# db_url (a scratch database per test, skipped without HIREDRA_TEST_DATABASE_URL) is in conftest.py.


def test_migrate_applies_each_migration_once(db_url):
    with pg.connect(db_url) as conn:
        assert pg.migrate(conn) == [
            "0001_init",
            "0002_cooldowns",
            "0003_api_usage_exhausted",
            "0004_search_on_demand",
            "0005_skill_source",
        ]
        assert pg.migrate(conn) == []  # idempotent: nothing pending the second time
        tables = {r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
    assert tables >= {
        "schema_migrations", "companies", "company_slug_aliases", "boards", "board_health", "runs", "board_crawls",
        "jobs", "job_skills", "job_events", "skill_vocab", "skill_alias", "api_usage", "cooldowns",
    }  # fmt: skip


def test_job_key_keeps_the_old_id_format(db_url):
    with pg.connect(db_url) as conn:
        pg.migrate(conn)
        key = conn.execute(
            "INSERT INTO jobs (source, external_id, url, title, company) "
            "VALUES ('greenhouse', '105736', 'https://x/1', 'Backend Engineer', 'Acme') RETURNING job_key"
        ).fetchone()[0]
        assert key == "greenhouse:105736"
        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute(
                "INSERT INTO jobs (source, external_id, url, title, company) "
                "VALUES ('greenhouse', '105736', 'https://x/2', 'Other', 'Acme')"
            )


def test_vectors_take_any_width_and_full_text_search_works_without_a_stored_column(db_url):
    """No fixed width (owner's call 2026-10-29, measured: exact matching beats an HNSW index here); embed_tag pins
    the model and width. Text sorts by code point, as SQLite and Python do (docker-compose initdb args)."""
    with pg.connect(db_url) as conn:
        pg.migrate(conn)
        conn.execute(
            "INSERT INTO skill_vocab (name, vector) VALUES ('python', array_fill(0.1::real, ARRAY[1024])::vector), "
            "('go', array_fill(0.1::real, ARRAY[8])::vector)"
        )
        assert conn.execute("SELECT name FROM skill_vocab WHERE vector IS NOT NULL ORDER BY name").fetchall() == [
            ("go",), ("python",)
        ]  # fmt: skip
        assert conn.execute("SELECT to_regclass('skill_vocab_vector')").fetchone() == (None,)  # no HNSW index
        conn.execute(
            "INSERT INTO jobs (source, external_id, url, title, company, description) "
            "VALUES ('lever', 'abc', 'https://x/3', 'LLM Engineer', 'Acme', 'Build retrieval pipelines')"
        )
        # 0004: nothing is stored for search; job_tsvector() is the one definition, and an index on it works
        assert conn.execute(
            "SELECT count(*) FROM information_schema.columns WHERE table_name = 'jobs' AND column_name = 'search'"
        ).fetchone() == (0,)
        match = "job_tsvector(title, company, description) @@ plainto_tsquery('english', %s)"
        assert conn.execute(f"SELECT job_key FROM jobs WHERE {match}", ("retrieval",)).fetchone() == ("lever:abc",)
        assert conn.execute(f"SELECT job_key FROM jobs WHERE {match}", ("acme",)).fetchone() == ("lever:abc",)
        conn.execute("CREATE INDEX jobs_search ON jobs USING gin (job_tsvector(title, company, description))")
        conn.execute("SET enable_seqscan = off")  # one row: force the planner to show the index is usable
        plan = " ".join(r[0] for r in conn.execute(f"EXPLAIN SELECT id FROM jobs WHERE {match}", ("retrieval",)))
        assert "jobs_search" in plan


def test_a_failed_migration_leaves_nothing_applied(db_url, tmp_path):
    (tmp_path / "0001_ok.sql").write_text("CREATE TABLE a (x int);", encoding="utf-8")
    (tmp_path / "0002_bad.sql").write_text("CREATE TABLE b (x int); SELECT * FROM missing_table;", encoding="utf-8")
    with pg.connect(db_url) as conn:
        with pytest.raises(psycopg.errors.UndefinedTable):
            pg.migrate(conn, tmp_path)
        conn.rollback()
        assert {r[0] for r in conn.execute("SELECT version FROM schema_migrations")} == {"0001_ok"}
        assert conn.execute("SELECT to_regclass('b')").fetchone()[0] is None  # 0002's first statement rolled back too


# ── D15.2: an applied migration is never edited ─────────────────────────────
def test_an_edited_applied_migration_is_refused(db_url, tmp_path):
    first = tmp_path / "0001_a.sql"
    first.write_text("CREATE TABLE a (x int);\n", encoding="utf-8")
    with pg.connect(db_url) as conn:
        assert pg.migrate(conn, tmp_path) == ["0001_a"]
        recorded = conn.execute("SELECT checksum FROM schema_migrations WHERE version = '0001_a'").fetchone()[0]
        assert recorded == pg.checksum(first)

        first.write_bytes(b"CREATE TABLE a (x int);\r\n")  # a CRLF checkout of the same text is not an edit
        assert pg.migrate(conn, tmp_path) == []

        first.write_text("CREATE TABLE a (x int, y int);\n", encoding="utf-8")
        (tmp_path / "0002_b.sql").write_text("CREATE TABLE b (x int);\n", encoding="utf-8")
        with pytest.raises(pg.MigrationChanged, match="0001_a: applied, but 0001_a.sql has been edited since"):
            pg.migrate(conn, tmp_path)
        with pytest.raises(pg.MigrationChanged):
            pg.pending(conn, tmp_path)
        assert conn.execute("SELECT to_regclass('b')").fetchone()[0] is None  # nothing after the refusal ran


def test_a_removed_applied_migration_is_refused(db_url, tmp_path):
    (tmp_path / "0001_a.sql").write_text("CREATE TABLE a (x int);\n", encoding="utf-8")
    with pg.connect(db_url) as conn:
        pg.migrate(conn, tmp_path)
        (tmp_path / "0001_a.sql").rename(tmp_path / "0001_renamed.sql")
        with pytest.raises(pg.MigrationChanged, match="0001_a: applied, but its file is gone"):
            pg.migrate(conn, tmp_path)


def test_rows_from_before_the_checksum_take_todays_hash(db_url, tmp_path):
    path = tmp_path / "0001_a.sql"
    path.write_text("CREATE TABLE a (x int);\n", encoding="utf-8")
    with pg.connect(db_url) as conn:
        # a database migrated before D15.2: the table has no checksum column
        conn.execute(
            "CREATE TABLE schema_migrations (version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
        )
        conn.execute("CREATE TABLE a (x int)")
        conn.execute("INSERT INTO schema_migrations (version) VALUES ('0001_a')")
        conn.commit()
        assert pg.migrate(conn, tmp_path) == []
        assert conn.execute("SELECT checksum FROM schema_migrations").fetchone()[0] == pg.checksum(path)
        path.write_text("CREATE TABLE a (y int);\n", encoding="utf-8")
        with pytest.raises(pg.MigrationChanged):
            pg.migrate(conn, tmp_path)
