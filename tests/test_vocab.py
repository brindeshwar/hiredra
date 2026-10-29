"""Tests for hiredra.vocab (build sub-slice 3.2: the skill vocabulary)."""

from __future__ import annotations

from hiredra import vocab
from hiredra.storage import connect, get_skill_vocab, init_db


def test_resolve_skill_inserts_a_new_folded_name(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    canonical = vocab.resolve_skill(conn, "FastAPI")
    assert canonical == "fastapi"
    assert get_skill_vocab(conn, "fastapi")["seen_count"] == 1
    conn.close()


def test_resolve_skill_exact_match_is_case_and_whitespace_insensitive(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    vocab.resolve_skill(conn, "Python")
    canonical = vocab.resolve_skill(conn, "  python  ")
    assert canonical == "python"
    assert get_skill_vocab(conn, "python")["seen_count"] == 2
    conn.close()


def test_resolve_skill_distinct_names_get_distinct_entries(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    vocab.resolve_skill(conn, "python")
    vocab.resolve_skill(conn, "kubernetes")
    assert get_skill_vocab(conn, "python")["seen_count"] == 1
    assert get_skill_vocab(conn, "kubernetes")["seen_count"] == 1
    conn.close()


def test_extraction_time_resolution_does_not_merge_near_duplicates(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    vocab.resolve_skill(conn, "fast api")
    # a genuinely different spelling of the same skill does NOT get merged
    # at extraction time -- that is the between-runs sync step (hiredra.graph.merge).
    canonical = vocab.resolve_skill(conn, "fastapi")
    assert canonical == "fastapi"
    assert get_skill_vocab(conn, "fast api") is not None
    assert get_skill_vocab(conn, "fastapi") is not None
    conn.close()
