"""resolve_skill's alias awareness (build sub-slice 4.2): a merged spelling
resolves to its canonical node with no embedder involved."""

from __future__ import annotations

from hiredra import vocab
from hiredra.storage import (
    add_market_skill,
    connect,
    get_skill_vocab,
    init_db,
    merge_skill,
    touch_skill_vocab,
)


def _db():
    conn = connect(":memory:")
    init_db(conn)
    return conn


def test_a_merged_spelling_resolves_to_its_canonical_node():
    conn = _db()
    touch_skill_vocab(conn, "fastapi")
    touch_skill_vocab(conn, "fast api")
    merge_skill(conn, "fast api", "fastapi")  # fastapi seen_count is now 2

    assert vocab.resolve_skill(conn, "Fast  API") == "fastapi"  # folded, then aliased
    assert get_skill_vocab(conn, "fastapi")["seen_count"] == 3  # bumped the survivor
    assert get_skill_vocab(conn, "fast api") is None  # no node resurrected
    conn.close()


def test_a_market_node_seen_in_a_jd_is_bumped_not_duplicated():
    conn = _db()
    add_market_skill(conn, "kafka")
    assert vocab.resolve_skill(conn, "Kafka") == "kafka"
    row = get_skill_vocab(conn, "kafka")
    assert row["seen_count"] == 1 and row["source"] == "market"
    conn.close()


def test_a_new_unaliased_name_is_still_added_as_a_jd_node():
    conn = _db()
    assert vocab.resolve_skill(conn, "Kubernetes") == "kubernetes"
    row = get_skill_vocab(conn, "kubernetes")
    assert row["seen_count"] == 1 and row["source"] == "jd"
    conn.close()


def test_extraction_time_resolution_never_needs_an_embedder():
    """resolve_skill takes no embedder and near-duplicate detection stays a
    deferred, between-runs step: two spellings are two nodes until a reviewed
    merge is applied."""
    conn = _db()
    vocab.resolve_skill(conn, "fast api")
    assert vocab.resolve_skill(conn, "fastapi") == "fastapi"
    assert get_skill_vocab(conn, "fast api") is not None
    assert get_skill_vocab(conn, "fastapi") is not None
    conn.close()
