"""Storage helpers for the skill graph's nodes and aliases (build sub-slice 4.2)."""

from __future__ import annotations

import struct
from datetime import datetime, timezone

import psycopg

import pytest

from hiredra.storage import (
    add_alias,
    add_market_skill,
    connect,
    get_skill_vocab,
    init_db,
    iter_skill_vocab,
    list_skill_aliases,
    merge_skill,
    resolve_alias,
    set_skill_vectors,
    skills_needing_embedding,
    touch_skill_vocab,
)

TAG = "gemini-embedding-2|d768|prefix=none"

# Vectors are float32 bytes (pgvector stores them; the graph codec owns the format).
KEEP, DROP, VX, VY, VZ = (struct.pack("<2f", a, b) for a, b in ((1, 0), (0, 1), (0.6, 0.8), (0.8, 0.6), (-1, 0)))
V3 = struct.pack("<3f", 0.25, -0.5, 1.0)


@pytest.fixture()
def conn(pg_url):
    c = connect(pg_url)
    init_db(c)
    yield c
    c.close()


def _set(conn, name, **cols):
    sets = ", ".join(f"{k} = %s" for k in cols)
    conn.execute(f"UPDATE skill_vocab SET {sets} WHERE name = %s", (*cols.values(), name))


# ── add_market_skill ─────────────────────────────────────────────────────────
def test_add_market_skill_inserts_an_unseen_market_node(conn):
    assert add_market_skill(conn, "kafka") is True
    row = get_skill_vocab(conn, "kafka")
    assert row["source"] == "market"
    assert row["seen_count"] == 0  # never seen in one of OUR jobs
    assert row["mine"] is False and row["level"] is None
    assert row["vector"] is None and row["embed_tag"] is None
    assert row["edges"] == []


def test_add_market_skill_is_insert_or_ignore(conn):
    add_market_skill(conn, "kafka")
    _set(conn, "kafka", seen_count=7)
    assert add_market_skill(conn, "kafka") is False
    assert get_skill_vocab(conn, "kafka")["seen_count"] == 7  # not clobbered


def test_add_market_skill_does_not_rewrite_an_existing_jd_node(conn):
    touch_skill_vocab(conn, "python")  # seen in a real JD -> source 'jd', count 1
    assert add_market_skill(conn, "python") is False
    row = get_skill_vocab(conn, "python")
    assert row["source"] == "jd" and row["seen_count"] == 1


def test_add_market_skill_skips_a_name_that_is_a_merged_alias(conn):
    add_market_skill(conn, "fastapi")
    add_market_skill(conn, "fast api")
    merge_skill(conn, "fast api", "fastapi")
    assert add_market_skill(conn, "fast api") is False
    assert get_skill_vocab(conn, "fast api") is None  # no node resurrected


def test_a_jd_sighting_of_a_market_node_bumps_the_count_but_keeps_its_source(conn):
    add_market_skill(conn, "kafka")
    touch_skill_vocab(conn, "kafka")
    row = get_skill_vocab(conn, "kafka")
    assert row["seen_count"] == 1 and row["source"] == "market"


# ── schema CHECKs (fresh DB) ─────────────────────────────────────────────────
@pytest.mark.parametrize("column,bad", [("source", "'bogus'"), ("mine", "2"), ("level", "'expert'")])
def test_checks_reject_bad_values(conn, column, bad):
    with pytest.raises((psycopg.errors.CheckViolation, psycopg.errors.DatatypeMismatch)):
        conn.execute(f"INSERT INTO skill_vocab (name, first_seen_at, {column}) VALUES ('x', now(), {bad})")


def test_profile_source_and_every_level_are_accepted(conn):
    for i, level in enumerate(("production", "built-with", "learned", "interview-ready")):
        conn.execute(
            "INSERT INTO skill_vocab (name, first_seen_at, mine, level, source) VALUES (%s, now(), true, %s, 'profile')",
            (f"s{i}", level),
        )


# ── aliases ──────────────────────────────────────────────────────────────────
def test_add_alias_then_resolve(conn):
    add_market_skill(conn, "fastapi")
    add_alias(conn, "fast api", "fastapi")
    assert resolve_alias(conn, "fast api") == "fastapi"
    assert resolve_alias(conn, "fastapi") is None  # a node is not an alias
    assert resolve_alias(conn, "unknown") is None


def test_add_alias_is_idempotent_for_the_same_target(conn):
    add_market_skill(conn, "fastapi")
    add_alias(conn, "fast api", "fastapi")
    add_alias(conn, "fast api", "fastapi")
    assert list_skill_aliases(conn) == {"fastapi": ["fast api"]}


def test_add_alias_refuses_to_retarget(conn):
    add_market_skill(conn, "fastapi")
    add_market_skill(conn, "flask")
    add_alias(conn, "fast api", "fastapi")
    with pytest.raises(ValueError, match="already points at"):
        add_alias(conn, "fast api", "flask")


def test_add_alias_refuses_a_name_that_is_a_node(conn):
    add_market_skill(conn, "fastapi")
    add_market_skill(conn, "fast api")
    with pytest.raises(ValueError, match="use merge_skill"):
        add_alias(conn, "fast api", "fastapi")


def test_add_alias_refuses_a_missing_canonical_and_self_alias(conn):
    with pytest.raises(ValueError, match="does not exist"):
        add_alias(conn, "fast api", "fastapi")
    add_market_skill(conn, "fastapi")
    with pytest.raises(ValueError, match="itself"):
        add_alias(conn, "fastapi", "fastapi")


def test_alias_foreign_key_is_enforced_at_the_db_level(conn):
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        conn.execute("INSERT INTO skill_alias (alias, canonical, created_at) VALUES ('a', 'nope', now())")


# ── merge_skill ──────────────────────────────────────────────────────────────
def test_merge_folds_counts_dates_and_records_the_alias(conn):
    touch_skill_vocab(conn, "fastapi")  # seen_count 1
    touch_skill_vocab(conn, "fast api")
    touch_skill_vocab(conn, "fast api")  # seen_count 2
    _set(conn, "fastapi", first_seen_at="2026-05-01T00:00:00+00:00")
    _set(conn, "fast api", first_seen_at="2026-03-01T00:00:00+00:00")

    merge_skill(conn, "fast api", "fastapi")

    assert get_skill_vocab(conn, "fast api") is None
    survivor = get_skill_vocab(conn, "fastapi")
    assert survivor["seen_count"] == 3
    assert survivor["first_seen_at"] == datetime(2026, 3, 1, tzinfo=timezone.utc)  # the earlier one
    assert resolve_alias(conn, "fast api") == "fastapi"


def test_merge_keeps_the_survivors_vector_and_drops_the_dups(conn):
    add_market_skill(conn, "fastapi")
    add_market_skill(conn, "fast api")
    set_skill_vectors(conn, [("fastapi", KEEP), ("fast api", DROP)], TAG)
    merge_skill(conn, "fast api", "fastapi")
    row = get_skill_vocab(conn, "fastapi")
    assert row["vector"] == KEEP and row["embed_tag"] == TAG


def test_merge_repoints_existing_aliases_so_they_stay_one_hop(conn):
    for n in ("fastapi", "fast api", "fast-api"):
        add_market_skill(conn, n)
    add_alias(conn, "fast-api-framework", "fast api")  # alias of the soon-to-be-merged dup
    merge_skill(conn, "fast api", "fastapi")
    assert resolve_alias(conn, "fast-api-framework") == "fastapi"
    assert resolve_alias(conn, "fast api") == "fastapi"
    assert list_skill_aliases(conn) == {"fastapi": ["fast api", "fast-api-framework"]}


def test_merge_never_absorbs_a_mine_skill(conn):
    add_market_skill(conn, "python")
    add_market_skill(conn, "python3")
    _set(conn, "python3", mine=True, level="production", source="profile")
    with pytest.raises(ValueError, match="never merged away"):
        merge_skill(conn, "python3", "python")
    assert get_skill_vocab(conn, "python3") is not None  # rolled back / untouched


def test_a_mine_node_may_absorb_a_market_duplicate(conn):
    add_market_skill(conn, "python")
    add_market_skill(conn, "python3")
    _set(conn, "python", mine=True, level="production", source="profile")
    merge_skill(conn, "python3", "python")
    survivor = get_skill_vocab(conn, "python")
    assert survivor["mine"] == 1 and survivor["level"] == "production"


def test_merge_rejects_self_and_missing(conn):
    add_market_skill(conn, "a")
    with pytest.raises(ValueError, match="itself"):
        merge_skill(conn, "a", "a")
    with pytest.raises(ValueError, match="does not exist"):
        merge_skill(conn, "ghost", "a")
    with pytest.raises(ValueError, match="does not exist"):
        merge_skill(conn, "a", "ghost")


# ── embedding work queue / vector writes ─────────────────────────────────────
def test_needing_embedding_lists_unembedded_nodes_valuable_first(conn):
    for n in ("zeta", "alpha", "beta", "gamma"):
        add_market_skill(conn, n)
    _set(conn, "beta", seen_count=9)
    _set(conn, "gamma", seen_count=2)
    _set(conn, "zeta", mine=True, source="profile")
    # mine first, then most-sighted, then by name
    assert skills_needing_embedding(conn, TAG) == ["zeta", "beta", "gamma", "alpha"]
    assert skills_needing_embedding(conn, TAG, limit=2) == ["zeta", "beta"]


def test_needing_embedding_excludes_current_and_includes_stale(conn):
    for n in ("a", "b", "c"):
        add_market_skill(conn, n)
    set_skill_vectors(conn, [("a", VX), ("b", VX)], TAG)
    set_skill_vectors(conn, [("c", VX)], "old-model|d768|prefix=none")
    assert skills_needing_embedding(conn, TAG) == ["c"]  # c is stale
    assert sorted(skills_needing_embedding(conn, "other-tag")) == ["a", "b", "c"]


def test_needing_embedding_is_empty_when_everything_is_current(conn):
    add_market_skill(conn, "a")
    set_skill_vectors(conn, [("a", VX)], TAG)
    assert skills_needing_embedding(conn, TAG) == []


def test_set_skill_vectors_stores_bytes_and_tag(conn):
    add_market_skill(conn, "a")
    set_skill_vectors(conn, [("a", V3)], TAG)
    row = get_skill_vocab(conn, "a")
    assert row["vector"] == V3 and row["embed_tag"] == TAG


def test_set_skill_vectors_is_all_or_nothing(conn):
    add_market_skill(conn, "a")
    add_market_skill(conn, "b")
    with pytest.raises(KeyError, match="ghost"):
        set_skill_vectors(conn, [("a", VX), ("ghost", VY), ("b", VZ)], TAG)
    assert get_skill_vocab(conn, "a")["vector"] is None  # 'a' rolled back with the batch
    assert get_skill_vocab(conn, "b")["vector"] is None


# ── listing ──────────────────────────────────────────────────────────────────
def test_iter_skill_vocab_and_aliases_shape(conn):
    for n in ("b", "a", "c"):
        add_market_skill(conn, n)
    merge_skill(conn, "c", "a")
    assert [r["name"] for r in iter_skill_vocab(conn)] == ["a", "b"]
    assert set(iter_skill_vocab(conn)[0]) >= {"name", "vector", "embed_tag", "mine", "level", "source", "edges"}
    assert list_skill_aliases(conn) == {"a": ["c"]}


# ── index status / invalidation / weighted queue (build 4.3) ─────────────────
def test_skill_index_status_counts_by_state_and_source(conn):
    from hiredra.storage import skill_index_status

    for n in ("a", "b", "c", "d"):
        add_market_skill(conn, n)
    touch_skill_vocab(conn, "jdskill")  # source 'jd'
    set_skill_vectors(conn, [("a", VX), ("b", VX)], TAG)
    set_skill_vectors(conn, [("c", VX)], "old-model|d768|prefix=none")
    add_alias(conn, "alias-of-a", "a")
    st = skill_index_status(conn, TAG)
    assert st == {
        "total": 5,
        "current": 2,
        "stale": 1,
        "unembedded": 2,
        "market": 4,
        "jd": 1,
        "profile": 0,
        "aliases": 1,
    }


def test_skill_index_status_on_an_empty_db_is_all_zero(conn):
    from hiredra.storage import skill_index_status

    assert set(skill_index_status(conn, TAG).values()) == {0}


def test_invalidate_skill_vectors_clears_every_vector_and_tag(conn):
    from hiredra.storage import invalidate_skill_vectors

    for n in ("a", "b", "c"):
        add_market_skill(conn, n)
    set_skill_vectors(conn, [("a", VX), ("b", VY)], TAG)
    assert invalidate_skill_vectors(conn) == 2  # 'c' never had one
    for n in ("a", "b", "c"):
        row = get_skill_vocab(conn, n)
        assert row["vector"] is None and row["embed_tag"] is None
    assert sorted(skills_needing_embedding(conn, TAG)) == ["a", "b", "c"]


def test_invalidate_leaves_everything_but_vectors_alone(conn):
    from hiredra.storage import invalidate_skill_vectors

    add_market_skill(conn, "a")
    _set(conn, "a", seen_count=7, mine=True, level="production", source="profile")
    set_skill_vectors(conn, [("a", VX)], TAG)
    invalidate_skill_vectors(conn)
    row = get_skill_vocab(conn, "a")
    assert (row["seen_count"], row["mine"], row["level"], row["source"]) == (7, 1, "production", "profile")


def test_queue_weights_order_market_nodes_but_never_outrank_mine_or_seen_count(conn):
    for n in ("rare", "common", "mine-one", "seen-one"):
        add_market_skill(conn, n)
    _set(conn, "mine-one", mine=True, source="profile")
    _set(conn, "seen-one", seen_count=3)
    order = skills_needing_embedding(conn, TAG, weights={"common": 500, "rare": 1, "seen-one": 0, "mine-one": 0})
    assert order == ["mine-one", "seen-one", "common", "rare"]


def test_queue_limit_applies_after_weighted_ordering(conn):
    for n in ("a", "b", "c"):
        add_market_skill(conn, n)
    assert skills_needing_embedding(conn, TAG, limit=2, weights={"c": 9, "b": 5, "a": 1}) == ["c", "b"]
