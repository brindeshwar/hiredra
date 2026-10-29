"""Tests for hiredra.graph.index: the Skill object and vector codec (sub-slice 4.2)."""

from __future__ import annotations

import numpy as np
import pytest

from hiredra.graph.config import GraphConfig
from hiredra.graph.index import Skill, blob_to_vector, load_skills, save_vectors, vector_to_blob
from hiredra.storage import (
    add_alias,
    add_market_skill,
    connect,
    get_skill_vocab,
    init_db,
    merge_skill,
    set_skill_vectors,
)

DIM = 8


def _cfg(**over) -> GraphConfig:
    base = dict(
        model="gemini-embedding-2",
        dimensions=DIM,
        prefix="",
        inputs_per_request=50,
        requests_per_minute=100,
        dedup_threshold=0.9,
        have=0.9,
        adj=0.7,
    )
    base.update(over)
    return GraphConfig(**base)


def _unit_rows(n: int, seed: int = 0) -> np.ndarray:
    m = np.random.default_rng(seed).normal(size=(n, DIM)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


@pytest.fixture()
def conn(pg_url):
    c = connect(pg_url)
    init_db(c)
    yield c
    c.close()


# ── codec ────────────────────────────────────────────────────────────────────
def test_blob_round_trip_is_exact():
    v = _unit_rows(1)[0]
    out = blob_to_vector(vector_to_blob(v), DIM)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, v)


def test_blob_is_little_endian_float32_and_dim_times_four_bytes():
    assert vector_to_blob(np.array([1.0], dtype=np.float32)) == b"\x00\x00\x80\x3f"
    assert len(vector_to_blob(np.zeros(768, dtype=np.float32))) == 768 * 4 == 3072


def test_blob_accepts_float64_input_and_stores_float32():
    blob = vector_to_blob(np.array([0.5, 0.25], dtype=np.float64))
    assert len(blob) == 8
    np.testing.assert_array_equal(blob_to_vector(blob, 2), np.array([0.5, 0.25], dtype=np.float32))


def test_blob_to_vector_returns_a_writable_copy():
    out = blob_to_vector(vector_to_blob(_unit_rows(1)[0]), DIM)
    out[0] = 99.0  # frombuffer views are read-only; this must not raise


@pytest.mark.parametrize("blob", [b"", b"\x00" * 7, b"\x00" * (DIM * 4 + 4)])
def test_blob_of_the_wrong_length_raises(blob):
    with pytest.raises(ValueError, match="expected"):
        blob_to_vector(blob, DIM)


def test_vector_to_blob_rejects_a_matrix():
    with pytest.raises(ValueError, match="1-D"):
        vector_to_blob(np.zeros((2, 3)))


# ── save / load ──────────────────────────────────────────────────────────────
def test_save_then_load_round_trips_vectors(conn):
    cfg = _cfg()
    names = ["kafka", "docker", "python"]
    for n in names:
        add_market_skill(conn, n)
    m = _unit_rows(3)
    save_vectors(conn, names, m, cfg.embed_tag)

    by_name = {s.name: s for s in load_skills(conn, cfg)}
    for n, row in zip(names, m):
        np.testing.assert_array_equal(by_name[n].vector, row)
    assert get_skill_vocab(conn, "kafka")["embed_tag"] == cfg.embed_tag


def test_load_skills_shape_and_order(conn):
    cfg = _cfg()
    for n in ("b", "a"):
        add_market_skill(conn, n)
    skills = load_skills(conn, cfg)
    assert [s.name for s in skills] == ["a", "b"]
    s = skills[0]
    assert isinstance(s, Skill)
    assert (s.aliases, s.mine, s.level, s.vector, s.edges) == ((), False, None, None, ())


def test_load_skills_attaches_aliases_mine_level_and_edges(conn):
    cfg = _cfg()
    for n in ("fastapi", "fast api", "python"):
        add_market_skill(conn, n)
    merge_skill(conn, "fast api", "fastapi")
    add_alias(conn, "fast-api", "fastapi")
    conn.execute("UPDATE skill_vocab SET mine=true, level='production', source='profile' WHERE name='python'")
    conn.execute(
        "UPDATE skill_vocab SET edges = %s WHERE name = 'fastapi'",
        ('[{"type": "used-in", "target": "python"}]',),
    )
    conn.commit()

    by_name = {s.name: s for s in load_skills(conn, cfg)}
    assert by_name["fastapi"].aliases == ("fast api", "fast-api")
    assert by_name["fastapi"].edges == ({"type": "used-in", "target": "python"},)
    assert by_name["python"].mine is True and by_name["python"].level == "production"
    assert by_name["fastapi"].mine is False


def test_a_vector_made_under_another_tag_loads_as_none(conn):
    """A stale index must look un-embedded -- old-model vectors never leak
    into a similarity query by accident."""
    cfg = _cfg()
    for n in ("a", "b"):
        add_market_skill(conn, n)
    m = _unit_rows(2)
    save_vectors(conn, ["a"], m[:1], "old-model|d8|prefix=none")
    save_vectors(conn, ["b"], m[1:], cfg.embed_tag)
    by_name = {s.name: s for s in load_skills(conn, cfg)}
    assert by_name["a"].vector is None
    np.testing.assert_array_equal(by_name["b"].vector, m[1])


def test_changing_the_prefix_makes_every_vector_stale(conn):
    bare = _cfg()
    prefixed = _cfg(prefix="task: sentence similarity | query: ")
    add_market_skill(conn, "a")
    save_vectors(conn, ["a"], _unit_rows(1), bare.embed_tag)
    assert load_skills(conn, bare)[0].vector is not None
    assert load_skills(conn, prefixed)[0].vector is None


def test_a_corrupt_blob_under_a_matching_tag_raises_rather_than_being_guessed(conn):
    cfg = _cfg()
    add_market_skill(conn, "a")
    set_skill_vectors(conn, [("a", b"\x00" * 20)], cfg.embed_tag)  # 5 floats under a tag that says 8
    with pytest.raises(ValueError, match="expected"):
        load_skills(conn, cfg)


def test_save_vectors_rejects_a_length_mismatch(conn):
    add_market_skill(conn, "a")
    with pytest.raises(ValueError, match="1 names but 2 vectors"):
        save_vectors(conn, ["a"], _unit_rows(2), _cfg().embed_tag)


def test_save_vectors_with_an_unknown_name_writes_nothing(conn):
    cfg = _cfg()
    add_market_skill(conn, "a")
    with pytest.raises(KeyError):
        save_vectors(conn, ["a", "ghost"], _unit_rows(2), cfg.embed_tag)
    assert load_skills(conn, cfg)[0].vector is None


def test_skill_equality_ignores_the_vector():
    a = Skill("x", (), False, None, vector=np.ones(3, dtype=np.float32))
    b = Skill("x", (), False, None, vector=np.zeros(3, dtype=np.float32))
    assert a == b  # ndarray == is elementwise; the vector is excluded from compare
