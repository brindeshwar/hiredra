"""Tests for hiredra.graph.merge (build sub-slice 4.3): merge proposals and the
reviewed apply. Vectors are placed at exact angles so every cosine is known."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hiredra.graph.config import GraphConfig
from hiredra.graph.index import save_vectors
from hiredra.graph.merge import apply_review, propose_merges, write_review
from hiredra.storage import (
    add_market_skill,
    connect,
    get_skill_vocab,
    init_db,
    iter_skill_vocab,
    list_skill_aliases,
    resolve_alias,
)

DIM = 8


def _cfg(**over) -> GraphConfig:
    base = dict(
        model="gemini-embedding-2",
        dimensions=DIM,
        prefix="",
        inputs_per_request=50,
        dedup_threshold=0.9,
        have=0.9,
        adj=0.7,
    )
    base.update(over)
    return GraphConfig(**base)


def _at(cosine: float, axis: int = 1, base: int = 0) -> np.ndarray:
    """Unit vector whose cosine with the basis vector e_base is exactly `cosine`,
    tilted toward e_axis. Vectors built on different `base`/`axis` pairs are
    orthogonal, so unrelated skill groups don't accidentally look alike."""
    v = np.zeros(DIM, dtype=np.float32)
    v[base] = cosine
    v[axis] = math.sqrt(max(0.0, 1.0 - cosine**2))
    return v


def _angle(deg: float) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[0], v[1] = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return v


@pytest.fixture()
def conn(pg_url):
    c = connect(pg_url)
    init_db(c)
    yield c
    c.close()


def _node(conn, cfg, name, vec, *, seen=0, mine=False):
    add_market_skill(conn, name)
    save_vectors(conn, [name], np.asarray([vec], dtype=np.float32), cfg.embed_tag)
    conn.execute("UPDATE skill_vocab SET seen_count = %s, mine = %s WHERE name = %s", (seen, bool(mine), name))
    if mine:
        conn.execute("UPDATE skill_vocab SET source = 'profile', level = 'production' WHERE name = %s", (name,))
    conn.commit()


def _pairs(report):
    return [(m.dup, m.survivor) for m in report.proposals]


# ── proposals ────────────────────────────────────────────────────────────────
def test_two_near_identical_spellings_produce_one_proposal(conn):
    cfg = _cfg()
    _node(conn, cfg, "fastapi", _at(1.0), seen=5)
    _node(conn, cfg, "fast api", _at(0.96), seen=1)
    _node(conn, cfg, "kafka", _at(0.30, axis=2))
    r = propose_merges(conn, cfg)
    assert _pairs(r) == [("fast api", "fastapi")]
    assert r.proposals[0].cosine == pytest.approx(0.96, abs=1e-5)
    assert (r.nodes_considered, r.nodes_skipped) == (3, 0)


@pytest.mark.parametrize(
    "cosine,listed_by_default,listed_with_show_low",
    [
        (0.96, True, True),
        (0.95, True, True),
        (0.93, True, True),  # AUTO / REVIEW: always listed
        (0.9299, False, True),
        (0.9001, False, True),
        (0.90, False, True),  # low band: hidden unless asked
        (0.8999, False, False),
        (0.85, False, False),  # below the threshold: never
    ],
)
def test_threshold_and_band_boundaries(conn, cosine, listed_by_default, listed_with_show_low):
    cfg = _cfg()
    _node(conn, cfg, "aaa", _at(1.0), seen=2)
    _node(conn, cfg, "bbb", _at(cosine), seen=1)
    assert bool(propose_merges(conn, cfg).proposals) is listed_by_default
    assert bool(propose_merges(conn, cfg, show_low=True).proposals) is listed_with_show_low


def test_the_react_vs_react_native_class_below_threshold_is_not_proposed_but_is_a_near_miss(conn):
    cfg = _cfg()
    _node(conn, cfg, "react", _at(1.0), seen=9)
    _node(conn, cfg, "react native", _at(0.765), seen=1)  # the real measured cosine
    _node(conn, cfg, "next.js", _at(0.87, axis=2), seen=1)
    r = propose_merges(conn, cfg)
    assert r.proposals == []
    assert [(a, b) for a, b, _ in r.near_misses] == [("react", "next.js")]  # 0.87 is in [0.85, 0.90); 0.765 is not


def test_survivor_priority_is_mine_then_seen_then_df_then_shorter_then_alpha(conn):
    cfg = _cfg()
    vec = _at(1.0)
    # mine beats a higher seen_count
    _node(conn, cfg, "py-mine", vec, seen=0, mine=True)
    _node(conn, cfg, "py-busy", vec, seen=50)
    # among non-mine: seen_count wins
    _node(conn, cfg, "go-a", _angle(90), seen=3)
    _node(conn, cfg, "go-b", _angle(90), seen=7)
    r = propose_merges(conn, cfg)
    assert ("py-busy", "py-mine") in _pairs(r)
    assert ("go-a", "go-b") in _pairs(r)


def test_df_breaks_seen_count_ties_and_shorter_name_breaks_df_ties(conn):
    cfg = _cfg()
    vec = _at(1.0)
    _node(conn, cfg, "node.js", vec)
    _node(conn, cfg, "nodejs", vec)
    weights = {"node.js": 5, "nodejs": 90}
    assert _pairs(propose_merges(conn, cfg, weights=weights)) == [("node.js", "nodejs")]  # higher df survives
    assert _pairs(propose_merges(conn, cfg)) == [("node.js", "nodejs")]  # tie: shorter name survives


def test_alphabetical_is_the_last_tiebreak(conn):
    cfg = _cfg()
    _node(conn, cfg, "bbb", _at(1.0))
    _node(conn, cfg, "aaa", _at(1.0))
    assert _pairs(propose_merges(conn, cfg)) == [("bbb", "aaa")]


def test_a_mine_node_is_never_proposed_as_the_duplicate(conn):
    cfg = _cfg()
    _node(conn, cfg, "python", _at(1.0), seen=0, mine=True)
    _node(conn, cfg, "python3", _at(0.97), seen=99)  # far busier, but the survivor is still `python`
    assert _pairs(propose_merges(conn, cfg)) == [("python3", "python")]


def test_two_mine_nodes_that_look_alike_are_left_alone(conn):
    cfg = _cfg()
    _node(conn, cfg, "aws", _at(1.0), mine=True)
    _node(conn, cfg, "amazon web services", _at(0.98), mine=True)
    assert propose_merges(conn, cfg).proposals == []


def test_star_clustering_never_chains_a_to_c_through_b(conn):
    """A~B = 0.95 and B~C = 0.95 but A~C = 0.81. B merges into A; C must NOT be
    dragged into A through B (that is how skill families collapse)."""
    cfg = _cfg()
    a, b, c = _angle(0), _angle(18.19), _angle(36.38)
    assert float(a @ b) > 0.94 and float(b @ c) > 0.94 and float(a @ c) < 0.85
    _node(conn, cfg, "a", a, seen=3)
    _node(conn, cfg, "b", b, seen=2)
    _node(conn, cfg, "c", c, seen=1)
    r = propose_merges(conn, cfg)
    assert _pairs(r) == [("b", "a")]  # c stays: it is only close to b, which is a dup
    assert all(m.dup != "c" for m in r.proposals)


def test_a_survivor_claims_every_node_directly_close_to_it(conn):
    cfg = _cfg()
    _node(conn, cfg, "kubernetes", _at(1.0), seen=9)
    _node(conn, cfg, "k8s", _at(0.95, axis=1), seen=1)
    _node(conn, cfg, "kube", _at(0.93, axis=2), seen=1)
    r = propose_merges(conn, cfg)
    assert sorted(_pairs(r)) == [("k8s", "kubernetes"), ("kube", "kubernetes")]


def test_each_duplicate_is_proposed_exactly_once(conn):
    cfg = _cfg()
    for i, name in enumerate(["s1", "s2", "d"]):
        _node(conn, cfg, name, _at(1.0), seen=10 - i)
    dups = [m.dup for m in propose_merges(conn, cfg).proposals]
    assert sorted(dups) == ["d", "s2"] and len(set(dups)) == len(dups)


def test_unembedded_and_stale_nodes_are_not_eligible(conn):
    cfg = _cfg()
    _node(conn, cfg, "a", _at(1.0))
    _node(conn, cfg, "b", _at(0.97))
    add_market_skill(conn, "no-vector-yet")
    _node(conn, _cfg(prefix="other: "), "stale-one", _at(0.99))  # tagged under another config
    r = propose_merges(conn, cfg)
    assert (r.nodes_considered, r.nodes_skipped) == (2, 2)
    assert _pairs(r) == [("b", "a")]


def test_fewer_than_two_embedded_nodes_is_an_empty_report(conn):
    cfg = _cfg()
    _node(conn, cfg, "only", _at(1.0))
    r = propose_merges(conn, cfg)
    assert r.proposals == [] and r.near_misses == []


def test_proposing_is_read_only(conn):
    cfg = _cfg()
    _node(conn, cfg, "fastapi", _at(1.0), seen=5)
    _node(conn, cfg, "fast api", _at(0.96))
    before = iter_skill_vocab(conn)
    propose_merges(conn, cfg)
    assert iter_skill_vocab(conn) == before and list_skill_aliases(conn) == {}


def test_proposal_carries_seen_and_df_for_the_reviewer(conn):
    cfg = _cfg()
    _node(conn, cfg, "fastapi", _at(1.0), seen=5)
    _node(conn, cfg, "fast api", _at(0.96), seen=2)
    m = propose_merges(conn, cfg, weights={"fastapi": 120, "fast api": 4}).proposals[0]
    assert (m.survivor_seen, m.dup_seen, m.survivor_df, m.dup_df) == (5, 2, 120, 4)


def test_near_miss_list_is_ranked_and_limited(conn):
    cfg = _cfg()
    _node(conn, cfg, "base", _at(1.0), seen=9)
    for i, c in enumerate([0.89, 0.88, 0.87, 0.86]):
        _node(conn, cfg, f"n{i}", _at(c, axis=1 + (i % 6)), seen=1)
    r = propose_merges(conn, cfg, near_miss_limit=3)
    assert len(r.near_misses) == 3
    cos = [c for _, _, c in r.near_misses]
    assert cos == sorted(cos, reverse=True) and all(0.85 <= c < 0.9 for c in cos)


# ── the review file ──────────────────────────────────────────────────────────
def _proposed(conn, cfg):
    _node(conn, cfg, "fastapi", _at(1.0), seen=5)
    _node(conn, cfg, "fast api", _at(0.96), seen=1)
    _node(conn, cfg, "kubernetes", _at(1.0, base=2, axis=3), seen=4)
    _node(conn, cfg, "k8s", _at(0.94, base=2, axis=3), seen=1)  # its own group: orthogonal to fastapi's
    return propose_merges(conn, cfg)


def test_review_file_is_readable_json_grouped_by_survivor(conn, tmp_path):
    cfg = _cfg()
    report = _proposed(conn, cfg)
    path = write_review(report, tmp_path / "nested" / "merge_proposals.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["embed_tag"] == cfg.embed_tag and doc["threshold"] == 0.9
    # AUTO first, then REVIEW; within a band grouped by survivor
    assert [(p["band"], p["dup"], p["survivor"]) for p in doc["proposals"]] == [
        ("auto", "fast api", "fastapi"),
        ("review", "k8s", "kubernetes"),
    ]
    assert [p["approve"] for p in doc["proposals"]] == [True, False]  # AUTO pre-approved, REVIEW is not
    assert {"cosine", "reason", "dup_seen", "survivor_seen", "dup_df", "survivor_df"} <= set(doc["proposals"][0])
    assert "approve" in doc["note"] and "Delete an entry" in doc["note"]


def test_writing_the_review_file_does_not_touch_the_database(conn, tmp_path):
    cfg = _cfg()
    report = _proposed(conn, cfg)
    before = iter_skill_vocab(conn)
    write_review(report, tmp_path / "r.json")
    assert iter_skill_vocab(conn) == before and list_skill_aliases(conn) == {}


# ── apply ────────────────────────────────────────────────────────────────────
def test_apply_merges_only_the_approved_pairs_and_archives_the_file(conn, tmp_path):
    cfg = _cfg()
    path = write_review(_proposed(conn, cfg), tmp_path / "r.json")
    r = apply_review(conn, cfg, path)

    assert r.applied == [("fast api", "fastapi")] and r.skipped == []  # AUTO went in...
    assert r.not_approved == 1  # ...the REVIEW pair did not
    assert resolve_alias(conn, "fast api") == "fastapi" and resolve_alias(conn, "k8s") is None
    assert get_skill_vocab(conn, "k8s") is not None
    assert get_skill_vocab(conn, "fastapi")["seen_count"] == 6  # 5 + 1
    assert not path.exists() and r.archived_to.exists() and ".applied-" in r.archived_to.name


def test_ticking_a_review_entry_makes_it_apply(conn, tmp_path):
    cfg = _cfg()
    path = write_review(_proposed(conn, cfg), tmp_path / "r.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    for entry in doc["proposals"]:
        entry["approve"] = True
    path.write_text(json.dumps(doc), encoding="utf-8")
    r = apply_review(conn, cfg, path)
    assert sorted(r.applied) == [("fast api", "fastapi"), ("k8s", "kubernetes")] and r.not_approved == 0
    assert resolve_alias(conn, "k8s") == "kubernetes"


def test_pairs_the_reviewer_deleted_are_not_merged_even_if_pre_approved(conn, tmp_path):
    cfg = _cfg()
    path = write_review(_proposed(conn, cfg), tmp_path / "r.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["proposals"] = [p for p in doc["proposals"] if p["dup"] != "fast api"]  # reviewer drops the AUTO pair
    for entry in doc["proposals"]:
        entry["approve"] = True  # and ticks the REVIEW one
    path.write_text(json.dumps(doc), encoding="utf-8")

    r = apply_review(conn, cfg, path)
    assert r.applied == [("k8s", "kubernetes")]
    assert get_skill_vocab(conn, "fast api") is not None and resolve_alias(conn, "fast api") is None


def test_apply_without_a_review_file_is_refused(conn, tmp_path):
    cfg = _cfg()
    _proposed(conn, cfg)
    before = iter_skill_vocab(conn)
    with pytest.raises(FileNotFoundError, match="dry-run"):
        apply_review(conn, cfg, tmp_path / "never-written.json")
    assert iter_skill_vocab(conn) == before  # nothing merged without a dry-run


def test_apply_refuses_a_file_from_another_embedding_version(conn, tmp_path):
    cfg = _cfg()
    path = write_review(_proposed(conn, cfg), tmp_path / "r.json")
    other = _cfg(prefix="task: x | query: ")
    before = iter_skill_vocab(conn)
    with pytest.raises(ValueError, match="different|embedding version"):
        apply_review(conn, other, path)
    assert iter_skill_vocab(conn) == before and path.exists()  # untouched, not archived


def test_apply_skips_a_hand_added_pair_below_the_threshold(conn, tmp_path):
    cfg = _cfg()
    _node(conn, cfg, "react", _at(1.0), seen=9)
    _node(conn, cfg, "react native", _at(0.765), seen=1)
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            {
                "embed_tag": cfg.embed_tag,
                "proposals": [{"dup": "react native", "survivor": "react", "approve": True}],  # reviewer force-added it
            }
        ),
        encoding="utf-8",
    )
    r = apply_review(conn, cfg, path)
    assert r.applied == [] and "below the floor" in r.skipped[0][2]
    assert get_skill_vocab(conn, "react native") is not None


def test_apply_skips_missing_nodes_mine_dups_and_stale_vectors(conn, tmp_path):
    cfg = _cfg()
    _node(conn, cfg, "python", _at(1.0), mine=True)
    _node(conn, cfg, "python3", _at(0.98))
    _node(conn, cfg, "kept", _at(0.99, axis=2))
    add_market_skill(conn, "unembedded")
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            {
                "embed_tag": cfg.embed_tag,
                "proposals": [
                    {"dup": "python", "survivor": "python3", "approve": True},  # dup is mine
                    {"dup": "ghost", "survivor": "python", "approve": True},  # dup gone
                    {"dup": "python3", "survivor": "ghost", "approve": True},  # survivor gone
                    {"dup": "unembedded", "survivor": "python", "approve": True},  # no vector
                ],
            }
        ),
        encoding="utf-8",
    )
    r = apply_review(conn, cfg, path)
    reasons = {(d, s): why for d, s, why in r.skipped}
    assert r.applied == []
    assert "mine" in reasons[("python", "python3")]
    assert "no longer exists" in reasons[("ghost", "python")]
    assert "no longer exists" in reasons[("python3", "ghost")]
    assert "missing or stale" in reasons[("unembedded", "python")]


def test_reapplying_after_a_crash_is_safe(conn, tmp_path):
    """If a run died after merging some pairs but before archiving the file, running
    it again skips the already-merged pairs rather than failing or double-counting."""
    cfg = _cfg()
    path = write_review(_proposed(conn, cfg), tmp_path / "r.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    for entry in doc["proposals"]:
        entry["approve"] = True
    path.write_text(json.dumps(doc), encoding="utf-8")
    saved = path.read_text(encoding="utf-8")
    first = apply_review(conn, cfg, path)
    assert len(first.applied) == 2
    path.write_text(saved, encoding="utf-8")  # simulate "file was never archived"
    second = apply_review(conn, cfg, path)
    assert second.applied == [] and all("no longer exists" in why for _, _, why in second.skipped)
    assert get_skill_vocab(conn, "fastapi")["seen_count"] == 6  # not double-counted
