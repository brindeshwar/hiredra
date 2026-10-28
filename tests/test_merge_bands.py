"""Tests for the three-band merge policy (build 4.3): AUTO / ALIAS / REVIEW / LEAVE,
the alias file, the never-merge list, and the approve-only apply.
Vectors sit at exact angles so every cosine is known. No network."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hiredra.graph.config import GraphConfig
from hiredra.graph.index import save_vectors
from hiredra.graph.merge import (
    SPELLING_FLOOR,
    apply_review,
    classify_pair,
    is_specialization,
    is_version_variant,
    normalized_key,
    propose_merges,
    same_spelling,
    spelling_keys,
    write_review,
)
from hiredra.storage import (
    add_alias,
    add_market_skill,
    connect,
    get_skill_vocab,
    init_db,
    iter_skill_vocab,
    resolve_alias,
)

DIM = 8


def _cfg(**over) -> GraphConfig:
    base = dict(
        model="gemini-embedding-2",
        dimensions=DIM,
        prefix="",
        inputs_per_request=50,
        requests_per_minute=100,
        dedup_threshold=0.90,
        review_cosine=0.93,
        auto_cosine=0.96,
        have=0.9,
        adj=0.7,
    )
    base.update(over)
    return GraphConfig(**base)


def _at(cosine: float, axis: int = 1, base: int = 0) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[base] = cosine
    v[axis] = math.sqrt(max(0.0, 1.0 - cosine**2))
    return v


@pytest.fixture()
def conn(pg_url):
    c = connect(pg_url)
    init_db(c)
    yield c
    c.close()


def _node(conn, cfg, name, vec=None, *, seen=0, mine=False, embed=True):
    add_market_skill(conn, name)
    if embed:
        save_vectors(conn, [name], np.asarray([vec], dtype=np.float32), cfg.embed_tag)
    conn.execute("UPDATE skill_vocab SET seen_count = %s, mine = %s WHERE name = %s", (seen, bool(mine), name))
    if mine:
        conn.execute("UPDATE skill_vocab SET source = 'profile', level = 'production' WHERE name = %s", (name,))
    conn.commit()


def _band(report, band):
    return sorted((m.dup, m.survivor) for m in report.by_band(band))


# ── the name rules ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "a,b,same",
    [
        ("github actions", "github-actions", True),
        ("rest api", "rest apis", True),
        ("Full Stack  Development", "full-stack development", True),
        ("oauth 2.0", "oauth2", True),
        ("node.js", "nodejs", True),
        ("pub/sub", "pubsub", True),
        ("system design", "systems design", True),
        ("c#", "csharp", True),
        ("c#", "c++", False),  # the bug the first read-through caught: must never collapse
        ("c#", "c", False),
        ("c++", "c", False),
        ("f#", "f", False),
        (".net", "net", False),
        ("oauth", "oauth2", False),
        ("sql", "sql server", False),
    ],
)
def test_same_spelling(a, b, same):
    assert same_spelling(a, b) is same
    assert same_spelling(b, a) is same  # symmetric


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("sql", "sql server", True),
        ("react", "react native", True),
        ("backend", "backend development", True),
        ("gitlab", "gitlab ci", True),
        ("kubernetes", "k8s", False),
        ("aws", "amazon web services", False),
        ("rest api", "rest apis", False),  # same words once plurals fold: not a PROPER subset
    ],
)
def test_is_specialization(a, b, expected):
    assert is_specialization(a, b) is expected


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("oauth", "oauth2", True),
        ("css", "css3", True),
        ("python", "python3", True),
        ("http", "http2", True),
        ("python 2", "python 3", True),
        ("ipv4", "ipv6", True),
        ("oauth 2.0", "oauth2", False),  # identical keys are a spelling match, not a variant
        ("react", "vue", False),
        ("s3", "s3", False),
    ],
)
def test_is_version_variant(a, b, expected):
    assert is_version_variant(normalized_key(a), normalized_key(b)) is expected


# ── classification table (default config: 0.90 / 0.93 / 0.96) ────────────────
@pytest.mark.parametrize(
    "a,b,cos,expected",
    [
        ("fastapi", "fast api", 0.965, ("auto", "same spelling after normalising")),
        ("github-actions", "github actions", 0.94, ("auto", "same spelling after normalising")),
        ("rest api", "rest apis", 0.96, ("auto", "same spelling after normalising")),
        ("oauth 2.0", "oauth2", 0.963, ("auto", "same spelling after normalising")),
        ("postgres", "postgresql", 0.974, ("auto", "cosine >= 0.96")),
        ("aws bedrock", "amazon bedrock", 0.979, ("auto", "cosine >= 0.96")),
        # the owner's change: a version variant is REVIEW even at very high cosine
        ("oauth", "oauth2", 0.961, ("review", "version variant (differs only by a number)")),
        ("css", "css3", 0.99, ("review", "version variant (differs only by a number)")),
        ("python", "python3", 0.99, ("review", "version variant (differs only by a number)")),
        ("python 2", "python 3", 0.99, ("review", "version variant (differs only by a number)")),
        # a specialization is REVIEW even at very high cosine, and LOW (hidden) under the review cutoff
        ("sql", "sql server", 0.97, ("review", "one name contains the other")),
        ("react", "react native", 0.99, ("review", "one name contains the other")),
        ("sql", "sql server", 0.925, ("low", "one name contains the other; below the 0.93 review cutoff")),
        # plain similar pairs
        ("cloud architecture", "cloud infrastructure", 0.94, ("review", "similar")),
        ("cloud computing", "cloud infrastructure", 0.907, ("low", "similar; below the 0.93 review cutoff")),
        ("x y z", "q r s", 0.965, ("auto", "cosine >= 0.96")),
        # left alone
        ("machine learning", "deep learning", 0.894, None),
        ("devops", "azure devops", 0.897, None),
        ("c#", "c++", 0.867, None),  # the collision the fix prevents
        ("c#", "csharp", 0.84, None),  # spelling match, but below the 0.85 floor
        ("a b", "a-b", 0.80, None),
    ],
)
def test_classify_pair(a, b, cos, expected):
    assert classify_pair(a, b, cos, _cfg()) == expected


def test_a_spelling_match_is_auto_down_to_the_spelling_floor_even_below_the_threshold():
    assert SPELLING_FLOOR == 0.85
    assert classify_pair("c#", "csharp", 0.86, _cfg())[0] == "auto"  # 0.86 < the 0.90 threshold, still AUTO
    assert classify_pair("c#", "csharp", 0.849, _cfg()) is None


def test_the_band_cutoffs_come_from_config():
    strict = _cfg(auto_cosine=0.99)
    assert classify_pair("x y z", "q r s", 0.97, strict) == ("review", "similar")
    loose = _cfg(review_cosine=0.90)
    assert classify_pair("cloud computing", "cloud infrastructure", 0.907, loose)[0] == "review"


# ── proposals through the database ───────────────────────────────────────────
def test_oauth_lands_in_review_not_auto(conn):
    cfg = _cfg()
    _node(conn, cfg, "oauth2", _at(1.0), seen=5)
    _node(conn, cfg, "oauth", _at(0.961), seen=1)
    r = propose_merges(conn, cfg)
    assert _band(r, "auto") == [] and _band(r, "review") == [("oauth", "oauth2")]
    assert r.by_band("review")[0].approve is False


def test_the_c_family_never_collapses_even_when_vectors_are_close(conn):
    cfg = _cfg()
    _node(conn, cfg, "c", _at(1.0))
    _node(conn, cfg, "c#", _at(0.867), seen=1)
    _node(conn, cfg, "c++", _at(0.867, axis=2), seen=1)
    assert propose_merges(conn, cfg, show_low=True).proposals == []


def test_auto_and_review_and_low_are_reported_separately(conn):
    cfg = _cfg()
    _node(conn, cfg, "fastapi", _at(1.0), seen=9)
    _node(conn, cfg, "fast api", _at(0.97), seen=1)  # spelling -> AUTO
    _node(conn, cfg, "sql", _at(1.0, base=2, axis=3), seen=9)
    _node(conn, cfg, "sql server", _at(0.925, base=2, axis=3), seen=1)  # specialization, low band
    _node(conn, cfg, "cloud architecture", _at(1.0, base=4, axis=5), seen=9)
    _node(conn, cfg, "cloud infrastructure", _at(0.94, base=4, axis=5), seen=1)  # review
    r = propose_merges(conn, cfg)
    assert _band(r, "auto") == [("fast api", "fastapi")]
    assert _band(r, "review") == [("cloud infrastructure", "cloud architecture")]
    assert r.hidden_low == 1  # sql ~ sql server, not listed
    shown = propose_merges(conn, cfg, show_low=True)
    assert ("sql server", "sql") in _band(shown, "review")
    assert shown.hidden_low == 0
    assert "below the 0.93 review cutoff" in [m.reason for m in shown.proposals if m.dup == "sql server"][0]


def test_review_pairs_are_never_pre_approved_and_auto_pairs_always_are(conn):
    cfg = _cfg()
    _node(conn, cfg, "fastapi", _at(1.0), seen=9)
    _node(conn, cfg, "fast api", _at(0.97), seen=1)
    _node(conn, cfg, "cloud architecture", _at(1.0, base=4, axis=5), seen=9)
    _node(conn, cfg, "cloud infrastructure", _at(0.94, base=4, axis=5), seen=1)
    r = propose_merges(conn, cfg)
    assert [(m.band, m.approve) for m in r.proposals] == [("auto", True), ("review", False)]


def test_a_node_that_survives_an_auto_cluster_is_never_merged_away_by_a_review_pair(conn):
    """j~k is a spelling match (AUTO, j survives). i~j is only REVIEW. The star rule says j
    stays a survivor: it is NOT also proposed as a duplicate of i (that would chain k into i)."""
    cfg = _cfg()
    _node(conn, cfg, "vector database", _at(1.0), seen=9)  # i (highest priority)
    _node(conn, cfg, "vector stores", _at(0.94), seen=5)  # j
    _node(conn, cfg, "vector store", _at(0.94, axis=2), seen=1)  # k: spelling match of j
    r = propose_merges(conn, cfg)
    assert _band(r, "auto") == [("vector store", "vector stores")]
    assert all(m.dup != "vector stores" for m in r.proposals)  # j is not merged into i


def test_auto_pairs_are_claimed_before_review_pairs_can_take_the_node(conn):
    cfg = _cfg()
    _node(conn, cfg, "vector database", _at(1.0), seen=9)  # highest priority; only REVIEW-close to `vector store`
    _node(conn, cfg, "vector stores", _at(0.94, axis=2), seen=5)
    _node(conn, cfg, "vector store", _at(0.94, axis=3), seen=1)
    r = propose_merges(conn, cfg)
    assert ("vector store", "vector stores") in _band(r, "auto")


def test_a_mine_node_is_still_never_a_duplicate_in_any_band(conn):
    cfg = _cfg()
    _node(conn, cfg, "python", _at(1.0), mine=True)
    _node(conn, cfg, "python3", _at(0.99), seen=9)  # version variant -> REVIEW
    r = propose_merges(conn, cfg)
    assert _band(r, "review") == [("python3", "python")]  # python (mine) survives
    r2 = propose_merges(conn, cfg)
    assert all(m.dup != "python" for m in r2.proposals)


# ── the never-merge list ─────────────────────────────────────────────────────
def test_a_never_merge_pair_is_never_proposed_in_any_band(conn):
    cfg = _cfg()
    _node(conn, cfg, "sql", _at(1.0), seen=9)
    _node(conn, cfg, "sql server", _at(0.97), seen=1)  # would be REVIEW
    _node(conn, cfg, "fastapi", _at(1.0, base=2, axis=3), seen=9)
    _node(conn, cfg, "fast api", _at(0.97, base=2, axis=3), seen=1)  # would be AUTO
    never = {frozenset(("sql", "sql server")), frozenset(("fast api", "fastapi"))}
    r = propose_merges(conn, cfg, never=never)
    assert r.proposals == [] and r.never_blocked == 2


def test_a_never_merge_pair_does_not_appear_as_a_near_miss(conn):
    cfg = _cfg()
    _node(conn, cfg, "machine learning", _at(1.0), seen=9)
    _node(conn, cfg, "deep learning", _at(0.894), seen=1)
    assert propose_merges(conn, cfg).near_misses == [
        ("machine learning", "deep learning", pytest.approx(0.894, abs=1e-4))
    ]
    never = {frozenset(("machine learning", "deep learning"))}
    assert propose_merges(conn, cfg, never=never).near_misses == []


def test_never_merge_is_order_independent(conn):
    cfg = _cfg()
    _node(conn, cfg, "sql", _at(1.0), seen=9)
    _node(conn, cfg, "sql server", _at(0.97), seen=1)
    assert propose_merges(conn, cfg, never={frozenset(("sql server", "sql"))}).proposals == []


# ── the alias band ───────────────────────────────────────────────────────────
def test_an_alias_that_is_not_a_node_is_proposed_as_alias_only(conn):
    cfg = _cfg()
    _node(conn, cfg, "kubernetes", _at(1.0), seen=3)
    r = propose_merges(conn, cfg, aliases={"kubernetes": ("k8s",)})
    m = r.by_band("alias")[0]
    assert (m.dup, m.survivor, m.cosine, m.approve) == ("k8s", "kubernetes", None, True)
    assert "alias only" in m.reason


def test_an_alias_that_is_a_node_is_proposed_as_a_merge_with_its_cosine(conn):
    cfg = _cfg()
    _node(conn, cfg, "go", _at(1.0), seen=3)
    _node(conn, cfg, "golang", _at(0.767), seen=1)  # far below any cosine band
    r = propose_merges(conn, cfg, aliases={"go": ("golang",)})
    m = r.by_band("alias")[0]
    assert (m.dup, m.survivor) == ("golang", "go") and m.cosine == pytest.approx(0.767, abs=1e-4)


def test_the_alias_band_needs_no_vectors_at_all(conn):
    cfg = _cfg()
    _node(conn, cfg, "kubernetes", embed=False)
    _node(conn, cfg, "kube", embed=False)
    r = propose_merges(conn, cfg, aliases={"kubernetes": ("kube", "k8s")})
    assert sorted(m.dup for m in r.by_band("alias")) == ["k8s", "kube"] and r.nodes_considered == 0


def test_an_alias_pair_is_not_proposed_a_second_time_by_cosine(conn):
    cfg = _cfg()
    _node(conn, cfg, "postgresql", _at(1.0), seen=9)
    _node(conn, cfg, "postgres", _at(0.974), seen=1)  # would also be AUTO by cosine
    r = propose_merges(conn, cfg, aliases={"postgresql": ("postgres",)})
    assert [m.band for m in r.proposals] == ["alias"]


def test_alias_entries_that_cannot_apply_are_reported_not_silently_dropped(conn):
    cfg = _cfg()
    _node(conn, cfg, "kubernetes", _at(1.0))
    _node(conn, cfg, "python", _at(1.0, base=2, axis=3), mine=True)
    _node(conn, cfg, "py", _at(0.5, base=2, axis=3))
    _node(conn, cfg, "sql", _at(1.0, base=4, axis=5))
    _node(conn, cfg, "sql server", _at(0.9, base=4, axis=5))
    _node(conn, cfg, "kube", _at(0.8, base=6, axis=7))
    add_alias(conn, "kube", "kubernetes") if False else None
    add_alias(conn, "ktes", "kubernetes")
    r = propose_merges(
        conn,
        cfg,
        aliases={
            "no-such-skill": ("x",),  # canonical missing
            "kubernetes": ("ktes",),  # already applied -> silent
            "python": ("py",),  # canonical is fine; alias is a plain node -> proposed
            "sql": ("sql server",),  # pair is in the never-merge list
        },
        never={frozenset(("sql", "sql server"))},
    )
    notes = {(a, c): why for a, c, why in r.alias_notes}
    assert "not a node" in notes[("x", "no-such-skill")]
    assert "never-merge" in notes[("sql server", "sql")]
    assert ("ktes", "kubernetes") not in notes and all(m.dup != "ktes" for m in r.proposals)
    assert [m.dup for m in r.by_band("alias")] == ["py"]


def test_an_alias_that_is_a_mine_skill_is_reported_not_proposed(conn):
    cfg = _cfg()
    _node(conn, cfg, "javascript", _at(1.0))
    _node(conn, cfg, "js", _at(0.5), mine=True)
    r = propose_merges(conn, cfg, aliases={"javascript": ("js",)})
    assert r.by_band("alias") == [] and "mine" in r.alias_notes[0][2]


def test_an_alias_already_pointing_elsewhere_is_a_reported_conflict(conn):
    cfg = _cfg()
    _node(conn, cfg, "javascript", _at(1.0))
    _node(conn, cfg, "typescript", _at(0.3))
    add_alias(conn, "js", "typescript")
    r = propose_merges(conn, cfg, aliases={"javascript": ("js",)})
    assert r.by_band("alias") == [] and "already an alias of 'typescript'" in r.alias_notes[0][2]


# ── the review file + apply ──────────────────────────────────────────────────
def _all_bands(conn, cfg):
    _node(conn, cfg, "fastapi", _at(1.0), seen=9)
    _node(conn, cfg, "fast api", _at(0.97), seen=1)  # AUTO
    _node(conn, cfg, "oauth2", _at(1.0, base=2, axis=3), seen=9)
    _node(conn, cfg, "oauth", _at(0.961, base=2, axis=3), seen=1)  # REVIEW (version variant)
    _node(conn, cfg, "kubernetes", _at(1.0, base=4, axis=5), seen=9)  # alias target
    return propose_merges(conn, cfg, aliases={"kubernetes": ("k8s",)})


def test_the_file_carries_band_approve_and_reason_for_every_entry(conn, tmp_path):
    cfg = _cfg()
    path = write_review(_all_bands(conn, cfg), tmp_path / "r.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    got = [(p["band"], p["dup"], p["approve"]) for p in doc["proposals"]]
    assert got == [("auto", "fast api", True), ("alias", "k8s", True), ("review", "oauth", False)]
    assert all(p["reason"] for p in doc["proposals"])
    assert doc["proposals"][1]["cosine"] is None  # alias-only: no cosine


def test_apply_applies_auto_and_alias_but_not_the_unticked_review_entry(conn, tmp_path):
    cfg = _cfg()
    path = write_review(_all_bands(conn, cfg), tmp_path / "r.json")
    r = apply_review(conn, cfg, path)
    assert sorted(r.applied) == [("fast api", "fastapi"), ("k8s", "kubernetes")] and r.not_approved == 1
    assert resolve_alias(conn, "fast api") == "fastapi"
    assert (
        resolve_alias(conn, "k8s") == "kubernetes" and get_skill_vocab(conn, "k8s") is None
    )  # alias only: no node made
    assert get_skill_vocab(conn, "oauth") is not None and resolve_alias(conn, "oauth") is None


def test_an_alias_only_entry_never_creates_a_node(conn, tmp_path):
    cfg = _cfg()
    path = write_review(_all_bands(conn, cfg), tmp_path / "r.json")
    before = {r["name"] for r in iter_skill_vocab(conn)}
    apply_review(conn, cfg, path)
    assert {r["name"] for r in iter_skill_vocab(conn)} == before - {"fast api"}  # only the merged dup is gone


@pytest.mark.parametrize("value", [False, None, "true", "yes", 1, 0, "True"])
def test_only_a_real_json_true_counts_as_approval(conn, tmp_path, value):
    cfg = _cfg()
    path = write_review(_all_bands(conn, cfg), tmp_path / "r.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    for entry in doc["proposals"]:
        entry["approve"] = value
    path.write_text(json.dumps(doc), encoding="utf-8")
    r = apply_review(conn, cfg, path)
    assert r.applied == [] and r.not_approved == 3


def test_a_missing_approve_field_means_not_approved(conn, tmp_path):
    cfg = _cfg()
    _all_bands(conn, cfg)
    path = tmp_path / "old-format.json"
    path.write_text(
        json.dumps(
            {"embed_tag": cfg.embed_tag, "proposals": [{"dup": "fast api", "survivor": "fastapi", "cosine": 0.97}]}
        ),
        encoding="utf-8",
    )  # the pre-band file format
    r = apply_review(conn, cfg, path)
    assert r.applied == [] and r.not_approved == 1


def test_a_ticked_review_entry_applies(conn, tmp_path):
    cfg = _cfg()
    path = write_review(_all_bands(conn, cfg), tmp_path / "r.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    for entry in doc["proposals"]:
        if entry["band"] == "review":
            entry["approve"] = True
    path.write_text(json.dumps(doc), encoding="utf-8")
    r = apply_review(conn, cfg, path)
    assert ("oauth", "oauth2") in r.applied and r.not_approved == 0


def test_apply_rechecks_the_never_merge_list(conn, tmp_path):
    cfg = _cfg()
    path = write_review(_all_bands(conn, cfg), tmp_path / "r.json")
    r = apply_review(conn, cfg, path, never={frozenset(("fast api", "fastapi"))})
    assert ("fast api", "fastapi") not in r.applied
    assert any("never-merge" in why for _, _, why in r.skipped)
    assert get_skill_vocab(conn, "fast api") is not None


def test_a_spelling_pair_applies_below_the_normal_threshold_but_only_if_the_names_really_match(conn, tmp_path):
    cfg = _cfg()
    _node(conn, cfg, "csharp", _at(1.0), seen=1)
    _node(conn, cfg, "c#", _at(0.86), seen=9)  # 0.86 < 0.90 but a spelling match
    _node(conn, cfg, "java", _at(1.0, base=2, axis=3), seen=1)
    _node(conn, cfg, "kotlin", _at(0.86, base=2, axis=3), seen=1)  # 0.86, NOT a spelling match
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            {
                "embed_tag": cfg.embed_tag,
                "proposals": [
                    {
                        "band": "auto",
                        "approve": True,
                        "reason": "same spelling after normalising",
                        "dup": "csharp",
                        "survivor": "c#",
                    },
                    # a tampered entry claiming to be a spelling match to dodge the cosine floor
                    {
                        "band": "auto",
                        "approve": True,
                        "reason": "same spelling after normalising",
                        "dup": "kotlin",
                        "survivor": "java",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    r = apply_review(conn, cfg, path)
    assert r.applied == [("csharp", "c#")]
    assert len(r.skipped) == 1 and "below the floor 0.9" in r.skipped[0][2]


def test_alias_entries_are_reverified_at_apply(conn, tmp_path):
    cfg = _cfg()
    _node(conn, cfg, "kubernetes", _at(1.0))
    _node(conn, cfg, "typescript", _at(0.3))
    add_alias(conn, "js", "typescript")
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            {
                "embed_tag": cfg.embed_tag,
                "proposals": [
                    {
                        "band": "alias",
                        "approve": True,
                        "dup": "js",
                        "survivor": "kubernetes",
                    },  # conflicts with the existing alias
                    {"band": "alias", "approve": True, "dup": "k8s", "survivor": "no-longer-there"},  # canonical gone
                    {"band": "alias", "approve": True, "dup": "js", "survivor": "typescript"},  # already applied
                ],
            }
        ),
        encoding="utf-8",
    )
    r = apply_review(conn, cfg, path)
    reasons = [why for _, _, why in r.skipped]
    assert r.applied == [] and len(reasons) == 3
    assert "already an alias of 'typescript'" in reasons[0] and "survivor no longer exists" in reasons[1]
    assert "already an alias of the survivor" in reasons[2]


def test_apply_of_an_alias_node_merges_it_and_keeps_its_count(conn, tmp_path):
    cfg = _cfg()
    _node(conn, cfg, "go", _at(1.0), seen=3)
    _node(conn, cfg, "golang", _at(0.767), seen=2)
    r = propose_merges(conn, cfg, aliases={"go": ("golang",)})
    path = write_review(r, tmp_path / "r.json")
    applied = apply_review(conn, cfg, path)
    assert applied.applied == [("golang", "go")]
    assert get_skill_vocab(conn, "go")["seen_count"] == 5 and resolve_alias(conn, "golang") == "go"


def test_proposing_with_aliases_and_never_is_still_read_only(conn):
    cfg = _cfg()
    _all_bands(conn, cfg)
    before = iter_skill_vocab(conn)
    propose_merges(conn, cfg, aliases={"kubernetes": ("k8s",)}, never={frozenset(("a", "b"))}, show_low=True)
    assert iter_skill_vocab(conn) == before and resolve_alias(conn, "k8s") is None


def test_spelling_keys_try_the_word_with_and_without_plural_folding():
    assert spelling_keys("nodejs") == {"nodejs", "nodej"}
    assert spelling_keys("node.js") == {"nodejs"}  # short words are not folded, so both keys agree
    assert spelling_keys("rest apis") == {"restapis", "restapi"}
    assert spelling_keys("c#") == {"csharp"} and "cplusplus" in spelling_keys("c++")
    # the whole point: the three C names never share a key
    assert spelling_keys("c#").isdisjoint(spelling_keys("c++"))
    assert spelling_keys("c").isdisjoint(spelling_keys("c#")) and spelling_keys("c").isdisjoint(spelling_keys("c++"))


def test_a_plural_only_difference_is_still_a_spelling_match_in_both_directions():
    for a, b in [("systems design", "system design"), ("rest api", "rest apis"), ("evaluation", "evaluations")]:
        assert same_spelling(a, b) and same_spelling(b, a)


def test_the_never_merge_count_is_distinct_pairs_not_attempts(conn):
    cfg = _cfg()
    _node(conn, cfg, "fastapi", _at(1.0), seen=9)
    _node(conn, cfg, "fast api", _at(0.97), seen=1)
    r = propose_merges(conn, cfg, never={frozenset(("fast api", "fastapi"))})
    assert r.never_blocked == 1
