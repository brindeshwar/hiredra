"""Tests for hiredra.graph.fit (build sub-slice 4.4): the per-skill fit
mechanic (docs/scoring.md sec 4). Hermetic -- exact-angle vectors, a fake
embedder, no DB and no network, so every cosine in these tests is exact."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hiredra.graph.config import GraphConfig
from hiredra.graph.fit import SkillFit, fit_skills
from hiredra.graph.index import Skill

DIM = 8


def _cfg(**over) -> GraphConfig:
    base = dict(
        model="gemini-embedding-2",
        dimensions=DIM,
        prefix="",
        inputs_per_request=50,
        dedup_threshold=0.9,
        have=0.90,
        adj=0.70,
    )
    base.update(over)
    return GraphConfig(**base)


def _basis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _tilt(cosine: float, base: int, axis: int) -> np.ndarray:
    """Unit vector with cosine `cosine` against e_base, tilted toward e_axis
    (a fresh axis keeps it at exactly 0 against every other reference)."""
    v = np.zeros(DIM, dtype=np.float32)
    v[base] = cosine
    v[axis] = math.sqrt(max(0.0, 1.0 - cosine**2))
    return v


def _ref(name: str, vec: np.ndarray, *, mine: bool = True) -> Skill:
    return Skill(name=name, aliases=(), mine=mine, level="production" if mine else None, vector=vec)


class FakeEmbedder:
    """Deterministic name -> the vector `vectors` maps it to; records every call
    so tests can assert exactly which (and how many) names were embedded."""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return np.stack([self.vectors[t] for t in texts])


# ── the worked example shape (docs/scoring.md sec 4) ──────────────────────────
def test_worked_example_shape(monkeypatch=None):
    cfg = _cfg()
    reference = [
        _ref("python", _basis(0)),
        _ref("fastapi", _basis(1)),
        _ref("postgresql", _basis(2)),
        _ref("langchain", _basis(3)),
        _ref("mcp", _basis(4)),
        _ref("docker", _basis(5)),
        _ref("aws", _basis(6)),
    ]
    jd = ["python", "fastapi", "postgresql", "kubernetes", "kafka"]
    embed = FakeEmbedder(
        {
            "python": _basis(0),
            "fastapi": _basis(1),
            "postgresql": _basis(2),
            "kubernetes": _tilt(0.82, base=5, axis=7),  # adjacent to docker
            "kafka": _tilt(0.31, base=6, axis=7),  # below ADJ vs aws -> floors to 0
        }
    )

    results = fit_skills(jd, reference, cfg, embed)

    by_skill = {r.skill: r for r in results}
    for name in ("python", "fastapi", "postgresql"):
        assert by_skill[name] == SkillFit(
            skill=name, nearest=name, similarity=pytest.approx(1.0), fit=pytest.approx(1.0)
        )
    k8s = by_skill["kubernetes"]
    assert k8s.nearest == "docker"
    assert k8s.similarity == pytest.approx(0.82)
    assert k8s.fit == pytest.approx(0.82)  # adjacent: fit == s itself, not flattened
    kafka = by_skill["kafka"]
    assert kafka.nearest == "aws"
    assert kafka.similarity == pytest.approx(0.31)
    assert kafka.fit == 0.0  # missing: hard 0, not the similarity


# ── threshold mapping ──────────────────────────────────────────────────────────
def test_similarity_at_or_above_have_is_full_credit():
    cfg = _cfg(have=0.90, adj=0.70)
    reference = [_ref("docker", _basis(0))]
    embed = FakeEmbedder({"kubernetes": _tilt(0.90, base=0, axis=1)})
    r = fit_skills(["kubernetes"], reference, cfg, embed)[0]
    assert r.similarity == pytest.approx(0.90)
    assert r.fit == 1.0


def test_similarity_below_adj_floors_to_zero_not_negative_credit():
    cfg = _cfg(have=0.90, adj=0.70)
    reference = [_ref("docker", _basis(0))]
    embed = FakeEmbedder({"kafka": _tilt(0.69, base=0, axis=1)})
    r = fit_skills(["kafka"], reference, cfg, embed)[0]
    assert r.similarity == pytest.approx(0.69)
    assert r.fit == 0.0


def test_similarity_exactly_at_adj_keeps_its_own_value():
    cfg = _cfg(have=0.90, adj=0.70)
    reference = [_ref("docker", _basis(0))]
    embed = FakeEmbedder({"kubernetes": _tilt(0.70, base=0, axis=1)})
    r = fit_skills(["kubernetes"], reference, cfg, embed)[0]
    assert r.fit == pytest.approx(0.70)


# ── nearest selection ──────────────────────────────────────────────────────────
def test_nearest_is_the_single_closest_reference_skill_not_first_or_last():
    cfg = _cfg()
    reference = [
        _ref("aws", _tilt(0.40, base=0, axis=1)),
        _ref("docker", _tilt(0.82, base=0, axis=1)),
        _ref("mcp", _tilt(0.10, base=0, axis=1)),
    ]
    embed = FakeEmbedder({"kubernetes": _basis(0)})
    r = fit_skills(["kubernetes"], reference, cfg, embed)[0]
    assert r.nearest == "docker"
    assert r.similarity == pytest.approx(0.82)


def test_empty_reference_gives_no_nearest_and_zero_fit():
    cfg = _cfg()
    embed = FakeEmbedder({"python": _basis(0)})
    r = fit_skills(["python"], [], cfg, embed)[0]
    assert r.nearest is None
    assert r.similarity == 0.0
    assert r.fit == 0.0


def test_reference_skills_without_a_vector_are_ignored():
    cfg = _cfg()
    reference = [
        Skill(name="unembedded", aliases=(), mine=True, level="production", vector=None),
        _ref("python", _basis(0)),
    ]
    embed = FakeEmbedder({"python": _basis(0)})
    r = fit_skills(["python"], reference, cfg, embed)[0]
    assert r.nearest == "python"
    assert r.fit == 1.0


# ── the cache: embed fresh only on a miss ──────────────────────────────────────
def test_a_repeated_skill_is_embedded_only_once():
    cfg = _cfg()
    reference = [_ref("python", _basis(0))]
    embed = FakeEmbedder({"python": _basis(0)})
    cache: dict[str, np.ndarray] = {}
    fit_skills(["python", "python", "python"], reference, cfg, embed, cache=cache)
    assert embed.calls == [["python"]]  # one embed call, not three


def test_a_name_already_in_the_cache_is_never_embedded():
    cfg = _cfg()
    reference = [_ref("python", _basis(0))]
    cache = {"kubernetes": _tilt(0.82, base=0, axis=1)}  # pre-seeded, e.g. from a stored graph node

    def _must_not_be_called(texts):
        raise AssertionError(f"embed() called for a cache hit: {texts}")

    r = fit_skills(["kubernetes"], reference, cfg, _must_not_be_called, cache=cache)[0]
    assert r.similarity == pytest.approx(0.82)


def test_cache_is_shared_and_updated_across_calls():
    cfg = _cfg()
    reference = [_ref("python", _basis(0))]
    embed = FakeEmbedder({"python": _basis(0), "kafka": _tilt(0.20, base=0, axis=1)})
    cache: dict[str, np.ndarray] = {}
    fit_skills(["python"], reference, cfg, embed, cache=cache)
    fit_skills(["kafka"], reference, cfg, embed, cache=cache)
    assert embed.calls == [["python"], ["kafka"]]
    assert set(cache) == {"python", "kafka"}
    # a third call for a name already cached must not embed again
    fit_skills(["python", "kafka"], reference, cfg, embed, cache=cache)
    assert embed.calls == [["python"], ["kafka"]]


def test_only_the_missing_names_are_sent_to_embed_in_one_batch():
    cfg = _cfg()
    reference = [_ref("python", _basis(0))]
    embed = FakeEmbedder(
        {"python": _basis(0), "kubernetes": _tilt(0.5, base=0, axis=1), "kafka": _tilt(0.2, base=0, axis=1)}
    )
    cache = {"python": _basis(0)}  # already cached
    fit_skills(["python", "kubernetes", "kafka", "kubernetes"], reference, cfg, embed, cache=cache)
    assert embed.calls == [["kubernetes", "kafka"]]  # one batch, de-duped, python excluded


def test_cache_none_default_still_dedupes_within_one_call():
    cfg = _cfg()
    reference = [_ref("python", _basis(0))]
    embed = FakeEmbedder({"python": _basis(0)})
    fit_skills(["python", "python"], reference, cfg, embed)  # cache=None
    assert embed.calls == [["python"]]


# ── name folding ────────────────────────────────────────────────────────────────
def test_jd_skill_names_are_folded_before_lookup_and_embedding():
    cfg = _cfg()
    reference = [_ref("python", _basis(0))]
    embed = FakeEmbedder({"python": _basis(0)})
    r = fit_skills(["  Python  "], reference, cfg, embed)[0]
    assert embed.calls == [["python"]]  # folded before it ever reaches the embedder
    assert r.skill == "python"  # the returned skill is the folded (canonical) form


def test_output_order_matches_input_order():
    cfg = _cfg()
    reference = [_ref("python", _basis(0)), _ref("docker", _basis(1))]
    embed = FakeEmbedder({"python": _basis(0), "docker": _basis(1)})
    results = fit_skills(["docker", "python", "docker"], reference, cfg, embed)
    assert [r.skill for r in results] == ["docker", "python", "docker"]


def test_no_jd_skills_returns_empty_without_calling_embed():
    cfg = _cfg()
    reference = [_ref("python", _basis(0))]

    def _must_not_be_called(texts):
        raise AssertionError("embed() called with no JD skills")

    assert fit_skills([], reference, cfg, _must_not_be_called) == []
