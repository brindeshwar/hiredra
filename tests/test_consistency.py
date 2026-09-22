"""Tests for hiredra.graph.consistency: batch vs one-at-a-time vs reversed
(build 4.3 fix). A fake embedder only; nothing here can spend quota."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from google.genai.errors import ServerError

from hiredra.graph.config import GraphConfig
from hiredra.graph.consistency import check_batch_consistency, pick_sample
from hiredra.graph.index import save_vectors
from hiredra.storage import add_market_skill, connect, init_db, iter_skill_vocab

DIM = 16


def _cfg(**over) -> GraphConfig:
    base = dict(
        model="gemini-embedding-2", dimensions=DIM, prefix="", inputs_per_request=50, requests_per_minute=100,
        dedup_threshold=0.9, have=0.9, adj=0.7, canary_min_cosine=0.999,
    )
    base.update(over)
    return GraphConfig(**base)


def _unit(seed: str) -> np.ndarray:
    rng = np.random.default_rng(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _perturbed(seed: str, amount: float) -> np.ndarray:
    v = _unit(seed) + np.float32(amount) * _unit("noise-" + seed)
    return (v / np.linalg.norm(v)).astype(np.float32)


class Fake:
    """Consistent by default. `mode` makes it misbehave in one specific way."""

    def __init__(self, mode=None):
        self.calls: list[list[str]] = []
        self.mode = mode

    def __call__(self, texts):
        self.calls.append(list(texts))
        out = []
        for i, t in enumerate(texts):
            if self.mode == "alone_differs" and len(texts) == 1:
                out.append(_perturbed(t, 0.5))
            elif self.mode == "batch_size_matters" and len(texts) > 1:
                out.append(_perturbed(t, 0.5))
            elif self.mode == "order_matters" and i > 0 and texts[0] > texts[-1]:
                out.append(_perturbed(t, 0.5))            # the reversed run only
            else:
                out.append(_unit(t))
        return np.stack(out)


@pytest.fixture()
def conn():
    c = connect(":memory:")
    init_db(c)
    yield c
    c.close()


def _index(conn, cfg, names, embedder=None):
    for n in names:
        add_market_skill(conn, n)
    vecs = np.stack([_unit(n) for n in names])
    save_vectors(conn, list(names), vecs, cfg.embed_tag)


NAMES = [f"skill-{i:02d}" for i in range(30)]


# ── the sample ───────────────────────────────────────────────────────────────
def test_the_sample_is_evenly_spread_sorted_and_deterministic():
    picks = pick_sample(list(reversed(NAMES)), 5)
    assert picks == pick_sample(NAMES, 5)                      # input order does not matter
    assert picks == sorted(picks) and picks[0] == NAMES[0] and picks[-1] == NAMES[-1]
    assert len(picks) == 5 and len(set(picks)) == 5


def test_asking_for_more_than_exist_returns_everything():
    assert pick_sample(["b", "a"], 10) == ["a", "b"]


# ── verdicts ─────────────────────────────────────────────────────────────────
def test_a_consistent_embedder_passes_all_three_ways(conn):
    cfg = _cfg()
    _index(conn, cfg, NAMES)
    emb = Fake()
    r = check_batch_consistency(conn, cfg, emb, sample=6)
    assert r.consistent is True and r.error is None and r.skipped_reason is None
    assert [c.name for c in r.conditions] == ["batch", "one_by_one", "reversed"]
    assert all(c.min_cosine == pytest.approx(1.0, abs=1e-5) for c in r.conditions)
    assert r.requests_used == 18                               # 3 ways x 6 skills, each input = 1 counted request


def test_it_makes_one_batch_call_then_single_calls_then_one_reversed_batch(conn):
    cfg = _cfg()
    _index(conn, cfg, NAMES)
    emb = Fake()
    r = check_batch_consistency(conn, cfg, emb, sample=4)
    sample = list(r.sample)
    assert emb.calls[0] == sample                              # the whole sample in ONE request
    assert emb.calls[1:5] == [[n] for n in sample]             # then one request per skill
    assert emb.calls[5] == sample[::-1]                        # then the same batch, reversed
    assert len(emb.calls) == 6


def test_a_skill_that_embeds_differently_alone_is_caught(conn):
    cfg = _cfg()
    _index(conn, cfg, NAMES)
    r = check_batch_consistency(conn, cfg, Fake("alone_differs"), sample=5)
    assert r.consistent is False
    by = {c.name: c for c in r.conditions}
    assert by["batch"].min_cosine > 0.999 and by["reversed"].min_cosine > 0.999
    assert by["one_by_one"].min_cosine < 0.999 and by["one_by_one"].worst in r.sample


def test_a_skill_that_embeds_differently_in_a_batch_is_caught(conn):
    cfg = _cfg()
    _index(conn, cfg, NAMES)
    r = check_batch_consistency(conn, cfg, Fake("batch_size_matters"), sample=5)
    assert r.consistent is False
    by = {c.name: c for c in r.conditions}
    assert by["one_by_one"].min_cosine > 0.999 and by["batch"].min_cosine < 0.999


def test_order_dependence_is_caught(conn):
    cfg = _cfg()
    _index(conn, cfg, NAMES)
    r = check_batch_consistency(conn, cfg, Fake("order_matters"), sample=5)
    by = {c.name: c for c in r.conditions}
    assert r.consistent is False and by["batch"].min_cosine > 0.999 and by["reversed"].min_cosine < 0.999


@pytest.mark.parametrize("threshold,expected", [(0.999, True), (0.99999, False)])
def test_the_threshold_comes_from_config(conn, threshold, expected):
    cfg = _cfg(canary_min_cosine=threshold)
    _index(conn, cfg, NAMES)

    class Slightly:
        def __call__(self, texts):
            return np.stack([_perturbed(t, 0.02) for t in texts])   # cosine ~0.9998 to the stored vectors

    r = check_batch_consistency(conn, cfg, Slightly(), sample=4)
    assert r.consistent is expected


# ── edge cases ───────────────────────────────────────────────────────────────
def test_nothing_to_compare_against_is_skipped_not_passed(conn):
    cfg = _cfg()
    emb = Fake()
    r = check_batch_consistency(conn, cfg, emb)
    assert r.skipped_reason and r.consistent is None and emb.calls == []


def test_only_nodes_with_a_current_vector_are_sampled(conn):
    cfg = _cfg()
    _index(conn, cfg, ["a", "b", "c"])
    add_market_skill(conn, "no-vector")
    _index(conn, _cfg(prefix="other: "), ["stale-one"])
    r = check_batch_consistency(conn, cfg, Fake(), sample=10)
    assert sorted(r.sample) == ["a", "b", "c"]


def test_an_api_failure_is_reported_as_unknown_not_as_a_pass(conn):
    cfg = _cfg()
    _index(conn, cfg, NAMES)

    def boom(texts):
        try:
            raise RuntimeError("daily quota") from ServerError(503, {"error": {"message": "unavailable"}})
        except RuntimeError as e:
            raise e

    r = check_batch_consistency(conn, cfg, boom, sample=4)
    assert r.consistent is None and "daily quota" in r.error


def test_it_never_changes_the_database(conn):
    cfg = _cfg()
    _index(conn, cfg, NAMES)
    before = iter_skill_vocab(conn)
    check_batch_consistency(conn, cfg, Fake("alone_differs"), sample=5)
    assert iter_skill_vocab(conn) == before


def test_an_unknown_exception_is_reraised_not_reported_as_could_not_tell(conn):
    cfg = _cfg()
    _index(conn, cfg, NAMES)

    def bug(texts):
        raise KeyError("a bug in the embed path")

    with pytest.raises(KeyError):
        check_batch_consistency(conn, cfg, bug, sample=4)
