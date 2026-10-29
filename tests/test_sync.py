"""Tests for hiredra.graph.sync (build sub-slice 4.3): the resumable embed
queue, the canary, and the version gate. Fake embedder -- no key, no network."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from google.genai.errors import ClientError, ServerError

from hiredra.graph.config import GraphConfig
from hiredra.graph.embedder import EmbeddingError
from hiredra.quota import DailyQuotaExceeded
from hiredra.graph.index import load_skills
from hiredra.graph.sync import check_canary, embed_pending, ensure_index_current
from hiredra.storage import add_market_skill, connect, get_skill_vocab, init_db, skill_index_status

DIM = 8


def _cfg(**over) -> GraphConfig:
    base = dict(
        model="gemini-embedding-2",
        dimensions=DIM,
        prefix="",
        inputs_per_request=2,
        dedup_threshold=0.9,
        have=0.9,
        adj=0.7,
        canary_skills=("python", "kubernetes", "postgresql"),
        canary_min_cosine=0.999,
    )
    base.update(over)
    return GraphConfig(**base)


def _unit(seed: str) -> np.ndarray:
    rng = np.random.default_rng(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    """Deterministic name -> unit vector; records every call. `override` swaps a
    name's vector (simulating provider drift); `script` maps a 1-based call number
    to an exception to raise on that call."""

    def __init__(self, override=None, script=None):
        self.calls: list[list[str]] = []
        self.override = override or {}
        self.script = script or {}

    def __call__(self, texts):
        self.calls.append(list(texts))
        exc = self.script.get(len(self.calls))
        if exc is not None:
            raise exc
        return np.stack([self.override.get(t, _unit(t)) for t in texts])

    @property
    def embedded_names(self) -> list[str]:
        return [n for call in self.calls for n in call]


def _wrapped(cause: BaseException, msg: str = "Error embedding content") -> Exception:
    """How langchain-google-genai surfaces an SDK error: wrapped, cause chained."""
    try:
        raise RuntimeError(msg) from cause
    except RuntimeError as e:
        return e


def _transient(msg: str = "Error embedding content (503 after retries)") -> Exception:
    return _wrapped(ServerError(503, {"error": {"message": "unavailable"}}), msg)


@pytest.fixture()
def conn(pg_url):
    c = connect(pg_url)
    init_db(c)
    yield c
    c.close()


def _seed(conn, *names):
    for n in names:
        add_market_skill(conn, n)


# ── embed_pending ────────────────────────────────────────────────────────────
def test_embeds_the_queue_in_batch_sized_chunks_and_stores_tagged_vectors(conn):
    cfg = _cfg()
    _seed(conn, "a", "b", "c", "d", "e")
    emb = FakeEmbedder()
    r = embed_pending(conn, cfg, emb)

    assert [len(c) for c in emb.calls] == [2, 2, 1]
    assert (r.queue_size, r.attempted, r.embedded, r.chunks, r.remaining, r.stopped) == (5, 5, 5, 3, 0, None)
    row = get_skill_vocab(conn, "c")
    assert row["embed_tag"] == cfg.embed_tag and len(row["vector"]) == DIM * 4
    by_name = {s.name: s for s in load_skills(conn, cfg)}
    np.testing.assert_array_equal(by_name["c"].vector, _unit("c"))


def test_an_empty_queue_makes_no_calls(conn):
    emb = FakeEmbedder()
    r = embed_pending(conn, _cfg(), emb)
    assert emb.calls == [] and (r.queue_size, r.embedded, r.remaining) == (0, 0, 0)


def test_queue_is_ordered_by_weight_so_common_skills_embed_first(conn):
    _seed(conn, "rare", "common", "mid")
    emb = FakeEmbedder()
    embed_pending(conn, _cfg(inputs_per_request=1), emb, weights={"common": 100, "mid": 10, "rare": 1})
    assert emb.embedded_names == ["common", "mid", "rare"]


def test_limit_caps_this_run_and_a_second_run_finishes_without_redoing_work(conn):
    cfg = _cfg()
    _seed(conn, "a", "b", "c", "d", "e")
    first = FakeEmbedder()
    r1 = embed_pending(conn, cfg, first, limit=3)
    assert (r1.attempted, r1.embedded, r1.remaining) == (3, 3, 2)
    assert [len(c) for c in first.calls] == [2, 1]

    second = FakeEmbedder()
    r2 = embed_pending(conn, cfg, second)
    assert (r2.queue_size, r2.embedded, r2.remaining) == (2, 2, 0)
    assert set(second.embedded_names).isdisjoint(first.embedded_names)  # nothing re-embedded
    assert sorted(first.embedded_names + second.embedded_names) == ["a", "b", "c", "d", "e"]


def test_a_failing_chunk_stops_the_run_and_keeps_finished_chunks(conn):
    cfg = _cfg()
    _seed(conn, "a", "b", "c", "d", "e")
    emb = FakeEmbedder(script={2: _transient("HTTP 503 after retries")})
    r = embed_pending(conn, cfg, emb)

    assert r.stopped == "transient" and "HTTP 503" in r.error
    assert (r.embedded, r.chunks, r.remaining) == (2, 1, 3)
    assert len(emb.calls) == 2  # stopped: did not skip ahead to chunk 3
    assert skill_index_status(conn, cfg.embed_tag)["current"] == 2


def test_resume_after_a_failure_only_embeds_what_is_left(conn):
    cfg = _cfg()
    _seed(conn, "a", "b", "c", "d", "e")
    embed_pending(conn, cfg, FakeEmbedder(script={2: _transient("boom")}))
    done = {r["name"] for r in conn.execute("SELECT name FROM skill_vocab WHERE vector IS NOT NULL")}
    assert len(done) == 2

    resumed = FakeEmbedder()
    r = embed_pending(conn, cfg, resumed)
    assert r.stopped is None and r.embedded == 3 and r.remaining == 0
    assert done.isdisjoint(resumed.embedded_names)
    assert skill_index_status(conn, cfg.embed_tag)["unembedded"] == 0


def test_a_daily_quota_stops_cleanly_with_the_reason_and_is_resumable(conn):
    cfg = _cfg()
    _seed(conn, "a", "b", "c", "d")
    emb = FakeEmbedder(script={2: DailyQuotaExceeded("daily embedding quota exhausted (PerDay)")})
    r = embed_pending(conn, cfg, emb)
    assert r.stopped == "daily_quota" and "PerDay" in r.error
    assert (r.embedded, r.remaining) == (2, 2)

    later = FakeEmbedder()  # "the next day"
    r2 = embed_pending(conn, cfg, later)
    assert (r2.stopped, r2.embedded, r2.remaining) == (None, 2, 0)


def test_ctrl_c_mid_run_keeps_finished_chunks(conn):
    cfg = _cfg()
    _seed(conn, "a", "b", "c", "d")
    with pytest.raises(KeyboardInterrupt):
        embed_pending(conn, cfg, FakeEmbedder(script={2: KeyboardInterrupt()}))
    assert skill_index_status(conn, cfg.embed_tag)["current"] == 2  # chunk 1 survived


def test_stale_vectors_are_re_embedded_under_the_new_tag(conn):
    old, new = _cfg(), _cfg(prefix="task: sentence similarity | query: ")
    _seed(conn, "a", "b")
    embed_pending(conn, old, FakeEmbedder())
    assert skill_index_status(conn, new.embed_tag) == {
        **skill_index_status(conn, new.embed_tag),
        "stale": 2,
        "current": 0,
    }

    emb = FakeEmbedder()
    r = embed_pending(conn, new, emb)
    assert r.embedded == 2
    assert skill_index_status(conn, new.embed_tag)["current"] == 2


def test_progress_is_logged_per_chunk(conn):
    _seed(conn, "a", "b", "c")
    lines: list[str] = []
    embed_pending(conn, _cfg(), FakeEmbedder(), log=lines.append)
    assert len(lines) == 2 and lines[-1].startswith("embedded 3/3")


# ── canary ───────────────────────────────────────────────────────────────────
def _indexed(conn, cfg, *names):
    _seed(conn, *names)
    embed_pending(conn, cfg, FakeEmbedder())


def test_canary_is_skipped_on_an_empty_index(conn):
    emb = FakeEmbedder()
    res = check_canary(conn, _cfg(), emb)
    assert res.skipped_reason and res.checked == () and not res.drifted
    assert emb.calls == []  # nothing to compare -> no API call


def test_canary_passes_when_the_provider_is_unchanged(conn):
    cfg = _cfg()
    _indexed(conn, cfg, "python", "kubernetes", "other")
    emb = FakeEmbedder()
    res = check_canary(conn, cfg, emb)
    assert res.checked == ("python", "kubernetes")  # 'postgresql' isn't a node -> skipped
    assert not res.drifted and res.min_cosine == pytest.approx(1.0, abs=1e-5)
    assert emb.calls == [["python", "kubernetes"]]  # ONE small request, only the canary names


def test_canary_detects_provider_drift(conn):
    cfg = _cfg()
    _indexed(conn, cfg, "python", "kubernetes")
    drifted = FakeEmbedder(override={"kubernetes": _unit("a-different-model")})
    res = check_canary(conn, cfg, drifted)
    assert res.drifted and res.min_cosine < cfg.canary_min_cosine


def test_canary_tolerates_float_noise_above_the_threshold(conn):
    cfg = _cfg()
    _indexed(conn, cfg, "python")
    noisy = _unit("python") + np.float32(1e-5) * _unit("noise")
    noisy = (noisy / np.linalg.norm(noisy)).astype(np.float32)
    assert not check_canary(conn, cfg, FakeEmbedder(override={"python": noisy})).drifted


def test_canary_ignores_nodes_whose_vector_is_from_another_tag(conn):
    old = _cfg()
    _indexed(conn, old, "python")
    new = _cfg(prefix="p: ")
    res = check_canary(conn, new, FakeEmbedder())
    assert res.skipped_reason and not res.drifted  # stale tag: nothing current to compare


def test_canary_that_cannot_run_reports_the_error_instead_of_raising(conn):
    cfg = _cfg()
    _indexed(conn, cfg, "python")
    res = check_canary(conn, cfg, FakeEmbedder(script={1: _transient("quota")}))
    assert res.error and "quota" in res.error and not res.drifted


# ── the gate ─────────────────────────────────────────────────────────────────
def test_gate_on_a_fresh_index_embeds_everything_and_is_ready(conn):
    cfg = _cfg()
    _seed(conn, "python", "a", "b")
    gate = ensure_index_current(conn, cfg, FakeEmbedder())
    assert gate.canary.skipped_reason and gate.invalidated == 0
    assert gate.embed.embedded == 3 and gate.ready
    assert gate.after["current"] == 3


def test_gate_with_no_drift_only_embeds_new_nodes(conn):
    cfg = _cfg()
    _indexed(conn, cfg, "python", "kubernetes")
    _seed(conn, "newskill")
    emb = FakeEmbedder()
    gate = ensure_index_current(conn, cfg, emb)
    assert gate.invalidated == 0 and not gate.canary.drifted
    assert emb.calls == [["python", "kubernetes"], ["newskill"]]  # canary, then just the new node
    assert gate.ready


def test_gate_on_drift_invalidates_and_re_embeds_the_whole_index(conn):
    cfg = _cfg()
    _indexed(conn, cfg, "python", "kubernetes", "docker", "kafka")
    drifted = FakeEmbedder(override={"python": _unit("new-model-python")})
    lines: list[str] = []
    gate = ensure_index_current(conn, cfg, drifted, log=lines.append)

    assert gate.canary.drifted and gate.invalidated == 4
    assert gate.embed.embedded == 4 and gate.ready
    assert any("CANARY DRIFT" in ln for ln in lines)
    # the re-embedded index carries the NEW model's vectors, not the old ones
    by_name = {s.name: s for s in load_skills(conn, cfg)}
    np.testing.assert_array_equal(by_name["python"].vector, _unit("new-model-python"))


def test_gate_after_a_config_change_re_embeds_stale_vectors_without_a_canary(conn):
    old, new = _cfg(), _cfg(dimensions=DIM, prefix="task: x | query: ")
    _indexed(conn, old, "python", "docker")
    emb = FakeEmbedder()
    gate = ensure_index_current(conn, new, emb)
    assert gate.before["stale"] == 2 and gate.canary.skipped_reason
    assert gate.embed.embedded == 2 and gate.ready


def test_gate_is_not_ready_when_the_daily_quota_stops_the_drain(conn):
    cfg = _cfg()
    _seed(conn, "a", "b", "c", "d")
    gate = ensure_index_current(conn, cfg, FakeEmbedder(script={2: DailyQuotaExceeded("PerDay")}))
    assert gate.embed.stopped == "daily_quota" and not gate.ready
    assert gate.after["unembedded"] == 2


def test_gate_is_not_ready_when_the_canary_could_not_run_even_if_the_drain_succeeds(conn):
    cfg = _cfg()
    _indexed(conn, cfg, "python")
    _seed(conn, "new")
    gate = ensure_index_current(conn, cfg, FakeEmbedder(script={1: _transient("canary call failed")}))
    assert gate.canary.error and gate.embed.embedded == 1
    assert not gate.ready  # unverified index is not "current"


# ── a real bug must not hide behind a status line (build 4.3 hardening) ──────
@pytest.mark.parametrize(
    "bug", [KeyError("missing"), TypeError("bad arg"), ValueError("bad value"), AttributeError("x")]
)
def test_an_unknown_exception_is_reraised_not_reported_as_a_stop(conn, bug):
    cfg = _cfg()
    _seed(conn, "a", "b", "c", "d")
    with pytest.raises(type(bug)):
        embed_pending(conn, cfg, FakeEmbedder(script={2: bug}))
    # ...but the chunk that finished before the bug is still saved
    assert skill_index_status(conn, cfg.embed_tag)["current"] == 2


def test_a_plain_runtime_error_is_also_unknown(conn):
    cfg = _cfg()
    _seed(conn, "a", "b")
    with pytest.raises(RuntimeError, match="not a google error"):
        embed_pending(conn, cfg, FakeEmbedder(script={1: RuntimeError("not a google error")}))


def test_google_saying_no_is_an_api_error_stop_with_its_own_label(conn):
    cfg = _cfg()
    _seed(conn, "a", "b", "c", "d")
    err = _wrapped(ClientError(403, {"error": {"message": "API key not valid", "status": "PERMISSION_DENIED"}}))
    r = embed_pending(conn, cfg, FakeEmbedder(script={2: err}))
    assert r.stopped == "api_error" and (r.embedded, r.remaining) == (2, 2)


def test_unusable_vectors_are_a_bad_response_stop(conn):
    cfg = _cfg()
    _seed(conn, "a", "b", "c")
    r = embed_pending(conn, cfg, FakeEmbedder(script={1: EmbeddingError("expected 768-dim vectors, got 3")}))
    assert r.stopped == "bad_response" and "768-dim" in r.error


def test_a_network_failure_is_a_transient_stop(conn):
    import httpx

    cfg = _cfg()
    _seed(conn, "a", "b")
    r = embed_pending(conn, cfg, FakeEmbedder(script={1: _wrapped(httpx.ConnectError("connection refused"))}))
    assert r.stopped == "transient"


def test_each_stop_reason_is_distinct(conn, pg_url):
    """quota / transient / api_error / bad_response are four different labels."""
    reasons = set()
    for exc in (
        DailyQuotaExceeded("PerDay"),
        _transient(),
        _wrapped(ClientError(400, {"error": {"message": "bad request"}})),
        EmbeddingError("zero norm"),
    ):
        c = connect(pg_url)
        init_db(c)
        _seed(c, "a")
        reasons.add(embed_pending(c, _cfg(), FakeEmbedder(script={1: exc})).stopped)
        c.close()
    assert reasons == {"daily_quota", "transient", "api_error", "bad_response"}


def test_the_canary_reraises_an_unknown_exception_but_reports_a_google_failure(conn):
    cfg = _cfg()
    _indexed(conn, cfg, "python")
    with pytest.raises(KeyError):
        check_canary(conn, cfg, FakeEmbedder(script={1: KeyError("bug")}))
    res = check_canary(conn, cfg, FakeEmbedder(script={1: _transient("quota")}))
    assert res.error and not res.drifted


def test_the_gate_lets_an_unknown_exception_escape(conn):
    cfg = _cfg()
    _seed(conn, "a", "b")
    with pytest.raises(TypeError):
        ensure_index_current(conn, cfg, FakeEmbedder(script={1: TypeError("bug in the embed path")}))
