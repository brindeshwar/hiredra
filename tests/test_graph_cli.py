"""Tests for the `python -m hiredra.graph` CLI (build sub-slice 4.3)."""

from __future__ import annotations

import hashlib
import json
import textwrap

import numpy as np
import pytest

from hiredra.graph import __main__ as cli
from hiredra.graph.embedder import DailyQuotaExceeded
from hiredra.graph.harvest import harvest as real_harvest
from hiredra.storage import (
    add_market_skill, connect, get_skill_vocab, init_db, iter_skill_vocab, resolve_alias, skill_index_status,
)

DIM = 128

CONFIG = textwrap.dedent("""
    embedding: {model: gemini-embedding-2, dimensions: 128, prefix: "", inputs_per_request: 2}
    limits: {requests_per_minute: 100, tokens_per_minute: 30000, requests_per_day: 1000, headroom: 0.9}
    dedup: {threshold: 0.90, review_cosine: 0.93, auto_cosine: 0.96}
    fit: {have: 0.90, adj: 0.70}
    harvest:
      export_base_url: https://x.test/exports
      export_date: 2026-09-19
      ats: [ashby]
      role_families: [software_engineering]
      min_df: 2
      max_words: 4
      max_chars: 40
    gate: {canary_skills: [python], canary_min_cosine: 0.999}
""")


def _unit(seed: str) -> np.ndarray:
    rng = np.random.default_rng(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeEmbedder:
    def __init__(self, script=None, override=None):
        self.calls = []
        self.script = script or {}
        self.override = override or {}

    def __call__(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) in self.script:
            raise self.script[len(self.calls)]
        return np.stack([self.override.get(t, _unit(t)) for t in texts])


@pytest.fixture()
def env(tmp_path):
    cfg = tmp_path / "graph.yml"
    cfg.write_text(CONFIG, encoding="utf-8")
    return {"db": tmp_path / "t.db", "cfg": cfg, "df": tmp_path / "df.json", "review": tmp_path / "review.json",
            "usage": tmp_path / "usage.json", "lock": tmp_path / "graph.lock",
            "alias": tmp_path / "aliases.yml", "never": tmp_path / "never.yml"}


def run(env, *args):
    extra = []
    if args and args[0] == "merges":   # tests never read the repo's real curation files
        extra = ["--alias-file", str(env["alias"]), "--never-merge-file", str(env["never"])]
    return cli.main(["--db", str(env["db"]), "--config", str(env["cfg"]), "--df-path", str(env["df"]),
                     "--usage-path", str(env["usage"]), "--lock-path", str(env["lock"]), *args, *extra])


def _conn(env):
    c = connect(env["db"])
    init_db(c)
    return c


# ── status ───────────────────────────────────────────────────────────────────
def test_status_on_a_new_db_reports_zeros(env, capsys):
    assert run(env, "status") == 0
    out = capsys.readouterr().out
    assert "nodes     : 0" in out and "queue     : 0" in out and "gemini-embedding-2|d128|prefix=none" in out


def test_status_counts_nodes_and_the_queue(env, capsys):
    c = _conn(env)
    for n in ("a", "b", "c"):
        add_market_skill(c, n)
    c.close()
    run(env, "status")
    out = capsys.readouterr().out
    assert "nodes     : 3  (market 3, jd 0, profile 0)" in out
    assert "3 not yet" in out and "3 node(s) need embedding" in out


# ── harvest ──────────────────────────────────────────────────────────────────
def _patch_reader(monkeypatch, jobs):
    def fake(conn, cfg, **kw):
        return real_harvest(conn, cfg, reader=lambda url: iter(jobs), **kw)
    monkeypatch.setattr(cli, "harvest", fake)


def _jobs(n=3):
    return [{"role_family": "software_engineering", "skills_required": ["python", "docker"], "skills_preferred": []}
            for _ in range(n)]


def test_harvest_prints_the_funnel_and_writes_nodes(env, monkeypatch, capsys):
    _patch_reader(monkeypatch, _jobs())
    assert run(env, "harvest") == 0
    out = capsys.readouterr().out
    assert "CANDIDATE MARKET NODES        : 2" in out and "new nodes inserted          : 2" in out
    assert "enriched jobs read            : 3" in out
    c = _conn(env)
    assert {r["name"] for r in iter_skill_vocab(c)} == {"python", "docker"}
    assert json.loads(env["df"].read_text(encoding="utf-8"))["df"] == {"docker": 3, "python": 3}


def test_harvest_dry_run_writes_nothing(env, monkeypatch, capsys):
    _patch_reader(monkeypatch, _jobs())
    assert run(env, "harvest", "--dry-run") == 0
    assert "dry run -- nothing written" in capsys.readouterr().out
    assert iter_skill_vocab(_conn(env)) == [] and not env["df"].exists()


def test_harvest_exits_nonzero_and_says_so_when_a_file_failed(env, monkeypatch, capsys):
    def boom(url):
        raise RuntimeError("HTTP 503")
    monkeypatch.setattr(cli, "harvest", lambda conn, cfg, **kw: real_harvest(conn, cfg, reader=boom, retry_wait_s=0, **kw))
    assert run(env, "harvest") == 1
    out = capsys.readouterr().out
    assert "FAILED FILES" in out and "NOTHING WRITTEN" in out
    assert iter_skill_vocab(_conn(env)) == [] and not env["df"].exists()


def test_harvest_allow_partial_writes_despite_a_failed_file(env, monkeypatch, capsys):
    def flaky(url):
        raise RuntimeError("HTTP 503")
    monkeypatch.setattr(cli, "harvest", lambda conn, cfg, **kw: real_harvest(conn, cfg, reader=flaky, retry_wait_s=0, **kw))
    assert run(env, "harvest", "--allow-partial") == 1          # still non-zero: the yield IS understated
    assert "FAILED FILES" in capsys.readouterr().out


# ── sync ─────────────────────────────────────────────────────────────────────
def _patch_embedder(monkeypatch, emb):
    monkeypatch.setattr("hiredra.graph.embedder.build_embedder", lambda cfg, **kw: emb)


def test_sync_embeds_everything_and_reports_ready(env, monkeypatch, capsys):
    c = _conn(env)
    for n in ("python", "a", "b"):
        add_market_skill(c, n)
    c.close()
    _patch_embedder(monkeypatch, FakeEmbedder())
    assert run(env, "sync") == 0
    out = capsys.readouterr().out
    assert "READY" in out and "embedded 3" in out
    assert skill_index_status(_conn(env), "gemini-embedding-2|d128|prefix=none")["current"] == 3


def test_sync_limit_then_resume_finishes_without_redoing_work(env, monkeypatch, capsys):
    c = _conn(env)
    for n in ("a", "b", "c", "d", "e"):
        add_market_skill(c, n)
    c.close()
    first = FakeEmbedder()
    _patch_embedder(monkeypatch, first)
    assert run(env, "sync", "--limit", "2") == 0
    assert "not ready yet" in capsys.readouterr().out
    second = FakeEmbedder()
    _patch_embedder(monkeypatch, second)
    assert run(env, "sync") == 0
    assert "READY" in capsys.readouterr().out
    done_first = [n for call in first.calls for n in call]
    done_second = [n for call in second.calls for n in call]
    assert len(done_first) == 2 and set(done_first).isdisjoint(done_second) and len(done_second) == 3


def test_sync_stops_with_exit_2_on_a_daily_quota_and_says_how_to_resume(env, monkeypatch, capsys):
    c = _conn(env)
    for n in ("a", "b", "c", "d"):
        add_market_skill(c, n)
    c.close()
    _patch_embedder(monkeypatch, FakeEmbedder(script={2: DailyQuotaExceeded("daily embedding quota exhausted (PerDay)")}))
    assert run(env, "sync") == 2
    out = capsys.readouterr().out
    assert "STOPPED (daily_quota)" in out and "Run `sync` again" in out
    assert skill_index_status(_conn(env), "gemini-embedding-2|d128|prefix=none")["current"] == 2   # progress kept


# ── merges ───────────────────────────────────────────────────────────────────
def _two_near_dups(env):
    from hiredra.graph.config import load_graph_config
    from hiredra.graph.index import save_vectors
    cfg = load_graph_config(env["cfg"])
    c = _conn(env)
    for n, v, seen in (("fastapi", [1, 0], 5), ("fast api", [0.96, 0.28], 1), ("kafka", [0, 0, 1], 2)):
        add_market_skill(c, n)
        vec = np.zeros(DIM, dtype=np.float32)
        vec[: len(v)] = v
        save_vectors(c, [n], np.asarray([vec / np.linalg.norm(vec)]), cfg.embed_tag)
        c.execute("UPDATE skill_vocab SET seen_count = ? WHERE name = ?", (seen, n))
    c.commit()
    c.close()


def test_merges_is_a_dry_run_by_default_and_changes_nothing(env, capsys):
    _two_near_dups(env)
    before = iter_skill_vocab(_conn(env))
    assert run(env, "merges", "--review-path", str(env["review"])) == 0
    out = capsys.readouterr().out
    assert "DRY RUN, nothing was changed" in out and "fastapi" in out and "<- fast api" in out
    assert "NEAR MISSES" in out and "Nothing applies until you do" in out
    assert env["review"].exists()
    after = _conn(env)
    assert iter_skill_vocab(after) == before and resolve_alias(after, "fast api") is None


def test_merges_apply_without_a_dry_run_is_refused_with_a_message_not_a_traceback(env, capsys):
    _two_near_dups(env)
    assert run(env, "merges", "--apply", "--review-path", str(env["review"])) == 1
    assert "NOT APPLIED" in capsys.readouterr().out
    assert resolve_alias(_conn(env), "fast api") is None


def test_merges_dry_run_then_apply_applies_the_reviewed_file(env, capsys):
    _two_near_dups(env)
    run(env, "merges", "--review-path", str(env["review"]))
    capsys.readouterr()
    assert run(env, "merges", "--apply", "--review-path", str(env["review"])) == 0
    assert "applied 1 merge(s)" in capsys.readouterr().out
    c = _conn(env)
    assert resolve_alias(c, "fast api") == "fastapi" and get_skill_vocab(c, "kafka") is not None


def test_reviewer_edits_are_honoured_through_the_cli(env, capsys):
    _two_near_dups(env)
    run(env, "merges", "--review-path", str(env["review"]))
    doc = json.loads(env["review"].read_text(encoding="utf-8"))
    doc["proposals"] = []                                   # reviewer rejects everything
    env["review"].write_text(json.dumps(doc), encoding="utf-8")
    capsys.readouterr()
    assert run(env, "merges", "--apply", "--review-path", str(env["review"])) == 0
    assert "applied 0 merge(s)" in capsys.readouterr().out
    assert resolve_alias(_conn(env), "fast api") is None


# ── today's budget, the exhausted-day guard, consistency, --reembed-all (build 4.3 fix) ──
def _usage_obj(env):
    from hiredra.graph.usage import DailyUsage
    return DailyUsage(900, path=env["usage"])           # same budget as the test config (1000 x 0.9)


def _exploding_embedder(monkeypatch):
    """Any attempt to BUILD an embedder (i.e. to get ready to call the API) fails the test."""
    def boom(cfg, **kw):
        raise AssertionError("an embedder was built: the CLI was about to call the API")
    monkeypatch.setattr("hiredra.graph.embedder.build_embedder", boom)


def test_status_shows_todays_budget_and_the_real_limits(env, capsys):
    _usage_obj(env).record(123)
    assert run(env, "status") == 0
    out = capsys.readouterr().out
    assert "123 of 900 requests used today" in out
    assert "100 req/min, 30000 tokens/min, 1000 req/day" in out and "each input in a batch = 1 request" in out
    assert "90/27000/900" in out


def test_sync_refuses_with_no_api_call_when_google_already_said_the_day_is_over(env, monkeypatch, capsys):
    c = _conn(env)
    add_market_skill(c, "a")
    c.close()
    _usage_obj(env).mark_exhausted("reported by Google")
    _exploding_embedder(monkeypatch)
    assert run(env, "sync") == 2
    out = capsys.readouterr().out
    assert "NOT RUN (no API call made)" in out and "already reported exhausted" in out and "IST" in out


def test_sync_stops_at_the_budget_before_googles_cap_and_says_how_to_resume(env, monkeypatch, capsys):
    c = _conn(env)
    for n in ("a", "b", "c", "d", "e"):
        add_market_skill(c, n)
    c.close()
    _usage_obj(env).record(897)                          # 3 requests left today
    _patch_embedder(monkeypatch, FakeEmbedder())
    assert run(env, "sync") == 2
    out = capsys.readouterr().out
    assert "STOPPED (daily_budget)" in out and "covers 3 of 5" in out
    assert "Run `sync` again" in out and "Pacific" in out
    assert skill_index_status(_conn(env), "gemini-embedding-2|d128|prefix=none")["current"] == 3


def test_sync_prints_the_units_and_pacing_in_googles_terms(env, monkeypatch, capsys):
    c = _conn(env)
    add_market_skill(c, "a")
    c.close()
    _patch_embedder(monkeypatch, FakeEmbedder())
    run(env, "sync")
    out = capsys.readouterr().out
    assert "2 skills per request" in out and "90 req/min" in out and "27000 tokens/min" in out
    assert "each skill = 1 request" in out and "requests used today" in out


def _index_with_vectors(env, names):
    from hiredra.graph.config import load_graph_config
    from hiredra.graph.index import save_vectors
    cfg = load_graph_config(env["cfg"])
    c = _conn(env)
    for n in names:
        add_market_skill(c, n)
    save_vectors(c, list(names), np.stack([_unit(n) for n in names]), cfg.embed_tag)
    c.close()


NAMES12 = [f"skill-{i:02d}" for i in range(12)]


def test_consistency_passes_and_says_batching_is_safe(env, monkeypatch, capsys):
    _index_with_vectors(env, NAMES12)
    emb = FakeEmbedder()
    _patch_embedder(monkeypatch, emb)
    assert run(env, "consistency", "--sample", "4") == 0
    out = capsys.readouterr().out
    assert "CONSISTENT" in out and "batching does not change the vectors" in out
    assert "batch " in out and "one_by_one" in out and "reversed" in out
    assert _usage_obj(env).used() == 0                   # fake embedder bypasses the real counter; nothing recorded


def test_consistency_failure_exits_3_and_prints_the_fallback_without_changing_anything(env, monkeypatch, capsys):
    _index_with_vectors(env, NAMES12)
    before = iter_skill_vocab(_conn(env))

    class Alone(FakeEmbedder):
        def __call__(self, texts):
            out = super().__call__(texts)
            return -out if len(texts) == 1 else out       # a single-skill call gives a different vector

    _patch_embedder(monkeypatch, Alone())
    assert run(env, "consistency", "--sample", "4") == 3
    out = capsys.readouterr().out
    assert "INCONSISTENT" in out and "Do NOT embed the pending skills yet" in out
    assert "sync --reembed-all" in out and "two days" in out
    assert iter_skill_vocab(_conn(env)) == before        # it only reports; it never re-embeds by itself


def test_consistency_refuses_with_no_api_call_when_the_day_is_exhausted(env, monkeypatch, capsys):
    _index_with_vectors(env, NAMES12)
    _usage_obj(env).mark_exhausted()
    _exploding_embedder(monkeypatch)
    assert run(env, "consistency") == 2
    assert "NOT RUN (no API call made)" in capsys.readouterr().out


def test_consistency_refuses_if_the_budget_cannot_cover_it(env, monkeypatch, capsys):
    _index_with_vectors(env, NAMES12)
    _usage_obj(env).record(890)                          # 10 left; the default check needs 24
    _exploding_embedder(monkeypatch)
    assert run(env, "consistency") == 2
    out = capsys.readouterr().out
    assert "NOT RUN (no API call made)" in out and "needs ~24 requests" in out


def test_consistency_with_nothing_embedded_is_skipped(env, monkeypatch, capsys):
    c = _conn(env)
    add_market_skill(c, "a")
    c.close()
    _patch_embedder(monkeypatch, FakeEmbedder())
    assert run(env, "consistency") == 2
    assert "SKIPPED" in capsys.readouterr().out


def test_reembed_all_clears_every_vector_then_embeds_everything_again(env, monkeypatch, capsys):
    _index_with_vectors(env, NAMES12[:6])
    emb = FakeEmbedder()
    _patch_embedder(monkeypatch, emb)
    assert run(env, "sync", "--reembed-all") == 0
    out = capsys.readouterr().out
    assert "cleared 6 stored vectors" in out and "READY" in out
    assert sorted(n for call in emb.calls for n in call) == sorted(NAMES12[:6])   # all six re-sent


def test_sync_without_reembed_all_leaves_current_vectors_alone(env, monkeypatch, capsys):
    _index_with_vectors(env, NAMES12[:6])
    emb = FakeEmbedder()
    _patch_embedder(monkeypatch, emb)
    run(env, "sync")
    assert emb.calls == []           # everything is current and no canary node exists here: zero requests


def test_a_missing_frequency_file_warns_for_sync_and_merges(env, monkeypatch, capsys):
    c = _conn(env)
    add_market_skill(c, "a")
    c.close()
    _patch_embedder(monkeypatch, FakeEmbedder())
    run(env, "sync")
    assert "WARNING: no market-frequency file" in capsys.readouterr().out
    run(env, "merges", "--review-path", str(env["review"]))
    assert "WARNING: no market-frequency file" in capsys.readouterr().out


def test_no_warning_when_the_frequency_file_exists(env, monkeypatch, capsys):
    c = _conn(env)
    add_market_skill(c, "a")
    c.close()
    env["df"].write_text(json.dumps({"df": {"a": 5}}), encoding="utf-8")
    _patch_embedder(monkeypatch, FakeEmbedder())
    run(env, "sync")
    assert "WARNING" not in capsys.readouterr().out


# ── three-band merges through the CLI (build 4.3) ────────────────────────────
def _vec(base: int, cosine: float, axis: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[base] = cosine
    v[axis] = float(np.sqrt(1 - cosine**2))
    return v


def _bands_db(env):
    """fastapi/fast api -> AUTO;  cloud architecture/infrastructure -> REVIEW;  sql/sql server -> hidden LOW;
    kubernetes (alias target);  machine learning/deep learning close but on the never-merge list."""
    from hiredra.graph.config import load_graph_config
    from hiredra.graph.index import save_vectors
    cfg = load_graph_config(env["cfg"])
    c = _conn(env)
    nodes = [
        ("fastapi", _vec(0, 1.0, 1), 9), ("fast api", _vec(0, 0.97, 1), 1),
        ("cloud architecture", _vec(2, 1.0, 3), 9), ("cloud infrastructure", _vec(2, 0.94, 3), 1),
        ("sql", _vec(4, 1.0, 5), 9), ("sql server", _vec(4, 0.925, 5), 1),
        ("kubernetes", _vec(6, 1.0, 7), 9),
        ("machine learning", _vec(8, 1.0, 9), 9), ("deep learning", _vec(8, 0.95, 9), 1),
    ]
    for name, vec, seen in nodes:
        add_market_skill(c, name)
        save_vectors(c, [name], np.asarray([vec]), cfg.embed_tag)
        c.execute("UPDATE skill_vocab SET seen_count = ? WHERE name = ?", (seen, name))
    c.commit()
    c.close()


def _write(path, text):
    path.write_text(textwrap.dedent(text), encoding="utf-8")


def test_merges_prints_the_three_bands_and_the_review_file_carries_approve_flags(env, capsys):
    _bands_db(env)
    _write(env["alias"], "aliases:\n  kubernetes: [k8s]\n")
    _write(env["never"], "never_merge:\n  - [machine learning, deep learning]\n")
    assert run(env, "merges", "--review-path", str(env["review"])) == 0
    out = capsys.readouterr().out
    assert "DRY RUN, nothing was changed" in out and '"approve": true' in out
    assert "AUTO -- pre-approved (approve: true)  --  1 pair(s)" in out and "<- fast api" in out
    assert "ALIAS FILE -- your entries, pre-approved (approve: true)  --  1 pair(s)" in out and "<- k8s" in out
    assert "REVIEW -- NOT approved (approve: false)" in out and "<- cloud infrastructure" in out
    assert "hidden: 1 more review-type pair(s)" in out and "--show-low" in out       # sql ~ sql server
    assert "sql server" not in out.split("REVIEW --")[1].split("hidden:")[0]          # ...not listed by default
    assert "never-merge list: 1 pair(s) suppressed" in out and "deep learning" not in out.split("NEAR MISSES")[0]
    assert "1 AUTO + 1 ALIAS (pre-approved)   |   1 REVIEW (unapproved)   |   1 hidden low-band" in out
    entries = json.loads(env["review"].read_text(encoding="utf-8"))["proposals"]
    assert [(e["band"], e["dup"], e["approve"]) for e in entries] == [
        ("auto", "fast api", True), ("alias", "k8s", True), ("review", "cloud infrastructure", False)]


def test_show_low_lists_the_hidden_band_as_unapproved_review(env, capsys):
    _bands_db(env)
    assert run(env, "merges", "--show-low", "--review-path", str(env["review"])) == 0
    out = capsys.readouterr().out
    assert "<- sql server" in out and "low band" in out and "hidden:" not in out
    entries = {e["dup"]: e for e in json.loads(env["review"].read_text(encoding="utf-8"))["proposals"]}
    assert entries["sql server"]["approve"] is False and entries["sql server"]["band"] == "review"


def test_merges_apply_applies_auto_and_alias_and_leaves_the_review_entry(env, capsys):
    _bands_db(env)
    _write(env["alias"], "aliases:\n  kubernetes: [k8s]\n")
    run(env, "merges", "--review-path", str(env["review"]))
    capsys.readouterr()
    assert run(env, "merges", "--apply", "--review-path", str(env["review"])) == 0
    out = capsys.readouterr().out
    # two REVIEW entries are left: cloud infrastructure, and machine/deep learning (no never-merge file in this test)
    assert "applied 2 merge(s), skipped 0, left 2 not approved" in out
    c = _conn(env)
    assert resolve_alias(c, "fast api") == "fastapi" and resolve_alias(c, "k8s") == "kubernetes"
    assert get_skill_vocab(c, "cloud infrastructure") is not None and resolve_alias(c, "cloud infrastructure") is None
    assert get_skill_vocab(c, "deep learning") is not None and resolve_alias(c, "deep learning") is None


def test_ticking_a_review_entry_in_the_file_makes_apply_merge_it(env, capsys):
    _bands_db(env)
    run(env, "merges", "--review-path", str(env["review"]))
    doc = json.loads(env["review"].read_text(encoding="utf-8"))
    for e in doc["proposals"]:
        if e["band"] == "review":
            e["approve"] = True
    env["review"].write_text(json.dumps(doc), encoding="utf-8")
    capsys.readouterr()
    assert run(env, "merges", "--apply", "--review-path", str(env["review"])) == 0
    assert "left 0 not approved" in capsys.readouterr().out
    assert resolve_alias(_conn(env), "cloud infrastructure") == "cloud architecture"


def test_apply_honours_a_never_merge_entry_added_after_the_dry_run(env, capsys):
    _bands_db(env)
    run(env, "merges", "--review-path", str(env["review"]))
    _write(env["never"], "never_merge:\n  - [fast api, fastapi]\n")          # the owner changes their mind
    capsys.readouterr()
    assert run(env, "merges", "--apply", "--review-path", str(env["review"])) == 0
    out = capsys.readouterr().out
    assert "never-merge list" in out and "applied 0 merge(s), skipped 1" in out
    assert get_skill_vocab(_conn(env), "fast api") is not None


@pytest.mark.parametrize("which,text", [
    ("alias", "aliases:\n  go: [go]\n"),                               # a self-alias
    ("alias", "aliases:\n  a: [x]\n  b: [x]\n"),                        # one alias under two canonicals
    ("never", "never_merge:\n  - [only-one]\n"),                        # not a pair
    ("alias", "surprise: 1\n"),                                         # unknown key
])
def test_an_invalid_curation_file_gives_a_clear_message_not_a_traceback(env, capsys, which, text):
    _bands_db(env)
    _write(env[which], text)
    assert run(env, "merges", "--review-path", str(env["review"])) == 1
    assert "NOT RUN: a curation file is invalid" in capsys.readouterr().out
    assert not env["review"].exists()


def test_a_missing_curation_file_just_means_empty_lists(env, capsys):
    _bands_db(env)                                           # env["alias"] / env["never"] were never written
    assert run(env, "merges", "--review-path", str(env["review"])) == 0
    assert "ALIAS FILE" in capsys.readouterr().out


# ── stop reasons: each gets its own next step; a bug is loud (build 4.3 hardening) ──
def _wrap(cause):
    try:
        raise RuntimeError("Error embedding content") from cause
    except RuntimeError as e:
        return e


def _server_error():
    from google.genai.errors import ServerError
    return _wrap(ServerError(503, {"error": {"message": "unavailable"}}))


def _client_error():
    from google.genai.errors import ClientError
    return _wrap(ClientError(403, {"error": {"message": "API key not valid"}}))


def _bad_response():
    from hiredra.graph.embedder import EmbeddingError
    return EmbeddingError("expected 768-dim vectors, got 3")


@pytest.mark.parametrize("label,make_exc,hint", [
    ("transient", _server_error, "in a few minutes"),
    ("api_error", _client_error, "check GEMINI_API_KEY"),
    ("bad_response", _bad_response, "may repeat"),
])
def test_each_stop_reason_prints_its_own_next_step_and_exits_2(env, monkeypatch, capsys, label, make_exc, hint):
    c = _conn(env)
    for n in ("a", "b", "c", "d"):
        add_market_skill(c, n)
    c.close()
    _patch_embedder(monkeypatch, FakeEmbedder(script={2: make_exc()}))
    assert run(env, "sync") == 2
    out = capsys.readouterr().out
    assert f"STOPPED ({label})" in out and "Every finished chunk is saved." in out and hint in out


def test_an_unknown_exception_escapes_the_cli_but_finished_chunks_and_the_lock_are_safe(env, monkeypatch):
    from hiredra.graph.lock import RunLock

    c = _conn(env)
    for n in ("a", "b", "c", "d"):
        add_market_skill(c, n)
    c.close()
    _patch_embedder(monkeypatch, FakeEmbedder(script={2: KeyError("bug in the embed path")}))
    with pytest.raises(KeyError, match="bug in the embed path"):
        run(env, "sync")
    assert skill_index_status(_conn(env), "gemini-embedding-2|d128|prefix=none")["current"] == 2   # chunk 1 kept
    with RunLock(env["lock"]):                                                                       # lock was released
        pass


# ── the run lock covers exactly the commands that write or spend budget ──────
def _held(env):
    from hiredra.graph.lock import RunLock
    return RunLock(env["lock"], what="sync (another terminal)")


def test_sync_refuses_while_another_run_holds_the_lock_and_makes_no_call(env, monkeypatch, capsys):
    c = _conn(env)
    add_market_skill(c, "a")
    c.close()
    _exploding_embedder(monkeypatch)
    with _held(env):
        assert run(env, "sync") == 4
    out = capsys.readouterr().out
    assert "NOT RUN: another graph maintenance run is in progress" in out and "sync (another terminal)" in out
    assert skill_index_status(_conn(env), "gemini-embedding-2|d128|prefix=none")["current"] == 0
    assert _usage_obj(env).used() == 0


def test_the_other_writing_commands_are_locked_out_too(env, monkeypatch, capsys):
    _bands_db(env)
    _exploding_embedder(monkeypatch)
    _patch_reader(monkeypatch, _jobs())
    before = iter_skill_vocab(_conn(env))
    with _held(env):
        assert run(env, "consistency") == 4
        assert run(env, "harvest") == 4
        assert run(env, "merges", "--apply", "--review-path", str(env["review"])) == 4
    out = capsys.readouterr().out
    assert out.count("NOT RUN: another graph maintenance run is in progress") == 3
    assert iter_skill_vocab(_conn(env)) == before and not env["df"].exists()


def test_read_only_commands_are_not_blocked_by_the_lock(env, monkeypatch, capsys):
    _bands_db(env)
    _patch_reader(monkeypatch, _jobs())
    with _held(env):
        assert run(env, "status") == 0
        assert run(env, "merges", "--review-path", str(env["review"])) == 0        # dry-run
        assert run(env, "harvest", "--dry-run") == 0
    assert "NOT RUN" not in capsys.readouterr().out


def test_the_lock_is_released_after_every_kind_of_finish(env, monkeypatch):
    c = _conn(env)
    add_market_skill(c, "a")
    c.close()
    _patch_embedder(monkeypatch, FakeEmbedder())
    assert run(env, "sync") == 0                       # normal finish
    with _held(env):
        pass
    _usage_obj(env).mark_exhausted()
    assert run(env, "sync") == 2                       # early refusal (non-zero exit)
    with _held(env):
        pass


def test_a_second_sync_started_while_the_first_runs_cannot_double_spend_the_budget(env, monkeypatch):
    """The scenario the lock exists for: run B starts while run A is mid-embed."""
    from hiredra.graph.lock import RunLock

    c = _conn(env)
    for n in ("a", "b", "c", "d"):
        add_market_skill(c, n)
    c.close()
    seen = {}

    class Reentrant(FakeEmbedder):
        def __call__(self, texts):
            if "during-run-a" not in seen:               # while A is embedding its first chunk, start B
                seen["during-run-a"] = run(env, "sync")  # B must be refused
            return super().__call__(texts)

    emb = Reentrant()
    monkeypatch.setattr("hiredra.graph.embedder.build_embedder", lambda cfg, **kw: emb)
    assert run(env, "sync") == 0
    assert seen["during-run-a"] == 4                     # B was locked out
    assert [n for call in emb.calls for n in call].count("a") == 1     # nothing was embedded twice
    with RunLock(env["lock"]):
        pass
