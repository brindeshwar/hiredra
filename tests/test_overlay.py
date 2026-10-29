"""Tests for hiredra.graph.overlay (plan unit PF1): the D12 match order, approve-only vector
links, the one-transaction apply, and the CLI. Vectors sit at exact cosines."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from hiredra.graph import overlay as ov
from hiredra.graph.__main__ import main as graph_main
from hiredra.graph.config import GraphConfig
from hiredra.graph.index import save_vectors
from hiredra.profile import parse_skills
from hiredra.storage import add_alias, add_market_skill, connect, get_skill_vocab, init_db, resolve_alias

DIM = 128


def _cfg() -> GraphConfig:
    return GraphConfig(
        model="gemini-embedding-2",
        dimensions=DIM,
        prefix="",
        inputs_per_request=50,
        dedup_threshold=0.9,
        have=0.9,
        adj=0.7,
    )


def _unit(i: int, cosine: float = 1.0) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i], v[i + 1] = cosine, math.sqrt(max(0.0, 1 - cosine**2))
    return v


@pytest.fixture()
def conn(pg_url):
    c = connect(pg_url)
    init_db(c)
    yield c
    c.close()


def _node(conn, name, vec=None, cfg=None):
    add_market_skill(conn, name)
    if vec is not None:
        save_vectors(conn, [name], np.asarray([vec], dtype=np.float32), (cfg or _cfg()).embed_tag)


def _profile(tmp_path, strong="python", working="docker", familiar="rust", name="p.md"):
    p = tmp_path / name
    p.write_text(
        f"Last updated: 2026-10-16\n## Skills\n**Strong:**\n{strong}\n**Working:**\n{working}\n**Familiar:**\n{familiar}\n",
        encoding="utf-8",
    )
    return p


class FakeEmbed:
    """Returns a fixed vector per name (unknown names: a far-away one); records every call."""

    def __init__(self, table):
        self.table = table
        self.calls: list[list[str]] = []

    def __call__(self, names):
        self.calls.append(list(names))
        return np.asarray([self.table.get(n, _unit(100)) for n in names], dtype=np.float32)  # unknown: far away


def _graph(conn):
    cfg = _cfg()
    _node(conn, "python", _unit(0), cfg)  # exact
    _node(conn, "scikit learn", _unit(2), cfg)  # hyphen target
    _node(conn, "postgresql", _unit(4), cfg)  # alias target
    add_alias(conn, "postgres", "postgresql")
    _node(conn, "large language models", _unit(6), cfg)  # vector target
    _node(conn, "go", _unit(8), cfg)  # unrelated
    return cfg


def _full_profile(tmp_path, **kw):
    return parse_skills(
        _profile(
            tmp_path,
            strong=kw.get("strong", "python, scikit-learn"),
            working=kw.get("working", "postgres, llms"),
            familiar=kw.get("familiar", "rust-lang"),
        )  # fmt: skip
    )


def _embed():
    return FakeEmbed({"llms": _unit(6, 0.93), "rust lang": _unit(8, 0.40)})


# ── the dry run ─────────────────────────────────────────────────────────────
def test_plan_follows_the_d12_order(conn, tmp_path):
    cfg = _graph(conn)
    emb = _embed()
    plan = ov.plan_overlay(conn, cfg, _full_profile(tmp_path), lambda: emb)
    got = {m.token: (m.step, m.node) for m in plan.matches}
    assert got == {
        "python": ("exact", "python"),
        "scikit-learn": ("hyphen", "scikit learn"),
        "postgres": ("alias", "postgresql"),
        "llms": ("vector", "large language models"),
        "rust-lang": ("new", None),
    }
    assert emb.calls == [["llms", "rust lang"]]  # only steps-1-3 misses, spaces spelling, one call
    assert plan.by_step("vector")[0].cosine == pytest.approx(0.93)


def test_alias_lookup_tries_the_spaces_spelling_too(conn, tmp_path):
    cfg = _cfg()
    _node(conn, "node js runtime", _unit(0), cfg)
    add_alias(conn, "node js", "node js runtime")
    plan = ov.plan_overlay(conn, cfg, parse_skills(_profile(tmp_path, strong="node-js")), lambda: FakeEmbed({}))
    m = plan.matches[0]
    assert (m.token, m.step, m.node) == ("node-js", "alias", "node js runtime")


def test_vector_at_the_threshold_is_proposed_just_under_is_new(conn, tmp_path):
    cfg = _graph(conn)
    emb = FakeEmbed({"llms": _unit(6, 0.90), "rust lang": _unit(8, 0.899)})
    plan = ov.plan_overlay(conn, cfg, _full_profile(tmp_path), lambda: emb)
    assert [m.token for m in plan.by_step("vector")] == ["llms"]
    assert [m.token for m in plan.by_step("new")] == ["rust-lang"]


def test_no_embedder_is_built_when_steps_1_to_3_place_everything(conn, tmp_path):
    cfg = _graph(conn)
    p = parse_skills(_profile(tmp_path, strong="python", working="postgres", familiar="scikit-learn"))
    plan = ov.plan_overlay(conn, cfg, p, lambda: pytest.fail("embedder built"))
    assert {m.step for m in plan.matches} == {"exact", "alias", "hyphen"}


def test_links_file_starts_unapproved(conn, tmp_path):
    cfg = _graph(conn)
    plan = ov.plan_overlay(conn, cfg, _full_profile(tmp_path), _embed)
    data = json.loads(ov.write_links(plan, tmp_path / "links.json").read_text(encoding="utf-8"))
    assert data["skills_hash"] == plan.skills_hash
    assert data["links"] == [
        {"token": "llms", "tier": "working", "node": "large language models", "cosine": 0.93, "approve": False}
    ]


# ── apply ───────────────────────────────────────────────────────────────────
def _plan_and_links(conn, tmp_path, approve: bool):
    cfg = _graph(conn)
    profile = _full_profile(tmp_path)
    links = ov.write_links(ov.plan_overlay(conn, cfg, profile, _embed), tmp_path / "links.json")
    if approve:
        data = json.loads(links.read_text(encoding="utf-8"))
        data["links"][0]["approve"] = True
        links.write_text(json.dumps(data), encoding="utf-8")
    return profile, links


def _mine(conn) -> dict[str, str]:
    return {r[0]: r[1] for r in conn.execute("SELECT name, level FROM skill_vocab WHERE mine = true")}


def test_apply_with_the_link_approved(conn, tmp_path):
    profile, links = _plan_and_links(conn, tmp_path, approve=True)
    out = tmp_path / "overlay.json"
    r = ov.apply_overlay(conn, profile, links, out)

    assert _mine(conn) == {
        "python": "production",
        "scikit learn": "production",
        "postgresql": "built-with",
        "large language models": "built-with",
        "rust-lang": "learned",
    }
    assert r.counts == {"exact": 1, "hyphen": 1, "alias": 1, "linked": 1, "new": 1}
    assert resolve_alias(conn, "scikit-learn") == "scikit learn"  # token-as-written -> node
    assert resolve_alias(conn, "llms") == "large language models"
    assert resolve_alias(conn, "rust lang") == "rust-lang"  # the other spelling of a new node
    new = get_skill_vocab(conn, "rust-lang")
    assert (new["source"], new["seen_count"], new["mine"]) == ("profile", 0, 1)
    state = json.loads(out.read_text(encoding="utf-8"))
    assert state["skills_hash"] == profile.skills_hash and state["file_hash"] == profile.file_hash
    assert state["last_updated"] == "2026-10-16" and state["counts"] == r.counts


def test_an_unapproved_link_becomes_a_new_node(conn, tmp_path):
    profile, links = _plan_and_links(conn, tmp_path, approve=False)
    r = ov.apply_overlay(conn, profile, links, tmp_path / "o.json")
    assert r.counts["linked"] == 0 and r.counts["new"] == 2
    assert get_skill_vocab(conn, "llms")["mine"] == 1
    assert get_skill_vocab(conn, "large language models")["mine"] == 0  # never linked without approval


@pytest.mark.parametrize("value", ["yes", 1, None, "missing"])
def test_only_a_literal_true_approves(conn, tmp_path, value):
    profile, links = _plan_and_links(conn, tmp_path, approve=False)
    data = json.loads(links.read_text(encoding="utf-8"))
    if value == "missing":
        del data["links"][0]["approve"]
    else:
        data["links"][0]["approve"] = value
    links.write_text(json.dumps(data), encoding="utf-8")
    r = ov.apply_overlay(conn, profile, links, tmp_path / "o.json")
    assert r.counts["linked"] == 0
    assert get_skill_vocab(conn, "large language models")["mine"] == 0


def test_apply_spends_no_embedding_and_needs_no_links_file(conn, tmp_path):
    cfg = _graph(conn)
    p = parse_skills(_profile(tmp_path, strong="python", working="postgres", familiar="brand-new"))
    r = ov.apply_overlay(conn, p, tmp_path / "missing.json", tmp_path / "o.json")
    assert r.counts == {"exact": 1, "hyphen": 0, "alias": 1, "linked": 0, "new": 1}
    del cfg


def test_reapply_rebuilds_mine_from_the_file(conn, tmp_path):
    profile, links = _plan_and_links(conn, tmp_path, approve=True)
    ov.apply_overlay(conn, profile, links, tmp_path / "o.json")
    smaller = parse_skills(_profile(tmp_path, strong="python", working="postgres", familiar="go", name="q.md"))
    ov.apply_overlay(conn, smaller, tmp_path / "none.json", tmp_path / "o.json")
    assert _mine(conn) == {"python": "production", "postgresql": "built-with", "go": "learned"}
    assert get_skill_vocab(conn, "rust-lang") is not None  # a dropped token's node stays, just not mine

    again = parse_skills(_profile(tmp_path, strong="python", working="postgres", familiar="go", name="r.md"))
    ov.apply_overlay(conn, again, tmp_path / "none.json", tmp_path / "o.json")  # idempotent
    assert _mine(conn) == {"python": "production", "postgresql": "built-with", "go": "learned"}


def test_a_links_file_for_another_profile_is_refused(conn, tmp_path):
    _, links = _plan_and_links(conn, tmp_path, approve=True)
    other = parse_skills(_profile(tmp_path, strong="python, go", name="other.md"))
    with pytest.raises(ov.OverlayError, match="different Skills section"):
        ov.apply_overlay(conn, other, links, tmp_path / "o.json")
    assert _mine(conn) == {}


def test_collision_keeps_one_node_at_the_strongest_tier(conn, tmp_path):
    _graph(conn)
    p = parse_skills(_profile(tmp_path, strong="postgresql", working="postgres", familiar="rust"))
    r = ov.apply_overlay(conn, p, tmp_path / "none.json", tmp_path / "o.json")
    assert r.collisions == {"postgresql": ["postgresql", "postgres"]}
    assert _mine(conn)["postgresql"] == "production"


def test_a_failed_apply_changes_nothing(conn, tmp_path, monkeypatch):
    profile, links = _plan_and_links(conn, tmp_path, approve=True)
    ov.apply_overlay(conn, profile, links, tmp_path / "o.json")
    before = _mine(conn)
    smaller = parse_skills(_profile(tmp_path, strong="python", working="docker", familiar="go", name="q.md"))
    real = ov.apply_profile_overlay

    def broken(conn_, **kw):
        kw["aliases"] = [*kw["aliases"], ("python", "go")]  # a string that is a node: breaks the invariant
        return real(conn_, **kw)

    monkeypatch.setattr(ov, "apply_profile_overlay", broken)
    with pytest.raises(ValueError):
        ov.apply_overlay(conn, smaller, tmp_path / "none.json", tmp_path / "o2.json")
    assert _mine(conn) == before and get_skill_vocab(conn, "docker") is None
    assert not (tmp_path / "o2.json").exists()


# ── CLI ─────────────────────────────────────────────────────────────────────
def _cli(tmp_path, pg_url, *extra):
    return [
        "--db", pg_url,
        "overlay", "--profile", str(tmp_path / "p.md"),
        "--links-path", str(tmp_path / "links.json"), "--overlay-path", str(tmp_path / "overlay.json"), *extra,
    ]  # fmt: skip


def test_cli_dry_run_then_apply(tmp_path, monkeypatch, capsys, pg_url):
    from hiredra.graph.config import load_graph_config

    cfg = load_graph_config()
    c = connect(pg_url)
    init_db(c)
    for name, axis in (("python", 0), ("large language models", 6)):
        v = np.zeros(cfg.dimensions, dtype=np.float32)
        v[axis] = 1.0
        add_market_skill(c, name)
        save_vectors(c, [name], np.asarray([v]), cfg.embed_tag)
    c.close()
    _profile(tmp_path, strong="python", working="llms", familiar="rust")

    def fake_builder(cfg_, **kw):
        def embed(names):
            out = np.zeros((len(names), cfg.dimensions), dtype=np.float32)
            for i, n in enumerate(names):
                out[i, 6 if n == "llms" else 40] = 1.0
            return out

        return embed

    monkeypatch.setattr("hiredra.graph.embedder.build_embedder", fake_builder)
    assert graph_main(_cli(tmp_path, pg_url)) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "[working] llms -> large language models" in out and "[familiar] rust" in out
    c = connect(pg_url)
    assert c.execute("SELECT COUNT(*) FROM skill_vocab WHERE mine = true").fetchone()[0] == 0  # dry run changed nothing
    c.close()

    assert graph_main(_cli(tmp_path, pg_url, "--apply")) == 0
    assert "overlay applied: 3 mine node(s)" in capsys.readouterr().out
    c = connect(pg_url)
    assert {r[0] for r in c.execute("SELECT name FROM skill_vocab WHERE mine = true")} == {"python", "llms", "rust"}
    c.close()


def test_cli_without_a_profile_names_the_owner_task(tmp_path, capsys, pg_url):
    assert graph_main(_cli(tmp_path, pg_url)) == 1
    assert "owner task O1a" in capsys.readouterr().out


def test_cli_bad_profile_shows_the_line(tmp_path, capsys, pg_url):
    (tmp_path / "p.md").write_text("## Skills\n**Strong:**\nPython\n", encoding="utf-8")
    assert graph_main(_cli(tmp_path, pg_url)) == 1
    assert "line 3: rule 4" in capsys.readouterr().out


def test_cli_model_stop_writes_nothing(tmp_path, monkeypatch, capsys, pg_url):
    from hiredra.graph.config import load_graph_config
    from hiredra.graph.embedder import EmbeddingError

    cfg = load_graph_config()
    c = connect(pg_url)
    init_db(c)
    add_market_skill(c, "python")
    save_vectors(
        c,
        [c.execute("SELECT 'python'").fetchone()[0]],
        np.asarray([np.eye(cfg.dimensions, dtype=np.float32)[0]]),
        cfg.embed_tag,
    )
    c.close()
    _profile(tmp_path, strong="python", working="llms", familiar="rust")

    def broken(cfg_, **kw):
        def embed(names):
            raise EmbeddingError("embedding has a zero or non-finite norm")

        return embed

    monkeypatch.setattr("hiredra.graph.embedder.build_embedder", broken)
    assert graph_main(_cli(tmp_path, pg_url)) == 2
    assert "STOPPED (bad_response)" in capsys.readouterr().out
    assert not (tmp_path / "links.json").exists()
