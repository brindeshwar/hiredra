"""Tests for hiredra.graph.descriptions and the per-skill embedding input (graph audit, 2026-11-12)."""

from __future__ import annotations

import dataclasses

import pytest

from hiredra.graph import descriptions as d
from hiredra.graph.sync import embed_pending, ensure_index_current
from hiredra.storage import add_market_skill, connect, get_skill_vocab, init_db, skill_index_status
from tests.test_sync import FakeEmbedder, _cfg

CFG = dataclasses.replace(_cfg(), inputs="described", description_sources=("override", "freehire"))


def _conn(pg_url, *names):
    conn = connect(pg_url)
    init_db(conn)
    for n in names:
        add_market_skill(conn, n)
    return conn


def _write(tmp_path, body: str):
    path = tmp_path / "skill_descriptions.yml"
    path.write_text(body, encoding="utf-8")
    return path


def test_overrides_are_strict_and_folded(tmp_path):
    good = _write(tmp_path, "overrides:\n  ' MCP ': 'Model Context Protocol.'\n  kafka: null\n")
    assert d.load_overrides(good) == {"mcp": "Model Context Protocol.", "kafka": None}
    with pytest.raises(ValueError):
        d.load_overrides(_write(tmp_path, "overides: {}\n"))
    with pytest.raises(ValueError):
        d.load_overrides(_write(tmp_path, "overrides:\n  kafka: ''\n"))


def test_refresh_ranks_override_then_freehire_then_generated(pg_url):
    conn = _conn(pg_url, "mcp", "kafka", "obscure tool", "described elsewhere")
    conn.execute(
        "UPDATE skill_vocab SET description = 'Written by the model.', description_source = 'generated' "
        "WHERE name IN ('obscure tool', 'kafka')"
    )
    counts = d.refresh_descriptions(conn, overrides={"mcp": "Model Context Protocol, for LLM tools."})
    rows = {n: get_skill_vocab(conn, n) for n in ("mcp", "kafka", "obscure tool", "described elsewhere")}
    assert (rows["mcp"]["description"], rows["mcp"]["description_source"]) == (
        "Model Context Protocol, for LLM tools.",
        "override",
    )
    assert rows["kafka"]["description_source"] == "freehire"  # freehire beats a generated one
    assert rows["obscure tool"]["description_source"] == "generated"  # kept as the describe step left it
    assert rows["described elsewhere"]["description"] is None
    assert counts == {"override": 1, "freehire": 1}
    assert d.refresh_descriptions(conn, overrides={"mcp": "Model Context Protocol, for LLM tools."}) == {}
    conn.close()


def test_the_lookup_follows_the_configured_sources(pg_url, monkeypatch):
    conn = _conn(pg_url, "obscure tool")
    conn.execute(
        "UPDATE skill_vocab SET description = 'Model text.', description_source = 'generated' WHERE name = 'obscure tool'"
    )
    monkeypatch.setattr(d, "load_overrides", lambda: {"mcp": "Model Context Protocol.", "kafka": None})
    lookup = d.embedding_descriptions(conn, CFG)
    assert lookup["mcp"] == "Model Context Protocol."  # the override replaces freehire's wrong text
    assert "kafka" not in lookup  # null: the bare name
    assert "python" in lookup and "obscure tool" not in lookup  # generated is not a configured source
    with_generated = dataclasses.replace(CFG, description_sources=("override", "freehire", "generated"))
    assert d.embedding_descriptions(conn, with_generated)["obscure tool"] == "Model text."
    assert d.embedding_descriptions(conn, dataclasses.replace(CFG, inputs="bare")) == {}
    conn.close()


def test_a_changed_description_re_embeds_only_that_skill(pg_url):
    conn = _conn(pg_url, "python", "kafka", "mcp")
    fake = FakeEmbedder()
    lookup = {"python": "A language.", "kafka": "A log."}
    first = embed_pending(conn, CFG, fake, descriptions=lookup)
    assert first.embedded == 3
    assert get_skill_vocab(conn, "python")["embed_input"] == "python: A language."
    assert get_skill_vocab(conn, "mcp")["embed_input"] == "mcp"
    lookup["mcp"] = "Model Context Protocol."
    assert skill_index_status(conn, CFG.embed_tag, {n: f"{n}: {t}" for n, t in lookup.items()})["stale"] == 1
    second = embed_pending(conn, CFG, fake, descriptions=lookup)
    assert (second.queue_size, second.embedded) == (1, 1)
    assert get_skill_vocab(conn, "mcp")["embed_input"] == "mcp: Model Context Protocol."
    conn.close()


def test_the_gate_records_old_inputs_and_skips_a_changed_canary(pg_url, monkeypatch):
    conn = _conn(pg_url, "python", "kubernetes")
    fake = FakeEmbedder()
    freehire = d.load_descriptions()
    embed_pending(conn, CFG, fake, descriptions=freehire)
    conn.execute("UPDATE skill_vocab SET embed_input = NULL, description = NULL, description_source = NULL")
    monkeypatch.setattr(d, "load_overrides", lambda: {"python": "Overridden."})
    lookup = {**freehire, "python": "Overridden."}
    gate = ensure_index_current(conn, CFG, FakeEmbedder(), descriptions=lookup)
    assert get_skill_vocab(conn, "kubernetes")["embed_input"] == f"kubernetes: {freehire['kubernetes']}"
    assert gate.canary.checked == ("kubernetes",)  # python's input changed on purpose: not drift
    assert not gate.canary.drifted and gate.invalidated == 0
    assert (gate.embed.queue_size, gate.embed.embedded) == (1, 1) and gate.ready
    assert get_skill_vocab(conn, "python")["description_source"] == "override"
    conn.close()
