"""Tests for hiredra.vocab: the skill vocabulary and the rules that clean a name on its way in."""

from __future__ import annotations

import pytest

from hiredra import vocab
from hiredra.storage import add_alias, connect, get_skill_vocab, init_db, resolve_alias, touch_skill_vocab


def test_resolve_skill_inserts_a_new_folded_name(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    canonical = vocab.resolve_skill(conn, "FastAPI")
    assert canonical == "fastapi"
    assert get_skill_vocab(conn, "fastapi")["seen_count"] == 1
    conn.close()


def test_resolve_skill_exact_match_is_case_and_whitespace_insensitive(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    vocab.resolve_skill(conn, "Python")
    canonical = vocab.resolve_skill(conn, "  python  ")
    assert canonical == "python"
    assert get_skill_vocab(conn, "python")["seen_count"] == 2
    conn.close()


# ── the pure rules ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("name", "base"),
    [
        ("angular 7", "angular"),
        ("angular 16+", "angular"),
        ("python3", "python"),
        ("python 3.11", "python"),
        (".net 6/7/8", ".net"),
        ("wcag 2.0", "wcag"),
        ("odata v4", "odata"),
        ("gpt-4", "gpt"),
        ("c++14", "c++"),
        ("qt 5", "qt"),
        ("db2", None),  # stuck-on digits need a 3+ character base
        ("s3", None),
        ("ec2", None),
        ("microsoft 365", None),  # three digits: a product name, not a version
        ("arinc653", None),
        ("iso 27001", None),
        ("kafka", None),
    ],
)
def test_version_base(name, base):
    assert vocab.version_base(name) == base


@pytest.mark.parametrize(
    ("short", "long", "expected"),
    [
        ("mcp", "model context protocol", True),
        ("dfm", "design for manufacturing", True),
        ("ddd", "domain-driven design", True),
        ("w&b", "weights & biases", True),
        ("llms", "large language models", True),
        ("aws", "model context protocol", False),
        ("mcp", "mcp", False),
    ],
)
def test_is_acronym(short, long, expected):
    assert vocab.is_acronym(short, long) is expected


def test_latin_only():
    assert vocab.latin_only("estatística") and vocab.latin_only("c++") and not vocab.latin_only("llamainデックス")


# ── the resolver ──────────────────────────────────────────────────────────────
def _store(pg_url, *names, mine=()):
    conn = connect(pg_url)
    init_db(conn)
    for n in names:
        touch_skill_vocab(conn, n)
    for n in mine:
        conn.execute("UPDATE skill_vocab SET mine = true, source = 'profile' WHERE name = %s", (n,))
    return conn


def test_same_spelling_becomes_an_alias_of_the_known_skill(pg_url):
    conn = _store(pg_url, "javascript", "node.js", "pandas")
    r = vocab.Resolver(conn)
    assert r.resolve("Java Script") == ["javascript"]
    assert r.resolve("node js") == ["node.js"]
    assert r.resolve("panda") == ["pandas"]
    assert resolve_alias(conn, "java script") == "javascript"
    assert get_skill_vocab(conn, "java script") is None
    assert get_skill_vocab(conn, "javascript")["seen_count"] == 2
    # the next run finds it by the alias table alone
    assert vocab.Resolver(conn).resolve("java script") == ["javascript"]
    conn.close()


def test_the_owners_skill_wins_a_shared_spelling(pg_url):
    conn = _store(pg_url, "neural-networks", "neural-networks", "neural network", mine=["neural network"])
    assert vocab.Resolver(conn).resolve("Neural Networks") == ["neural network"]
    conn.close()


def test_a_version_of_a_known_skill_is_that_skill(pg_url):
    conn = _store(pg_url, "angular", "python", ".net")
    r = vocab.Resolver(conn)
    assert r.resolve("Angular 7") == ["angular"]
    assert r.resolve("python3") == ["python"]
    assert r.resolve(".NET 6/7/8") == [".net"]
    assert r.resolve("react 18") == ["react 18"]  # base unknown: a new name, not a guess
    conn.close()


def test_never_merge_pairs_and_keep_whole_names_are_respected(pg_url):
    conn = _store(pg_url, "ada", "ci", "cd", "web", "do-178b", "c")
    r = vocab.Resolver(conn, never=[frozenset(("ada", "adas"))], keep_whole=["ci/cd", "web3", "do-178b/c"])
    assert r.resolve("ADAS") == ["adas"]
    assert r.resolve("CI/CD") == ["ci/cd"]
    assert r.resolve("web3") == ["web3"]
    assert r.resolve("DO-178B/C") == ["do-178b/c"]
    conn.close()


def test_lists_split_only_into_known_skills(pg_url):
    conn = _store(pg_url, "docker", "kubernetes", "c", "c++", "distributed systems", "html", "css")
    r = vocab.Resolver(conn, skip=["collaboration", "communication"])
    assert r.resolve("Docker & Kubernetes") == ["docker", "kubernetes"]
    assert r.resolve("C/C++") == ["c", "c++"]
    assert r.resolve("html/css") == ["html", "css"]
    assert r.resolve("trust and safety") == ["trust and safety"]  # parts unknown: one skill
    assert r.resolve("building large, distributed systems") == ["distributed systems"]
    assert r.resolve("collaboration and communication") == []  # every part generic
    conn.close()


def test_bracketed_acronym_is_one_skill_with_every_form_an_alias(pg_url):
    conn = _store(pg_url, "mcp")
    r = vocab.Resolver(conn)
    assert r.resolve("Model Context Protocol (MCP)") == ["mcp"]
    assert resolve_alias(conn, "model context protocol") == "mcp"
    assert r.resolve("domain-driven design (ddd)") == ["domain-driven design"]
    assert resolve_alias(conn, "ddd") == "domain-driven design"
    assert r.resolve("DDD") == ["domain-driven design"]
    conn.close()


def test_brackets_with_examples_give_each_known_example(pg_url):
    conn = _store(pg_url, "prometheus", "grafana")
    r = vocab.Resolver(conn)
    assert r.resolve("observability (e.g., Prometheus, Grafana)") == ["observability", "prometheus", "grafana"]
    conn.close()


def test_junk_names_are_dropped_and_reported(pg_url):
    conn = _store(pg_url)
    r = vocab.Resolver(conn, max_words=4)
    assert r.resolve("llamainデックス") == []
    assert r.resolve("networking protocols in layers 3 and 4") == []
    assert r.resolve("1553") == []
    assert [why for _, why in r.dropped] == ["non-Latin letters", "too many words", "no letters"]
    assert get_skill_vocab(conn, "1553") is None
    conn.close()


def test_generic_words_are_dropped_without_being_stored(pg_url):
    conn = _store(pg_url)
    assert vocab.Resolver(conn, skip=["communication"]).resolve("Communication") == []
    assert get_skill_vocab(conn, "communication") is None
    conn.close()


def test_an_existing_alias_still_resolves_first(pg_url):
    conn = _store(pg_url, "kubernetes")
    add_alias(conn, "k8s", "kubernetes")
    assert vocab.Resolver(conn).resolve("K8s") == ["kubernetes"]
    conn.close()
