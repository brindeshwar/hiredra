"""Tests for hiredra.graph.curation: the owner-editable alias and never-merge files."""

from __future__ import annotations

import textwrap

import pytest

from hiredra.graph.curation import (
    DEFAULT_ALIAS_PATH,
    DEFAULT_NEVER_MERGE_PATH,
    load_aliases,
    load_never_merge,
)


def _file(tmp_path, text, name="f.yml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


# ── aliases ──────────────────────────────────────────────────────────────────
def test_aliases_load_and_fold_names(tmp_path):
    p = _file(
        tmp_path,
        """
        aliases:
          Kubernetes: [K8S, "Kube  Native"]
          go: [golang]
    """,
    )
    assert load_aliases(p) == {"kubernetes": ("k8s", "kube native"), "go": ("golang",)}


def test_a_missing_alias_file_is_empty(tmp_path):
    assert load_aliases(tmp_path / "nope.yml") == {}


def test_an_empty_alias_file_is_empty(tmp_path):
    assert load_aliases(_file(tmp_path, "")) == {}
    assert load_aliases(_file(tmp_path, "aliases: {}\n", "b.yml")) == {}


@pytest.mark.parametrize(
    "text,match",
    [
        ("surprise: 1\n", "unknown top-level key"),
        ("aliases: [a, b]\n", "must be a mapping"),
        ("aliases:\n  go: golang\n", "non-empty list"),
        ("aliases:\n  go: []\n", "non-empty list"),
        ("aliases:\n  go: [go]\n", "alias of itself"),
        ("aliases:\n  go: [GO]\n", "alias of itself"),  # folded before comparing
        ("aliases:\n  a: [x]\n  b: [x]\n", "alias of both"),
        ("aliases:\n  a: [x]\n  x: [y]\n", "both a canonical and an alias"),  # would make a chain
        ("aliases:\n  a: [ '' ]\n", "empty or non-text"),
        ("aliases:\n  a: [1]\n", "empty or non-text"),
        ("- just\n- a list\n", "top level must be a mapping"),
    ],
)
def test_invalid_alias_files_are_rejected_loudly(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_aliases(_file(tmp_path, text))


def test_a_canonical_listed_twice_is_rejected(tmp_path):
    # YAML itself keeps the last duplicate key; a differently-cased repeat survives to our check
    with pytest.raises(ValueError, match="appears twice"):
        load_aliases(_file(tmp_path, "aliases:\n  Go: [golang]\n  go: [gopher]\n"))


# ── never-merge ──────────────────────────────────────────────────────────────
def test_never_merge_pairs_load_unordered_and_folded(tmp_path):
    p = _file(
        tmp_path,
        """
        never_merge:
          - [SQL, sql server]
          - [react, React Native]
    """,
    )
    pairs = load_never_merge(p)
    assert pairs == {frozenset(("sql", "sql server")), frozenset(("react", "react native"))}
    assert frozenset(("sql server", "sql")) in pairs  # order does not matter


def test_a_missing_or_empty_never_merge_file_is_empty(tmp_path):
    assert load_never_merge(tmp_path / "nope.yml") == frozenset()
    assert load_never_merge(_file(tmp_path, "never_merge: []\n")) == frozenset()


@pytest.mark.parametrize(
    "text,match",
    [
        ("surprise: 1\n", "unknown top-level key"),
        ("never_merge: {a: b}\n", "must be a list"),
        ("never_merge:\n  - [only-one]\n", "exactly two names"),
        ("never_merge:\n  - [a, b, c]\n", "exactly two names"),
        ("never_merge:\n  - just-a-string\n", "exactly two names"),
        ("never_merge:\n  - [a, '']\n", "exactly two names"),
        ("never_merge:\n  - [Same, same]\n", "with itself"),
    ],
)
def test_invalid_never_merge_files_are_rejected_loudly(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        load_never_merge(_file(tmp_path, text))


# ── the starter files that ship with the repo ────────────────────────────────
def test_the_shipped_alias_file_is_valid_and_conservative():
    aliases = load_aliases(DEFAULT_ALIAS_PATH)
    assert aliases["kubernetes"] == ("k8s",) and aliases["go"] == ("golang",)
    assert aliases["c#"] == ("csharp",)
    # ambiguous abbreviations are deliberately absent ("node" -> node.js was added by owner decision)
    every_alias = {a for v in aliases.values() for a in v}
    assert every_alias.isdisjoint({"ai", "ci", "ux"})
    assert aliases["node.js"] == ("nodejs", "node")
    # the owner's short/long forms that embeddings score below ADJ
    assert "model context protocol" in aliases["mcp"] and aliases["rag"] == ("retrieval augmented generation",)
    assert not (set(aliases) & every_alias)  # one hop, no chains


def test_the_shipped_never_merge_file_is_valid_and_keeps_the_c_family_apart():
    pairs = load_never_merge(DEFAULT_NEVER_MERGE_PATH)
    for a, b in [
        ("sql", "sql server"),
        ("react", "react native"),
        ("c", "c++"),
        ("c", "c#"),
        ("c++", "c#"),
        ("machine learning", "deep learning"),
    ]:
        assert frozenset((a, b)) in pairs


def test_no_shipped_alias_pair_is_also_on_the_never_merge_list():
    pairs = load_never_merge(DEFAULT_NEVER_MERGE_PATH)
    for canonical, aliases in load_aliases(DEFAULT_ALIAS_PATH).items():
        for a in aliases:
            assert frozenset((canonical, a)) not in pairs
