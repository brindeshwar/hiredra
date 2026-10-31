"""Tests for hiredra.skilltag (the port of freehire's skill dictionary, D10.3) and the dictionary producer.

The parity fixture is freehire's own test texts tagged by freehire's Go ``skilltag.Parse`` at commit 0e715dfc
(the expected output came from running the Go code, not from reading it). The port also matched the Go code
on all 665 real job descriptions of the step-9 comparison (docs/freehire-merge/09-step10-vocabulary.md).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hiredra import extraction, skilltag
from hiredra.storage import (
    add_alias,
    connect,
    get_job,
    init_db,
    set_dictionary_skills,
    set_extraction,
    touch_skill_vocab,
)

CASES = json.loads((Path(__file__).parent / "fixtures" / "freehire_skilltag_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=range(len(CASES)))
def test_parse_matches_freehire_go(case):
    assert skilltag.parse(case["text"], case["category"]) == case["tags"]


@pytest.mark.parametrize(
    ("text", "tags"),
    [
        ("We use Golang and PostgreSQL.", ["go", "postgresql"]),
        ("Objective-C developer", ["objective-c"]),  # the "c" after "-" is not C
        ("see docs at foo.asp.net for details", []),  # nor an alias after "."
        ("distributed-systems and distributed_systems", ["distributed-systems"]),  # separator-insensitive phrase
        ("<p>Kubernetes</p><a href='k8s'>apply</a> https://example.com/about.html", ["kubernetes"]),  # markup, URLs
        ("ML and Python", ["machine-learning", "python"]),  # an UPPERCASE acronym tags
        ("add 5 ml of water", []),  # its lowercase form does not
        ("You must react quickly to customers in Zendesk.", ["zendesk"]),  # lowercase proper noun: not React
        ("React and TypeScript", ["react", "typescript"]),
        ("Spring in the park", []),  # an ambiguous word alone is not a skill
    ],
)
def test_parse_rules(text, tags):
    assert skilltag.parse(text) == tags


def test_rag_needs_an_ai_category():
    text = "Build RAG pipelines in Python."
    assert "rag" not in skilltag.parse(text)  # "RAG status" in general job text
    assert "rag" in skilltag.parse(text, skilltag.DOMAIN_CATEGORY["agent_llm"])


# --- the producer ---------------------------------------------------------------------------------
CFG = extraction.ExtractionConfig(
    min_description_chars=200,
    requests_per_minute=15,
    generic_skills=frozenset({"ai"}),
    dictionary_skip=frozenset({"slack"}),
)
NAMES = {
    "python": "python",
    "kubernetes": "kubernetes",
    "apache-spark": "apache spark",
    "stripe": "stripe",
    "slack": "slack",
    "artificial-intelligence": "ai",
    "docker": "docker",
    "postgresql": "postgresql",
}


def test_dictionary_extras_adds_what_the_llm_left_out_and_says_why_the_rest_went(monkeypatch):
    monkeypatch.setattr(skilltag, "parse", lambda text, category=None: sorted(NAMES))
    extras, dropped = extraction.dictionary_extras(
        ["python", "spark", "postgres"], "jd", "Stripe, Inc.", "backend", NAMES, CFG
    )
    assert extras == ["docker", "kubernetes", "postgresql"]
    assert dropped == {
        "listed": ["apache spark", "python"],  # whole-word containment: "spark" is inside "apache spark"
        "generic": ["ai", "slack"],
        "company": ["stripe"],
    }


def test_dictionary_version_follows_both_stoplists():
    base = extraction.dictionary_version(CFG)
    assert base.startswith("skilltag:")
    assert extraction.dictionary_version(dataclasses.replace(CFG, dictionary_skip=frozenset())) != base
    assert extraction.dictionary_version(dataclasses.replace(CFG, generic_skills=frozenset())) != base
    assert extraction.dictionary_version(dataclasses.replace(CFG, skill_synonyms={"a": ("b",)})) == base


def test_vocab_names_prefers_our_names_and_aliases(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    for name in ("node.js", "ci/cd", "kubernetes"):
        touch_skill_vocab(conn, name)
    add_alias(conn, "k8s", "kubernetes")
    names = skilltag.vocab_names(conn, ["nodejs", "ci-cd", "kubernetes", "dspy"])
    assert names == {"nodejs": "node.js", "ci-cd": "ci/cd", "kubernetes": "kubernetes", "dspy": "dspy"}
    conn.close()


def _extracted(conn, job_id="greenhouse:d1"):
    from hiredra.storage import set_status, upsert_job

    upsert_job(
        conn,
        {
            "job_id": job_id,
            "url": f"https://example.com/{job_id}",
            "title": "Backend Engineer",
            "company": "Acme",
            "source": "greenhouse",
            "description": "Python services on Kubernetes with Docker and PostgreSQL. " * 6,
        },
    )
    set_status(conn, job_id, "filtered")
    set_extraction(
        conn,
        job_id,
        skills=[{"name": "python", "requirement": "required"}],
        domain_bucket="backend",
        legitimacy="legitimate",
        extraction_confidence="high",
        extraction_version="v",
    )
    set_status(conn, job_id, "extracted")
    return job_id


def test_dictionary_rows_follow_the_llm_rows_and_are_replaced_on_rerun(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    jid = _extracted(conn)
    set_dictionary_skills(conn, jid, ["docker", "python", "kubernetes"], "skilltag:1")
    row = get_job(conn, jid)
    assert [(s["name"], s["requirement"], s["source"]) for s in row["skills"]] == [
        ("python", "required", "llm"),  # the LLM's row stays; the dictionary's "python" is skipped
        ("docker", "good_to_have", "dictionary"),
        ("kubernetes", "good_to_have", "dictionary"),
    ]
    assert row["dictionary_version"] == "skilltag:1"
    set_dictionary_skills(conn, jid, ["postgresql"], "skilltag:2")
    row = get_job(conn, jid)
    assert [s["name"] for s in row["skills"]] == ["python", "postgresql"]
    assert row["dictionary_version"] == "skilltag:2"
    conn.close()


def test_add_dictionary_skills_backfills_once_per_stamp(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    jid = _extracted(conn)
    counts = extraction.add_dictionary_skills(conn, CFG)
    row = get_job(conn, jid)
    found = [s["name"] for s in row["skills"] if s["source"] == "dictionary"]
    assert counts == {"jobs": 1, "scored": 0, "skills_added": len(found)}
    assert {"docker", "kubernetes", "postgresql"} <= set(found) and "python" not in found
    assert row["dictionary_version"] == extraction.dictionary_version(CFG)
    assert extraction.add_dictionary_skills(conn, CFG)["jobs"] == 0  # already stamped: nothing to do
    conn.close()
