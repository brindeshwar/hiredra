"""Tests for hiredra.filter (build sub-slice 2b: seniority gate + role-type).

Location gate tests live in tests/test_location.py; liveness tests live in
tests/test_liveness.py -- each gate's tests sit beside the module they cover."""

from __future__ import annotations

from hiredra import filter as funnel
from hiredra.storage import connect, get_job, init_db, upsert_job

CONFIG = funnel.load_filter_config()


def _job(**overrides):
    job = {
        "job_id": overrides.pop("job_id", "j1"),
        "url": overrides.pop("url", "https://example.com/j1"),
        "title": "Backend Engineer",
        "company": "Acme",
        "description": "We need a backend engineer.",
        "source": "greenhouse",
    }
    job.update(overrides)
    return job


# --- extract_max_years -------------------------------------------------
def test_extract_max_years_plain():
    assert funnel.extract_max_years("3+ years of experience required") == 3


def test_extract_max_years_takes_the_max_of_multiple_matches():
    assert funnel.extract_max_years("2 years minimum, 5+ years preferred") == 5


def test_extract_max_years_none_when_absent():
    assert funnel.extract_max_years("Great team, great mission, no year stated") is None


def test_extract_max_years_none_is_not_zero():
    # locked rule: unknown is None, never a stand-in for an explicit 0
    assert funnel.extract_max_years("") is None
    assert funnel.extract_max_years(None) is None


def test_extract_max_years_ignores_education_boilerplate():
    # real Accenture posting text: the true requirement is 3 years, but a
    # naive max-of-all-matches picks up "15" from the education line instead.
    text = (
        "**Must have skills :** \n Large Language Models (LLMs)\n\n"
        "**Minimum 3 Year(s) Of Experience Is Required**\n"
        "**Educational Qualification :** \n 15 years full time education\n\n"
        "Summary: As an AI / ML Engineer, a typical day involves..."
    )
    assert funnel.extract_max_years(text) == 3


def test_extract_max_years_ignores_full_time_education_phrasing_either_order():
    # the boilerplate sometimes has no separate "Educational Qualification"
    # label -- the exclusion keys off "full time education" alone too.
    text = "2 years of experience required. 15 years full-time education."
    assert funnel.extract_max_years(text) == 2


def test_extract_max_years_still_counts_unrelated_high_numbers():
    # the exclusion is narrow -- a genuine high experience requirement
    # elsewhere in the same JD, with no education phrasing nearby, still counts.
    text = "10+ years of professional software engineering experience required."
    assert funnel.extract_max_years(text) == 10


# --- word-boundary matching (the approved correctness fix) --------------
def test_contains_term_does_not_match_across_a_word_boundary():
    # "forward deployed engineer" must not match inside "...Engineering" --
    # a real title from the data review (a hardware role, not the AI-industry
    # eng title the term was added for).
    title = "Physical Design Engineer, Forward Deployed Engineering"
    assert not funnel._contains_term(title.lower(), "forward deployed engineer")


def test_contains_term_matches_the_whole_phrase():
    assert funnel._contains_term("forward deployed engineer (fde) - sf".lower(), "forward deployed engineer")


def test_contains_term_handles_a_punctuation_edged_term():
    # "sr." ends in a non-word char -- the naive \b approach fails here.
    assert funnel._contains_term("sr. engineer".lower(), "sr.")
    assert not funnel._contains_term("sre engineer".lower(), "sr.")


def test_agent_does_not_match_agentic():
    assert not funnel._contains_term("agentic workflows engineer".lower(), "agent")
    assert funnel._contains_term("ai agent platform engineer".lower(), "agent")


# --- the MTS carve-out ---------------------------------------------------
def test_staff_engineer_is_caught_by_the_senior_term():
    term = funnel.has_senior_title_term("Staff Software Engineer", CONFIG.senior_title_terms)
    assert term == "staff"


def test_member_of_technical_staff_survives_the_title_term_check():
    term = funnel.has_senior_title_term("Member of Technical Staff - MTS", CONFIG.senior_title_terms)
    assert term is None


def test_mts_abbreviation_alone_also_survives():
    term = funnel.has_senior_title_term("MTS, Applied AI", CONFIG.senior_title_terms)
    assert term is None


def test_senior_member_of_technical_staff_is_still_caught_via_senior():
    # neutralizing "staff" doesn't neutralize other senior terms in the title
    term = funnel.has_senior_title_term("Senior Member of Technical Staff", CONFIG.senior_title_terms)
    assert term == "senior"


def test_mts_role_type_addition_actually_works_end_to_end():
    # the whole point of adding "member of technical staff" to ALLOW: an
    # otherwise-plain MTS title, with no years/anti-signal in the body, must
    # survive both gates.
    seniority = funnel.passes_seniority_gate(
        "Member of Technical Staff - MTS", "Join our team building great products.", CONFIG
    )
    assert seniority.keep
    role = funnel.passes_role_type("Member of Technical Staff - MTS", CONFIG)
    assert role.keep


# --- seniority gate -------------------------------------------------------
def test_seniority_gate_drops_on_years_over_cutoff():
    r = funnel.passes_seniority_gate("Backend Engineer", "5+ years of experience", CONFIG)
    assert not r.keep and r.years == 5 and "requires" in r.reason


def test_seniority_gate_keeps_unknown_years():
    r = funnel.passes_seniority_gate("Backend Engineer", "Great mission, no years stated", CONFIG)
    assert r.keep and r.years is None


def test_seniority_gate_keeps_at_the_cutoff():
    r = funnel.passes_seniority_gate("Backend Engineer", "2 years of experience", CONFIG)
    assert r.keep and r.years == 2


def test_seniority_gate_drops_on_title_term_even_with_low_years():
    r = funnel.passes_seniority_gate("Engineering Manager", "1 year of experience", CONFIG)
    assert not r.keep and r.reason == "title term 'manager'"


def test_seniority_gate_drops_on_anti_signal_phrase_with_no_years_stated():
    r = funnel.passes_seniority_gate("Backend Engineer", "You will mentor junior engineers on the team.", CONFIG)
    assert not r.keep and "anti-signal phrase" in r.reason
    assert r.years is None  # still populated (as None) even though it's what drops it


def test_data_architect_addition_was_dropped_and_is_dead_via_seniority_anyway():
    # confirms the reasoning for NOT adding "data architect" to role-type
    # ALLOW: "architect" already gates it out via the seniority title-term
    # check before role-type ever runs.
    assert "data architect" not in CONFIG.allow
    r = funnel.passes_seniority_gate("Data Architect", "No years stated.", CONFIG)
    assert not r.keep and r.reason == "title term 'architect'"


# --- role-type filter -----------------------------------------------------
def test_role_type_keeps_on_allow():
    r = funnel.passes_role_type("Backend Engineer", CONFIG)
    assert r.keep


def test_role_type_drops_on_block():
    r = funnel.passes_role_type("Sales Development Representative", CONFIG)
    assert not r.keep and "block term" in r.reason


def test_role_type_drops_neither():
    r = funnel.passes_role_type("Executive Assistant", CONFIG)
    assert not r.keep and r.reason == "no allow term matched"


def test_role_type_block_wins_on_both():
    r = funnel.passes_role_type("Backend Sales Engineer", CONFIG)
    assert not r.keep and "block term" in r.reason


# --- wiring ----------------------------------------------------------------
def test_run_funnel_advances_and_drops_with_reasons():
    conn = connect(":memory:")
    init_db(conn)
    upsert_job(
        conn, _job(job_id="keep1", url="u1", title="Backend Engineer", description="Great team, no years stated.")
    )
    upsert_job(conn, _job(job_id="drop-sen", url="u2", title="Engineering Manager", description="5+ years required."))
    upsert_job(conn, _job(job_id="drop-role", url="u3", title="Content Writer", description="Write great content."))

    stats = funnel.run_funnel(conn, CONFIG)

    assert stats.seen == 3
    assert stats.kept == 1
    assert stats.dropped_seniority == 1
    assert stats.dropped_role_type == 1

    keep = get_job(conn, "keep1")
    assert keep["status"] == "filtered"
    assert keep["years_required"] is None

    dropped_sen = get_job(conn, "drop-sen")
    assert dropped_sen["status"] == "dropped"
    assert dropped_sen["years_required"] == 5
    assert dropped_sen["status_detail"].startswith("seniority:")

    dropped_role = get_job(conn, "drop-role")
    assert dropped_role["status"] == "dropped"
    assert dropped_role["status_detail"].startswith("role_type:")

    conn.close()


def test_run_funnel_is_idempotent_on_a_rerun():
    conn = connect(":memory:")
    init_db(conn)
    upsert_job(conn, _job(job_id="keep1", url="u1"))
    funnel.run_funnel(conn, CONFIG)
    stats2 = funnel.run_funnel(conn, CONFIG)
    assert stats2.seen == 0  # nothing left at status='fetched'
    conn.close()


# --- markdown-escaped JobSpy text (M0.5 finding) -----------------------------
def test_escaped_plus_years_is_read():
    from hiredra.filter import load_filter_config, passes_seniority_gate

    cfg = load_filter_config()
    r = passes_seniority_gate("AI Engineer", r"**MUST HAVES** \- **4\+ years of experience** in Python", cfg)
    assert (r.keep, r.years) == (False, 4)
    assert passes_seniority_gate("AI Engineer", r"0\-2 years of experience", cfg).keep is True


def test_unescape_markdown_leaves_plain_text_alone():
    from hiredra.filter import unescape_markdown

    assert unescape_markdown("C++ and 3+ years") == "C++ and 3+ years"
    assert unescape_markdown(r"3\+ years, \(remote\)") == "3+ years, (remote)"
    assert unescape_markdown(None) is None and unescape_markdown("") == ""
