"""Tests for hiredra.filter (build sub-slice 2b: seniority gate + role-type).

Location gate tests live in tests/test_location.py; liveness tests live in
tests/test_liveness.py -- each gate's tests sit beside the module they cover."""

from __future__ import annotations

import pytest

from hiredra import filter as funnel
from hiredra.storage import connect, get_job, init_db, upsert_job

CONFIG = funnel.load_filter_config()


def _job(**overrides):
    job = {
        "job_id": overrides.pop("job_id", "greenhouse:j1"),
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


# Not-a-requirement figures (2026-10-26). Every phrase below is quoted from a real job in our database that
# the old max-of-everything parser dropped.
def test_extract_max_years_ignores_company_history():
    # the false drop that found the bug: a 2027 new-grad role
    text = (
        "With over 20 years of game development expertise and teams all over North America our commitment to "
        "balanced and sustainable development is well known in the game community. "
        "1+ years of experience in software development."
    )
    assert funnel.extract_max_years(text) == 1
    assert funnel.extract_max_years("For more than 20 years, we’ve helped hotels attract guests.") is None
    assert funnel.extract_max_years("A company that has been providing quality childcare for over 20 years!") is None


def test_extract_max_years_ignores_company_size_beside_its_years():
    # the Keka false drop (2026-11-01): an "Associate AI Engineer" asking 1-2 years, dropped as 13
    text = (
        "With 13+ years of experience, 350+ professionals, and 400+ agency partners globally, E2M offers services "
        "across WordPress, Shopify, SEO, PPC, content marketing, and AI automation.\n"
        "1–2 years of experience in software development, AI/ML, backend development."
    )
    assert funnel.extract_max_years(text) == 1
    assert funnel.extract_max_years("Over 20 years of experience and 5,000 employees in 30 countries.") is None
    # about the candidate, it counts
    assert funnel.extract_max_years("You bring 5+ years of experience and 2+ projects shipped to production.") == 5
    assert funnel.extract_max_years("5+ years of experience leading 10+ engineers.") == 5


def test_extract_max_years_ignores_service_agreements():
    # the Keka false drop (2026-11-01): a fresher role dropped as needing 3 years
    text = "Fresher role.\n• Agreement Period: 3 Years from the date of joining (including probation)."
    assert funnel.extract_max_years(text) is None
    assert funnel.extract_max_years("There is a service bond of 2 years.") is None
    assert funnel.extract_max_years("Lock-in period: 18 months; 3 years after joining.") is None
    assert funnel.extract_max_years("Contract duration: 3 years") is None
    # experience still counts, even beside an agreement
    assert funnel.extract_max_years("Bond: 2 years. 4 years of experience in Java.") == 4
    assert funnel.extract_max_years("3 years of contract experience required") == 3


def test_extract_max_years_ignores_age_phrases():
    assert funnel.extract_max_years("Availability on evenings and weekends 18+ years old") is None
    assert funnel.extract_max_years("Must be at least 21 years of age; 2 years of experience.") == 2
    assert funnel.extract_max_years("Age has to be 18+ years") is None


def test_extract_max_years_ignores_time_spans():
    assert funnel.extract_max_years("What started 13 years ago is now backed by evidence.") is None
    assert funnel.extract_max_years("Over the last 5 years, BAMKO has won countless awards.") is None
    assert funnel.extract_max_years("become a $10B business in the next 4 years") is None


def test_extract_max_years_ignores_business_age_but_not_business_fields():
    assert funnel.extract_max_years("Proudly 20 years in business, and growing.") is None
    # a field, not the company's age, so it counts; as a range it counts by its lower end (was 7: the en dash
    # was not read as a range before 2026-10-29)
    assert funnel.extract_max_years("3–7 years in business operations, accounting or payroll") == 3
    assert funnel.extract_max_years("12+ years in business intelligence, analytics engineering") == 12


def test_extract_max_years_counts_operation_and_typo_requirements():
    # rules that once removed these real requirements: "of operation" as business age, "at last" as a span
    assert funnel.extract_max_years("2-6 years of operation of production systems") == 2
    assert funnel.extract_max_years("Have at last 8 years of experience in Solvency 2 regulatory framework") == 8
    assert funnel.extract_max_years("Have at last 8 years in the Solvency 2 regulatory framework") == 8


def test_extract_max_years_counts_experience_even_after_a_history_lead():
    assert funnel.extract_max_years("for over 5 years of hands-on experience we have built") == 5
    # a history-shaped clause that is about the candidate still counts
    assert funnel.extract_max_years("Our ideal candidate has worked for over 5 years in fintech") == 5


def test_extract_max_years_caps_unbelievable_figures():
    assert funnel.extract_max_years("Payment veterans with more than 100 years of combined experience") is None
    assert funnel.extract_max_years("25+ years of experience") == 25


def test_extract_max_years_reads_decimals_whole_and_rounds_up():
    # these were read from the digit after the point: "1.5+ years" as 5 (a false drop), "5.1-7 years" as 1
    assert funnel.extract_max_years("* 1.5+ years of software engineering experience") == 2
    assert funnel.extract_max_years("Experience: 0.5-2 years") == 1
    assert funnel.extract_max_years("Possesses 2.5 years of experience") == 3
    assert funnel.extract_max_years("Preferred experience: - 5.1-7 years") == 6


def test_extract_max_years_reads_a_figure_right_after_a_full_stop():
    # list items joined without a space: the point is a sentence end, not a decimal
    assert funnel.extract_max_years("license in TX, FL and/or PA.3+ years of clinical experience.1+ years") == 3


def test_extract_max_years_range_counts_its_lower_end():
    assert funnel.extract_max_years("2-5 years of experience in working with AI models") == 2


@pytest.mark.parametrize(
    "text,years",
    [
        ("**Experience:** 0–3 years", 0),  # en dash: a junior posting was dropped as needing 3
        ("2—3 years of experience in AI/ML development", 2),  # em dash
        ("Have 2 to 5 years of hands-on experience building ML models", 2),
        ("Minimum requirements 2-12+ years of experience in backend software development", 2),
        ("2 - 4 years experience", 2),
        ("up to 5 years of experience", 5),  # no lower number: not a range
    ],
)
def test_extract_max_years_reads_every_way_of_writing_a_range(text, years):
    assert funnel.extract_max_years(text) == years


def test_a_range_still_loses_to_a_higher_requirement_elsewhere():
    # the max over the posting stays: freehire's minimum would keep this 6-year role
    text = "* 6-8 years of experience in Software Engineering\n* Minimum 2 years of hands-on experience with LLMs"
    assert funnel.extract_max_years(text) == 6


def test_extract_max_years_does_not_read_yearly_as_years():
    assert funnel.extract_max_years("€20 yearly learning budget") is None


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


@pytest.mark.parametrize(
    "title,keep",
    [
        ("Software Engineer (Frontend)", False),  # F6: frontend-only gets in through "software engineer"
        ("SDE II - QA, (Fixed term Contract)", False),
        ("Golang SRE / Production Engineer", False),
        ("Frontend/Backend Engineer", True),  # names backend: not frontend-only
        ("Full Stack Engineer (Frontend Oriented)", True),
        ("Backend Engineer", True),
    ],
)
def test_off_bucket_terms_drop_frontend_test_and_sre_roles_unless_backend_or_full_stack(title, keep):
    r = funnel.passes_role_type(title, CONFIG)
    assert r.keep is keep
    if not keep:
        assert "off-bucket term" in r.reason


@pytest.mark.parametrize(
    "title,keep",
    [
        ("Collections Agent", False),  # F3: the bare "agent" kept call-centre and insurance agents
        ("Tier 1 - Support Agent", False),
        ("AI Agent Observability Engineer", True),
        ("Binance Accelerator Program - AI Agent Engineer", True),
        ("Summer 2027 Software Engineering Internship/Co-op", True),  # F5 additions
        ("Java Developer", True),
        ("Prompt Engineer", True),
    ],
)
def test_role_type_after_the_freehire_comparison(title, keep):
    assert funnel.passes_role_type(title, CONFIG).keep is keep


@pytest.mark.parametrize(
    "title,keep",
    [
        ("Software Engineer II", True),  # F4: II is the second level; the years decide
        ("Software Engineer III", False),
        ("Android Software Engineer Sênior", False),  # F7
        ("Machine Learning Engineer, Vice President", False),
    ],
)
def test_senior_title_terms_after_the_freehire_comparison(title, keep):
    assert (funnel.has_senior_title_term(title, CONFIG.senior_title_terms) is None) is keep


# --- wiring ----------------------------------------------------------------
def test_run_funnel_advances_and_drops_with_reasons(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    upsert_job(
        conn,
        _job(job_id="greenhouse:keep1", url="u1", title="Backend Engineer", description="Great team, no years stated."),
    )
    upsert_job(
        conn,
        _job(job_id="greenhouse:drop-sen", url="u2", title="Engineering Manager", description="5+ years required."),
    )
    upsert_job(
        conn, _job(job_id="greenhouse:drop-role", url="u3", title="Content Writer", description="Write great content.")
    )

    stats = funnel.run_funnel(conn, CONFIG)

    assert stats.seen == 3
    assert stats.kept == 1
    assert stats.dropped_seniority == 1
    assert stats.dropped_role_type == 1

    keep = get_job(conn, "greenhouse:keep1")
    assert keep["status"] == "filtered"
    assert keep["years_required"] is None

    dropped_sen = get_job(conn, "greenhouse:drop-sen")
    assert dropped_sen["status"] == "dropped"
    assert dropped_sen["years_required"] == 5
    assert dropped_sen["status_detail"].startswith("seniority:")

    dropped_role = get_job(conn, "greenhouse:drop-role")
    assert dropped_role["status"] == "dropped"
    assert dropped_role["status_detail"].startswith("role_type:")

    conn.close()


def test_run_funnel_is_idempotent_on_a_rerun(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    upsert_job(conn, _job(job_id="greenhouse:keep1", url="u1"))
    funnel.run_funnel(conn, CONFIG)
    stats2 = funnel.run_funnel(conn, CONFIG)
    assert stats2.seen == 0  # nothing left at status='fetched'
    conn.close()


def test_a_crash_keeps_the_committed_chunks_and_leaves_the_rest_fetched(pg_url, monkeypatch):
    """Writes commit per chunk: a failure mid-chunk keeps the earlier chunks and rolls back only its own, whose jobs
    stay 'fetched' for the next run."""
    conn = connect(pg_url)
    for n in range(5):
        upsert_job(conn, _job(job_id=f"greenhouse:j{n}", url=f"u{n}"))
    monkeypatch.setattr(funnel, "COMMIT_EVERY", 2)
    real = funnel.set_status

    def flaky(conn, job_id, *a, **kw):
        if job_id == "greenhouse:j3":
            raise RuntimeError("crash")
        real(conn, job_id, *a, **kw)

    monkeypatch.setattr(funnel, "set_status", flaky)
    with pytest.raises(RuntimeError):
        funnel.run_funnel(conn, CONFIG)
    status = {f"greenhouse:j{n}": get_job(conn, f"greenhouse:j{n}")["status"] for n in range(5)}
    assert status == {
        "greenhouse:j0": "filtered",
        "greenhouse:j1": "filtered",
        "greenhouse:j2": "fetched",  # same chunk as j3: rolled back
        "greenhouse:j3": "fetched",
        "greenhouse:j4": "fetched",
    }
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
