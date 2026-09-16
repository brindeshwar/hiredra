"""Tests for hiredra.filter (build sub-slice 2b: seniority gate + role-type +
location + liveness)."""

from __future__ import annotations

import pytest

from hiredra import filter as funnel
from hiredra.sourcing.http import HTTPError, HttpClient, HttpConfig
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import FetchContext
from hiredra.storage import connect, get_events, get_job, init_db, upsert_job

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
    term = funnel.has_senior_title_term(
        "Member of Technical Staff - MTS", CONFIG.senior_title_terms
    )
    assert term is None


def test_mts_abbreviation_alone_also_survives():
    term = funnel.has_senior_title_term("MTS, Applied AI", CONFIG.senior_title_terms)
    assert term is None


def test_senior_member_of_technical_staff_is_still_caught_via_senior():
    # neutralizing "staff" doesn't neutralize other senior terms in the title
    term = funnel.has_senior_title_term(
        "Senior Member of Technical Staff", CONFIG.senior_title_terms
    )
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
    r = funnel.passes_seniority_gate(
        "Backend Engineer", "You will mentor junior engineers on the team.", CONFIG
    )
    assert not r.keep and "anti-signal phrase" in r.reason
    assert r.years is None  # still populated (as None) even though it's what drops it


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


def test_data_architect_addition_was_dropped_and_is_dead_via_seniority_anyway():
    # confirms the reasoning for dropping the addition: "architect" already
    # gates it out before role-type ever runs.
    assert "data architect" not in CONFIG.allow
    r = funnel.passes_seniority_gate("Data Architect", "No years stated.", CONFIG)
    assert not r.keep and r.reason == "title term 'architect'"


# --- wiring ----------------------------------------------------------------
def test_run_funnel_advances_and_drops_with_reasons():
    conn = connect(":memory:")
    init_db(conn)
    upsert_job(conn, _job(job_id="keep1", url="u1", title="Backend Engineer",
                           description="Great team, no years stated."))
    upsert_job(conn, _job(job_id="drop-sen", url="u2", title="Engineering Manager",
                           description="5+ years required."))
    upsert_job(conn, _job(job_id="drop-role", url="u3", title="Content Writer",
                           description="Write great content."))

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


# --- check_liveness ----------------------------------------------------------
class _FakeLivenessProvider:
    """Mirrors tests/test_orchestrator.py's FakeProvider shape, plus the
    _board_url() hook check_liveness reuses to group jobs by board."""

    def __init__(self, name, board_url="https://fake/board", jobs=None, error=None):
        self.name = name
        self._board_url_value = board_url
        self._jobs = jobs if jobs is not None else []
        self._error = error
        self.fetch_calls = 0

    def detect(self, entry):
        return entry.provider == self.name

    def _board_url(self, entry):
        return self._board_url_value

    def fetch(self, entry, ctx):
        self.fetch_calls += 1
        if self._error is not None:
            raise self._error
        return list(self._jobs)


@pytest.fixture()
def ctx():
    c = FetchContext(http=HttpClient(HttpConfig()))
    yield c
    c.http.close()


def _filtered_job(conn, **overrides):
    """A job already sitting at status='filtered' -- what check_liveness reads."""
    job = _job(**overrides)
    upsert_job(conn, job)
    from hiredra.storage import set_status

    set_status(conn, job["job_id"], "filtered")
    return job["job_id"]


def test_liveness_keeps_a_job_still_in_the_fresh_listing(ctx):
    conn = connect(":memory:")
    init_db(conn)
    job_id = _filtered_job(conn, job_id="p1", url="https://fake.example/p1")
    fresh = [Job(job_id=job_id, url="https://fake.example/p1", title="x", company="Acme", source="greenhouse")]
    provider = _FakeLivenessProvider("greenhouse", jobs=fresh)

    stats = funnel.check_liveness(conn, ctx=ctx, registry=[provider])

    assert stats.checked == 1 and stats.boards_checked == 1
    assert stats.live == 1 and stats.gone == 0 and stats.ambiguous == 0
    assert get_job(conn, job_id)["status"] == "filtered"
    conn.close()


def test_liveness_drops_a_job_absent_from_the_fresh_listing(ctx):
    conn = connect(":memory:")
    init_db(conn)
    job_id = _filtered_job(conn, job_id="p1", url="https://fake.example/p1")
    provider = _FakeLivenessProvider("greenhouse", jobs=[])  # board is alive, this posting isn't listed anymore

    stats = funnel.check_liveness(conn, ctx=ctx, registry=[provider])

    assert stats.gone == 1 and stats.live == 0
    row = get_job(conn, job_id)
    assert row["status"] == "dropped" and row["status_detail"] == "liveness: gone"
    conn.close()


def test_liveness_drops_the_whole_board_on_a_404(ctx):
    conn = connect(":memory:")
    init_db(conn)
    j1 = _filtered_job(conn, job_id="p1", url="https://fake.example/p1")
    j2 = _filtered_job(conn, job_id="p2", url="https://fake.example/p2", company="Acme")
    provider = _FakeLivenessProvider("greenhouse", error=HTTPError("gone", status=404))

    stats = funnel.check_liveness(conn, ctx=ctx, registry=[provider])

    assert stats.boards_checked == 1  # both jobs grouped into one board, one fetch attempt
    assert stats.gone == 2
    assert get_job(conn, j1)["status_detail"] == "liveness: gone"
    assert get_job(conn, j2)["status_detail"] == "liveness: gone"
    conn.close()


def test_liveness_keeps_jobs_when_the_recheck_is_ambiguous(ctx):
    conn = connect(":memory:")
    init_db(conn)
    job_id = _filtered_job(conn, job_id="p1", url="https://fake.example/p1")
    provider = _FakeLivenessProvider("greenhouse", error=RuntimeError("timed out"))

    stats = funnel.check_liveness(conn, ctx=ctx, registry=[provider])

    assert stats.ambiguous == 1 and stats.gone == 0
    assert get_job(conn, job_id)["status"] == "filtered"  # unchanged -- inconclusive is never a drop
    conn.close()


def test_liveness_groups_multiple_jobs_at_the_same_board_into_one_fetch(ctx):
    conn = connect(":memory:")
    init_db(conn)
    _filtered_job(conn, job_id="p1", url="https://fake.example/p1", company="Acme")
    _filtered_job(conn, job_id="p2", url="https://fake.example/p2", company="Acme")
    _filtered_job(conn, job_id="p3", url="https://fake.example/p3", company="Acme")
    fresh = [
        Job(job_id="p1", url="https://fake.example/p1", title="x", company="Acme", source="greenhouse"),
        Job(job_id="p2", url="https://fake.example/p2", title="x", company="Acme", source="greenhouse"),
        Job(job_id="p3", url="https://fake.example/p3", title="x", company="Acme", source="greenhouse"),
    ]
    provider = _FakeLivenessProvider("greenhouse", jobs=fresh)

    stats = funnel.check_liveness(conn, ctx=ctx, registry=[provider])

    assert provider.fetch_calls == 1  # one board, one fetch, three jobs checked against it
    assert stats.boards_checked == 1 and stats.live == 3
    conn.close()


def test_liveness_skips_and_notes_jobs_with_no_board_mechanism(ctx):
    conn = connect(":memory:")
    init_db(conn)
    job_id = _filtered_job(conn, job_id="js1", url="https://www.linkedin.com/jobs/view/123", source="jobspy-linkedin")

    stats = funnel.check_liveness(conn, ctx=ctx, registry=[])  # no provider handles "jobspy-linkedin"

    assert stats.unchecked == 1 and stats.gone == 0
    row = get_job(conn, job_id)
    assert row["status"] == "filtered"  # kept, never fetched
    notes = [e for e in get_events(conn, job_id) if e["kind"] == "note"]
    assert len(notes) == 1 and notes[0]["detail"] == "liveness: unchecked"
    conn.close()


def test_liveness_never_touches_jobs_already_dropped_by_an_earlier_gate(ctx):
    conn = connect(":memory:")
    init_db(conn)
    upsert_job(conn, _job(job_id="d1", url="https://fake.example/d1"))
    from hiredra.storage import set_status

    set_status(conn, "d1", "dropped", detail="role_type: no allow term matched")
    provider = _FakeLivenessProvider("greenhouse", jobs=[])

    stats = funnel.check_liveness(conn, ctx=ctx, registry=[provider])

    assert stats.checked == 0 and provider.fetch_calls == 0
    assert get_job(conn, "d1")["status_detail"] == "role_type: no allow term matched"  # untouched
    conn.close()
