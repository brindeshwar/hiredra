"""Tests for hiredra.extraction (build step 3: skill-extraction agent).

No live network/API key anywhere here -- a fake chain stands in for the
LangChain structured-output chain, same style as tests/test_liveness.py's
_FakeLivenessProvider. The live smoke test (real Gemini call) is a separate,
manually-run script (scripts/smoke_test_extraction.py).
"""

from __future__ import annotations

import json


from hiredra import extraction
from hiredra.storage import connect, get_events, get_job, get_skill_vocab, init_db, upsert_job

CONFIG = extraction.ExtractionConfig(
    min_description_chars=200,
    requests_per_minute=15,
    boilerplate_footers={"Acme": "FOOTER: equal opportunity employer etc."},
)


def _job(**overrides):
    job = {
        "job_id": overrides.pop("job_id", "j1"),
        "url": overrides.pop("url", "https://example.com/j1"),
        "title": "Backend Engineer",
        "company": "Acme",
        "description": "x" * 500,
        "source": "greenhouse",
    }
    job.update(overrides)
    return job


def _filtered_job(conn, **overrides):
    job = _job(**overrides)
    upsert_job(conn, job)
    from hiredra.storage import set_status

    set_status(conn, job["job_id"], "filtered")
    return job["job_id"]


class _FakeChain:
    """Mirrors tests/test_liveness.py's fake-provider shape."""

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if self._error is not None:
            raise self._error
        return self._result


def _result(**overrides):
    base = dict(
        skills=[
            extraction.SkillItem(name="FastAPI", requirement="required"),
            extraction.SkillItem(name="python", requirement="good_to_have"),
        ],
        domain_bucket="backend",
        legitimacy="legitimate",
        extraction_confidence="high",
    )
    base.update(overrides)
    return extraction.ExtractionResult(**base)


# --- preprocessing: boilerplate footer strip --------------------------------
def test_strip_boilerplate_footer_exact_match():
    text = "Real JD content here.\n\nFOOTER: equal opportunity employer etc."
    clean, log = extraction.preprocess(text, "Acme", CONFIG)
    assert clean == "Real JD content here."
    assert any("boilerplate footer" in note for note in log)


def test_strip_boilerplate_footer_does_not_touch_a_near_miss():
    # the footer text is close but not byte-identical -- must not strip.
    text = "Real JD content here.\n\nFOOTER: equal opportunity employer, etc."
    clean, log = extraction.preprocess(text, "Acme", CONFIG)
    assert clean == text
    assert log == []


def test_strip_boilerplate_footer_no_entry_for_company_is_a_noop():
    text = "Real JD content here.\n\nFOOTER: equal opportunity employer etc."
    clean, log = extraction.preprocess(text, "SomeOtherCo", CONFIG)
    assert clean == text
    assert log == []


# --- preprocessing: leading "About <Company>" paragraph ---------------------
def test_strip_leading_about_paragraph_bare_header_plus_mission_paragraph():
    # Stripe/OpenAI's real shape: "About X" as its own bare header line,
    # then the mission text as a separate paragraph -- both get consumed.
    text = (
        "About Stripe\n\nStripe is a financial infrastructure platform for "
        "businesses and does many things.\n\nWhat you'll do\n\nBuild things."
    )
    clean, log = extraction.preprocess(text, "Stripe", CONFIG)
    assert clean == "What you'll do\n\nBuild things."
    assert any("About Stripe" in note for note in log)


def test_strip_leading_about_paragraph_single_paragraph_shape():
    # "About X" with real content in the SAME paragraph (no bare header
    # line) -- only that one paragraph is consumed.
    text = "About Acme is a great place to build things.\n\nWhat you'll do\n\nBuild things."
    clean, log = extraction.preprocess(text, "Acme", CONFIG)
    assert clean == "What you'll do\n\nBuild things."
    assert any("About Acme" in note for note in log)


def test_strip_leading_about_paragraph_only_touches_the_leading_one():
    # "About" mentioned later in the text must survive untouched.
    text = "What you'll do\n\nBuild things. Learn more about Stripe on our site."
    clean, log = extraction.preprocess(text, "Stripe", CONFIG)
    assert clean == text
    assert log == []


def test_strip_leading_about_paragraph_no_blank_line_leaves_text_alone():
    text = "About Stripe is a company with no blank line anywhere in this JD"
    clean, log = extraction.preprocess(text, "Stripe", CONFIG)
    assert clean == text
    assert log == []


# --- preprocessing: markdown unescape + blank-line collapse -----------------
def test_unescape_markdown_punctuation():
    # the verified characterization set: \- \. \( \) \* \#
    text = "open\\-weight LLMs \\(on\\-premise\\)\\. Rated \\#1\\."
    clean, _ = extraction.preprocess(text, "Acme", CONFIG)
    assert clean == "open-weight LLMs (on-premise). Rated #1."


def test_collapse_excessive_blank_lines():
    text = "A bullet\n\n\n\n\nAnother bullet"
    clean, _ = extraction.preprocess(text, "Acme", CONFIG)
    assert clean == "A bullet\n\nAnother bullet"


def test_preprocess_strips_are_independent_and_composable():
    text = (
        "About Acme\n\nWe build things.\n\nRequirements\n\nPython\\-based work.\n\n\n\n"
        "FOOTER: equal opportunity employer etc."
    )
    clean, log = extraction.preprocess(text, "Acme", CONFIG)
    assert clean == "Requirements\n\nPython-based work."
    assert len(log) == 2


# --- run_extraction: skip-thin gate ------------------------------------------
def test_run_extraction_skips_a_too_thin_body_without_calling_the_model():
    conn = connect(":memory:")
    init_db(conn)
    job_id = _filtered_job(conn, description="too short")
    chain = _FakeChain(result=_result())

    stats = extraction.run_extraction(conn, CONFIG, chain=chain)

    assert chain.calls == 0
    assert stats.skipped_thin == 1 and stats.extracted == 0
    row = get_job(conn, job_id)
    assert row["status"] == "extracted"
    assert row["extraction_confidence"] == "low"
    assert row["domain_bucket"] is None and row["legitimacy"] is None
    assert row["skills"] == "[]"
    notes = [e["detail"] for e in get_events(conn, job_id) if e["kind"] == "note"]
    assert "extraction: skipped-thin" in notes
    conn.close()


def test_run_extraction_skip_threshold_is_measured_post_preprocessing():
    # Real content alone is under 200 chars; only the (stripped) footer
    # pushes the RAW length over 200. The skip check must fire on the
    # post-strip length, not the raw one.
    real_content = "Short real bit. " * 10  # 170 chars, under the threshold
    footer = CONFIG.boilerplate_footers["Acme"]
    text = f"{real_content}\n\n{footer}"
    assert len(real_content) < 200 <= len(text)  # confirms the raw-vs-post-strip gap this test needs

    conn = connect(":memory:")
    init_db(conn)
    job_id = _filtered_job(conn, description=text, company="Acme")
    chain = _FakeChain(result=_result())

    extraction.run_extraction(conn, CONFIG, chain=chain)

    assert chain.calls == 0
    assert get_job(conn, job_id)["extraction_confidence"] == "low"
    conn.close()


# --- run_extraction: normal call path ----------------------------------------
def test_run_extraction_writes_all_four_fields_and_advances_status():
    conn = connect(":memory:")
    init_db(conn)
    job_id = _filtered_job(conn)
    chain = _FakeChain(result=_result())

    stats = extraction.run_extraction(conn, CONFIG, chain=chain)

    assert chain.calls == 1
    assert stats.extracted == 1
    row = get_job(conn, job_id)
    assert row["status"] == "extracted"
    assert row["domain_bucket"] == "backend"
    assert row["legitimacy"] == "legitimate"
    assert row["extraction_confidence"] == "high"
    skills = json.loads(row["skills"])
    assert skills == [
        {"name": "fastapi", "requirement": "required"},
        {"name": "python", "requirement": "good_to_have"},
    ]
    conn.close()


def test_run_extraction_canonicalizes_skill_names_through_the_vocab():
    conn = connect(":memory:")
    init_db(conn)
    job_id = _filtered_job(conn)
    chain = _FakeChain(
        result=_result(
            skills=[
                extraction.SkillItem(name="  FastAPI  ", requirement="required"),
            ]
        )
    )

    extraction.run_extraction(conn, CONFIG, chain=chain)

    assert get_skill_vocab(conn, "fastapi") is not None
    skills = json.loads(get_job(conn, job_id)["skills"])
    assert skills == [{"name": "fastapi", "requirement": "required"}]
    conn.close()


def test_run_extraction_never_calls_the_model_for_already_extracted_jobs():
    conn = connect(":memory:")
    init_db(conn)
    _filtered_job(conn)
    chain = _FakeChain(result=_result())
    extraction.run_extraction(conn, CONFIG, chain=chain)  # first pass: extracts it

    stats2 = extraction.run_extraction(conn, CONFIG, chain=chain)  # resumed pass

    assert stats2.seen == 0  # nothing left at status='filtered'
    assert chain.calls == 1  # not called again
    conn.close()


# --- run_extraction: call failure --------------------------------------------
def test_run_extraction_leaves_a_failed_call_at_filtered_for_retry():
    conn = connect(":memory:")
    init_db(conn)
    job_id = _filtered_job(conn)
    chain = _FakeChain(error=RuntimeError("boom"))

    stats = extraction.run_extraction(conn, CONFIG, chain=chain)

    assert stats.failed == 1 and stats.extracted == 0
    row = get_job(conn, job_id)
    assert row["status"] == "filtered"  # unchanged -- a rerun retries it
    notes = [e["detail"] for e in get_events(conn, job_id) if e["kind"] == "note"]
    assert any("call failed" in n for n in notes)
    conn.close()
