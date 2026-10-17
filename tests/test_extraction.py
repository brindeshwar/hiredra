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


# --- run_extraction: failure classes (M0.1) -----------------------------------
# Each test proves one row of run_extraction's failure table.
import httpx  # noqa: E402
import pytest  # noqa: E402
from google.genai.errors import ClientError, ServerError  # noqa: E402
from langchain_core.exceptions import OutputParserException  # noqa: E402

from hiredra.lock import RunLock  # noqa: E402
from hiredra.quota import DailyUsage  # noqa: E402

_PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
_PER_DAY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def _wrapped(cause: BaseException) -> Exception:
    """How langchain-google-genai surfaces an SDK error: wrapped, cause chained."""
    try:
        raise RuntimeError("Error calling model") from cause
    except RuntimeError as e:
        return e


def _quota_429(quota_id: str, retry_delay: str = "20s") -> Exception:
    details = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": quota_id, "quotaValue": "500"}],
        },
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay},
    ]
    return _wrapped(ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": details}}))


class _ScriptChain:
    """Answers from a script, one item per call: a result, ``None``, or an exception to raise.
    Once the script runs out it keeps answering with a good result."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        item = self.script.pop(0) if self.script else _result()
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture()
def db():
    c = connect(":memory:")
    init_db(c)
    yield c
    c.close()


def _usage(tmp_path, budget=450):
    return DailyUsage(budget, tmp_path / "usage.json")


def _run(conn, chain, **kw):
    kw.setdefault("sleep", lambda s: None)
    return extraction.run_extraction(conn, CONFIG, chain=chain, **kw)


def _status(conn, job_id):
    return get_job(conn, job_id)["status"]


def _notes(conn, job_id):
    return [e["detail"] for e in get_events(conn, job_id) if e["kind"] == "note"]


def test_unknown_error_is_reraised_and_the_job_stays_filtered(db):
    job_id = _filtered_job(db)
    with pytest.raises(RuntimeError, match="boom"):
        _run(db, _ScriptChain(RuntimeError("boom")))
    assert _status(db, job_id) == "filtered"


def test_finished_jobs_stay_saved_when_a_bug_is_raised(db):
    first = _filtered_job(db, job_id="a", url="u://a")
    second = _filtered_job(db, job_id="b", url="u://b")
    with pytest.raises(KeyError):
        _run(db, _ScriptChain(_result(), KeyError("bug")))
    assert _status(db, first) == "extracted" and _status(db, second) == "filtered"


def test_googles_daily_quota_stops_the_run_marks_the_counter_and_is_not_retried(db, tmp_path):
    a = _filtered_job(db, job_id="a", url="u://a")
    b = _filtered_job(db, job_id="b", url="u://b")
    usage = _usage(tmp_path)
    chain = _ScriptChain(_quota_429(_PER_DAY, retry_delay="41000s"))

    stats = _run(db, chain, usage=usage)

    assert stats.stopped == "daily_quota" and chain.calls == 1
    assert usage.exhausted() and usage.used() == 0
    assert _status(db, a) == "filtered" and _status(db, b) == "filtered"


def test_our_own_budget_stops_before_the_call(db, tmp_path):
    a = _filtered_job(db, job_id="a", url="u://a")
    b = _filtered_job(db, job_id="b", url="u://b")
    usage = _usage(tmp_path, budget=1)
    chain = _ScriptChain()

    stats = _run(db, chain, usage=usage)

    assert stats.stopped == "daily_quota" and stats.extracted == 1
    assert chain.calls == 1  # the second job never reached Google
    assert usage.used() == 1 and not usage.exhausted()  # our cap, not Google's: nothing marked
    assert _status(db, a) == "extracted" and _status(db, b) == "filtered"


def test_a_spent_budget_means_zero_calls(db, tmp_path):
    _filtered_job(db)
    usage = _usage(tmp_path)
    usage.mark_exhausted("earlier today")
    chain = _ScriptChain()

    stats = _run(db, chain, usage=usage)

    assert stats.stopped == "daily_quota" and stats.seen == 0 and chain.calls == 0


def test_a_per_minute_429_is_retried_then_succeeds(db, tmp_path):
    job_id = _filtered_job(db)
    usage = _usage(tmp_path)
    slept: list[float] = []
    chain = _ScriptChain(_quota_429(_PER_MINUTE))

    stats = _run(db, chain, usage=usage, sleep=slept.append)

    assert stats.extracted == 1 and chain.calls == 2
    assert 21.0 in slept  # the server's 20s retryDelay + 1s margin
    assert usage.used() == 1  # the rejected attempt is not counted
    assert _status(db, job_id) == "extracted"


@pytest.mark.parametrize(
    "error",
    [
        _quota_429(_PER_MINUTE),
        _wrapped(ServerError(503, {"error": {"message": "unavailable"}})),
        _wrapped(httpx.ConnectError("connection reset")),
    ],
    ids=["429-per-minute", "503", "network"],
)
def test_transient_trouble_that_outlasts_4_tries_stops_the_run(db, error):
    job_id = _filtered_job(db)
    chain = _ScriptChain(error, error, error, error)

    stats = _run(db, chain)

    assert stats.stopped == "transient" and chain.calls == 4
    assert _status(db, job_id) == "filtered"


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_an_api_error_stops_at_once(db, code):
    _filtered_job(db, job_id="a", url="u://a")
    _filtered_job(db, job_id="b", url="u://b")
    chain = _ScriptChain(_wrapped(ClientError(code, {"error": {"message": "no"}})))

    stats = _run(db, chain)

    assert stats.stopped == "api_error" and chain.calls == 1 and stats.seen == 1


@pytest.mark.parametrize(
    "bad",
    [None, OutputParserException("not JSON"), "validation"],
    ids=["None", "unparsable", "validation"],
)
def test_a_schema_error_is_noted_and_the_run_goes_on(db, tmp_path, bad):
    if bad == "validation":
        try:
            extraction.ExtractionResult(
                skills=[], domain_bucket="nope", legitimacy="legitimate", extraction_confidence="high"
            )
        except Exception as err:  # pydantic.ValidationError
            bad = err
    a = _filtered_job(db, job_id="a", url="u://a")
    b = _filtered_job(db, job_id="b", url="u://b")
    usage = _usage(tmp_path)

    stats = _run(db, _ScriptChain(bad), usage=usage)

    assert stats.schema_failed == 1 and stats.extracted == 1 and stats.stopped is None
    assert _status(db, a) == "filtered" and _status(db, b) == "extracted"
    assert any(n.startswith("extraction: schema error (") for n in _notes(db, a))
    assert usage.used() == 2  # Google answered both times, so both count


def test_three_schema_errors_in_a_row_stop_the_run(db):
    ids = [_filtered_job(db, job_id=f"j{i}", url=f"u://{i}") for i in range(4)]
    chain = _ScriptChain(None, None, None)

    stats = _run(db, chain)

    assert stats.stopped == "schema_error" and stats.schema_failed == 3
    assert chain.calls == 3 and _status(db, ids[3]) == "filtered"  # the 4th job was never sent


def test_a_success_resets_the_schema_error_streak(db):
    for i in range(5):
        _filtered_job(db, job_id=f"j{i}", url=f"u://{i}")
    stats = _run(db, _ScriptChain(None, None, _result(), None, None))
    assert stats.stopped is None and stats.schema_failed == 4 and stats.extracted == 1


def test_limit_caps_the_jobs_processed(db):
    for i in range(3):
        _filtered_job(db, job_id=f"j{i}", url=f"u://{i}")
    chain = _ScriptChain()
    stats = _run(db, chain, limit=2)
    assert stats.seen == 2 and chain.calls == 2


# --- the extraction stamp ------------------------------------------------------
def test_every_written_job_carries_the_stamp(db):
    called = _filtered_job(db, job_id="a", url="u://a")
    thin = _filtered_job(db, job_id="b", url="u://b", description="too short")
    _run(db, _ScriptChain())
    stamp = extraction.extraction_version()
    assert stamp.startswith("gemini-3.5-flash-lite:") and len(stamp.split(":")[1]) == 8
    assert get_job(db, called)["extraction_version"] == stamp
    assert get_job(db, thin)["extraction_version"] == stamp


def test_changing_the_prompt_changes_the_stamp_and_stale_lists_exactly_those_jobs(db, monkeypatch):
    old = _filtered_job(db, job_id="old", url="u://old")
    _run(db, _ScriptChain())
    old_stamp = extraction.extraction_version()
    assert extraction.stale_jobs(db) == {}

    monkeypatch.setattr(extraction, "_PROMPT_PREFIX", extraction._PROMPT_PREFIX + "Be brief.\n")
    assert extraction.extraction_version() != old_stamp
    new = _filtered_job(db, job_id="new", url="u://new")
    _run(db, _ScriptChain())

    groups = extraction.stale_jobs(db)
    assert list(groups) == [old_stamp]
    assert [j["job_id"] for j in groups[old_stamp]] == [old]
    assert new not in {j["job_id"] for jobs in groups.values() for j in jobs}


def test_changing_the_schema_changes_the_stamp(monkeypatch):
    before = extraction.extraction_version()
    monkeypatch.setattr(extraction.ExtractionResult, "model_json_schema", classmethod(lambda cls: {"changed": True}))
    assert extraction.extraction_version() != before


# --- config + CLI ----------------------------------------------------------------
def test_the_shipped_config_gives_a_450_budget():
    assert extraction.load_extraction_config().daily_budget == 450


def _cli_paths(tmp_path):
    path = tmp_path / "h.db"
    conn = connect(path)
    init_db(conn)
    _filtered_job(conn)
    conn.close()
    return [
        "--db",
        str(path),
        "--usage-path",
        str(tmp_path / "usage.json"),
        "--lock-path",
        str(tmp_path / "llm.lock"),
    ]


def test_cli_ok_run_prints_the_final_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(extraction, "build_extraction_chain", lambda **kw: _ScriptChain())
    monkeypatch.setattr(extraction._Pacer, "wait", lambda self: None)
    assert extraction.main(_cli_paths(tmp_path)) == 0
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert out == "seen=1 extracted=1 skipped_thin=0 schema_failed=0 stopped=none used_today=1/450"


def test_cli_stop_exits_2_with_the_reason(tmp_path, monkeypatch, capsys):
    error = _wrapped(ClientError(403, {"error": {"message": "API key not valid"}}))
    monkeypatch.setattr(extraction, "build_extraction_chain", lambda **kw: _ScriptChain(error))
    assert extraction.main(_cli_paths(tmp_path)) == 2
    out = capsys.readouterr().out
    assert "STOPPED (api_error)" in out and "stopped=api_error" in out


def test_cli_a_spent_budget_exits_2_without_building_a_chain(tmp_path, monkeypatch, capsys):
    args = _cli_paths(tmp_path)
    DailyUsage(450, tmp_path / "usage.json").mark_exhausted("earlier")

    def no_chain(**kw):
        raise AssertionError("a chain was built")

    monkeypatch.setattr(extraction, "build_extraction_chain", no_chain)
    assert extraction.main(args) == 2
    assert "NOT RUN" in capsys.readouterr().out


def test_cli_second_process_gets_exit_4(tmp_path, monkeypatch, capsys):
    args = _cli_paths(tmp_path)
    monkeypatch.setattr(extraction, "build_extraction_chain", lambda **kw: _ScriptChain())
    with RunLock(tmp_path / "llm.lock", what="tailoring"):
        assert extraction.main(args) == 4
    out = capsys.readouterr().out
    assert "NOT RUN: another flash-lite run is in progress" in out and "tailoring" in out


def test_cli_missing_key_is_a_config_error(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert extraction.main(_cli_paths(tmp_path)) == 1


def test_cli_stale_lists_without_any_call(tmp_path, monkeypatch, capsys):
    args = _cli_paths(tmp_path)
    monkeypatch.setattr(extraction, "build_extraction_chain", lambda **kw: _ScriptChain())
    monkeypatch.setattr(extraction._Pacer, "wait", lambda self: None)
    extraction.main(args)
    monkeypatch.setattr(extraction, "_PROMPT_PREFIX", "a new prompt\n")
    monkeypatch.setattr(extraction, "build_extraction_chain", lambda **kw: pytest.fail("no call expected"))
    assert extraction.main([*args, "--stale"]) == 0
    out = capsys.readouterr().out
    assert "stale: 1" in out and "j1  Acme -- Backend Engineer" in out
