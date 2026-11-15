"""Tests for piece 5, the application filler: facts, form reading, answers, batches and decisions, the API, and the
fill driver on local test forms (it must never submit)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hiredra.api.app import create_app
from hiredra.api.runs import Runs
from hiredra.autofill import batch
from hiredra.autofill.answers import DraftAnswer, DraftAnswers, build_answers
from hiredra.autofill.facts import FactsError, load_facts
from hiredra.autofill.scan import Field, classify, parse_greenhouse, parse_lever, scan_job
from hiredra.resume.claims import Gate
from hiredra.resume.source import read_profile
from hiredra.storage import add_artifact, connect, get_job, init_db

PROFILE = Path(__file__).parent / "fixtures" / "profile_resume.md"


# ── facts ─────────────────────────────────────────────────────────────────────
def _facts(tmp_path, text: str) -> Path:
    p = tmp_path / "facts.yml"
    p.write_text(text, encoding="utf-8")
    return p


def test_facts_load_and_derive_the_name(tmp_path):
    f = load_facts(
        _facts(tmp_path, "contact:\n  full_name: Ada Q Example\n  email: a@x.io\nfacts:\n  country: India\n")
    )
    assert f == {"full_name": "Ada Q Example", "email": "a@x.io", "country": "India", "first_name": "Ada",
                 "last_name": "Example"}  # fmt: skip
    assert load_facts(tmp_path / "missing.yml") == {}


@pytest.mark.parametrize(
    "text",
    [
        "facts:\n  visa_status: none needed\n",
        "facts:\n  current_status: needs H1B sponsorship\n",
        "facts:\n  expected_salary: 10\n",
        "contact:\n  gender: x\n",
        "facts:\n  date_of_birth: 2000-01-01\n",
        "facts:\n  favourite_colour: blue\n",  # unknown key
        "secrets:\n  a: b\n",  # unknown block
    ],
)
def test_facts_refuse_sensitive_topics_and_unknown_keys(tmp_path, text):
    with pytest.raises(FactsError):
        load_facts(_facts(tmp_path, text))


# ── classification ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("label", "kind", "cls", "detail"),
    [
        ("First Name", "text", "FACT", "first_name"),
        ("LinkedIn Profile", "text", "FACT", "linkedin"),
        ("Linkedln Profile", "text", "FACT", "linkedin"),  # a real typo on a real form
        ("Current company", "text", "FACT", "current_company"),
        ("Resume/CV", "file", "FILE", "cv.pdf"),
        ("Cover Letter", "file", "FILE", "cover.pdf"),
        ("Why do you want to join us?", "textarea", "DRAFT", None),
        ("Are you legally authorized to work in India?", "select", "PAUSE", "work_auth"),
        ("Expected CTC", "text", "PAUSE", "salary"),
        ("Notice Period", "select", "PAUSE", "notice"),
        ("Are you ready to relocate to Pune", "select", "PAUSE", "relocation"),
        ("Gender", "select", "PAUSE", "demographic"),
        ("I agree to the privacy notice", "checkbox", "PAUSE", "legal"),
        ("How did you hear about us?", "text", "PAUSE", "judgment"),
        ("Preferred location", "select", "PAUSE", "unknown"),  # not "referral"
        ("If you answered Yes above, when do you graduate?", "text", "PAUSE", "conditional"),
        ("JEE Advanced Rank", "text", "PAUSE", "unknown"),  # unknown is always PAUSE
    ],
)
def test_classification(label, kind, cls, detail):
    f = classify(Field(label=label, key="q", type=kind))
    assert f.cls == cls and (f.facts_key or f.file or f.category) == detail


GREENHOUSE = {
    "questions": [
        {"label": "First Name", "required": True, "fields": [{"name": "first_name", "type": "input_text"}]},
        {"label": "Resume/CV", "required": True, "fields": [{"name": "resume", "type": "input_file"},
                                                           {"name": "resume_text", "type": "textarea"}]},
        {"label": "Notice Period", "required": True, "fields": [
            {"name": "question_1", "type": "multi_value_single_select", "values": [{"label": "Immediate", "value": 1}]}]},
    ],
    "location_questions": [{"label": "Location", "required": False, "fields": [{"name": "location", "type": "input_text"}]}],
    "demographic_questions": {"questions": [{"label": "Gender"}]},
}  # fmt: skip

LEVER = """<form><div class="h-captcha-spacing"></div>
<input type="file" name="resume">
<ul><li class="application-question"><label><div class="application-label">Full name<span class="required">\u2731</span></div>
<div class="application-field"><input type="text" name="name" required></div></label></li>
<li class="application-question"><label><div class="application-label">Pronouns</div>
<div class="application-field"><input type="checkbox" name="pronouns" value="he/him"><input type="checkbox" name="pronouns" value="she/her"></div></label></li>
</ul></form>"""


def test_greenhouse_and_lever_forms_are_read():
    gh = [classify(f) for f in parse_greenhouse(GREENHOUSE)]
    assert [(f.label, f.cls) for f in gh] == [("First Name", "FACT"), ("Resume/CV", "FILE"), ("Notice Period", "PAUSE"),
                                              ("Location", "FACT"), ("demographic questions questions", "PAUSE")]  # fmt: skip
    assert gh[2].options == ["Immediate"]
    fields, captcha = parse_lever(LEVER)
    assert captcha and [(f.key, f.label, f.type) for f in fields] == [
        ("resume", "Resume/CV", "file"), ("name", "Full name", "text"), ("pronouns", "Pronouns", "checkbox")]  # fmt: skip
    assert fields[2].options == ["he/him", "she/her"] and fields[1].required


class FakeHttp:
    def __init__(self, data=None, fail=None):
        self.data, self.fail, self.urls = data, fail, []

    def get_json(self, url):
        self.urls.append(url)
        if self.fail:
            raise self.fail
        return self.data

    get_text = get_json


def test_scan_job_reads_greenhouse_marks_a_gone_posting_and_skips_others():
    job = {"job_id": "greenhouse:9", "source": "greenhouse", "company_slug": "acme", "external_id": "9",
           "url": "https://job-boards.greenhouse.io/acme/jobs/9"}  # fmt: skip
    http = FakeHttp(GREENHOUSE)
    scan = scan_job(job, http)
    assert http.urls == ["https://boards-api.greenhouse.io/v1/boards/acme/jobs/9?questions=true"]
    assert scan["via"] == "api" and len(scan["fields"]) == 5 and not scan["dead"]

    class Gone(Exception):
        status = 404

    assert scan_job(job, FakeHttp(fail=Gone("HTTP 404")))["dead"] is True
    lever = scan_job({"job_id": "lever:1", "source": "lever", "url": "https://jobs.lever.co/a/1"}, FakeHttp(LEVER))
    assert lever["via"] == "static" and lever["url"].endswith("/apply") and lever["fields"][-1]["category"] == "captcha"
    assert scan_job({"job_id": "ashby:1", "source": "ashby"}, FakeHttp())["via"] == "unsupported"


# ── answers ───────────────────────────────────────────────────────────────────
class FakeCaller:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def call(self, prompt, schema, *, stage, job_id=None):
        self.calls += 1
        return self.result


def _scan_with_draft():
    fields = [Field("First Name", "first_name", "text"), Field("Portfolio", "portfolio", "text"),
              Field("Resume", "resume", "file"), Field("Why do you want to join?", "why", "textarea"),
              Field("Expected CTC", "ctc", "text")]  # fmt: skip
    from dataclasses import asdict

    return {"job_id": "greenhouse:9", "via": "api", "fields": [asdict(classify(f)) for f in fields]}


def test_answers_fill_facts_files_and_checked_drafts(tmp_path):
    profile = read_profile(PROFILE)
    gate = Gate(profile, {"python", "fastapi", "kubernetes"})
    job = {"job_id": "greenhouse:9", "title": "Backend Engineer", "company": "Acme"}
    good = DraftAnswers.model_construct(answers=[DraftAnswer.model_construct(
        label="Why do you want to join?", answer="I built a FastAPI service for search and want to keep building backends.")])  # fmt: skip
    a = build_answers(_scan_with_draft(), {"first_name": "Ada"}, job=job, profile=profile, gate=gate,
                      caller=FakeCaller(good), packet_dir=tmp_path)  # fmt: skip
    by = {f["label"]: f for f in a["fields"]}
    assert by["First Name"]["value"] == "Ada"
    assert (by["Portfolio"]["class"], by["Portfolio"]["category"]) == ("PAUSE", "missing_fact")  # never a guess
    assert by["Resume"]["value"] == str(tmp_path / "cv.pdf")
    assert by["Why do you want to join?"]["class"] == "DRAFT" and "FastAPI" in by["Why do you want to join?"]["value"]
    assert by["Expected CTC"]["class"] == "PAUSE" and by["Expected CTC"]["value"] is None
    bad = DraftAnswers.model_construct(answers=[DraftAnswer.model_construct(
        label="Why do you want to join?", answer="I ran Kubernetes clusters for five years at scale here.")])  # fmt: skip
    a = build_answers(_scan_with_draft(), {}, job=job, profile=profile, gate=gate, caller=FakeCaller(bad))
    assert {f["label"]: f["category"] for f in a["fields"]}["Why do you want to join?"] == "draft_failed"
    a = build_answers(_scan_with_draft(), {}, job=job, profile=profile, gate=gate, caller=None)
    assert {f["label"]: f["category"] for f in a["fields"]}["Why do you want to join?"] == "not_drafted"


# ── batches and decisions ─────────────────────────────────────────────────────
def _tailored(conn, n: int, score: float, scanned: bool = True) -> str:
    from tests.test_graph_clean import _job

    key = _job(conn, n, [("python", "required")])
    conn.execute("UPDATE jobs SET status = 'tailored', score = %s WHERE job_key = %s", (score, key))
    if scanned:
        add_artifact(conn, key, "form_scan", f"data/packets/x{n}/form_scan.json", "0" * 64)
    return key


def test_batches_take_up_to_ten_scanned_jobs_best_first_never_twice(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    keys = [_tailored(conn, n, score=n / 10) for n in range(1, 13)]
    _tailored(conn, 99, score=5.0, scanned=False)  # no form read yet: not batched
    assert batch.make_batch(conn, min_size=20) is None
    b1 = batch.make_batch(conn)
    assert b1["batch_id"] == "B0001" and [i["job_id"] for i in b1["items"]] == keys[::-1][:10]
    assert all(get_job(conn, k)["status"] == "queued" for k in keys[2:])
    b2 = batch.make_batch(conn)
    assert b2["batch_id"] == "B0002" and [i["job_id"] for i in b2["items"]] == [keys[1], keys[0]]
    assert batch.make_batch(conn) is None
    conn.close()


def test_decisions_follow_the_status_machine_and_close_the_batch(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    a, b, c, d = (_tailored(conn, n, score=n) for n in range(1, 5))
    batch.make_batch(conn)
    with pytest.raises(ValueError, match="confirmation"):
        batch.decide(conn, a, "applied", "")
    assert batch.decide(conn, a, "applied", "Thanks! Your application was received.")["status"] == "applied"
    assert get_job(conn, a)["status_detail"].startswith("confirmed: Thanks!")
    assert batch.decide(conn, b, "not_yet")["status"] == "queued"
    assert batch.decide(conn, b, "deferred")["status"] == "tailored"
    assert batch.decide(conn, c, "skipped")["status"] == "dropped"
    assert batch.get_batch(conn, "B0001")["status"] == "in_review"
    assert batch.decide(conn, d, "closed")["status"] == "dropped"
    assert batch.get_batch(conn, "B0001")["status"] == "closed"
    with pytest.raises(ValueError, match="only a queued job"):
        batch.decide(conn, a, "applied", "again")
    with pytest.raises(ValueError):
        batch.decide(conn, d, "submitted")
    conn.close()


def test_attention_items_hold_labels_only(pg_url):
    conn = connect(pg_url)
    init_db(conn)
    key = _tailored(conn, 1, 1.0)
    n = batch.record_attention(conn, key, [{"label": "Expected CTC", "class": "PAUSE", "category": "salary", "value": None},
                                           {"label": "Email", "class": "FACT", "value": "a@x.io"}])  # fmt: skip
    rows = conn.execute("SELECT category, field_label FROM attention").fetchall()
    assert n == 1 and [(r["category"], r["field_label"]) for r in rows] == [("salary", "Expected CTC")]
    conn.close()


# ── the API ───────────────────────────────────────────────────────────────────
def test_api_batches_apply_view_and_decisions(pg_url, tmp_path):
    conn = connect(pg_url)
    init_db(conn)
    key = _tailored(conn, 1, 1.0)
    auth = {"Authorization": "Bearer " + "t" * 32}
    client = TestClient(create_app(token="t" * 32, db_url=pg_url, runs=Runs({}), packets_dir=tmp_path))
    assert client.get("/batches", headers=auth).json() == []
    assert client.get(f"/apply/{key}", headers=auth).status_code == 404
    folder = tmp_path / key.replace(":", "_")
    folder.mkdir()
    (folder / "answers.json").write_text(json.dumps({"job_id": key, "fields": [
        {"label": "Resume", "class": "FILE", "file": "cv.pdf", "value": "C:/private/path/cv.pdf"}]}), encoding="utf-8")  # fmt: skip
    view = client.get(f"/apply/{key}", headers=auth).json()
    assert view["fields"][0]["value"] == "cv.pdf"  # a file name, never a path on this machine
    assert (
        client.post(
            f"/apply/{key}/decision", json={"decision": "applied", "confirmation": "ok ok"}, headers=auth
        ).status_code
        == 409
    )
    made = client.post("/batches", json={"min": 1}, headers=auth).json()
    assert (
        made["batch_id"] == "B0001" and client.get("/batches/B0001", headers=auth).json()["items"][0]["job_id"] == key
    )
    assert client.post("/batches", json={"min": 1}, headers=auth).status_code == 409  # nothing left to batch
    r = client.post(f"/apply/{key}/decision", json={"decision": "applied", "confirmation": "Application received"},
                    headers=auth)  # fmt: skip
    assert r.json()["status"] == "applied"
    conn.close()


# ── the fill driver (local test forms only) ───────────────────────────────────
def _chromium_ready() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            pw.chromium.launch(headless=True).close()
        return True
    except Exception:
        return False


needs_browser = pytest.mark.skipif(not _chromium_ready(), reason="Playwright's Chromium is not installed")


@needs_browser
def test_the_driver_fills_three_pages_and_never_submits():
    from hiredra.autofill.driver import run_fixture

    report, counters = run_fixture("three_page", headless=True)
    assert counters == {"submits": 0, "finals": 0}
    assert report.pages == 3 and report.last_page_reached
    assert {
        "First name",
        "Resume/CV",
        "Country",
        "Do you have a GitHub profile?",
        "Why do you want to work here?",
    } <= set(report.filled)
    assert report.paused == ["Expected salary", "I agree to the privacy policy"]
    refused = " ".join(report.refused)
    assert "Submit application" in refused and "Apply now" in refused and "Finish" in refused


@needs_browser
def test_refusal_rules_on_real_controls():
    from playwright.sync_api import sync_playwright

    from hiredra.autofill.driver import refusal

    html = """<button type="button" id="a">Next</button><button type="submit" id="b">Continue</button>
    <button type="submit" id="c">Save</button><button id="d">Submit</button><input type="submit" id="e" value="Apply">
    <button id="f" aria-label="Review and submit">Go</button><button id="g">Finish</button>"""  # fmt: skip
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(html)
        got = {i: refusal(page.locator(f"#{i}")) for i in "abcdefg"}
        browser.close()
    assert got["a"] is None and got["b"] is None  # Next / Continue, whatever their type
    assert all(got[i] for i in "cdefg")  # everything else is refused
