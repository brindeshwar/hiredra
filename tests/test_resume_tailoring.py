"""Tests for piece 2: the skills section, the claim gate, tailoring with its repair path, the cover letter and the
packet. A fake caller stands in for the model: no network."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from hiredra.resume import build as build_mod
from hiredra.resume import cover, packet, tailor
from hiredra.resume.__main__ import main as resume_main
from hiredra.resume.claims import Gate, numbers, singular
from hiredra.resume.llm import Caller
from hiredra.resume.settings import ResumeConfig, SkillDisplay, load_resume_config, load_skill_display
from hiredra.resume.skills import build_skills
from hiredra.resume.source import load_candidate, read_profile
from hiredra.storage import add_alias, artifacts_for, connect, get_job, init_db, touch_skill_vocab

PROFILE = Path(__file__).parent / "fixtures" / "profile_resume.md"
CFG = ResumeConfig(max_projects=2)
DISPLAY = SkillDisplay(
    categories=("Languages", "Backend", "Infra & Tools"),
    category={"python": "Languages", "fastapi": "Backend", "sql": "Languages"},
    name={"python": "Python", "fastapi": "FastAPI", "sql": "SQL", "llm-integration": "LLM integration"},
    fallback=(("Infra & Tools", ()),),
    hide=frozenset({"docker"}),
)
ACME = "Acme Labs (Acme Pvt Ltd) — Backend Intern, May–Aug 2025"
PIPELINE = "Pipeline — job matching system"


def _profile():
    return read_profile(PROFILE)


def _gate(profile, extra=()):
    vocab = {"python", "fastapi", "docker", "sql", "rust", "kubernetes", "mcp", "mcp server", "llm integration", *extra}
    return Gate(profile, vocab, aliases={"mcp server": "mcp"}, banned=("full-stack",))


class FakeCaller:
    """Answers from a script: a list of answers per stage, used in order; records every call."""

    def __init__(self, **answers):
        self.answers = {k: list(v) for k, v in answers.items()}
        self.calls = 0
        self.prompts = []

    def call(self, prompt, schema, *, stage, job_id=None):
        self.calls += 1
        self.prompts.append((stage, prompt))
        queue = self.answers.get(stage) or [None]
        return queue.pop(0) if len(queue) > 1 else queue[0]


def _cv(summary="Backend intern who built **FastAPI** search services.", acme=None, pipeline=None):
    return tailor.TailoredCv.model_construct(
        summary=summary,
        experience=[tailor.ExperienceOut.model_construct(ref=ACME, bullets=acme or ["Built a **FastAPI** search service"])],
        projects=[tailor.ProjectOut.model_construct(ref="Pipeline", tagline="Job matcher", tech="Python",
                                                    bullets=pipeline or ["Sources jobs and scores them"])],
    )  # fmt: skip


JOB = {"job_id": "greenhouse:1", "title": "Backend Engineer", "company": "Initech.com", "description": "We use Python.",
       "skills": [{"name": "sql"}, {"name": "python"}]}  # fmt: skip


# ── the skills section ────────────────────────────────────────────────────────
def test_skills_are_claimable_only_grouped_job_first_and_hidden_ones_left_out():
    groups = build_skills(_profile().skills, ["sql", "python"], DISPLAY)
    got = {g.category: g.items for g in groups}
    assert got["Languages"] == ["Python", "SQL"]  # the job's skills first, both Strong/Working
    assert got["Backend"] == ["FastAPI"]
    assert "rust" not in str(got) and "docker" not in str(got)  # Familiar never; hidden never
    assert [g.category for g in groups] == ["Languages", "Backend", "Infra & Tools"]


def test_the_shipped_display_config_loads():
    display, cfg = load_skill_display(), load_resume_config()
    assert display.categories[0] == "Languages" and display.category_of("langchain") == "AI / LLM"
    assert display.category_of("some-new-agent-tool") == "AI / LLM" and "full-stack" in cfg.banned_phrases


# ── the claim gate ────────────────────────────────────────────────────────────
def test_numbers_and_plurals():
    assert numbers("cut cost by ~20% over 116,712 jobs, 50x faster, F1 0.795") == {
        "20%",
        "116712",
        "50x",
        "0.795",
    }  # "F1" is a name
    assert singular("mcp servers") == "mcp server" and singular("class") == "class"


def test_the_gate_passes_the_profiles_own_facts_and_broad_phrases():
    p = _profile()
    g = _gate(p, extra={"search service", "search"})
    assert g.check("b", "Built a **FastAPI** search service and cut cost by 20%", ACME) == []
    assert g.check("s", "Engineer who built search services in Python.") == []
    assert g.check("s", "Shipped MCP servers.", context="") == [
        {"rule": "unsupported_skill", "where": "s", "detail": "mcp"}
    ]  # MCP is not one of this profile's skills


def test_the_gate_fails_new_numbers_unsupported_familiar_banned_and_bad_refs():
    p = _profile()
    g = _gate(p)
    rules = lambda hits: [(h["rule"], h["detail"]) for h in hits]  # noqa: E731
    assert rules(g.check("b", "Cut cost by 35%", ACME)) == [("new_number", "35%")]
    assert rules(g.check("b", "Deployed with Kubernetes", ACME)) == [("unsupported_skill", "kubernetes")]
    assert rules(g.check("s", "Strong in Rust.")) == [("unsupported_skill", "rust")]  # Familiar: never in a summary
    assert rules(g.check("s", "A full-stack engineer.")) == [("banned_phrase", "full-stack")]
    assert rules(g.check("b", "x", "No such heading")) == [("bad_ref", "No such heading")]


def test_the_gate_lets_a_letter_name_the_job_and_company():
    g = _gate(_profile(), extra={"initech"})
    assert g.check("p", "I am applying to Initech.", context="Backend Engineer Initech.com") == []
    assert g.check("p", "I am applying to Initech.") != []


# ── tailoring ─────────────────────────────────────────────────────────────────
def test_tailoring_places_model_text_on_profile_facts():
    p = _profile()
    caller = FakeCaller(tailor=[_cv()])
    t = tailor.tailor(JOB, p, load_candidate(p, "none.yml"), _gate(p), caller, DISPLAY, CFG)
    assert t.claims["pass"] and t.flags == [] and caller.calls == 1
    acme = next(e for e in t.payload.experience if e.company.startswith("Acme"))
    assert (acme.role, acme.dates, acme.about) == (
        "Backend Intern",
        "May 2025 - Aug 2025",
        "Acme builds meeting tools.",
    )
    assert acme.bullets == ["Built a **FastAPI** search service"]
    beta = next(e for e in t.payload.experience if e.company == "Beta Corp")
    assert beta.bullets == ["Built the SDK core"]  # the model skipped it: the profile's own words
    assert [pr.name for pr in t.payload.projects] == ["Pipeline"] and t.payload.projects[0].url.endswith("pipeline")


def test_a_failing_answer_is_retried_once_then_repaired_from_the_profile():
    p = _profile()
    bad = _cv(summary="A full-stack Kubernetes expert with 99 years.", acme=["Ran Kubernetes for 35% savings"])
    caller = FakeCaller(tailor=[bad, bad])
    t = tailor.tailor(JOB, p, load_candidate(p, "none.yml"), _gate(p), caller, DISPLAY, CFG)
    assert caller.calls == 2 and "YOUR LAST ANSWER BROKE THESE RULES" in caller.prompts[1][1]
    assert t.flags == ["critic_unresolved"] and t.claims["pass"] and t.claims["fixed"]
    assert t.payload.summary == ""  # a failing summary is dropped, never printed
    acme = next(e for e in t.payload.experience if e.company.startswith("Acme"))
    assert acme.bullets[0] == "Built a FastAPI service for search"  # the profile's own bullet


def test_no_answer_twice_falls_back_to_the_profile():
    p = _profile()
    caller = FakeCaller(tailor=[None])
    t = tailor.tailor(JOB, p, load_candidate(p, "none.yml"), _gate(p), caller, DISPLAY, CFG)
    assert caller.calls == 2 and "model_gave_nothing" in t.flags
    assert t.payload.projects and t.payload.experience[0].bullets


def test_refs_are_matched_by_entry_name_and_the_schema_refuses_empty_answers():
    p = _profile()
    assert tailor.resolve_ref("Pipeline", p) == PIPELINE
    assert tailor.resolve_ref("acme labs (acme pvt ltd) - backend intern", p) == ACME
    assert tailor.resolve_ref("Nowhere Inc", p) is None
    try:
        tailor.TailoredCv.model_validate({"summary": "x" * 50, "experience": [], "projects": []})
        raise AssertionError("empty lists must be refused")
    except ValidationError:
        pass


# ── the cover letter ──────────────────────────────────────────────────────────
def test_the_cover_letter_is_checked_and_falls_back_to_plain_words():
    p = _profile()
    cv = tailor.tailor(JOB, p, load_candidate(p, "none.yml"), _gate(p), FakeCaller(tailor=[_cv()]), DISPLAY, CFG)
    good = cover.CoverOut.model_construct(paragraphs=["I am applying for the Backend Engineer role at Initech." * 2,
                                                      "At Acme I built a FastAPI service for search." * 2,
                                                      "Thank you for your time and consideration." * 2])  # fmt: skip
    letter, claims, flags = cover.write_letter(JOB, p, cv.payload, _gate(p), FakeCaller(cover=[good]), CFG)
    assert claims["pass"] and flags == [] and len(letter.paragraphs) == 3 and letter.company == "Initech.com"
    bad = cover.CoverOut.model_construct(paragraphs=["I led Kubernetes migrations at Initech." * 3] * 3)
    letter, claims, flags = cover.write_letter(JOB, p, cv.payload, _gate(p), FakeCaller(cover=[bad, bad]), CFG)
    assert flags == ["critic_unresolved"] and claims["pass"] and "Kubernetes" not in " ".join(letter.paragraphs)
    try:
        cover.CoverOut.model_validate({"paragraphs": [".", ".", "."]})
        raise AssertionError("one-character paragraphs must be refused")
    except ValidationError:
        pass


# ── the caller ────────────────────────────────────────────────────────────────
def test_the_caller_paces_retries_passing_failures_and_logs_every_call(pg_url):
    import httpx

    conn = connect(pg_url)
    init_db(conn)
    state = {"n": 0}

    class Chain:
        def invoke(self, prompt):
            state["n"] += 1
            if state["n"] == 1:
                raise httpx.ConnectError("reset")
            return cover.CoverOut.model_construct(paragraphs=["x"])

    waits = []
    c = Caller(conn, CFG, chain_factory=lambda schema: Chain(), sleep=waits.append, clock=lambda: 0.0)
    assert c.call("hi", cover.CoverOut, stage="cover").paragraphs == ["x"]
    rows = conn.execute("SELECT ok, error_class FROM llm_calls ORDER BY id").fetchall()
    assert [(r["ok"], r["error_class"]) for r in rows] == [(False, "transient"), (True, None)] and waits
    conn.close()


# ── the packet ────────────────────────────────────────────────────────────────
def _scored_job(conn, key="greenhouse:1"):
    from tests.test_graph_clean import _job

    job_id = _job(conn, int(key.split(":")[1]), [("python", "required"), ("sql", "good_to_have")])
    conn.execute("UPDATE jobs SET status = 'scored', score = 1.5 WHERE job_key = %s", (job_id,))
    return job_id


def test_a_packet_writes_its_files_records_artifacts_and_tailors_the_job(pg_url, tmp_path, monkeypatch):
    conn = connect(pg_url)
    init_db(conn)
    job_id = _scored_job(conn)
    for n in ("python", "fastapi"):
        touch_skill_vocab(conn, n)
    add_alias(conn, "py", "python")

    def no_converter(docx):
        raise build_mod.PdfUnavailable("no converter")

    monkeypatch.setattr(build_mod, "to_pdf", no_converter)
    monkeypatch.setattr(packet, "_REPO_ROOT", tmp_path)
    p = _profile()
    gate = packet.build_gate(conn, p)
    letter = cover.CoverOut.model_construct(paragraphs=["I am applying for this role and would like to talk." * 2] * 3)
    caller = FakeCaller(tailor=[_cv()], cover=[letter])
    assert packet.select_jobs(conn, 5) == [job_id]
    r = packet.build_packet(conn, job_id, profile=p, candidate=load_candidate(p, "none.yml"), gate=gate, caller=caller,
                            display=DISPLAY, cfg=CFG, out_root=tmp_path / "packets")  # fmt: skip
    files = sorted(x.name for x in r.folder.iterdir())
    assert files == ["claims.json", "cover.docx", "cv.docx", "payload.json"] and "no_pdf" in r.flags
    assert get_job(conn, job_id)["status"] == "tailored"
    assert {a["kind"] for a in artifacts_for(conn, job_id)} == {"cv_docx", "cover_docx", "payload", "claims"}
    assert packet.select_jobs(conn, 5) == []  # a job with a packet is not picked again
    conn.close()


def test_cli_packets_dry_run_makes_no_model_call(pg_url, capsys):
    conn = connect(pg_url)
    init_db(conn)
    job_id = _scored_job(conn)
    conn.close()
    assert resume_main(["packets", "--top", "3", "--dry-run", "--db", pg_url, "--profile", str(PROFILE)]) == 0
    assert f"would build: {job_id}" in capsys.readouterr().out
