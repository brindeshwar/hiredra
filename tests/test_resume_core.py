"""Tests for the résumé core: the payload contract, the profile reader, the Word renderer and the page fit."""

from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document
from pydantic import ValidationError

from hiredra.resume import build
from hiredra.resume.__main__ import main as resume_main
from hiredra.resume.payload import Candidate, CoverLetter, CvPayload, Experience, Project
from hiredra.resume.source import baseline_payload, education_items, load_candidate, read_profile
from hiredra.resume.word import Layout, clean, render_cover, render_cv

PROFILE = Path(__file__).parent / "fixtures" / "profile_resume.md"


def _payload() -> CvPayload:
    profile = read_profile(PROFILE)
    return baseline_payload(profile, load_candidate(profile, "no-such-file.yml"))


def _bullets(payload: CvPayload) -> int:
    return sum(len(e.bullets) for e in [*payload.experience, *payload.projects])


def test_the_payload_refuses_unknown_keys():
    with pytest.raises(ValidationError):
        CvPayload.model_validate({"candidate": {"name": "A"}, "hobbies": ["x"]})
    with pytest.raises(ValidationError):
        Experience.model_validate({"company": "A", "role": "B", "salary": "1"})
    assert Project(name="x").bullets == []


def test_the_profile_reader_finds_entries_bullets_links_and_keeps_guidance_apart():
    p = read_profile(PROFILE)
    assert p.name == "Ada Example"
    assert p.links == {"github": "github.com/ada-example", "linkedin": "linkedin.com/in/ada-example"}
    acme, beta = p.section("experience")
    assert (acme.name, acme.role, acme.dates) == ("Acme Labs (Acme Pvt Ltd)", "Backend Intern", "May–Aug 2025")
    assert acme.bullets == ["Built a FastAPI service for search", "Cut cost by 20%", "Deployed with Docker"]
    assert acme.stack == "Python, FastAPI, Docker" and acme.intro == "Acme builds meeting tools."
    assert beta.role == "AI Engineer / Software Engineer"
    (overlay,) = p.section("projects")  # "Other work" is not an entry
    assert (overlay.name, overlay.role, overlay.url) == (
        "Overlay",
        "desktop assistant",
        "github.com/ada-example/overlay",
    )
    assert p.section("flagship")[0].url == "github.com/ada-example/pipeline"
    assert p.guidance == ["Experience level: fresher."] and p.rules == ["Resume: strictly 1 page"]
    assert education_items(p)[0].model_dump() == {
        "title": "B.Tech Computer Science", "org": "Example University", "location": "", "year": "2026",
        "description": "CGPA 8.0.",
    }  # fmt: skip


def test_the_candidate_comes_from_the_facts_file_with_the_profile_as_fallback(tmp_path):
    p = read_profile(PROFILE)
    assert load_candidate(p, tmp_path / "missing.yml").github == "github.com/ada-example"
    facts = tmp_path / "facts.yml"
    facts.write_text("contact:\n  full_name: Ada E.\n  email: ada@example.com\n", encoding="utf-8")
    c = load_candidate(p, facts)
    assert (c.name, c.email, c.linkedin) == ("Ada E.", "ada@example.com", "linkedin.com/in/ada-example")


def test_the_baseline_uses_only_the_owners_words_and_claimable_skills():
    payload = _payload()
    assert [e.company for e in payload.experience] == ["Beta Corp", "Acme Labs (Acme Pvt Ltd)"]  # newest first
    assert payload.experience[0].role == "AI Engineer"
    assert payload.projects[0].name == "Pipeline"  # the flagship leads
    listed = {i for g in payload.skills for i in g.items}
    assert listed == {"python", "fastapi", "llm integration", "docker", "sql"}  # Familiar ("rust") is never listed


def test_the_word_file_is_single_column_with_bold_terms_and_no_dashes(tmp_path):
    payload = _payload()
    payload.summary = "Backend engineer — built **FastAPI** services."
    doc = Document(str(render_cv(payload, tmp_path / "cv.docx")))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert doc.tables == [] and "—" not in text and "**" not in text
    assert "EXPERIENCE" in text and "Ada Example" in text and "Aug 2026-present" in text
    summary = next(p for p in doc.paragraphs if p.text.startswith("Backend engineer"))
    assert [r.text for r in summary.runs if r.bold] == ["FastAPI"]
    assert {r.font.name for p in doc.paragraphs for r in p.runs if r.text.strip()} == {"Arial"}
    assert "SUMMARY" in text and "PROJECTS" in text
    payload.summary, payload.projects = "", []
    again = "\n".join(p.text for p in Document(str(render_cv(payload, tmp_path / "cv2.docx"))).paragraphs)
    assert "SUMMARY" not in again and "PROJECTS" not in again  # empty sections are left out


def test_clean_and_the_layout_steps():
    assert clean("a — b  – c") == "a, b - c"
    steps = [Layout()]
    while (nxt := steps[-1].tighter()) is not None:
        steps.append(nxt)
    assert steps[-1].body_pt == 8.5 and len(steps) == 4


def test_the_cover_letter_renders(tmp_path):
    letter = CoverLetter(candidate=Candidate(name="Ada"), company="Acme", role="Engineer", paragraphs=["One.", "Two."])
    text = "\n".join(p.text for p in Document(str(render_cover(letter, tmp_path / "c.docx"))).paragraphs)
    assert "Re: Engineer" in text and "Dear Hiring Team," in text and text.strip().endswith("Ada")


def _fake_pdf(monkeypatch, pages_for):
    """Stand in for the converter: the page count is a function of the layout and the number of bullets."""
    seen = {}

    def render(payload, path, layout=None):
        seen["layout"], seen["bullets"] = layout, _bullets(payload)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"docx")
        return Path(path)

    monkeypatch.setattr(build, "render_cv", render)
    monkeypatch.setattr(build, "to_pdf", lambda docx: Path(docx).with_suffix(".pdf"))
    monkeypatch.setattr(build, "page_count", lambda pdf: pages_for(seen["layout"], seen["bullets"]))


def test_page_fit_tightens_the_layout_before_it_drops_a_bullet(tmp_path, monkeypatch):
    _fake_pdf(monkeypatch, lambda layout, bullets: 1 if layout.body_pt <= 9.0 else 2)
    result = build.write_cv(_payload(), tmp_path)
    assert (result.pages, result.layout.body_pt, result.dropped) == (1, 9.0, 0)


def test_page_fit_drops_whole_bullets_only_when_the_layout_is_at_its_smallest(tmp_path, monkeypatch):
    payload = _payload()
    total = _bullets(payload)
    _fake_pdf(monkeypatch, lambda layout, bullets: 1 if layout.body_pt <= 8.5 and bullets <= total - 2 else 2)
    result = build.write_cv(payload, tmp_path)
    assert (result.pages, result.dropped, result.note) == (1, 2, "")
    assert _bullets(payload) == total  # the caller's copy is intact


def test_page_fit_reports_when_it_cannot_fit_and_when_there_is_no_converter(tmp_path, monkeypatch):
    _fake_pdf(monkeypatch, lambda layout, bullets: 2)
    result = build.write_cv(_payload(), tmp_path)
    assert result.pages == 2 and "still 2 pages" in result.note

    def no_converter(docx):
        raise build.PdfUnavailable("no converter")

    monkeypatch.setattr(build, "to_pdf", no_converter)
    result = build.write_cv(_payload(), tmp_path)
    assert result.pdf is None and result.pages is None and "no converter" in result.note


def test_cli_sample_and_render_write_the_files(tmp_path, monkeypatch, capsys):
    def no_converter(docx):
        raise build.PdfUnavailable("no converter")

    monkeypatch.setattr(build, "to_pdf", no_converter)
    assert resume_main(["sample", "--profile", str(PROFILE), "--facts", "none.yml", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "cv.docx").exists() and "pages: unknown" in capsys.readouterr().out
    payload = tmp_path / "p.json"
    payload.write_text(_payload().model_dump_json(), encoding="utf-8")
    assert resume_main(["render", str(payload), "--out", str(tmp_path / "r")]) == 0
    assert (tmp_path / "r" / "cv.docx").exists()
