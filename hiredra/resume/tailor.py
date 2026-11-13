"""The tailoring step: a résumé for one job, written by a model from the master profile and checked by code.

The model writes only text: a summary, and for each profile entry it uses, its bullets (rephrased and reordered
from what the entry says) and, for a project, a tagline and a tech line. It names each entry by its ``###`` heading
(``ref``). Code does the rest (plan decision F1):

- company, role, dates and the one line about the company come from the profile entry, never from the model;
- the skills section is built by ``hiredra.resume.skills``;
- the claim gate (``hiredra.resume.claims``) checks every piece. A failing round is sent back once with the findings
  listed; whatever still fails is replaced by the profile's own words (the entry's first bullets; no summary), and
  the result is flagged ``critic_unresolved``. A failing piece is never printed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from hiredra.resume.claims import Gate, report
from hiredra.resume.payload import CvPayload, Experience, Project
from hiredra.resume.settings import ResumeConfig, SkillDisplay
from hiredra.resume.skills import build_skills
from hiredra.resume.source import Entry, ProfileText, education_items, first_sentence, format_dates

PROMPT_VERSION = "tailor-1"


class ExperienceOut(BaseModel):
    ref: str
    bullets: list[str] = Field(min_length=1)


class ProjectOut(BaseModel):
    ref: str
    tagline: str = ""
    tech: str = ""
    bullets: list[str] = Field(min_length=1)


class TailoredCv(BaseModel):
    """What the model must return. Required parts and minimum lengths are in the JSON schema the model is held to,
    so it cannot answer with empty lists."""

    summary: str = Field(min_length=40)
    experience: list[ExperienceOut] = Field(min_length=1)
    projects: list[ProjectOut] = Field(min_length=1)


@dataclass
class Tailored:
    payload: CvPayload
    claims: dict
    flags: list[str] = field(default_factory=list)
    rounds: int = 0


def _entry_block(e: Entry) -> str:
    lines = [f"### {e.heading}"]
    if e.intro:
        lines.append(e.intro)
    lines += [f"- {b}" for b in e.bullets]
    if e.stack:
        lines.append(f"Stack: {e.stack}")
    return "\n".join(lines)


def build_prompt(job: dict, profile: ProfileText, cfg: ResumeConfig, feedback: list[dict] | None = None) -> str:
    experience = "\n\n".join(_entry_block(e) for e in profile.section("experience"))
    projects = "\n\n".join(_entry_block(e) for e in profile.section("flagship") + profile.section("projects"))
    rules = "\n".join(f"- {r}" for r in profile.rules)
    guidance = "\n".join(f"- {g}" for g in profile.guidance)
    jd = (job.get("description") or "")[: cfg.max_jd_chars]
    text = (
        "You tailor a one-page resume to a job. You may only rephrase, shorten, reorder and select what the "
        "candidate's profile below says. Never add a skill, tool, number, employer, title, result or claim that the "
        "profile entry you are rewriting does not state.\n\n"
        f"JOB: {job.get('title') or ''} at {job.get('company') or ''}\n{jd}\n\n"
        f"CANDIDATE GUIDANCE (never printed):\n{guidance}\n\nSTANDING RULES:\n{rules}\n\n"
        f"EXPERIENCE ENTRIES:\n{experience}\n\nPROJECT ENTRIES:\n{projects}\n\n"
        "WRITE:\n"
        "- summary: 2 or 3 sentences for this job, only from facts in the entries above.\n"
        f"- experience: every experience entry, newest first, each with at most {cfg.max_bullets_per_experience} "
        "bullets rewritten from that entry only, strongest for this job first.\n"
        f"- projects: the {cfg.max_projects} projects that fit this job best, each with a tagline (a few words on "
        f"what it is), a tech line (tools the entry names, comma separated) and at most {cfg.max_bullets_per_project} "
        "bullets from that entry only.\n"
        "- ref: copy the entry's heading exactly as written after '### '.\n"
        "- Start bullets with a past-tense verb for finished work. Put the 2-4 terms that matter most for this job in "
        "each bullet inside **double asterisks**. No em dashes, no Oxford commas, no first person.\n"
        "- Keep every number exactly as the entry writes it; never round or invent one.\n"
    )
    if feedback:
        found = "\n".join(f"- {h['where']}: {h['rule']} ({h['detail']})" for h in feedback)
        text += f"\nYOUR LAST ANSWER BROKE THESE RULES. Fix each one (remove the claim if the entry does not support it):\n{found}\n"
    return text


def resolve_ref(ref: str, profile: ProfileText) -> str | None:
    """The profile heading a model's ``ref`` means: exact, then case-insensitive, then by the entry's name (the part
    before " — "), so "Anchor8" or "anchor8 - AI Engineer" still finds its entry. None when nothing matches."""
    want = ref.strip().removeprefix("### ").strip()
    low = want.casefold()
    for e in profile.entries:
        if e.heading == want or e.heading.casefold() == low:
            return e.heading
    name = re.split(r"\s+[\u2014\u2013-]\s+|:\s|\s\(", want, maxsplit=1)[0].strip().casefold()
    for e in profile.entries:
        if e.name.casefold() == name or e.heading.casefold().startswith(low) or low.startswith(e.name.casefold()):
            return e.heading
    return None


def _resolve(out: TailoredCv, profile: ProfileText) -> TailoredCv:
    experience = [e.model_copy(update={"ref": r}) for e in out.experience if (r := resolve_ref(e.ref, profile))]
    projects = [p.model_copy(update={"ref": r}) for p in out.projects if (r := resolve_ref(p.ref, profile))]
    return TailoredCv.model_construct(summary=out.summary, experience=experience, projects=projects)


def _context(job: dict) -> str:
    """What a summary may echo from the job without it being a claim: its title and company."""
    return f"{job.get('title') or ''} {job.get('company') or ''}"


def _check(gate: Gate, out: TailoredCv, context: str = "") -> list[dict]:
    hits = gate.check("summary", out.summary, context=context) if out.summary else []
    for i, e in enumerate(out.experience):
        for j, b in enumerate(e.bullets):
            hits += gate.check(f"experience[{i}].bullets[{j}]", b, e.ref)
    for i, p in enumerate(out.projects):
        for part, text in (("tagline", p.tagline), ("tech", p.tech)):
            if text:
                hits += gate.check(f"projects[{i}].{part}", text, p.ref)
        for j, b in enumerate(p.bullets):
            hits += gate.check(f"projects[{i}].bullets[{j}]", b, p.ref)
    return hits


def _repair(gate: Gate, out: TailoredCv, profile: ProfileText, cfg: ResumeConfig, context: str = "") -> TailoredCv:
    """Replace every piece that still fails with the profile's own words."""
    headings = {e.heading: e for e in profile.entries}
    summary = "" if (out.summary and gate.check("summary", out.summary, context=context)) else out.summary
    experience, projects = [], []
    for e in out.experience:
        if e.ref not in headings:
            continue
        bullets = [b for b in e.bullets if not gate.check("", b, e.ref)]
        if len(bullets) < len(e.bullets):
            bullets = headings[e.ref].bullets[: cfg.max_bullets_per_experience]
        experience.append(ExperienceOut.model_construct(ref=e.ref, bullets=bullets))
    for p in out.projects:
        if p.ref not in headings:
            continue
        clean = [b for b in p.bullets if not gate.check("", b, p.ref)]
        bullets = clean if len(clean) == len(p.bullets) else headings[p.ref].bullets[: cfg.max_bullets_per_project]
        projects.append(ProjectOut.model_construct(
            ref=p.ref,
            tagline=p.tagline if not gate.check("", p.tagline, p.ref) else headings[p.ref].role,
            tech=p.tech if not gate.check("", p.tech, p.ref) else headings[p.ref].stack,
            bullets=bullets,
        ))  # fmt: skip
    return TailoredCv.model_construct(summary=summary, experience=experience, projects=projects)


def assemble(
    out: TailoredCv, job: dict, profile: ProfileText, candidate, display: SkillDisplay, cfg: ResumeConfig
) -> CvPayload:
    """The payload: the model's text placed on the profile's facts."""
    headings = {e.heading: e for e in profile.entries}
    wrote = {e.ref: e for e in out.experience if e.ref in headings}
    experience = []
    for entry in reversed(profile.section("experience")):  # every job, newest first, even if the model skipped one
        bullets = wrote[entry.heading].bullets if entry.heading in wrote else entry.bullets
        experience.append(Experience(
            company=entry.name, role=entry.role.split(" / ")[0], dates=format_dates(entry.dates),
            about=first_sentence(entry.intro), bullets=bullets[: cfg.max_bullets_per_experience],
        ))  # fmt: skip
    projects = []
    if not out.projects:  # the model gave none we could use: the profile's own first projects
        out = out.model_copy(update={"projects": [
            ProjectOut.model_construct(ref=e.heading, tagline=e.role, tech=e.stack,
                                       bullets=e.bullets[: cfg.max_bullets_per_project])
            for e in (profile.section("flagship") + profile.section("projects"))[: cfg.max_projects]
        ]})  # fmt: skip
    for p in out.projects[: cfg.max_projects]:
        entry = headings.get(p.ref)
        if entry is None or entry.section == "experience":
            continue
        projects.append(Project(
            name=entry.name, tagline=p.tagline or entry.role, tech=p.tech or entry.stack, url=entry.url,
            bullets=p.bullets[: cfg.max_bullets_per_project] or entry.bullets[:1],
        ))  # fmt: skip
    job_skills = [s["name"] for s in job.get("skills") or []]
    return CvPayload(
        candidate=candidate,
        summary=out.summary,
        experience=experience,
        projects=projects,
        education=education_items(profile),
        skills=build_skills(profile.skills, job_skills, display, per_category=cfg.max_skills_per_category),
    )


def tailor(job: dict, profile: ProfileText, candidate, gate: Gate, caller, display: SkillDisplay,
           cfg: ResumeConfig) -> Tailored:  # fmt: skip
    """Ask, check, ask once more with the findings, then repair whatever still fails."""
    flags: list[str] = []
    out = caller.call(build_prompt(job, profile, cfg), TailoredCv, stage="tailor", job_id=job["job_id"])
    rounds = 1
    if out is None:  # no usable answer: ask once more before giving up on the model
        out = caller.call(build_prompt(job, profile, cfg), TailoredCv, stage="tailor", job_id=job["job_id"])
        rounds = 2
    if out is None:
        out, flags = TailoredCv.model_construct(summary="", experience=[], projects=[]), ["model_gave_nothing"]
    out = _resolve(out, profile)
    context = _context(job)
    hits = _check(gate, out, context)
    if hits:
        retry = caller.call(build_prompt(job, profile, cfg, hits), TailoredCv, stage="tailor", job_id=job["job_id"])
        rounds = 2
        if retry is not None:
            out = _resolve(retry, profile)
            hits = _check(gate, out, context)
    if hits:
        flags.append("critic_unresolved")
        out = _repair(gate, out, profile, cfg, context)
    payload = assemble(out, job, profile, candidate, display, cfg)
    final = _check(gate, out, context)
    return Tailored(payload=payload, claims={**report(final), "fixed": hits}, flags=flags, rounds=rounds)


__all__ = ["PROMPT_VERSION", "Tailored", "TailoredCv", "assemble", "build_prompt", "tailor"]
