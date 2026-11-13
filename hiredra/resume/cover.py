"""The cover letter: three short paragraphs written by a model, checked by the same claim gate as the résumé.

Paragraph one says why this role; two gives two pieces of evidence from the profile; three closes. A paragraph that
fails the gate twice is replaced by a plain one built by code from the profile's own words, and the letter is
flagged ``critic_unresolved``.
"""

from __future__ import annotations

from datetime import date

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from hiredra.resume.claims import Gate, report
from hiredra.resume.payload import CoverLetter, CvPayload
from hiredra.resume.settings import ResumeConfig
from hiredra.resume.source import ProfileText

PROMPT_VERSION = "cover-1"


class CoverOut(BaseModel):
    """Exactly three real paragraphs: the schema the model is held to refuses empty or one-word ones."""

    paragraphs: list[Annotated[str, StringConstraints(min_length=60)]] = Field(min_length=3, max_length=3)


def build_prompt(job: dict, profile: ProfileText, cv: CvPayload, cfg: ResumeConfig, feedback=None) -> str:
    evidence = "\n".join(f"- {b}" for e in cv.experience for b in e.bullets)
    evidence += "\n" + "\n".join(f"- {p.name}: {b}" for p in cv.projects for b in p.bullets)
    rules = "\n".join(f"- {r}" for r in profile.rules)
    text = (
        "Write a short cover letter body for this job: exactly 3 paragraphs, 2 to 4 sentences each. Paragraph 1: "
        "the role and why the candidate fits it. Paragraph 2: two concrete pieces of evidence taken from the "
        "facts below. Paragraph 3: a brief, plain close. Use only the facts below; never add a skill, number, "
        "employer or result. No greeting and no sign-off (code adds them). No em dashes, no Oxford commas.\n\n"
        f"JOB: {job.get('title') or ''} at {job.get('company') or ''}\n{(job.get('description') or '')[: cfg.max_jd_chars]}\n\n"
        f"CANDIDATE SUMMARY: {cv.summary}\n\nFACTS:\n{evidence}\n\nSTANDING RULES:\n{rules}\n"
    )
    if feedback:
        text += "\nYOUR LAST ANSWER BROKE THESE RULES; fix each one:\n" + "\n".join(
            f"- {h['where']}: {h['rule']} ({h['detail']})" for h in feedback
        )
    return text


def _check(gate: Gate, paragraphs: list[str], context: str = "") -> list[dict]:
    return [h for i, p in enumerate(paragraphs) for h in gate.check(f"paragraphs[{i}]", p, context=context)]


def plain_letter(job: dict, cv: CvPayload) -> list[str]:
    """A letter from the résumé's own (already checked) words, for when the model's text cannot be used."""
    role, company = job.get("title") or "this role", job.get("company") or "your team"
    facts = [b.replace("**", "") for e in cv.experience for b in e.bullets][:2]
    first = f"I am applying for the {role} position at {company}."
    if cv.summary:
        first += " " + cv.summary.replace("**", "")
    second = " ".join(f.rstrip(".") + "." for f in facts) if facts else ""
    third = "I would welcome the chance to discuss how this experience fits your team. Thank you for your time."
    return [p for p in (first, second, third) if p]


def write_letter(job: dict, profile: ProfileText, cv: CvPayload, gate: Gate, caller,
                 cfg: ResumeConfig) -> tuple[CoverLetter, dict, list[str]]:  # fmt: skip
    flags: list[str] = []
    context = f"{job.get('title') or ''} {job.get('company') or ''}"
    out = caller.call(build_prompt(job, profile, cv, cfg), CoverOut, stage="cover", job_id=job["job_id"])
    paragraphs = out.paragraphs[:3] if out is not None else []
    hits = _check(gate, paragraphs, context) if paragraphs else [{"rule": "empty", "where": "paragraphs", "detail": ""}]
    if hits:
        retry = caller.call(build_prompt(job, profile, cv, cfg, hits), CoverOut, stage="cover", job_id=job["job_id"])
        if retry is not None and retry.paragraphs:
            paragraphs, hits = retry.paragraphs[:3], _check(gate, retry.paragraphs[:3], context)
    if hits:
        flags.append("critic_unresolved")
        paragraphs = plain_letter(job, cv)
    letter = CoverLetter(
        candidate=cv.candidate,
        date=date.today().strftime("%d %B %Y"),
        company=job.get("company") or "",
        role=job.get("title") or "",
        paragraphs=paragraphs,
    )
    return letter, {**report(_check(gate, paragraphs, context)), "fixed": hits}, flags
