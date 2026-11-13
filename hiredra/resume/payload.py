"""The résumé and cover-letter payloads (plan contract 3.5): the one shape every renderer reads.

Who writes what (locked, F1): ``candidate`` comes from the owner's facts file; ``skills`` is built by CODE from
the master profile (Strong and Working only); everything else is written by the tailoring agent and checked by
the claim gate before it is rendered. Unknown keys are refused, so a model can never smuggle a field in.

Text may mark an important term as ``**bold**``; renderers turn that into bold type and print nothing else of it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Candidate(_Strict):
    name: str
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    portfolio: str = ""


class Experience(_Strict):
    company: str
    role: str
    location: str = ""
    dates: str = ""
    bullets: list[str] = Field(default_factory=list)


class Project(_Strict):
    name: str
    description: str = ""
    bullets: list[str] = Field(default_factory=list)
    tech: str = ""
    url: str = ""


class Education(_Strict):
    title: str
    org: str = ""
    location: str = ""
    year: str = ""
    description: str = ""


class SkillGroup(_Strict):
    category: str
    items: list[str]


class CvPayload(_Strict):
    lang: str = "en"
    page_format: str = "a4"
    candidate: Candidate
    summary: str = ""
    experience: list[Experience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    skills: list[SkillGroup] = Field(default_factory=list)


class CoverLetter(_Strict):
    candidate: Candidate
    date: str = ""
    company: str
    role: str
    greeting: str = "Dear Hiring Team,"
    paragraphs: list[str]
    closing: str = "Sincerely,"
