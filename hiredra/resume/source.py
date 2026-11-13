"""What a résumé is written FROM: the master profile's prose sections and the owner's contact facts.

``hiredra.profile`` reads the Skills section under a strict grammar (the graph depends on it). This reads the rest
tolerantly, for the tailoring step: the header links, Education, and every ``###`` entry under Experience,
Flagship project and Projects, with its bullets, its ``Stack:`` line and its first link. Nothing under
"For tailoring only" is returned as printable text (the profile's own standing rule); it is kept apart as
``guidance`` for the tailoring prompt.

Contact details come from ``data/autofill_facts.yml`` (private, git-ignored; shape in
``config/autofill_facts.example.yml``). The profile supplies the name and links when that file is missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from hiredra.profile import DEFAULT_PROFILE_PATH, ProfileSkills, parse_skills
from hiredra.resume.payload import Candidate, CvPayload, Education, Experience, Project, SkillGroup

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACTS_PATH = _REPO_ROOT / "data" / "autofill_facts.yml"
_URL = re.compile(r"(?:https?://)?((?:github\.com|linkedin\.com|zenodo\.org)/[^\s·)]+)")
_META = re.compile(r"^(Dossier|Repo|Repos|GitHub|Status|One line|Paid|Full-time)\b", re.IGNORECASE)
_STATUS_TAIL = re.compile(r"\s*\((?:ongoing|complete|planned)[^)]*\)\s*$", re.IGNORECASE)
_SPLIT = " — "  # the profile writes "Name — rest"


@dataclass
class Entry:
    section: str  # experience | flagship | projects
    heading: str  # the full "### ..." text, the reference a claim check points at
    name: str
    role: str = ""  # experience: the role; projects: the tagline
    dates: str = ""
    intro: str = ""
    bullets: list[str] = field(default_factory=list)
    stack: str = ""
    url: str = ""


@dataclass
class ProfileText:
    name: str
    links: dict[str, str]
    education: list[str]
    entries: list[Entry]
    guidance: list[str]  # "For tailoring only": steers the writer, never printed
    rules: list[str]  # the standing rules
    skills: ProfileSkills

    def section(self, name: str) -> list[Entry]:
        return [e for e in self.entries if e.section == name]


def _entry(section: str, heading: str) -> Entry:
    name, _, rest = heading.partition(_SPLIT)
    rest = _STATUS_TAIL.sub("", rest).strip()
    if section == "experience" and ", " in rest:
        role, dates = rest.rsplit(", ", 1)
        return Entry(section, heading, name.strip(), role.strip(), dates.strip())
    return Entry(section, heading, name.strip(), rest)


def read_profile(path: Path | str = DEFAULT_PROFILE_PATH) -> ProfileText:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    name = lines[0].lstrip("# ").partition(_SPLIT)[0].strip() if lines else ""
    links: dict[str, str] = {}
    education: list[str] = []
    guidance: list[str] = []
    rules: list[str] = []
    entries: list[Entry] = []
    sections = {"## Experience": "experience", "## Flagship project": "flagship", "## Projects": "projects"}
    section, current = "", None
    for line in lines[1:]:
        text = line.strip()
        if text.startswith("Links:"):
            for url in _URL.findall(text):
                links[url.split(".")[0]] = url
        if text.startswith("## "):
            section = sections.get(text, text[3:].lower())
            current = None
            continue
        if not text:
            continue
        if section.startswith("for tailoring only"):
            guidance.append(text.lstrip("- "))
        elif section.startswith("standing rules"):
            rules.append(text.lstrip("- "))
        elif section == "education":
            education.append(text.lstrip("- "))
        elif section in sections.values():
            if text.startswith("### "):
                current = _entry(section, text[4:])
                entries.append(current)
            elif current is None or current.name == "Other work":
                continue
            elif text.startswith("- Stack:"):
                current.stack = text[len("- Stack:") :].strip().rstrip(".")
            elif text.startswith("- "):
                current.bullets.append(text[2:])
            else:
                if not current.url and (found := _URL.search(text)) and "github.com" in found.group(1):
                    current.url = found.group(1).rstrip(".,")
                if not _META.match(text):
                    current.intro = f"{current.intro} {text}".strip()
    entries = [e for e in entries if e.name != "Other work"]
    return ProfileText(name, links, education, entries, guidance, rules, parse_skills(path))


def load_candidate(profile: ProfileText, facts_path: Path | str = DEFAULT_FACTS_PATH) -> Candidate:
    """The résumé header: the facts file's ``contact`` block, with the profile's name and links as the fallback."""
    facts = {}
    p = Path(facts_path)
    if p.exists():
        facts = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("contact") or {}
    get = lambda key, default="": str(facts.get(key) or default)  # noqa: E731
    return Candidate(
        name=get("full_name", profile.name),
        email=get("email"),
        phone=get("phone"),
        location=get("location"),
        linkedin=get("linkedin", profile.links.get("linkedin", "")),
        github=get("github", profile.links.get("github", "")),
        portfolio=get("portfolio"),
    )


def education_items(profile: ProfileText) -> list[Education]:
    """The degree line: "<title>, <org>, completed <month year>. <the rest>"."""
    if not profile.education:
        return []
    first, _, rest = profile.education[0].partition(". ")
    parts = [x.strip() for x in first.split(",")]
    year = re.search(r"(?:19|20)\d{2}", first)
    return [
        Education(
            title=parts[0],
            org=parts[1] if len(parts) > 1 else "",
            year=year.group(0) if year else "",
            description=rest.strip(),
        )
    ]


def display(token: str) -> str:
    """A skill token as a reader sees it: "llm-integration" -> "LLM integration" comes with the display table in
    the tailoring step; here hyphens become spaces."""
    return token.replace("-", " ")


def baseline_payload(profile: ProfileText, candidate: Candidate, *, bullets: int = 3, projects: int = 3) -> CvPayload:
    """A résumé straight from the profile, no model involved: every entry's first bullets, as written. It is the
    renderer's test input and the fallback when tailoring is unavailable; every word is the owner's own."""
    experience = [
        Experience(company=e.name, role=e.role.split(" / ")[0], dates=e.dates, bullets=e.bullets[:bullets])
        for e in reversed(profile.section("experience"))
    ]
    chosen = (profile.section("flagship") + profile.section("projects"))[:projects]
    return CvPayload(
        candidate=candidate,
        experience=experience,
        projects=[Project(name=e.name, bullets=e.bullets[: max(1, bullets - 1)], url=e.url) for e in chosen],
        education=education_items(profile),
        skills=[
            SkillGroup(category="Core", items=[display(t) for t in profile.skills.tiers["strong"][:22]]),
            SkillGroup(category="Working knowledge", items=[display(t) for t in profile.skills.tiers["working"][:18]]),
        ],
    )
