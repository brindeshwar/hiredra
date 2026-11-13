"""The résumé's skills section, built by code alone (plan decision F1: the model never writes it).

Only the master profile's Strong and Working skills are printed (the profile's standing rule; Familiar skills count
for matching only). They are grouped under ``config/skill_display.yml``'s categories in that file's order, and inside
each group the job's own skills come first, then the rest in the profile's order, at most
``max_skills_per_category`` per group.
"""

from __future__ import annotations

from collections.abc import Iterable

from hiredra.profile import ProfileSkills
from hiredra.resume.payload import SkillGroup
from hiredra.resume.settings import SkillDisplay


def fold(name: str) -> str:
    """One spelling for comparing a profile token ("llm-integration") with a graph name ("llm integration")."""
    return " ".join(name.lower().replace("-", " ").split())


def claimable(skills: ProfileSkills) -> list[str]:
    """The tokens a résumé may assert: Strong, then Working, in the profile's order."""
    return [*skills.tiers["strong"], *skills.tiers["working"]]


def build_skills(
    skills: ProfileSkills, job_skills: Iterable[str], display: SkillDisplay, *, per_category: int = 12
) -> list[SkillGroup]:
    wanted = {fold(n) for n in job_skills}
    tokens = claimable(skills)
    ordered = [t for t in tokens if fold(t) in wanted] + [t for t in tokens if fold(t) not in wanted]
    groups: dict[str, list[str]] = {c: [] for c in display.categories}
    for token in ordered:
        if token in display.hide:
            continue
        items = groups[display.category_of(token)]
        if len(items) < per_category:
            items.append(display.display(token))
    return [SkillGroup(category=c, items=items) for c, items in groups.items() if items]
