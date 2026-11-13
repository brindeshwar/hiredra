"""``config/resume.yml`` and ``config/skill_display.yml``, read once per run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
RESUME_CONFIG = _REPO_ROOT / "config" / "resume.yml"
SKILL_DISPLAY = _REPO_ROOT / "config" / "skill_display.yml"


@dataclass(frozen=True)
class ResumeConfig:
    provider: str = "nvidia"
    model: str = "nvidia/nemotron-3.5-super-vl-120b-a12b"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    requests_per_minute: int = 35
    requests_per_day: int = 10000
    timeout_seconds: float = 180.0
    max_tokens: int = 4096
    max_jd_chars: int = 6000
    max_projects: int = 4
    max_bullets_per_experience: int = 5
    max_bullets_per_project: int = 3
    max_skills_per_category: int = 12
    banned_phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillDisplay:
    categories: tuple[str, ...]
    category: dict[str, str] = field(default_factory=dict)
    name: dict[str, str] = field(default_factory=dict)
    fallback: tuple[tuple[str, tuple[str, ...]], ...] = ()
    hide: frozenset[str] = frozenset()  # tokens never printed in the skills section

    def category_of(self, token: str) -> str:
        if token in self.category:
            return self.category[token]
        for cat, words in self.fallback:
            if not words or any(w in token for w in words):
                return cat
        return self.categories[-1]

    def display(self, token: str) -> str:
        return self.name.get(token, token.replace("-", " "))


def load_resume_config(path: Path | str = RESUME_CONFIG) -> ResumeConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = set(ResumeConfig.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{Path(path).name}: unknown key(s) {sorted(unknown)}")
    if "banned_phrases" in data:
        data["banned_phrases"] = tuple(str(x) for x in data["banned_phrases"] or ())
    return ResumeConfig(**data)


def load_skill_display(path: Path | str = SKILL_DISPLAY) -> SkillDisplay:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cats = tuple(data.get("categories") or ())
    category = {str(k): str(v) for k, v in (data.get("category") or {}).items()}
    bad = {v for v in category.values() if v not in cats} | {
        r["category"] for r in data.get("fallback") or () if r["category"] not in cats
    }
    if not cats or bad:
        raise ValueError(f"{Path(path).name}: categories missing or unknown: {sorted(bad)}")
    return SkillDisplay(
        categories=cats,
        category=category,
        name={str(k): str(v) for k, v in (data.get("name") or {}).items()},
        fallback=tuple((r["category"], tuple(r.get("words") or ())) for r in data.get("fallback") or ()),
        hide=frozenset(str(x) for x in data.get("hide") or ()),
    )
