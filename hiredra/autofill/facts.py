"""The owner's verified facts for application forms: ``data/autofill_facts.yml`` (private, git-ignored; its shape is
``config/autofill_facts.example.yml``).

Only facts a form may be filled with go here: contact details, links, education, the current employer. Some topics
are never filled by the system, whatever the file says (plan decision D4 and the pause list): work authorization,
visas and sponsorship, salary and compensation, demographics, date of birth and age, and government IDs. The loader
refuses a file that holds any of them, by key name or by value, so they can only ever be answered by the human in the
form itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACTS_PATH = _REPO_ROOT / "data" / "autofill_facts.yml"

#: Topics the system never fills (case-insensitive; matched in key names and in values).
FORBIDDEN = re.compile(
    r"authori[sz]|visa|sponsor|work.?permit|citizen|right.to.work|salary|\bctc\b|compensation|remuneration|"
    r"\bpay\b|stipend|gender|\brace\b|ethnic|caste|religio|disabilit|veteran|date.of.birth|\bdob\b|\bage\b|"
    r"birth|passport|aadhaar|\bpan\b|\bssn\b|social.security|national.id|government.id|password|secret",
    re.IGNORECASE,
)
#: The keys a form may use, in two blocks.
CONTACT_KEYS = ("full_name", "email", "phone", "location", "linkedin", "github", "portfolio")
FACT_KEYS = ("preferred_name", "first_name", "last_name", "country", "city", "degree", "institution",
             "graduation_year", "current_company", "current_title", "current_status")  # fmt: skip


class FactsError(ValueError):
    """The facts file holds something the system must never fill, or a key it does not know."""


def load_facts(path: Path | str = DEFAULT_FACTS_PATH) -> dict[str, str]:
    """Every fact as one flat ``{key: value}`` (empty values left out). First and last name are derived from the full
    name when not given."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise FactsError(f"{p.name}: the top level must be a mapping")
    problems = [f"unknown block {k!r}" for k in data if k not in ("contact", "facts")]
    flat: dict[str, str] = {}
    for block, allowed in (("contact", CONTACT_KEYS), ("facts", FACT_KEYS)):
        for key, value in (data.get(block) or {}).items():
            key = str(key)
            text = "" if value is None else str(value).strip()
            if FORBIDDEN.search(key) or (text and FORBIDDEN.search(text)):
                problems.append(f"{block}.{key}: a topic the system never fills (answer it in the form yourself)")
            elif key not in allowed:
                problems.append(f"{block}.{key}: not a known key ({', '.join(allowed)})")
            elif text:
                flat[key] = text
    if problems:
        raise FactsError(f"{p.name}: " + "; ".join(problems))
    if "full_name" in flat:
        first, _, last = flat["full_name"].partition(" ")
        flat.setdefault("first_name", first)
        if last:
            flat.setdefault("last_name", last.split()[-1] if " " in last else last)
    return flat
