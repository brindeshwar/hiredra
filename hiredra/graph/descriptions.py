"""Skill descriptions: the "description" half of the embedding input "name: description" (decision D1.15).

The source is ``hiredra/graph/skill_descriptions.tsv``: freehire's plain-language descriptions (MIT, see
THIRD_PARTY_NOTICES.md) with their display labels. A skill is described only when its own name matches an entry
exactly -- the slug, the slug with spaces, or the label. freehire's alias table is not used: it would give
"entity framework" the .NET description, pulling two different skills together once embedded.

``source_tag`` fingerprints the file, so the embedding version tag moves (and every vector is re-embedded) when
the descriptions change.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

DESCRIPTIONS_PATH = Path(__file__).with_name("skill_descriptions.tsv")


@lru_cache(maxsize=4)
def load_descriptions(path: Path = DESCRIPTIONS_PATH) -> dict[str, str]:
    """Lowercased skill name -> description, keyed by the slug, the slug with spaces, and the label."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        slug, label, text = line.split("\t")
        for form in (slug, slug.replace("-", " "), label.lower()):
            if form:
                out.setdefault(form, text)
    return out


def source_tag(path: Path = DESCRIPTIONS_PATH) -> str:
    """8 hex characters of the sha256 of the file's text, line endings normalised: a checkout that writes CRLF
    (git's autocrlf on Windows) must not make every stored vector look stale."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def described(name: str, descriptions: dict[str, str]) -> str:
    """The embedding input for ``name``: "name: description" when it has one, else the bare name."""
    text = descriptions.get(name.lower())
    return f"{name}: {text}" if text else name
