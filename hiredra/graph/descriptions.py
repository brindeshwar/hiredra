"""Skill descriptions: the "description" half of the embedding input "name: description" (decision D1.15).

Where a node's description comes from, in order (graph audit, 2026-11-12):

1. ``config/skill_descriptions.yml`` -- the owner's overrides (``null`` = no description, embed the bare name);
2. ``hiredra/graph/skill_descriptions.tsv`` -- freehire's plain-language descriptions (MIT, see
   THIRD_PARTY_NOTICES.md) with their display labels. A skill is described only when its own name matches an entry
   exactly -- the slug, the slug with spaces, or the label. freehire's alias table is not used: it would give
   "entity framework" the .NET description, pulling two different skills together once embedded;
3. a description the describe step wrote (``generated``), or its refusal (``declined``), kept in the database.

``refresh_descriptions`` writes the winner into ``skill_vocab.description`` / ``description_source``, so the
database is the one place to read a node's description. ``embedding_descriptions`` is the lookup the embedder uses;
only the sources listed in ``embedding.description_sources`` feed vectors. Each vector records its exact input
(``embed_input``), so a changed description re-embeds that one skill.

``source_tag`` fingerprints freehire's file (fixed, third-party), which stays part of the embedding version tag.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import yaml

DESCRIPTIONS_PATH = Path(__file__).with_name("skill_descriptions.tsv")
OVERRIDES_PATH = Path(__file__).resolve().parents[2] / "config" / "skill_descriptions.yml"


@lru_cache(maxsize=4)
def load_descriptions(path: Path = DESCRIPTIONS_PATH) -> dict[str, str]:
    """Lowercased skill name -> freehire's description, keyed by the slug, the slug with spaces, and the label."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        slug, label, text = line.split("\t")
        for form in (slug, slug.replace("-", " "), label.lower()):
            if form:
                out.setdefault(form, text)
    return out


def load_overrides(path: Path = OVERRIDES_PATH) -> dict[str, str | None]:
    """The owner's overrides: folded name -> description, or None for "no description". Strict: a non-text value
    or an unknown top-level key raises, so a typo can't silently do nothing."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or set(data) - {"overrides"}:
        raise ValueError(f"{p.name}: the only top-level key is 'overrides'")
    raw = data.get("overrides") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{p.name}: 'overrides' must map a skill name to a description or null")
    out: dict[str, str | None] = {}
    for name, text in raw.items():
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise ValueError(f"{p.name}: {name!r} needs a description, or null")
        out[" ".join(str(name).split()).lower()] = " ".join(text.split()) if text else None
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


def refresh_descriptions(conn, *, overrides: dict[str, str | None] | None = None) -> dict[str, int]:
    """Write each node's winning description and its source (override > freehire > generated/declined, which stay
    as the describe step left them). Returns how many rows changed, by new source."""
    overrides = load_overrides() if overrides is None else overrides
    freehire = load_descriptions()
    rows = conn.execute("SELECT name, description, description_source FROM skill_vocab").fetchall()
    changes: list[tuple[str | None, str | None, str]] = []
    counts: dict[str, int] = {}
    for r in rows:
        name, old = r["name"], (r["description"], r["description_source"])
        if name in overrides:
            new = (overrides[name], "override")
        elif name in freehire:
            new = (freehire[name], "freehire")
        elif r["description_source"] in ("generated", "declined"):
            new = old
        else:
            new = (None, None)
        if new != old:
            changes.append((*new, name))
            counts[new[1] or "none"] = counts.get(new[1] or "none", 0) + 1
    if changes:
        with conn.transaction():
            conn.cursor().executemany(
                "UPDATE skill_vocab SET description = %s, description_source = %s WHERE name = %s", changes
            )
    return counts


def embedding_descriptions(conn, cfg) -> dict[str, str]:
    """The lookup the embedder uses: name -> description, from the sources ``cfg.description_sources`` allows,
    in the same order of precedence. Empty for ``inputs: bare``. Covers names that are not nodes too (a calibration
    pair, a skill scored before it is stored), exactly as freehire's file and the overrides describe them."""
    if cfg.inputs != "described":
        return {}
    sources = set(cfg.description_sources)
    out: dict[str, str] = {}
    if "generated" in sources:
        for r in conn.execute(
            "SELECT name, description FROM skill_vocab WHERE description_source = 'generated' AND description IS NOT NULL"
        ):
            out[r["name"]] = r["description"]
    if "freehire" in sources:
        out.update(load_descriptions())
    if "override" in sources:
        for name, text in load_overrides().items():
            if text:
                out[name] = text
            else:
                out.pop(name, None)
    return out
