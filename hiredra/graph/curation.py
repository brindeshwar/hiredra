"""Owner-editable merge curation files (build 4.3): the alias list and the
never-merge list.

Cosine can't reach true abbreviations (k8s ~ kubernetes, golang ~ go) and
can't stop a wrong-but-similar pair (sql ~ sql server) from coming back every
time. These two small YAML files are where the owner's judgement lives:

* ``config/skill_aliases.yml``      -- ``canonical: [alias, ...]``
* ``config/skill_never_merge.yml``  -- pairs that must never be proposed

Both are STRICT: an unknown key, a self-alias, an alias listed under two
canonicals, or a name that is both a canonical and an alias raises, so a typo
can't silently do nothing (or worse, chain). Names are folded the same way as
everywhere else (case + whitespace). A missing file just means "empty".

Neither file changes anything by itself. Alias entries appear in the merge
dry-run as their own band and are applied only by ``merges --apply``.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from hiredra.vocab import normalize

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALIAS_PATH = _REPO_ROOT / "config" / "skill_aliases.yml"
DEFAULT_NEVER_MERGE_PATH = _REPO_ROOT / "config" / "skill_never_merge.yml"


def _load(path: Path | str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p.name}: top level must be a mapping")
    return data


def load_aliases(path: Path | str = DEFAULT_ALIAS_PATH) -> dict[str, tuple[str, ...]]:
    """``{canonical: (alias, ...)}``, all names folded. Empty if the file is missing."""
    name = Path(path).name
    data = _load(path)
    unknown = set(data) - {"aliases"}
    if unknown:
        raise ValueError(f"{name}: unknown top-level key(s) {sorted(unknown)} (only 'aliases' is allowed)")
    raw = data.get("aliases") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{name}: 'aliases' must be a mapping of canonical: [aliases]")

    out: dict[str, tuple[str, ...]] = {}
    seen_alias: dict[str, str] = {}
    for canonical, aliases in raw.items():
        c = normalize(str(canonical))
        if not c:
            raise ValueError(f"{name}: an empty canonical name")
        if not isinstance(aliases, list) or not aliases:
            raise ValueError(f"{name}: {canonical!r} must map to a non-empty list of aliases")
        folded: list[str] = []
        for a in aliases:
            if not isinstance(a, str) or not normalize(a):
                raise ValueError(f"{name}: {canonical!r} has an empty or non-text alias")
            fa = normalize(a)
            if fa == c:
                raise ValueError(f"{name}: {a!r} is listed as an alias of itself")
            if fa in seen_alias:
                raise ValueError(f"{name}: {a!r} is an alias of both {seen_alias[fa]!r} and {c!r}")
            seen_alias[fa] = c
            folded.append(fa)
        if c in out:
            raise ValueError(f"{name}: canonical {canonical!r} appears twice")
        out[c] = tuple(folded)
    clash = set(out) & set(seen_alias)
    if clash:
        raise ValueError(f"{name}: {sorted(clash)} are both a canonical and an alias (aliases must be one hop)")
    return out


def load_never_merge(path: Path | str = DEFAULT_NEVER_MERGE_PATH) -> frozenset[frozenset[str]]:
    """The set of unordered name pairs that must never be proposed."""
    name = Path(path).name
    data = _load(path)
    unknown = set(data) - {"never_merge"}
    if unknown:
        raise ValueError(f"{name}: unknown top-level key(s) {sorted(unknown)} (only 'never_merge' is allowed)")
    raw = data.get("never_merge") or []
    if not isinstance(raw, list):
        raise ValueError(f"{name}: 'never_merge' must be a list of [a, b] pairs")
    pairs: set[frozenset[str]] = set()
    for item in raw:
        if not (isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) and normalize(x) for x in item)):
            raise ValueError(f"{name}: each entry must be a list of exactly two names, got {item!r}")
        a, b = normalize(item[0]), normalize(item[1])
        if a == b:
            raise ValueError(f"{name}: {item!r} pairs a name with itself")
        pairs.add(frozenset((a, b)))
    return frozenset(pairs)
