"""One pass of the skill-name rules (``hiredra.vocab``) over the names ALREADY stored (owner, 2026-11-12).

Extraction now cleans every new name on its way in. This applies the same rules to the vocabulary built before
that: each node that is not the owner's own is looked at as if it were new.

    python -m hiredra.graph clean            # dry run: writes data/graph-clean/plan.md + plan.json, changes nothing
    python -m hiredra.graph clean --apply    # applies exactly that plan (refuses if the database moved since)

Four kinds of change:

* ``merge``  -- a version or same-spelling variant of another node ("angular 7", "java script"), or "x (y)" where
  the other form is known: folded into it (``merge_skill``; its sightings move, it becomes an alias);
* ``rename`` -- "x (y)" where neither form is known: renamed to x, with y and the full form as aliases;
* ``split``  -- a list ("c/c++", "docker & kubernetes"): every job listing it gets the parts instead, then it is
  removed. A comma list keeps only its recognised parts;
* ``drop``   -- junk (non-Latin letters, too many words or characters): removed from every job, then removed.

Jobs that list a changed name have their skill rows rewritten to the new names (duplicates kept once), so a job
never counts one skill twice. Scored jobs keep their score until ``python -m hiredra.scoring run --force``.
Run it again after applying: a second pass catches what the first one changed (normally nothing).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from hiredra.extraction import name_resolver
from hiredra.storage import merge_skill, remove_skill, rename_skill, select_jobs, set_skills
from hiredra.vocab import _BRACKET, _EXAMPLES_LEAD, Resolver, is_acronym

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_DIR = _REPO_ROOT / "data" / "graph-clean"


@dataclass(frozen=True)
class Change:
    name: str
    kind: str  # merge | rename | split | drop
    targets: tuple[str, ...]
    why: str
    seen: int = 0


def _part_targets(r: Resolver, parts: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for part in parts:
        if part in r.skip:
            continue
        found = r.find(part)
        target = found[0] if found else (None if r.junk_reason(part) else part)
        if target and target not in out:
            out.append(target)
    return tuple(out)


def plan_node(r: Resolver, name: str) -> Change | None:
    """What the rules would do with node ``name`` if it arrived today (None: it stays as it is)."""
    if name in r.keep_whole:
        return None
    saved = r._canon.pop(name)  # look at it as if new: its own entry must not answer for it
    try:
        m = _BRACKET.fullmatch(name)
        if m:
            outer, inner = m.group(1).strip(), m.group(2).strip()
            if is_acronym(inner, outer) or is_acronym(outer, inner):
                known = next((f[0] for f in map(r.find, (inner, outer)) if f and f[0] != name), None)
                if known:
                    return Change(name, "merge", (known,), "acronym of a known skill")
                return Change(name, "rename", (outer,), f"acronym; {inner} becomes an alias")
            targets = _part_targets(r, r._split(outer) + r._split(_EXAMPLES_LEAD.sub("", inner)))
            return Change(name, "split" if targets else "drop", targets, "a name with examples in brackets")
        parts = r._split(name)
        if len(parts) > 1:
            targets = _part_targets(r, parts)
            return Change(name, "split" if targets else "drop", targets, "a list of skills")
        found = r.find(name)
        if found and found[0] != name:
            return Change(name, "merge", (found[0],), f"{found[1]} variant")
        reason = r.junk_reason(name)
        return Change(name, "drop", (), reason) if reason else None
    finally:
        r._canon[name] = saved


def plan(conn, resolver: Resolver) -> list[Change]:
    """Every change, for nodes that are not ``mine`` (the owner's skills are edited in the profile, never here)."""
    rows = conn.execute("SELECT name, seen_count FROM skill_vocab WHERE NOT mine ORDER BY name").fetchall()
    changes = []
    for row in rows:
        change = plan_node(resolver, row["name"])
        if change and change.targets != (row["name"],):
            changes.append(Change(**{**asdict(change), "seen": row["seen_count"]}))
    # A change whose target is itself changing waits for the next pass, so nothing points at a removed node.
    changing = {c.name for c in changes}
    return [c for c in changes if not changing.intersection(c.targets)]


def write_plan(changes: list[Change], plan_dir: Path) -> tuple[Path, Path]:
    plan_dir.mkdir(parents=True, exist_ok=True)
    js, md = plan_dir / "plan.json", plan_dir / "plan.md"
    js.write_text(json.dumps([asdict(c) for c in changes], indent=1), encoding="utf-8")
    lines = ["# Skill-name cleanup plan", "", "Applied with `python -m hiredra.graph clean --apply`.", ""]
    for kind in ("merge", "rename", "split", "drop"):
        group = [c for c in changes if c.kind == kind]
        lines += [f"## {kind} ({len(group)})", "", "| name | becomes | why | seen |", "|---|---|---|---|"]
        lines += [f"| {c.name} | {' + '.join(c.targets) or '-'} | {c.why} | {c.seen} |" for c in group]
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    return js, md


def load_plan(plan_dir: Path) -> list[Change]:
    path = plan_dir / "plan.json"
    return [Change(**{**c, "targets": tuple(c["targets"])}) for c in json.loads(path.read_text(encoding="utf-8"))]


def apply(conn, resolver: Resolver, changes: list[Change]) -> dict[str, int]:
    """Apply ``changes``: merges and renames first (so every target exists), then each affected job's skill rows,
    then the split and dropped nodes."""
    counts = {"merged": 0, "renamed": 0, "jobs_rewritten": 0, "scored_jobs": 0, "removed": 0}
    by_name = {c.name: c for c in changes}
    for c in changes:
        if c.kind == "merge":
            merge_skill(conn, c.name, c.targets[0])
            counts["merged"] += 1
        elif c.kind == "rename":
            rename_skill(conn, c.name, c.targets[0])
            counts["renamed"] += 1
    affected = [c.name for c in changes]
    jobs = select_jobs(conn, "j.id IN (SELECT job_id FROM job_skills WHERE skill = ANY(%s))", (affected,), "j.id")
    for job in jobs:
        rows = []
        for item in job["skills"] or []:
            c = by_name.get(item["name"])
            names = [item["name"]] if c is None else list(c.targets)
            for n in names:
                if c is not None and c.kind == "split":  # a part may be new: resolving creates it and counts it
                    n = next(iter(resolver.resolve(n)), None)
                if n:
                    rows.append({"name": n, "requirement": item["requirement"], "source": item.get("source", "llm")})
        rows.sort(key=lambda r: r["requirement"] != "required")  # a skill required anywhere stays required
        set_skills(conn, job["job_id"], rows)
        counts["jobs_rewritten"] += 1
        counts["scored_jobs"] += job["status"] == "scored"
    for c in changes:
        if c.kind in ("split", "drop"):
            remove_skill(conn, c.name)
            counts["removed"] += 1
    conn.commit()
    return counts


def build_resolver(conn) -> Resolver:
    from hiredra.extraction import load_extraction_config

    return name_resolver(conn, load_extraction_config())
