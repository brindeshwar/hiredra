"""One company per employer (decisions D2.4, D6.5; build slice 2.2). Ported in idea from freehire's
``cmd/merge-companies`` and docs/agents/company-identity.md (MIT, see THIRD_PARTY_NOTICES.md).

The same employer reaches the store under more than one key:

- **rule**: the key rule changed after the company was stored (e.g. India's "Pvt Ltd" form, 2.2), so its stored
  slug is no longer what ``company_slug(name)`` gives;
- **fold**: one employer spelled two ways that differ only by spaces and hyphens ("Together AI" from Indeed,
  "Togetherai" from its board slug). No rule can pick the right side, so they are grouped by the squashed key;
- **curated**: pairs neither rule reaches ("Razorpay" vs "Razorpay Software"), listed by the owner in
  ``config/company_merges.yml``. A judgement, so never automatic.

``plan_merges`` is pure: it reads company rows and returns renames and merge groups. ``apply_merges`` writes them in
one transaction: every job, board and alias of a retired company moves to the canon, the retired rows go, and every
retired spelling becomes an alias, so the next crawl that writes it lands on the canon (``storage.db._company_id``).
A canon is frozen once chosen: an alias is never re-pointed by a later run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hiredra.sourcing.identity import company_slug

_REPO_ROOT = Path(__file__).resolve().parents[2]
CURATED_PATH = _REPO_ROOT / "config" / "company_merges.yml"


@dataclass(frozen=True, slots=True)
class Rename:
    id: int
    old_slug: str
    new_slug: str


@dataclass(frozen=True, slots=True)
class MergeGroup:
    canon_id: int  # the row that stays: the one with the most jobs and boards
    slug: str  # the key it ends with: the most readable member key (most hyphens)
    name: str  # the display name of the member that key came from ("Together AI", not "Togetherai")
    retired: tuple[tuple[int, str], ...]  # (id, stored slug) of every row folded into the canon
    old_slug: str  # the canon row's stored slug before the merge
    reason: str  # fold | curated
    jobs: int  # jobs across the group


def load_curated(path: Path | str = CURATED_PATH) -> list[list[str]]:
    """``merges:`` as lists of names or slugs; each name is keyed with ``company_slug``. A missing file is empty."""
    path = Path(path)
    if not path.exists():
        return []
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    groups = doc.get("merges") or []
    if not isinstance(groups, list) or not all(isinstance(g, list) and len(g) >= 2 for g in groups):
        raise ValueError(f"{path}: 'merges' must be a list of lists of two or more names")
    return [[company_slug(str(n)) for n in g] for g in groups]


def _readable(key: str) -> tuple[int, int, str]:
    return (-key.count("-"), len(key), key)


def plan_merges(
    rows: Iterable[Mapping[str, Any]], curated: Sequence[Sequence[str]] = ()
) -> tuple[list[Rename], list[MergeGroup]]:
    """``rows``: companies with ``id``, ``slug``, ``name``, ``jobs``, ``boards``. Returns the renames (a lone company
    whose key changed) and the merge groups (two or more companies that are one employer)."""
    rows = [dict(r) for r in rows]
    for r in rows:
        r["key"] = company_slug(r["name"]) or r["slug"]
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    by_fold: dict[str, int] = {}
    by_key: dict[str, int] = {}
    for i, r in enumerate(rows):
        for k in {r["key"], r["slug"]}:
            by_key.setdefault(k, i)
        fold = r["key"].replace("-", "")
        if fold in by_fold:
            union(i, by_fold[fold])
        else:
            by_fold[fold] = i
    curated_members: set[int] = set()
    named: list[int] = []  # the first name of each curated group: the owner's choice of name for the employer
    for group in curated:
        members = [by_key[k] for k in group if k in by_key]
        for m in members[1:]:
            union(m, members[0])
        if len(members) > 1:
            curated_members.update(members)
            if by_key.get(group[0]) == members[0]:
                named.append(members[0])

    clusters: dict[int, list[int]] = {}
    for i in range(len(rows)):
        clusters.setdefault(find(i), []).append(i)

    renames: list[Rename] = []
    groups: list[MergeGroup] = []
    for members in clusters.values():
        if len(members) == 1:
            r = rows[members[0]]
            if r["key"] != r["slug"]:
                renames.append(Rename(r["id"], r["slug"], r["key"]))
            continue
        ranked = sorted(members, key=lambda i: (-(rows[i]["jobs"] + rows[i]["boards"]), rows[i]["id"]))
        canon = rows[ranked[0]]
        chosen = [i for i in named if i in members]
        best = chosen[0] if chosen else min(members, key=lambda i: _readable(rows[i]["key"]))
        folds = {rows[i]["key"].replace("-", "") for i in members}
        reason = "curated" if set(members) & curated_members and len(folds) > 1 else "fold"
        groups.append(
            MergeGroup(
                canon_id=canon["id"],
                slug=rows[best]["key"],
                name=rows[best]["name"],
                retired=tuple((rows[i]["id"], rows[i]["slug"]) for i in ranked[1:]),
                old_slug=canon["slug"],
                reason=reason,
                jobs=sum(rows[i]["jobs"] for i in members),
            )
        )
    renames.sort(key=lambda x: x.old_slug)
    groups.sort(key=lambda g: (-g.jobs, g.slug))
    return renames, groups


def suggestions(rows: Iterable[Mapping[str, Any]], limit: int = 20) -> list[tuple[str, str, int]]:
    """Pairs a rule can't decide, for the owner to judge into ``config/company_merges.yml``: one key is another plus
    one more word ("accenture" / "accenture-services"), both with jobs. Biggest first."""
    rows = [r for r in rows if r["jobs"]]
    keys = {company_slug(r["name"]) or r["slug"]: r for r in rows}
    out = []
    for key, r in keys.items():
        head, _, tail = key.rpartition("-")
        if head and tail and head in keys:
            out.append((keys[head]["name"], r["name"], keys[head]["jobs"] + r["jobs"]))
    return sorted(out, key=lambda x: -x[2])[:limit]


def company_rows(conn) -> list[Any]:
    return conn.execute(
        "SELECT c.id, c.slug, c.name, "
        "(SELECT count(*) FROM jobs j WHERE j.company_id = c.id) AS jobs, "
        "(SELECT count(*) FROM boards b WHERE b.company_id = c.id) AS boards "
        "FROM companies c ORDER BY c.id"
    ).fetchall()


def apply_merges(conn, renames: Sequence[Rename], groups: Sequence[MergeGroup]) -> None:
    """Write the plan in one transaction. Retired and renamed spellings become aliases of the company they now mean."""
    with conn.transaction():
        for g in groups:
            retired = [i for i, _ in g.retired]
            for table in ("jobs", "boards", "company_slug_aliases"):
                conn.execute(f"UPDATE {table} SET company_id = %s WHERE company_id = ANY(%s)", (g.canon_id, retired))
            conn.execute("DELETE FROM companies WHERE id = ANY(%s)", (retired,))
            conn.execute("UPDATE companies SET slug = %s, name = %s WHERE id = %s", (g.slug, g.name, g.canon_id))
            _aliases(conn, g.canon_id, [g.old_slug, *(s for _, s in g.retired)], g.slug)
        for r in renames:
            conn.execute("UPDATE companies SET slug = %s WHERE id = %s", (r.new_slug, r.id))
            _aliases(conn, r.id, [r.old_slug], r.new_slug)


def _aliases(conn, company_id: int, spellings: Iterable[str], current: str) -> None:
    for alias in dict.fromkeys(spellings):
        if alias != current:
            conn.execute(
                "INSERT INTO company_slug_aliases (alias, company_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (alias, company_id),
            )
    conn.execute("DELETE FROM company_slug_aliases WHERE alias = %s", (current,))  # a live key is never an alias
