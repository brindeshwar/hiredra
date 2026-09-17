"""Sub-slice 3.2: the growing canonical skill vocabulary (build step 3).

A flat, growing list of canonical skill names (docs/graph.md's eventual
`market` nodes are seeded from this table) -- never an enum, since the
vocabulary is discovered from JD text and grows every run. Step 3 only
dedupes exact (case/whitespace-folded) repeats here; catching near-duplicate
SPELLINGS of the same skill ("fastapi" vs "fast api") is Step 4 machinery
(docs/graph.md sec 5's overlay match: exact name first, then >=0.9 cosine)
and needs the embedder that Step 4 builds -- `_find_near_duplicate` is a
stub seam for it, not a real implementation, so Step 3 doesn't block on the
embedder existing.
"""

from __future__ import annotations

from hiredra.storage import touch_skill_vocab


def _normalize(raw_name: str) -> str:
    """Case/whitespace fold only -- no synonym logic here."""
    return " ".join(raw_name.split()).lower()


def _find_near_duplicate(name: str, conn) -> str | None:
    """Step-4 seam: find an existing vocab entry that's the same skill under
    a different spelling, via embedding similarity (docs/graph.md sec 5).

    Always returns None for now -- there is no embedder yet, and Step 3 is
    not blocked on one existing. Step 4 replaces this body with the real
    embed-and-compare check; the call site in `resolve_skill` does not
    change.
    """
    return None


def resolve_skill(conn, raw_name: str) -> str:
    """Canonicalize `raw_name` against `skill_vocab`, recording the sighting.

    Exact match (case/whitespace-folded) reuses the existing entry. No exact
    match: the near-duplicate check is the Step-4 stub above, which always
    misses today -- so right now, no exact match always means "first sighting
    of a new skill," and it's added as its own vocab entry under its folded
    name.
    """
    name = _normalize(raw_name)
    near = _find_near_duplicate(name, conn)
    canonical = near if near is not None else name
    touch_skill_vocab(conn, canonical)
    return canonical
