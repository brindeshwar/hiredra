"""Sub-slice 3.2: the growing canonical skill vocabulary (build step 3).

A flat, growing list of canonical skill names (docs/graph.md's `market`
nodes live in this same table) -- never an enum, since the vocabulary is
discovered from JD text and grows every run.

Extraction-time resolution is deliberately cheap and offline: case/whitespace
fold, then an exact lookup in the merged-spelling alias table (Step 4, v4).
It never calls the embedder. Catching near-duplicate SPELLINGS of the same
skill ("fastapi" vs "fast api") is the between-runs sync step's job
(`hiredra.graph.merge`; docs/graph.md sec 6's freeze rule: nodes are added and
merged between runs, never mid-run, and merges are dry-run-reviewed, never
auto-applied). Once a merge is applied, the alias table makes that spelling
resolve here without an embed call, so extraction stays independent of the
embedding API being up.
"""

from __future__ import annotations

from hiredra.storage import resolve_alias, touch_skill_vocab


def normalize(raw_name: str) -> str:
    """Case/whitespace fold only -- no synonym logic here."""
    return " ".join(raw_name.split()).lower()


def resolve_skill(conn, raw_name: str) -> str:
    """Canonicalize `raw_name` against `skill_vocab`, recording the sighting.

    Order: fold; if that exact string is a merged alias, use the node it was
    merged into; otherwise it is its own canonical name. Either way the
    canonical node's `seen_count` is bumped (or the node is created, on first
    sighting). No embedder is involved.
    """
    name = normalize(raw_name)
    canonical = resolve_alias(conn, name) or name
    touch_skill_vocab(conn, canonical)
    return canonical
