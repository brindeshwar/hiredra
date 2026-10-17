"""The master-profile overlay: turn the owner's Skills section into ``mine`` nodes (plan unit
PF1, decision D12, docs/graph.md sec 5).

For each profile token, in this order:

1. ``exact``   the token is a node name.
2. ``hyphen``  the token with hyphens turned into spaces is a node name.
3. ``alias``   either spelling is a recorded alias -> its canonical node.
4. ``vector``  embed the spaces spelling; if the nearest node's cosine is >= 0.90, PROPOSE a
               link. Proposals start ``"approve": false`` in ``data/profile_links.json`` and
               are never applied unless the owner sets them true (D12b).
5. ``new``     otherwise the token becomes a new ``source='profile'`` node named exactly as
               written, plus an alias for the other spelling (D12a).

Steps 1-3 are plain normalization and need no approval. The dry run (``plan_overlay``) is the
only step that may spend embedding requests; ``apply_overlay`` spends none -- it re-derives
steps 1-3, reads the owner's decisions on step 4 from the links file, and rebuilds ``mine`` in
one transaction. The graph is rebuilt from the profile file, never edited by hand.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hiredra.graph.config import GraphConfig
from hiredra.graph.embedder import Embedder
from hiredra.graph.index import load_skills
from hiredra.profile import LEVELS, TIERS, ProfileSkills
from hiredra.storage import apply_profile_overlay, get_skill_vocab, resolve_alias

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LINKS_PATH = _REPO_ROOT / "data" / "profile_links.json"
DEFAULT_OVERLAY_PATH = _REPO_ROOT / "data" / "profile_overlay.json"

#: The vector check only proposes a link at or above this cosine (D12).
LINK_COSINE = 0.90
#: When two tokens land on one node, the node keeps the strongest tier.
_TIER_RANK = {tier: i for i, tier in enumerate(TIERS)}
_EPS = 1e-6


class OverlayError(ValueError):
    """The overlay can't be applied as asked (stale links file, a vanished node, ...)."""


@dataclass
class Match:
    token: str
    tier: str
    step: str  # exact | hyphen | alias | vector | new
    node: str | None  # the node it resolves to (None for new; the proposed node for vector)
    cosine: float | None = None


@dataclass
class OverlayPlan:
    skills_hash: str
    file_hash: str
    matches: list[Match] = field(default_factory=list)

    def by_step(self, step: str) -> list[Match]:
        return [m for m in self.matches if m.step == step]

    def collisions(self) -> dict[str, list[str]]:
        """Nodes that two or more tokens resolve to: {node: [tokens]}."""
        groups: dict[str, list[str]] = {}
        for m in self.matches:
            if m.node is not None and m.step != "vector":
                groups.setdefault(m.node, []).append(m.token)
        return {node: tokens for node, tokens in groups.items() if len(tokens) > 1}


def _is_node(conn, name: str) -> bool:
    return get_skill_vocab(conn, name) is not None


def _normalized(conn, token: str, tier: str) -> Match | None:
    """Steps 1-3 (no embedding): exact, hyphens -> spaces, alias on either spelling."""
    spaces = token.replace("-", " ")
    if _is_node(conn, token):
        return Match(token, tier, "exact", token)
    if spaces != token and _is_node(conn, spaces):
        return Match(token, tier, "hyphen", spaces)
    for form in dict.fromkeys((token, spaces)):
        canonical = resolve_alias(conn, form)
        if canonical is not None:
            return Match(token, tier, "alias", canonical)
    return None


def plan_overlay(
    conn,
    cfg: GraphConfig,
    profile: ProfileSkills,
    embed_factory: Callable[[], Embedder] | None = None,
) -> OverlayPlan:
    """The dry run. Embeds only the tokens steps 1-3 can't place, in one call. A budget stop
    raises before anything is written (the caller writes the links file afterwards)."""
    plan = OverlayPlan(skills_hash=profile.skills_hash, file_hash=profile.file_hash)
    leftover: list[tuple[str, str]] = []
    for token, tier in profile.tokens():
        m = _normalized(conn, token, tier)
        if m is not None:
            plan.matches.append(m)
        else:
            leftover.append((token, tier))
    if not leftover:
        return plan

    nodes = [s for s in load_skills(conn, cfg) if s.vector is not None]
    if not nodes:
        plan.matches.extend(Match(t, tier, "new", None) for t, tier in leftover)
        return plan
    if embed_factory is None:
        raise OverlayError(f"{len(leftover)} token(s) need the vector check but no embedder was given")
    vectors = embed_factory()([t.replace("-", " ") for t, _ in leftover])
    matrix = np.array([s.vector for s in nodes], dtype=np.float32)
    for (token, tier), vec in zip(leftover, vectors):
        sims = matrix @ np.asarray(vec, dtype=np.float32)
        best = int(np.argmax(sims))
        cosine = float(sims[best])
        if cosine >= LINK_COSINE - _EPS:
            plan.matches.append(Match(token, tier, "vector", nodes[best].name, round(cosine, 4)))
        else:
            plan.matches.append(Match(token, tier, "new", None, round(cosine, 4)))
    return plan


def write_links(plan: OverlayPlan, path: Path | str = DEFAULT_LINKS_PATH) -> Path:
    """The review file: every vector proposal, starting ``"approve": false``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "skills_hash": plan.skills_hash,
                "note": (
                    'Only links with "approve": true are applied; the rest become new profile nodes. '
                    "Then run: python -m hiredra.graph overlay --apply"
                ),
                "links": [
                    {"token": m.token, "tier": m.tier, "node": m.node, "cosine": m.cosine, "approve": False}
                    for m in plan.by_step("vector")
                ],
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return p


@dataclass
class OverlayResult:
    counts: dict[str, int]
    collisions: dict[str, list[str]]
    mine: dict[str, str]
    new_nodes: dict[str, str]
    aliases: list[tuple[str, str]]


def apply_overlay(
    conn,
    profile: ProfileSkills,
    links_path: Path | str = DEFAULT_LINKS_PATH,
    overlay_path: Path | str = DEFAULT_OVERLAY_PATH,
) -> OverlayResult:
    """Apply the profile: steps 1-3 re-derived, step 4 from the owner's approvals, the rest new.
    Spends no embedding requests. Refuses if the links file was made for a different profile."""
    links_file = Path(links_path)
    approved: dict[str, str] = {}
    if links_file.exists():
        data = json.loads(links_file.read_text(encoding="utf-8"))
        if data.get("skills_hash") != profile.skills_hash:
            raise OverlayError(
                f"{links_file} was made for a different Skills section -- re-run `python -m hiredra.graph overlay`"
            )
        for link in data.get("links") or []:
            if link.get("approve") is True:
                if not _is_node(conn, str(link.get("node"))):
                    raise OverlayError(
                        f"approved link {link.get('token')!r} -> {link.get('node')!r}: that node is gone"
                    )
                approved[str(link["token"])] = str(link["node"])

    counts: Counter = Counter()
    mine: dict[str, str] = {}  # node -> tier (strongest wins on a collision)
    new_nodes: dict[str, str] = {}
    aliases: list[tuple[str, str]] = []
    resolved_to: dict[str, list[str]] = {}

    def claim(node: str, tier: str) -> None:
        if node not in mine or _TIER_RANK[tier] < _TIER_RANK[mine[node]]:
            mine[node] = tier

    for token, tier in profile.tokens():
        m = _normalized(conn, token, tier)
        if m is None and token in approved:
            m = Match(token, tier, "linked", approved[token])
        if m is None:
            counts["new"] += 1
            new_nodes[token] = tier
            other = token.replace("-", " ")
            if other != token and not _is_node(conn, other) and resolve_alias(conn, other) is None:
                aliases.append((other, token))
            continue
        counts[m.step] += 1
        claim(m.node, tier)
        resolved_to.setdefault(m.node, []).append(token)
        if token != m.node and not _is_node(conn, token) and resolve_alias(conn, token) is None:
            aliases.append((token, m.node))

    apply_profile_overlay(
        conn,
        mine={node: LEVELS[tier] for node, tier in mine.items()},
        new_nodes={name: LEVELS[tier] for name, tier in new_nodes.items()},
        aliases=aliases,
    )
    result = OverlayResult(
        counts={k: counts[k] for k in ("exact", "hyphen", "alias", "linked", "new")},
        collisions={node: tokens for node, tokens in resolved_to.items() if len(tokens) > 1},
        mine={node: LEVELS[tier] for node, tier in mine.items()},
        new_nodes={name: LEVELS[tier] for name, tier in new_nodes.items()},
        aliases=aliases,
    )
    out = Path(overlay_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "skills_hash": profile.skills_hash,
                "file_hash": profile.file_hash,
                "last_updated": profile.last_updated,
                "applied_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "counts": result.counts,
                "collisions": result.collisions,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return result
