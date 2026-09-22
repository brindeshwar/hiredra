"""Sub-slice 4.3: near-duplicate merge proposals and their reviewed application
(docs/graph.md sec 5; owner rulings: merges are NEVER auto-applied, and the
approach is three bands, not one cosine cutoff).

Cosine alone can't tell a synonym from a sibling (genai ~ generative ai 0.896 is
the same skill; machine learning ~ deep learning 0.894 is not). So cosine only
SORTS pairs into bands, and a few cheap name rules decide which band is safe:

  AUTO    (pre-approved in the file)
          names that are the same after normalising (case, hyphen/space, plural,
          "2.0", keeping + and # so c# / c++ / c never collide) -- or cosine >=
          ``auto_cosine`` when neither name contains the other and they are not a
          version variant (oauth / oauth2).
  ALIAS   (pre-approved) the owner's own ``config/skill_aliases.yml`` entries.
  REVIEW  (``approve: false`` -- the owner ticks it) cosine >= ``review_cosine``
          that is not AUTO: one name contains the other (sql / sql server), or a
          version variant (css / css3). Pairs between ``threshold`` and
          ``review_cosine`` are hidden unless ``show_low`` (about half were wrong).
  LEAVE   everything else, and every pair in ``config/skill_never_merge.yml``.

A missed merge is cheap (fit already gives full credit at cosine >= have, so
"css" and "css3" score alike unmerged); a wrong merge loses a real distinction.

Two steps joined only by a file a human can read and edit:

1. ``propose_merges`` (read-only) + ``write_review`` -> ``data/merge_proposals.json``.
2. ``apply_review`` applies exactly the entries with ``"approve": true`` -- nothing
   else, ever. Each is re-verified at apply time (nodes exist, not ``mine``, not in
   the never-merge list, vectors current, cosine still above the band's floor, the
   file made under the live embedding version).

Cluster shape: greedy STAR clusters. Nodes are walked in survivor-priority order;
a survivor claims only nodes DIRECTLY related to it, so A~B and B~C never chain C
into A. AUTO pairs are claimed first (pass 1), then REVIEW pairs (pass 2), and a
node that already survives an AUTO cluster is never itself merged away in pass 2.
Survivor priority: ``mine`` (never the duplicate), most-sighted in our jobs, market
document frequency, shorter name, alphabetical.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hiredra.graph.config import GraphConfig
from hiredra.graph.index import blob_to_vector
from hiredra.storage import add_alias, get_skill_vocab, iter_skill_vocab, merge_skill, resolve_alias

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW_PATH = _REPO_ROOT / "data" / "merge_proposals.json"

#: float32 dot products of unit vectors can land a hair under an exact cosine.
_EPS = 1e-6
#: Same-spelling pairs are AUTO down to this cosine. Below it the embedding
#: disagrees so strongly that something is wrong, so we leave the pair alone.
SPELLING_FLOOR = 0.85

_BAND_ORDER = {"auto": 0, "alias": 1, "review": 2}


# ── the name rules ────────────────────────────────────────────────────────────
def _words(name: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", name.lower()) if w]


def _sing(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def normalized_key(name: str) -> str:
    """Spelling-insensitive key: lower-case, hyphens/spaces/slashes/dots ignored,
    simple plurals folded, ``2.0`` -> ``2``. ``+`` and ``#`` keep their meaning
    (``c#`` -> csharp, ``c++`` -> cplusplus) so they can never collapse into ``c``,
    and a leading dot too (``.net`` -> dotnet)."""
    n = name.lower().replace("+", " plus ").replace("#", " sharp ")
    n = re.sub(r"^\.", "dot ", n)
    n = re.sub(r"\.0\b", "", n)
    return "".join(_sing(w) for w in _words(n))


def spelling_keys(name: str) -> frozenset[str]:
    """The spellings a name can be compared by: with and WITHOUT plural folding.
    Folding alone mangles a joined word ("nodejs" -> "nodej") so it would never match
    "node.js" ("nodejs"); trying both means each real match is found either way."""
    n = name.lower().replace("+", " plus ").replace("#", " sharp ")
    n = re.sub(r"^\.", "dot ", n)
    n = re.sub(r"\.0\b", "", n)
    words = _words(n)
    return frozenset({"".join(words), "".join(_sing(w) for w in words)})


def same_spelling(a: str, b: str) -> bool:
    """True if the two names are one spelling apart: case, hyphen/space/slash/dot,
    a plural, "2.0". ``+`` and ``#`` keep their meaning (c# / c++ / c never match)."""
    return bool(spelling_keys(a) & spelling_keys(b))


def is_specialization(a: str, b: str) -> bool:
    """One name's words are a proper subset of the other's (sql / sql server,
    react / react native, backend / backend development)."""
    wa, wb = {_sing(w) for w in _words(a)}, {_sing(w) for w in _words(b)}
    return wa < wb or wb < wa


def is_version_variant(key_a: str, key_b: str) -> bool:
    """Different keys that are equal once digits are removed: oauth / oauth2,
    css / css3, python / python3, ipv4 / ipv6. A version is a real difference."""
    if key_a == key_b:
        return False
    stripped = re.sub(r"\d+", "", key_a)
    return bool(stripped) and stripped == re.sub(r"\d+", "", key_b)


def classify_pair(a: str, b: str, cosine: float, cfg: GraphConfig) -> tuple[str, str] | None:
    """``(band, reason)`` for a pair, or None to leave it alone. ``band`` is
    ``auto``, ``review`` or ``low`` (a review-type pair under ``review_cosine``,
    hidden unless ``show_low``)."""
    if same_spelling(a, b):
        return ("auto", "same spelling after normalising") if cosine >= SPELLING_FLOOR - _EPS else None
    ka, kb = normalized_key(a), normalized_key(b)
    if cosine < cfg.dedup_threshold - _EPS:
        return None
    if is_version_variant(ka, kb):
        reason = "version variant (differs only by a number)"
    elif is_specialization(a, b):
        reason = "one name contains the other"
    elif cosine >= cfg.auto_cosine - _EPS:
        return ("auto", f"cosine >= {cfg.auto_cosine}")
    else:
        reason = "similar"
    if cosine >= cfg.review_cosine - _EPS:
        return ("review", reason)
    return ("low", f"{reason}; below the {cfg.review_cosine} review cutoff")


# ── proposals ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MergeProposal:
    dup: str
    survivor: str
    cosine: float | None  # None for an alias-file entry whose alias has no vector / isn't a node
    dup_seen: int = 0
    survivor_seen: int = 0
    dup_df: int = 0
    survivor_df: int = 0
    band: str = "review"  # auto | alias | review
    reason: str = ""

    @property
    def approve(self) -> bool:
        """Pre-approved in the review file? AUTO and ALIAS yes; REVIEW never."""
        return self.band in ("auto", "alias")


@dataclass
class ProposalReport:
    embed_tag: str
    threshold: float
    nodes_considered: int  # embedded under the live tag
    nodes_skipped: int  # un-embedded or stale -- not eligible for cosine bands yet
    proposals: list[MergeProposal] = field(default_factory=list)
    near_misses: list[tuple[str, str, float]] = field(default_factory=list)  # just under the threshold
    hidden_low: int = 0  # review-type pairs between threshold and review_cosine (not listed)
    alias_notes: list[tuple[str, str, str]] = field(default_factory=list)  # alias-file entries that can't apply
    never_blocked: int = 0  # distinct pairs dropped because they are in the never-merge list

    def by_band(self, band: str) -> list[MergeProposal]:
        return [p for p in self.proposals if p.band == band]


def propose_merges(
    conn,
    cfg: GraphConfig,
    *,
    weights: Mapping[str, int] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
    never: Collection[frozenset[str]] | None = None,
    show_low: bool = False,
    near_miss_floor: float = 0.85,
    near_miss_limit: int = 40,
) -> ProposalReport:
    w = weights or {}
    never = never or frozenset()
    rows = iter_skill_vocab(conn)
    by_name = {r["name"]: r for r in rows}
    usable = [r for r in rows if r["vector"] is not None and r["embed_tag"] == cfg.embed_tag]
    report = ProposalReport(
        embed_tag=cfg.embed_tag,
        threshold=cfg.dedup_threshold,
        nodes_considered=len(usable),
        nodes_skipped=len(rows) - len(usable),
    )
    usable.sort(key=lambda r: (-r["mine"], -r["seen_count"], -w.get(r["name"], 0), len(r["name"]), r["name"]))
    names = [r["name"] for r in usable]
    index = {n: i for i, n in enumerate(names)}
    count = len(usable)
    blocked: set[frozenset[str]] = set()  # never-merge pairs that would otherwise have been considered
    assigned = np.zeros(count, dtype=bool)  # already merged into a survivor
    survives = np.zeros(count, dtype=bool)  # already survives an AUTO cluster
    sims = None
    if count >= 2:
        vectors = np.stack([blob_to_vector(r["vector"], cfg.dimensions) for r in usable])
        sims = vectors @ vectors.T

    def proposal(dup_row, surv_row, cosine, band, reason) -> MergeProposal:
        return MergeProposal(
            dup=dup_row["name"] if dup_row else "", survivor=surv_row["name"], cosine=cosine,
            dup_seen=dup_row["seen_count"] if dup_row else 0, survivor_seen=surv_row["seen_count"],
            dup_df=w.get(dup_row["name"], 0) if dup_row else 0, survivor_df=w.get(surv_row["name"], 0),
            band=band, reason=reason,
        )

    # -- ALIAS band: the owner's own list; needs no vectors --------------------
    for canonical, alist in (aliases or {}).items():
        c_row = by_name.get(canonical)
        for alias in alist:
            if c_row is None:
                report.alias_notes.append((alias, canonical, "canonical is not a node (yet)"))
                continue
            if frozenset((alias, canonical)) in never:
                report.alias_notes.append((alias, canonical, "the pair is also in the never-merge list"))
                continue
            existing = resolve_alias(conn, alias)
            if existing == canonical:
                continue  # already applied
            if existing is not None:
                report.alias_notes.append((alias, canonical, f"already an alias of {existing!r}"))
                continue
            row = by_name.get(alias)
            if row is None:
                report.proposals.append(MergeProposal(
                    dup=alias, survivor=canonical, cosine=None, survivor_seen=c_row["seen_count"],
                    survivor_df=w.get(canonical, 0), band="alias", reason="alias file (not a node: alias only)"))
                continue
            if row["mine"]:
                report.alias_notes.append((alias, canonical, "the alias is a 'mine' skill (never merged away)"))
                continue
            cos = float(sims[index[alias], index[canonical]]) if sims is not None and alias in index and canonical in index else None
            report.proposals.append(proposal(row, c_row, cos, "alias", "alias file"))
            if alias in index:
                assigned[index[alias]] = True

    # -- AUTO then REVIEW bands: cosine plus name rules ------------------------
    if sims is not None:
        lowest = min(SPELLING_FLOOR, cfg.dedup_threshold) - _EPS

        def related(i: int) -> list[int]:
            return [int(j) for j in np.flatnonzero(sims[i] >= lowest) if j > i and not usable[int(j)]["mine"]]

        def classify(i: int, j: int):
            if frozenset((names[i], names[j])) in never:
                blocked.add(frozenset((names[i], names[j])))
                return None
            return classify_pair(names[i], names[j], float(sims[i, j]), cfg)

        for i in range(count):  # pass 1: AUTO
            if assigned[i]:
                continue
            for j in related(i):
                if assigned[j]:
                    continue
                cls = classify(i, j)
                if cls and cls[0] == "auto":
                    assigned[j], survives[i] = True, True
                    report.proposals.append(proposal(usable[j], usable[i], float(sims[i, j]), "auto", cls[1]))

        for i in range(count):  # pass 2: REVIEW (and, with show_low, the low band)
            if assigned[i]:
                continue
            for j in related(i):
                if assigned[j] or survives[j]:
                    continue
                cls = classify(i, j)
                if not cls or cls[0] == "auto":
                    continue
                if cls[0] == "low" and not show_low:
                    report.hidden_low += 1
                    continue
                assigned[j] = True
                report.proposals.append(proposal(usable[j], usable[i], float(sims[i, j]), "review", cls[1]))

        upper = np.triu(np.ones_like(sims, dtype=bool), k=1)
        near = np.argwhere(upper & (sims >= near_miss_floor) & (sims < cfg.dedup_threshold - _EPS))
        ranked = sorted(((float(sims[a, b]), int(a), int(b)) for a, b in near), reverse=True)
        for c, a, b in ranked:
            if len(report.near_misses) >= near_miss_limit:
                break
            if frozenset((names[a], names[b])) in never or classify_pair(names[a], names[b], c, cfg):
                continue  # never-merge pairs and same-spelling pairs are not "near misses"
            report.near_misses.append((names[a], names[b], c))

    report.never_blocked = len(blocked)
    report.proposals.sort(key=lambda m: (_BAND_ORDER[m.band], m.survivor, -(m.cosine or 0.0), m.dup))
    return report


# ── the review file ───────────────────────────────────────────────────────────
def write_review(report: ProposalReport, path: Path | str = DEFAULT_REVIEW_PATH) -> Path:
    """Write the proposals as an editable JSON file. AUTO and ALIAS entries are
    written ``"approve": true``; REVIEW entries ``"approve": false``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "embed_tag": report.embed_tag,
                "threshold": report.threshold,
                "note": (
                    "Only entries with \"approve\": true are applied. AUTO and ALIAS entries start true; REVIEW "
                    "entries start false -- set one to true to merge it. Delete an entry to drop it. Then run: "
                    "python -m hiredra.graph merges --apply"
                ),
                "proposals": [
                    {
                        "band": m.band,
                        "approve": m.approve,
                        "reason": m.reason,
                        "dup": m.dup,
                        "survivor": m.survivor,
                        "cosine": None if m.cosine is None else round(m.cosine, 4),
                        "dup_seen": m.dup_seen,
                        "survivor_seen": m.survivor_seen,
                        "dup_df": m.dup_df,
                        "survivor_df": m.survivor_df,
                    }
                    for m in report.proposals
                ],
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return p


@dataclass
class ApplyReport:
    applied: list[tuple[str, str]] = field(default_factory=list)  # (dup, survivor)
    skipped: list[tuple[str, str, str]] = field(default_factory=list)  # (dup, survivor, reason)
    not_approved: int = 0  # entries left as approve != true
    archived_to: Path | None = None


def apply_review(
    conn,
    cfg: GraphConfig,
    path: Path | str = DEFAULT_REVIEW_PATH,
    *,
    never: Collection[frozenset[str]] | None = None,
    log: Callable[[str], None] = lambda _msg: None,
) -> ApplyReport:
    """Apply exactly the entries with ``"approve": true``, then archive the file.

    Refuses (``FileNotFoundError``) if there is no review file -- there is no way
    to apply merges without first having produced (and been able to read) the
    dry-run list. Refuses (``ValueError``) a file made under a different embedding
    version. Anything not ``approve: true`` (missing, false, or any other value)
    is left alone. Each approved entry is re-verified; failures are skipped and
    reported, never forced. Re-running after a crash is safe: already-merged pairs
    skip as "no longer exists" and the file is archived only once the loop ends.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"no review file at {p} -- run `python -m hiredra.graph merges` (dry-run) first and review it"
        )
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("embed_tag") != cfg.embed_tag:
        raise ValueError(
            f"review file was made under embedding version {doc.get('embed_tag')!r}, live is "
            f"{cfg.embed_tag!r} -- regenerate it with a fresh dry-run"
        )
    never = never or frozenset()
    report = ApplyReport()
    for entry in doc.get("proposals", []):
        if entry.get("approve") is not True:
            report.not_approved += 1
            continue
        dup, survivor = entry["dup"], entry["survivor"]
        band = entry.get("band", "review")
        reason = _apply_one(conn, cfg, dup, survivor, band, str(entry.get("reason", "")), never)
        if reason is None:
            report.applied.append((dup, survivor))
            log(f"merged {dup!r} -> {survivor!r}  [{band}]")
        else:
            report.skipped.append((dup, survivor, reason))
            log(f"skipped {dup!r} -> {survivor!r}: {reason}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = p.with_name(f"{p.stem}.applied-{stamp}{p.suffix}")
    n = 0
    while target.exists():  # never overwrite an earlier review record (and Windows can't rename onto one)
        n += 1
        target = p.with_name(f"{p.stem}.applied-{stamp}-{n}{p.suffix}")
    p.rename(target)
    report.archived_to = target
    return report


def _apply_one(conn, cfg, dup, survivor, band, reason, never) -> str | None:
    """Verify then apply one approved entry. Returns None on success, else why not."""
    if frozenset((dup, survivor)) in never:
        return "the pair is in the never-merge list"
    d, s = get_skill_vocab(conn, dup), get_skill_vocab(conn, survivor)
    if s is None:
        return "survivor no longer exists"

    if band == "alias":  # the owner's alias file: no cosine involved
        existing = resolve_alias(conn, dup)
        if existing == survivor:
            return "already an alias of the survivor"
        if existing is not None:
            return f"already an alias of {existing!r}"
        try:
            if d is None:
                add_alias(conn, dup, survivor)
            else:
                merge_skill(conn, dup, survivor)
        except ValueError as exc:
            return str(exc)
        return None

    if d is None:
        return "dup no longer exists"
    if d["mine"]:
        return "dup is a 'mine' skill"
    if d["vector"] is None or s["vector"] is None or d["embed_tag"] != cfg.embed_tag or s["embed_tag"] != cfg.embed_tag:
        return "a vector is missing or stale"
    spelling = band == "auto" and reason.startswith("same spelling") and same_spelling(dup, survivor)
    floor = SPELLING_FLOOR if spelling else cfg.dedup_threshold
    cosine = float(blob_to_vector(d["vector"], cfg.dimensions) @ blob_to_vector(s["vector"], cfg.dimensions))
    if cosine < floor - _EPS:
        return f"cosine {cosine:.3f} is below the floor {floor} for this kind of pair"
    try:
        merge_skill(conn, dup, survivor)
    except ValueError as exc:
        return str(exc)
    return None
