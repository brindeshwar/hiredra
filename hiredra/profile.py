"""The master profile's Skills section: a strict parser for the frozen grammar (plan unit PF1,
decision D11, contract in the plan's sec 3.7).

``data/master_profile.md`` (git-ignored) is an EXPORT of the owner's project doc, which is the
source of truth (A2): code reads it, never edits it. Only the ``## Skills`` section is parsed;
everything else is prose for the tailoring and critic agents (D4). One more line is read: the
first ``Last updated:`` line, reported beside the file hash.

    ## Skills <any text>
    **Strong <any text>:**
    python, fastapi, postgresql
    **Working <any text>:**
    docker, scikit-learn
    **Familiar <any text>:**
    rust
    Explicitly NOT claimed: <any text>

Every rejection names its line number and the rule it broke; nothing partial is returned.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = _REPO_ROOT / "data" / "master_profile.md"

TIERS = ("strong", "working", "familiar")
#: Tier -> graph ``level`` (D4). All three count as ``mine`` for scoring.
LEVELS = {"strong": "production", "working": "built-with", "familiar": "learned"}

_HEADER_RE = re.compile(r"^## Skills")
_SECTION_END_RE = re.compile(r"^## ")
_LABEL_RE = re.compile(r"^\*\*(Strong|Working|Familiar).*:\*\*$")
_ANY_LABEL_RE = re.compile(r"^\*\*.*:\*\*$")
_NOT_CLAIMED = "Explicitly NOT claimed:"
_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9+#./-]*$")
_LAST_UPDATED_RE = re.compile(r"^\s*\**\s*Last updated\s*:?\**\s*(.+)$", re.IGNORECASE)


class ProfileError(ValueError):
    """The Skills section breaks the grammar. The message lists every problem with its line."""


@dataclass(frozen=True)
class ProfileSkills:
    tiers: dict[str, tuple[str, ...]]  # tier -> tokens, in written order
    skills_hash: str  # changes only when the SET of tokens per tier changes -> rescore
    file_hash: str  # the whole file's bytes, for tracing only
    last_updated: str | None
    warnings: tuple[str, ...] = field(default=())

    def tokens(self) -> list[tuple[str, str]]:
        """Every ``(token, tier)``, Strong first, each tier in written order."""
        return [(t, tier) for tier in TIERS for t in self.tiers[tier]]


def skills_hash(tiers: dict[str, tuple[str, ...]]) -> str:
    """sha256 of ``strong:a,b|working:c|familiar:d`` with each tier's tokens SORTED, so reordering
    tokens or editing prose outside the section never triggers a rescore."""
    canonical = "|".join(f"{tier}:{','.join(sorted(tiers[tier]))}" for tier in TIERS)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_skills(path: Path | str = DEFAULT_PROFILE_PATH) -> ProfileSkills:
    raw = Path(path).read_bytes()
    lines = raw.decode("utf-8").splitlines()
    errors: list[str] = []

    def reject(lineno: int, rule: int, why: str) -> None:
        errors.append(f"line {lineno}: rule {rule}: {why}")

    headers = [i for i, line in enumerate(lines, start=1) if _HEADER_RE.match(line)]
    if len(headers) != 1:
        where = ", ".join(f"line {i}" for i in headers) if headers else "none found"
        raise ProfileError(f"{path}: rule 1: need exactly one line starting '## Skills' ({where})")
    header = headers[0]

    tiers: dict[str, list[str]] = {}
    tier_line: dict[str, int] = {}
    token_tier: dict[str, tuple[str, int]] = {}
    pending: tuple[str, int] | None = None  # a tier label still waiting for its token line
    not_claimed_at: int | None = None

    for lineno in range(header + 1, len(lines) + 1):
        line = lines[lineno - 1]
        if _SECTION_END_RE.match(line):
            break
        text = line.strip()
        if not text:
            continue
        label = _LABEL_RE.match(text)
        if label:
            if pending is not None:
                reject(pending[1], 3, f"the {pending[0].title()} label has no token line before line {lineno}")
            tier = label.group(1).lower()
            if tier in tier_line:
                reject(lineno, 2, f"{tier.title()} appears again (first at line {tier_line[tier]})")
                pending = None
                continue
            tier_line[tier] = lineno
            pending = (tier, lineno)
            continue
        if text.startswith(_NOT_CLAIMED):
            if pending is not None:
                reject(pending[1], 3, f"the {pending[0].title()} label has no token line before line {lineno}")
                pending = None
            if not_claimed_at is not None:
                reject(lineno, 7, f"'{_NOT_CLAIMED}' appears again (first at line {not_claimed_at})")
            not_claimed_at = not_claimed_at or lineno
            continue
        if _ANY_LABEL_RE.match(text):
            reject(lineno, 6, f"unknown label {text!r} (only Strong, Working and Familiar)")
            continue
        if pending is None:
            if all(_TOKEN_RE.match(t.strip()) for t in text.split(",")):
                reject(lineno, 3, "a token line must come right after a tier label (one token line per tier)")
            else:
                reject(lineno, 6, f"unexpected line in the Skills section: {text[:60]!r}")
            continue
        tier, _ = pending
        pending = None
        parts = [t.strip() for t in text.split(",")]
        tokens: list[str] = []
        for t in parts:
            if not t:
                reject(lineno, 4, "empty token (a doubled or trailing comma)")
            elif not _TOKEN_RE.match(t):
                reject(lineno, 4, f"token {t!r} doesn't match [a-z0-9][a-z0-9+#./-]*")
            elif t in token_tier:
                other, at = token_tier[t]
                reject(lineno, 5, f"token {t!r} is already listed under {other.title()} (line {at})")
            else:
                token_tier[t] = (tier, lineno)
                tokens.append(t)
        tiers[tier] = tokens

    if pending is not None:
        reject(pending[1], 3, f"the {pending[0].title()} label has no token line")
    for tier in TIERS:
        if tier not in tier_line:
            reject(header, 2, f"the {tier.title()} tier is missing (all three are required)")
    if errors:
        raise ProfileError(f"{path}: {len(errors)} problem(s) in the Skills section\n  " + "\n  ".join(errors))

    last_updated = None
    for line in lines:
        m = _LAST_UPDATED_RE.match(line)
        if m:
            last_updated = m.group(1).strip().strip("*").strip() or None
            break
    warnings = () if last_updated else ("no 'Last updated:' line found -- scoring status can't show the export date",)

    frozen = {tier: tuple(tiers.get(tier, ())) for tier in TIERS}
    return ProfileSkills(
        tiers=frozen,
        skills_hash=skills_hash(frozen),
        file_hash=hashlib.sha256(raw).hexdigest(),
        last_updated=last_updated,
        warnings=warnings,
    )
