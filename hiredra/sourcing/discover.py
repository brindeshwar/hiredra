"""Finding a company's board by guessing its slug and asking the ATS (decision D2.2; build slices 2.4 and 2.6).
Ported in idea from freehire's ``cmd/harvest-boards`` (MIT, see THIRD_PARTY_NOTICES.md): a candidate slug is valid
only if the live ATS answers for it.

Greenhouse, Lever and Ashby all answer an unknown board with a 404 and a real one with 200 (measured 2026-10-31),
so a probe is one ordinary fetch through the provider a run uses. The jobs it returns are handed back, so the caller
can measure a board's worth without fetching it twice.

The YC directory (slice 2.4) is read from yc-oss's public JSON, which states no licence: it is used only on the
owner's machine as a list of facts (name, website, hiring regions) and is never committed or republished
(owner decision, 2026-10-31).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.identity import company_slug
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext, Provider

#: Probe order: YC startups use Ashby most, then Greenhouse, then Lever. The first board found wins.
PROBE_ORDER = ("ashby", "greenhouse", "lever")
#: The YC regions that can hold an India- or remote-eligible job.
YC_REGIONS = frozenset({"India", "South Asia", "Remote", "Fully Remote", "Partly Remote"})


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    slugs: tuple[str, ...]  # guesses, best first
    website: str | None = None
    team_size: int | None = None


@dataclass(slots=True)
class Probe:
    candidate: Candidate
    provider: str | None = None  # None: no board found
    slug: str | None = None
    jobs: list[Job] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # anything but a 404, per guess


def slug_guesses(name: str, hints: Iterable[str] = ()) -> tuple[str, ...]:
    """The board slugs to try for a company: the hints first (e.g. YC's own slug), then the company key squashed
    ("Together AI" -> ``togetherai``) and hyphenated (``together-ai``). Lowercase, no duplicates."""
    key = company_slug(name)
    out = [h.strip().lower() for h in hints if h and h.strip()]
    out += [key.replace("-", ""), key]
    return tuple(dict.fromkeys(s for s in out if s))


def yc_candidates(path: Path | str, known: set[str]) -> list[Candidate]:
    """yc-oss ``companies/all.json`` entries that are hiring in an India- or remote-eligible region, minus companies
    whose squashed key is in ``known`` (they already have a board)."""
    out = []
    for c in json.loads(Path(path).read_text(encoding="utf-8")):
        if not c.get("isHiring") or not YC_REGIONS & set(c.get("regions") or ()):
            continue
        name = (c.get("name") or "").strip()
        if not name:
            continue
        if {company_slug(name).replace("-", ""), (c.get("slug") or "").replace("-", "")} & known:
            continue
        out.append(
            Candidate(
                name=name,
                slugs=slug_guesses(name, [c.get("slug") or ""]),
                website=c.get("website"),
                team_size=c.get("team_size") or None,
            )
        )
    return out


#: Below this many jobs a board can't out-hire a startup convincingly, so the team-size guard stays off (freehire's
#: ``collisionJobFloor`` idea in ``cmd/import-yc``).
HOMONYM_JOB_FLOOR = 5


def _host(url: str | None) -> str | None:
    host = urlsplit(url or "").hostname or ""
    host = host.removeprefix("www.")
    return host or None


def confirm(result: Probe) -> str:
    """Is the board found really this company's? A guessed slug can belong to a namesake ("Apollo").

    ``domain``  the company's own web host appears in a job's URL or text: the board is theirs.
    ``name``    only the company's name appears, and the board lists no more jobs than the company has people
                (freehire's homonym guard): probably theirs.
    ``none``    an empty board (nothing to check), a board out-hiring the startup, or no mention at all.
    """
    jobs = result.jobs
    if not jobs:
        return "none"
    cand = result.candidate
    texts = [f"{j.url} {getattr(j, 'apply_url', '') or ''} {j.description or ''}".lower() for j in jobs]
    host = _host(cand.website)
    if host and any(host in t for t in texts):
        return "domain"
    size = cand.team_size or 0
    if size and len(jobs) > max(size, HOMONYM_JOB_FLOOR):
        return "none"
    name = cand.name.lower()
    return "name" if any(name in t for t in texts) else "none"


def probe(
    candidate: Candidate,
    providers: Sequence[Provider],
    ctx: FetchContext,
    *,
    sleep: Callable[[float], None] = lambda s: None,
    pause: float = 0.0,
) -> Probe:
    """Try each guess on each provider in ``PROBE_ORDER`` until one answers 200. A 404 means "not here"; anything
    else is recorded and the search goes on (a failed guess must not hide a board under the next one)."""
    result = Probe(candidate)
    by_name = {p.name: p for p in providers}
    for name in PROBE_ORDER:
        provider = by_name.get(name)
        if provider is None:
            continue
        for slug in candidate.slugs:
            sleep(pause)
            entry = CompanyEntry(name=candidate.name, provider=name, extra={"slug": slug})
            try:
                jobs = provider.fetch(entry, ctx)
            except HTTPError as err:
                if err.status != 404:
                    result.errors.append(f"{name}/{slug}: HTTP {err.status}")
                continue
            except FetchError as err:
                result.errors.append(f"{name}/{slug}: {err}")
                continue
            result.provider, result.slug, result.jobs = name, slug, list(jobs)
            return result
    return result


def known_folds(rows: Iterable[Any]) -> set[str]:
    """Squashed company keys that already have a board (``slug`` per row)."""
    return {str(r[0]).replace("-", "") for r in rows}
