"""Sub-slice 2b: filter funnel (build step 2b, docs/scoring.md sec 1).

Location eligibility, then the seniority gate, then role-type filter, then
liveness (deferred to its own sub-slice). Location runs first -- on a global
intake it removes the largest share, and CLAUDE.md sec 3.5 says filter hard
and early. Reads jobs at status='fetched'; survivors of all three stages
advance to 'filtered', failures move to 'dropped' with the killing stage
recorded in status_detail. Code only -- no LLM call anywhere here.

    python -m hiredra.filter [--db data/hiredra.db] [--config config/filters.yml]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from hiredra.location import eligibility, parse
from hiredra.storage import (
    DEFAULT_DB_PATH,
    add_note,
    connect,
    init_db,
    iter_jobs,
    set_status,
    set_years_required,
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "filters.yml"

# "N years" / "N+ yrs" / "N-M years" -- takes the max of every match, per job.
_YEARS_RE = re.compile(r"(\d+)\s*\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs?)", re.IGNORECASE)

# Neutralized before generic senior-term matching so a bare "staff" term
# catches "Staff Engineer" without also catching this generic AI-lab IC title.
_MTS_RE = re.compile(r"\bmember of (?:the )?technical staff\b|\bmts\b", re.IGNORECASE)

# Indian-market boilerplate: "Educational Qualification: 15 years full time
# education" means "has a bachelor's degree," not 15 years of work experience.
# A years match is excluded when "educational qualification" precedes it or
# "full time education" follows it, each within a short window -- checked
# directionally (not one wide window centered on the match) so a genuine
# "N years of experience" match earlier in the same posting (e.g. "Minimum 3
# Year(s) Of Experience Is Required ... Educational Qualification: 15 years
# full time education") isn't also caught by the boilerplate a few dozen
# characters downstream. This is deliberately narrow (two specific phrases),
# not general experience-context anchoring (deferred to calibration).
_EDU_QUALIFICATION_RE = re.compile(r"educational qualification", re.IGNORECASE)
_FULL_TIME_EDUCATION_RE = re.compile(r"full[- ]?time education", re.IGNORECASE)
_EDUCATION_WINDOW = 45


@dataclass(frozen=True)
class FilterConfig:
    allow: tuple[str, ...]
    block: tuple[str, ...]
    senior_title_terms: tuple[str, ...]
    anti_signal_phrases: tuple[str, ...]
    max_years: int
    location_eligible_for: str


def load_filter_config(path: Path | str = DEFAULT_CONFIG_PATH) -> FilterConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    location = data.get("location") or {}
    return FilterConfig(
        allow=tuple(data.get("allow") or ()),
        block=tuple(data.get("block") or ()),
        senior_title_terms=tuple(data.get("senior_title_terms") or ()),
        anti_signal_phrases=tuple(data.get("anti_signal_phrases") or ()),
        max_years=int(data.get("max_years", 2)),
        location_eligible_for=str(location.get("eligible_for", "India")),
    )


def extract_max_years(text: str | None) -> int | None:
    """Max of every 'N years'/'N+ yrs' match in `text`, or None if none found.

    None means unknown -- never a stand-in for zero (docs/repos.md sec 6).
    A match near "educational qualification" / "full time education" is
    excluded (real example: "Educational Qualification: 15 years full time
    education" -- a bachelor's degree, not 15 years of experience); this is a
    narrow, specific exclusion, not general experience-context anchoring.
    """
    if not text:
        return None
    matches = []
    for m in _YEARS_RE.finditer(text):
        before = text[max(0, m.start() - _EDUCATION_WINDOW) : m.start()]
        after = text[m.end() : m.end() + _EDUCATION_WINDOW]
        if _EDU_QUALIFICATION_RE.search(before) or _FULL_TIME_EDUCATION_RE.search(after):
            continue
        matches.append(int(m.group(1)))
    return max(matches) if matches else None


def _contains_term(text: str, term: str) -> bool:
    """Whole-phrase, case-insensitive membership.

    Not a bare substring: uses non-word lookaround (not \\b) so a term ending
    in punctuation (e.g. "sr.") still requires a real boundary on both sides.
    This is what keeps "forward deployed engineer" from also matching
    "forward deployed engineering", and "agent" from matching "agentic"
    (which stays its own separate term).
    """
    pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


def title_hits(title: str | None, terms: tuple[str, ...]) -> str | None:
    """First term in `terms` that appears in `title` as a whole phrase, else None."""
    title = title or ""
    for term in terms:
        if _contains_term(title, term):
            return term
    return None


def has_senior_title_term(title: str | None, terms: tuple[str, ...]) -> str | None:
    """Same as title_hits, but neutralizes "Member of Technical Staff" / "MTS"
    first, so that title's seniority is decided by the years regex and the
    anti-signal phrases, not by the bare "staff" term."""
    working = _MTS_RE.sub(" ", title or "")
    return title_hits(working, terms)


def has_anti_signal_phrase(text: str | None, phrases: tuple[str, ...]) -> str | None:
    """First phrase in `phrases` found in `text`, else None."""
    return title_hits(text, phrases)


@dataclass(frozen=True)
class LocationResult:
    keep: bool
    reason: str | None  # drop reason if not keep; "unknown" if kept with no location signal; None if kept on a definite match
    note: str | None = None  # a job_events note to record regardless of keep/drop (e.g. onsite/hybrid with no stated place)


def passes_location(
    title: str | None, location: str | None, description: str | None, config: FilterConfig
) -> LocationResult:
    """Is this job workable from ``config.location_eligible_for``?

    A job with no location signal at all (``eligibility`` returns ``None``) is
    KEPT, not dropped -- same "never guess" posture as the seniority gate's
    unknown years. Callers can tell the two keep cases apart via `reason`:
    ``"unknown"`` (no signal, kept by default) vs ``None`` (a definite match).

    Every "unknown" keep gets a `job_events` note, so a later consumer (e.g.
    the discovery harvest's confirmed-vs-unknown-only company tagging) can
    tell a definite match from a kept-by-default one without re-deriving the
    decision. A specific unknown shape -- the posting says "Hybrid"/
    "In-Office" (so we know it's NOT remote) but names no city or country at
    all -- gets its own more specific note: the owner wants these retained
    and distinguishable, not lost or silently folded into the ordinary
    "unknown" bucket.
    """
    ok, reason = eligibility(config.location_eligible_for, location, description or "", title or "")
    note = None
    if ok is None:
        parsed = parse(location or "", description or "", title or "")
        # Checking countries only (not cities too): parse()'s city fallback
        # can catch the arrangement word itself as a fake placeholder "city"
        # (e.g. "In-Office" -> cities=["In-Office"]), which would otherwise
        # hide a genuinely place-less posting from this flag.
        if parsed["remote"] in ("onsite", "hybrid") and not parsed["countries"]:
            note = f"location: {parsed['remote']}, place unknown"
        else:
            note = "location: unknown"
        return LocationResult(True, "unknown", note)
    if ok is False:
        return LocationResult(False, reason)
    return LocationResult(True, None)


@dataclass(frozen=True)
class SeniorityResult:
    keep: bool
    years: int | None
    reason: str | None  # None iff keep


def passes_seniority_gate(
    title: str | None, description: str | None, config: FilterConfig
) -> SeniorityResult:
    """Gate only -- not a score term. `years` is always populated when
    extractable, even when a title term or anti-signal phrase is what
    actually drops the job."""
    years = extract_max_years(description)

    term = has_senior_title_term(title, config.senior_title_terms)
    if term:
        return SeniorityResult(False, years, f"title term '{term}'")

    phrase = has_anti_signal_phrase(description, config.anti_signal_phrases)
    if phrase:
        return SeniorityResult(False, years, f"anti-signal phrase '{phrase}'")

    if years is not None and years > config.max_years:
        return SeniorityResult(False, years, f"requires {years}y (max {config.max_years})")

    return SeniorityResult(True, years, None)


@dataclass(frozen=True)
class RoleTypeResult:
    keep: bool
    reason: str | None


def passes_role_type(title: str | None, config: FilterConfig) -> RoleTypeResult:
    """Title only, no JD-body fallback. BLOCK wins if a title hits both."""
    block_term = title_hits(title, config.block)
    if block_term:
        return RoleTypeResult(False, f"block term '{block_term}'")
    allow_term = title_hits(title, config.allow)
    if allow_term:
        return RoleTypeResult(True, None)
    return RoleTypeResult(False, "no allow term matched")


@dataclass
class FunnelStats:
    seen: int = 0
    dropped_location: int = 0
    dropped_seniority: int = 0
    dropped_role_type: int = 0
    kept: int = 0
    unknown_location: int = 0  # of `kept` + drops past this stage: how many had no location signal at all
    onsite_unknown_place: int = 0  # subset of unknown_location: "Hybrid"/"In-Office" but no city/country stated


def run_funnel(conn, config: FilterConfig) -> FunnelStats:
    """Read status='fetched', run the three text stages, advance/drop.

    Liveness is deferred to its own sub-slice -- a survivor here lands at
    'filtered', not (yet) a liveness-checked status.
    """
    stats = FunnelStats()
    # Materialized up front: writes below change `status`, and iterating a
    # cursor while mutating the table it's reading from is asking for trouble.
    # Fine at this scale (hundreds-to-thousands of rows per CLAUDE.md sec 3.5).
    jobs = list(iter_jobs(conn, status="fetched"))
    for job in jobs:
        stats.seen += 1
        job_id = job["job_id"]

        location = passes_location(job["title"], job["location"], job["description"], config)
        if location.reason == "unknown":
            stats.unknown_location += 1
        if location.note:
            add_note(conn, job_id, location.note)
            if location.note.endswith("place unknown"):  # the onsite/hybrid sub-case specifically,
                stats.onsite_unknown_place += 1           # not every "location: unknown" note
        if not location.keep:
            set_status(conn, job_id, "dropped", detail=f"location: {location.reason}")
            stats.dropped_location += 1
            continue

        seniority = passes_seniority_gate(job["title"], job["description"], config)
        set_years_required(conn, job_id, seniority.years)
        if not seniority.keep:
            set_status(conn, job_id, "dropped", detail=f"seniority: {seniority.reason}")
            stats.dropped_seniority += 1
            continue

        role = passes_role_type(job["title"], config)
        if not role.keep:
            set_status(conn, job_id, "dropped", detail=f"role_type: {role.reason}")
            stats.dropped_role_type += 1
            continue

        set_status(conn, job_id, "filtered")
        stats.kept += 1

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hiredra.filter")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args(argv)

    config = load_filter_config(args.config)
    conn = connect(args.db)
    try:
        init_db(conn)
        stats = run_funnel(conn, config)
    finally:
        conn.close()

    print(
        f"seen={stats.seen} kept={stats.kept} "
        f"dropped_location={stats.dropped_location} "
        f"dropped_seniority={stats.dropped_seniority} "
        f"dropped_role_type={stats.dropped_role_type} "
        f"unknown_location={stats.unknown_location} "
        f"onsite_unknown_place={stats.onsite_unknown_place}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
