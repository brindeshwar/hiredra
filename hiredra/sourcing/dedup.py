"""Dedup -- collapse the combined sourcing pile, reconcile it against the store.

Runs on the in-memory pile from one sourcing run (registry providers + JobSpy),
BEFORE persist (docs/sourcing.md sec. 8-9). Two stages:

1. **intra-run collapse** -- group by normalized url, then by
   (normalized company, normalized role);
2. **cross-run reconciliation** -- match survivors against EVERY stored job
   (all statuses, including ``dropped``), so a job the funnel already rejected
   is never re-processed.

Layered keys are **exact-match on normalized strings; no fuzzy matching**. A
false merge loses a real job; a missed near-dup just shows twice in the batch
of 10 -- so the bias is deliberately toward under-merging. On any collision the
ATS row wins over an aggregator row (docs/sourcing.md sec. 8).

The (company, role) key alone is not identity (2026-10-27, measured on 675 boards
that post in India: it merged 24% of 49,906 jobs and threw away 624 India-eligible
postings for a foreign copy of the same title, before the location filter ran).
So a (company, role) match only merges two jobs that are the same role:
- from different sources (an aggregator copy of a board's posting, or two
  aggregators) -- two postings on one ATS board are two jobs by the board's own
  ids, and a true duplicate there already shares a url;
- in the same countries, as ``hiredra.location.parse`` reads them -- a role in
  Bangalore and the same title in London are two jobs to us.
Compare freehire's jobhash/jobdedup (MIT), which never deletes on (company, title)
and clusters reposts only with the description in the key.
"""

from __future__ import annotations

import re
import psycopg
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import parse_qsl, urlsplit

from hiredra.location import parse
from hiredra.sourcing.job import Job

_AGG_PREFIX = "jobspy-"
_WS = re.compile(r"\s+")


# ── result ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class MergeTrace:
    kept: Job
    dropped: Job
    on: str  # "url" | "company+role"


@dataclass(frozen=True, slots=True)
class DedupResult:
    new: list[Job]  # unique this run AND not stored -> persist_jobs
    seen: list[str]  # distinct stored job_ids matched -> touch_job
    collapsed_in_run: int  # dupes removed within this run's pile
    matched_in_store: int  # distinct stored jobs re-matched (== len(seen))
    merges: list[MergeTrace] = field(default_factory=list)  # intra-run, for inspection
    store_matches: list[tuple[Job, str]] = field(default_factory=list)  # (incoming, stored job_id)


# ── url normalization ──────────────────────────────────────────────────────
# ONLY params that are unambiguously tracking. When unsure, keep it -- under-merge
# is the safe direction. (Real corpus of 809 jobs showed only gh_jid / jk / sid,
# all identity or harmless.)
_TRACKING_PARAMS = frozenset(
    {
        "gclid",
        "gclsrc",
        "dclid",
        "fbclid",
        "msclkid",
        "yclid",
        "twclid",
        "mc_cid",
        "mc_eid",
        "_ga",
        "_gl",
        "igshid",
        "ref_src",
        "spm",
        "scid",
    }
)
# ATS host variants that address the same board.
_HOST_CANON = {
    "job-boards.greenhouse.io": "greenhouse.io",
    "boards.greenhouse.io": "greenhouse.io",
    "job-boards.eu.greenhouse.io": "greenhouse.io",
    "jobs.eu.lever.co": "jobs.lever.co",
    "in.linkedin.com": "linkedin.com",
    "in.indeed.com": "indeed.com",
}


def _canon_host(host: str) -> str:
    host = host.lower().removeprefix("www.")
    return _HOST_CANON.get(host, host)


def _norm_url(url: str | None) -> str | None:
    if not url or not isinstance(url, str):
        return None
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower() or None
    host = _canon_host(parts.hostname or "")
    if not host:
        return url.strip().lower() or None
    path = parts.path.rstrip("/") or "/"
    kept = sorted(
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k not in _TRACKING_PARAMS and not k.lower().startswith("utm_")
    )
    query = "&".join(f"{k}={v}" for k, v in kept)
    return f"{host}{path}" + (f"?{query}" if query else "")


# ── company / role normalization ───────────────────────────────────────────
# Trailing LEGAL suffixes only. No "technologies"/"labs"/"solutions" -- those
# can be the real name.
_COMPANY_SUFFIX = re.compile(
    r"[\s,]+(?:inc|incorporated|llc|l\.l\.c|ltd|limited|corp|corporation|co|company"
    r"|gmbh|ag|plc|pte|pvt|private|bv|b\.v|s\.a|s\.a\.s)\.?$",
    re.I,
)
_PUNCT_COMPANY = re.compile(r"[^a-z0-9 ]")
_PUNCT_ROLE = re.compile(r"[^a-z0-9+#/ ]")  # keep + # / for c++, c#, ci/cd

_LOC = (
    r"remote|hybrid|on-?site|wfh|work from home|anywhere|"
    r"india|united states|usa|u\.s\.|uk|u\.k\.|canada|europe|emea|apac|anz|latam|"
    r"bengaluru|bangalore|mumbai|new delhi|delhi|ncr|gurgaon|gurugram|noida|hyderabad"
    r"|pune|chennai|kolkata|ahmedabad|jaipur|kochi|"
    r"new york|nyc|san francisco|sf|bay area|seattle|austin|boston|chicago|denver"
    r"|atlanta|los angeles|"
    r"london|dublin|berlin|munich|amsterdam|paris|madrid|barcelona|lisbon|warsaw|"
    r"singapore|tokyo|hong kong|sydney|melbourne|toronto|vancouver"
)
# A trailing location tag after a separator -- but ONLY when everything after the
# separator is location words (optionally a comma/slash list). "Backend - London
# Markets" is NOT stripped ("Markets" isn't a location); "Backend - Bengaluru,
# India" IS.
_ROLE_LOC_TAIL = re.compile(
    rf"\s*[\-–—|(/,]\s*(?:{_LOC})"
    rf"(?:\s*[,/;&]\s*(?:{_LOC})|\s+(?:and|or)\s+(?:{_LOC}))*\s*\)?\s*$",
    re.I,
)
_ROLE_PAREN_TAG = re.compile(
    r"\s*\(\s*(?:remote|hybrid|on-?site|contract|full[\- ]?time|part[\- ]?time"
    r"|intern(?:ship)?|permanent|temp(?:orary)?|freelance|w2|c2c|"
    r"\d+\s*(?:months?|years?|yrs?))\s*\)",
    re.I,
)


def _norm_company(name: str) -> str:
    s = _WS.sub(" ", str(name).strip().lower())
    for _ in range(4):
        stripped = _COMPANY_SUFFIX.sub("", s).strip(" .,-")
        if stripped == s:
            break
        s = stripped
    return _WS.sub(" ", _PUNCT_COMPANY.sub(" ", s)).strip()


def _norm_role(title: str) -> str:
    s = _WS.sub(" ", str(title).strip().lower())
    s = _ROLE_PAREN_TAG.sub(" ", s)
    s = _ROLE_LOC_TAIL.sub("", s)
    return _WS.sub(" ", _PUNCT_ROLE.sub(" ", s)).strip()


# ── winner selection ("ATS wins" -- sec. 8) ────────────────────────────────
def _is_aggregator(job: Job) -> bool:
    return job.source.startswith(_AGG_PREFIX)


@lru_cache(maxsize=65536)
def _countries(location: str | None, title: str | None) -> frozenset[str]:
    return frozenset(parse(location or "", "", title or "")["countries"])


def _same_role(a_source: str, a_countries: frozenset[str], b_source: str, b_countries: frozenset[str]) -> bool:
    """May two jobs with the same (company, role) key be merged? See the module docstring."""
    one_ats_board = a_source == b_source and not a_source.startswith(_AGG_PREFIX)
    return not one_ats_board and a_countries == b_countries


def _job_countries(job: Job) -> frozenset[str]:
    return _countries(job.location, job.title)


def _completeness(job: Job) -> int:
    return sum(v is not None for v in (job.description, job.date_posted, job.salary, job.location, job.is_remote))


def _pick_winner(a: Job, b: Job) -> Job:
    a_agg, b_agg = _is_aggregator(a), _is_aggregator(b)
    if a_agg != b_agg:
        return a if b_agg else b  # a non-aggregator (ATS) row always wins
    return b if _completeness(b) > _completeness(a) else a


def _merge(current: Job, incoming: Job, on: str, trace: list[MergeTrace]) -> Job:
    winner = _pick_winner(current, incoming)
    loser = incoming if winner is current else current
    trace.append(MergeTrace(kept=winner, dropped=loser, on=on))
    return winner


def _collapse(jobs: list[Job], trace: list[MergeTrace]) -> tuple[list[Job], int]:
    collapsed = 0

    by_url: dict[str, int] = {}
    slots: list[Job] = []
    for job in jobs:
        key = _norm_url(job.url)
        pos = by_url.get(key) if key else None
        if pos is not None:
            slots[pos] = _merge(slots[pos], job, "url", trace)
            collapsed += 1
        else:
            if key:
                by_url[key] = len(slots)
            slots.append(job)

    by_ct: dict[tuple[str, str], list[int]] = {}
    out: list[Job] = []
    for job in slots:
        ct = (_norm_company(job.company), _norm_role(job.title))
        if not ct[0] or not ct[1]:
            out.append(job)
            continue
        here = _job_countries(job)
        pos = next(
            (i for i in by_ct.get(ct, ()) if _same_role(out[i].source, _job_countries(out[i]), job.source, here)), None
        )
        if pos is not None:
            out[pos] = _merge(out[pos], job, "company+role", trace)
            collapsed += 1
        else:
            by_ct.setdefault(ct, []).append(len(out))
            out.append(job)

    return out, collapsed


# ── cross-run index ────────────────────────────────────────────────────────
class _SeenIndex:
    __slots__ = ("by_job_id", "by_url", "by_ct")

    def __init__(self) -> None:
        self.by_job_id: set[str] = set()
        self.by_url: dict[str, str] = {}
        self.by_ct: dict[tuple[str, str], list[tuple[str, str, frozenset[str]]]] = {}

    def match(self, job: Job) -> str | None:
        if job.job_id in self.by_job_id:
            return job.job_id
        for candidate in (_norm_url(job.url), _norm_url(job.apply_url)):
            if candidate and candidate in self.by_url:
                return self.by_url[candidate]
        ct = (_norm_company(job.company), _norm_role(job.title))
        if ct[0] and ct[1]:
            here = _job_countries(job)
            for jid, source, countries in self.by_ct.get(ct, ()):
                if _same_role(source, countries, job.source, here):
                    return jid
        return None


def _load_seen_index(conn: psycopg.Connection) -> _SeenIndex:
    idx = _SeenIndex()
    # Insertion order, as SQLite gave it: the first stored job claims a url / role key (setdefault, first match).
    rows = conn.execute("SELECT job_key, url, apply_url, company, title, source, location FROM jobs ORDER BY id")
    for row in rows:
        jid = row["job_key"]
        idx.by_job_id.add(jid)
        for candidate in (_norm_url(row["url"]), _norm_url(row["apply_url"])):
            if candidate:
                idx.by_url.setdefault(candidate, jid)
        ct = (_norm_company(row["company"]), _norm_role(row["title"]))
        if ct[0] and ct[1]:
            idx.by_ct.setdefault(ct, []).append((jid, row["source"], _countries(row["location"], row["title"])))
    return idx


# ── entry point ────────────────────────────────────────────────────────────
def dedup(conn: psycopg.Connection, jobs: Iterable[Job]) -> DedupResult:
    trace: list[MergeTrace] = []
    unique, collapsed = _collapse(list(jobs), trace)

    idx = _load_seen_index(conn)
    new: list[Job] = []
    seen: list[str] = []
    store_matches: list[tuple[Job, str]] = []
    for job in unique:
        hit = idx.match(job)
        if hit is None:
            new.append(job)
        else:
            seen.append(hit)
            store_matches.append((job, hit))

    seen_unique = list(dict.fromkeys(seen))
    return DedupResult(
        new=new,
        seen=seen_unique,
        collapsed_in_run=collapsed,
        matched_in_store=len(seen_unique),
        merges=trace,
        store_matches=store_matches,
    )
