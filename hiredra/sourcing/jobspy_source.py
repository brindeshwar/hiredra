"""Path 3 -- the JobSpy aggregator source (docs/sourcing.md sec. 1, 5, 9).

NOT a ``Provider``: it is company-agnostic (a search, not a per-company board),
takes a search config rather than a ``CompanyEntry``, and is not in the
registry. The orchestrator (sub-slice 8) runs it in parallel with the registry
loop. It produces the same ``Job`` objects.

Design ported from the owner's Track 1 service (reference/jobspy-service-track1):
the per-``(term, site)`` loop with per-pair fault isolation, and the production
defaults (``hours_old=336``, ``results_wanted=20``, ``country_indeed="India"``,
``linkedin_fetch_description=True``). What Track 1 did in-service and Hiredra
does NOT do here: title/seniority filtering (-> sub-slice 2b funnel) and dedup
(-> sub-slice 7). See docs/repos.md.

Hiredra additions over Track 1: blocked-vs-empty detection (Track 1 swallowed
exceptions and read a silent block as ``count: 0``) and cross-run per-site
cooldowns.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from hiredra.sourcing.http import Blocked
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import FetchContext
from hiredra.sourcing.providers._common import clean, infer_remote, iso_date

log = logging.getLogger("hiredra.sourcing.jobspy")

# Sites whose blocks are opaque and long (hours-days), per docs/sourcing.md sec. 5.
_LONG_COOLDOWN_SITES = frozenset({"glassdoor", "naukri", "bayt", "bdjobs"})

# JobSpy per-scraper logger names (JobSpy:LinkedIn, ...); propagate is False, so
# we attach directly. getLogger() is idempotent -- safe before jobspy imports.
_JOBSPY_LOGGERS = tuple(
    f"JobSpy:{s}" for s in ("LinkedIn", "Indeed", "Google", "Glassdoor", "Naukri", "ZipRecruiter", "Bayt", "BDJobs")
)

# A JobSpy log line (or exception text) that means "the board is blocking us".
# Note: `captcha` has no leading boundary on purpose -- it must match "recaptcha"
# (Naukri returns HTTP 406 `{"message":"recaptcha required"}`).
_BLOCK_RE = re.compile(
    r"status code:?\s*(?:401|403|406|407|409|429|5\d\d)"
    r"|\bblocked\b|\bforbidden\b|\bunauthoriz\w*"
    r"|captcha"
    r"|rate.?limit|\btoo many requests\b|\baccess denied\b"
    r"|\bchallenge\b|\bbad proxy\b|\bproxy error\b|\bcloudflare\b",
    re.I,
)


# ── config ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True, kw_only=True)
class SiteQuery:
    """The search parameters for one tier of sites."""

    search_terms: tuple[str, ...]
    location: str | None = "India"
    is_remote: bool = False
    job_type: str | None = None  # "fulltime" | "internship" | "contract" | ...
    hours_old: int | None = 336  # Track 1 default (14 days)
    results_per_site: int = 20  # Track 1 default
    distance: int | None = None  # radius in miles/km (LinkedIn/Indeed)
    google_search_term: str | None = None  # when set, google scrapes once with it


# Minimal fallbacks only. The real, curated term lists live in
# `config/scoring.yml` and are loaded into JobSpyConfig by the config loader.
_DEFAULT_QUERY = SiteQuery(search_terms=("Software Engineer",))

# LinkedIn caps at ~10 pages / a few hundred results before throttling, so it
# gets its own tighter query in config/scoring.yml; this is only the fallback.
_LINKEDIN_QUERY = SiteQuery(
    search_terms=("Software Engineer",),
    job_type="fulltime",
    hours_old=168,
    results_per_site=25,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class JobSpyConfig:
    sites: tuple[str, ...] = ("indeed", "linkedin", "google")
    default_query: SiteQuery = _DEFAULT_QUERY
    linkedin_query: SiteQuery | None = _LINKEDIN_QUERY  # overrides default_query for LinkedIn only
    country_indeed: str = "India"
    linkedin_fetch_description: bool = True  # LinkedIn needs the body for the seniority gate (2b)
    between_sites_delay: float = 3.0
    cooldown_short: float = 1800.0  # LinkedIn/Indeed/Google block -> 30 min
    cooldown_long: float = 21600.0  # Glassdoor/Naukri block -> 6 h
    proxies: tuple[str, ...] = ()  # wired, empty = OFF (decisions.md #20)
    verbose: int = 2


# ── results ────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class SiteOutcome:
    site: str
    status: str  # "ok" | "empty" | "blocked" | "cooldown" | "error"
    jobs: int
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class JobSpyResult:
    jobs: list[Job] = field(default_factory=list)
    outcomes: tuple[SiteOutcome, ...] = ()


# ── log capture ────────────────────────────────────────────────────────────
class _RecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.messages.append(record.getMessage())
        except Exception:  # noqa: BLE001 -- a broken record must not break a scrape
            pass


@contextmanager
def _capture_jobspy_logs() -> Iterator[_RecordCollector]:
    collector = _RecordCollector()
    saved: list[tuple[logging.Logger, int]] = []
    for name in _JOBSPY_LOGGERS:
        logger = logging.getLogger(name)
        saved.append((logger, logger.level))
        logger.addHandler(collector)
        logger.setLevel(logging.DEBUG)
    try:
        yield collector
    finally:
        for logger, level in saved:
            logger.removeHandler(collector)
            logger.setLevel(level)


def _block_marker(messages: list[str]) -> str | None:
    for msg in messages:
        if _BLOCK_RE.search(msg):
            return msg
    return None


# ── row -> Job ─────────────────────────────────────────────────────────────
def _salary(row: dict) -> str | None:
    def num(value: object) -> str | None:
        try:
            f = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return None if f != f else f"{f:,.0f}"  # f != f -> NaN

    low, high = num(row.get("min_amount")), num(row.get("max_amount"))
    if not low and not high:
        return None
    rng = f"{low}-{high}" if low and high else (low or high)
    ccy = clean(row.get("currency"))
    interval = clean(row.get("interval"))
    return " ".join(p for p in (ccy, rng, f"/{interval}" if interval else "") if p) or None


def _row_to_job(row: dict, site: str) -> Job | None:
    url = clean(row.get("job_url"))
    title = clean(row.get("title"))
    company = clean(row.get("company"))
    if not url or not title or not company:
        return None

    direct = clean(row.get("job_url_direct"))
    apply_url = direct if direct and direct != url else None
    location = clean(row.get("location"))

    raw_remote = row.get("is_remote")
    remote: bool | None
    if raw_remote is not None and bool(raw_remote):
        remote = True  # JobSpy's True is trusted
    else:
        remote = infer_remote(f"{location or ''} {title}")  # its False/NaN is not

    job_id_raw = row.get("id")
    try:
        return Job(
            job_id=f"jobspy-{site}:{job_id_raw}" if job_id_raw else "",
            url=url,
            title=title,
            company=company,
            source=f"jobspy-{site}",
            description=clean(row.get("description")),
            location=location,
            date_posted=iso_date(row.get("date_posted")),
            job_type=clean(row.get("job_type")),
            salary=_salary(row),
            apply_url=apply_url,
            is_remote=remote,
        )
    except (ValueError, TypeError) as err:
        log.warning("jobspy: skipping %s row -- %s", site, err)
        return None


def _frames_to_jobs(frames: list, site: str) -> list[Job]:
    if not frames:
        return []
    import pandas as pd

    combined = pd.concat(frames, ignore_index=True)
    # cast to object first, else None written back into a float/datetime column
    # silently becomes NaN/NaT again (pandas dtype coercion).
    combined = combined.astype(object).where(pd.notna(combined), None)
    jobs: list[Job] = []
    for row in combined.to_dict("records"):
        job = _row_to_job(row, site)
        if job is not None:
            jobs.append(job)
    return jobs


# ── the source ─────────────────────────────────────────────────────────────
class JobSpySource:
    """Config-driven. ``JobSpySource(config).fetch(ctx) -> JobSpyResult``."""

    name = "jobspy"

    def __init__(
        self,
        config: JobSpyConfig | None = None,
        *,
        sleep=None,
        scrape=None,
    ) -> None:
        self.config = config or JobSpyConfig()
        self._sleep = sleep if sleep is not None else time.sleep
        self._injected_scrape = scrape

    @property
    def _scrape(self):
        if self._injected_scrape is not None:
            return self._injected_scrape
        try:
            from jobspy import scrape_jobs
        except ImportError as err:  # pragma: no cover
            raise RuntimeError("python-jobspy is not installed (run `uv sync`)") from err
        return scrape_jobs

    # -----------------------------------------------------------------
    def fetch(self, ctx: FetchContext) -> JobSpyResult:
        if not self.config.sites:
            raise ValueError("JobSpyConfig.sites is empty")

        all_jobs: list[Job] = []
        outcomes: list[SiteOutcome] = []
        for i, site in enumerate(self.config.sites):
            if i:
                self._sleep(self.config.between_sites_delay)
            key = f"jobspy-{site}"
            try:
                ctx.http.cooldowns.check(key)
            except Blocked as blk:
                outcomes.append(SiteOutcome(site, "cooldown", 0, str(blk)))
                continue

            query = (
                self.config.linkedin_query
                if site == "linkedin" and self.config.linkedin_query is not None
                else self.config.default_query
            )
            jobs, outcome = self._scrape_site(site, query, ctx)
            all_jobs.extend(jobs)
            outcomes.append(outcome)

        return JobSpyResult(jobs=all_jobs, outcomes=tuple(outcomes))

    # -----------------------------------------------------------------
    def _terms_for(self, site: str, query: SiteQuery) -> list[str]:
        if site == "google" and query.google_search_term:
            return [query.google_search_term]  # one call; JobSpy uses google_search_term
        return list(query.search_terms)

    def _params(self, site: str, term: str, query: SiteQuery) -> dict:
        params: dict = {
            "site_name": [site],
            "search_term": term,
            "location": query.location,
            "results_wanted": query.results_per_site,
            "country_indeed": self.config.country_indeed,
            "verbose": self.config.verbose,
            "description_format": "markdown",
        }
        if query.hours_old is not None:
            params["hours_old"] = query.hours_old
        if query.is_remote:
            params["is_remote"] = True
        if query.job_type:
            params["job_type"] = query.job_type
        if query.distance is not None:
            params["distance"] = query.distance
        if site == "google" and query.google_search_term:
            params["google_search_term"] = query.google_search_term
        if site == "linkedin" and self.config.linkedin_fetch_description:
            params["linkedin_fetch_description"] = True
        if self.config.proxies:
            params["proxies"] = list(self.config.proxies)
        return params

    def _trip(self, ctx: FetchContext, site: str, reason: str) -> None:
        seconds = self.config.cooldown_long if site in _LONG_COOLDOWN_SITES else self.config.cooldown_short
        ctx.http.cooldowns.trip(f"jobspy-{site}", seconds, reason=reason[:200])
        log.warning("jobspy: %s blocked -- cooling down %.0fs", site, seconds)

    def _scrape_site(self, site: str, query: SiteQuery, ctx: FetchContext) -> tuple[list[Job], SiteOutcome]:
        frames: list = []
        errors: list[str] = []

        for term in self._terms_for(site, query):
            with _capture_jobspy_logs() as cap:
                try:
                    df = self._scrape(**self._params(site, term, query))
                except Exception as err:  # noqa: BLE001 -- fail-soft per sec. 5
                    text = f"{term}: {err}"
                    if _BLOCK_RE.search(str(err)):
                        self._trip(ctx, site, text)
                        jobs = _frames_to_jobs(frames, site)
                        return jobs, SiteOutcome(site, "blocked", len(jobs), text)
                    errors.append(text)
                    continue

            if df is not None and not getattr(df, "empty", True):
                frames.append(df)

            marker = _block_marker(cap.messages)
            if marker:
                self._trip(ctx, site, marker)
                jobs = _frames_to_jobs(frames, site)
                return jobs, SiteOutcome(site, "blocked", len(jobs), marker)

        jobs = _frames_to_jobs(frames, site)
        if jobs:
            return jobs, SiteOutcome(site, "ok", len(jobs), "; ".join(errors) or None)
        if errors:
            return jobs, SiteOutcome(site, "error", 0, "; ".join(errors))
        return jobs, SiteOutcome(site, "empty", 0, None)
