"""Ashby provider -- public posting-api JSON, no key.

Ported from career-ops ``providers/ashby.mjs``. Ashby's public API has a latency
floor and rate-limits repeated unauthenticated hits, so this provider passes a
longer per-request timeout. A 429 still becomes a cooldown (docs/sourcing.md
sec. 5), not a retry. The list endpoint ships ``descriptionPlain`` for free.
"""

from __future__ import annotations

import logging
import re

from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import (
    salary_range,
    assert_allowed_url,
    clean,
    infer_remote,
    iso_date,
    work_mode,
)

log = logging.getLogger("hiredra.sourcing.ashby")

_ALLOWED_HOSTS = frozenset({"api.ashbyhq.com"})
_CAREERS_RE = re.compile(r"jobs\.ashbyhq\.com/([^/?#]+)", re.I)
_API = "https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true"
_TIMEOUT_S = 30.0


def _fold_locations(job: dict) -> str | None:
    """``location`` + ``secondaryLocations[]`` (label + address), deduped, ``" · "``."""
    parts: list[str] = []

    def add(value: object) -> None:
        c = clean(value)
        if c and c.lower() not in {p.lower() for p in parts}:
            parts.append(c)

    add(job.get("location"))
    secondary = job.get("secondaryLocations")
    if isinstance(secondary, list):
        for entry in secondary:
            if not isinstance(entry, dict):
                continue
            add(entry.get("location"))
            address = entry.get("address")
            postal = address.get("postalAddress") if isinstance(address, dict) else None
            if isinstance(postal, dict):
                add(postal.get("addressLocality"))
                add(postal.get("addressCountry"))
    return " · ".join(parts) or None


def _salary(job: dict) -> str | None:
    comp = job.get("compensation")
    if not isinstance(comp, dict):
        return None
    for key in ("compensationTierSummary", "scrapeableCompensationSalarySummary"):
        summary = clean(comp.get(key))
        if summary:
            return summary
    return None


def _salary_range(job: dict) -> dict:
    """The ``Salary`` component of ``compensation.summaryComponents`` (equity and bonus are not pay) as the Job
    salary fields."""
    comp = job.get("compensation")
    parts = comp.get("summaryComponents") if isinstance(comp, dict) else None
    for part in parts if isinstance(parts, list) else ():
        if isinstance(part, dict) and part.get("compensationType") == "Salary":
            return salary_range(
                part.get("minValue"), part.get("maxValue"), part.get("currencyCode"), part.get("interval")
            )
    return {}


class AshbyProvider:
    name = "ashby"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        if entry.api:
            try:
                assert_allowed_url(entry.api, _ALLOWED_HOSTS, self.name)
            except FetchError:
                return None
            return entry.api
        slug = entry.extra.get("slug") if entry.extra else None
        if not slug and entry.careers_url:
            m = _CAREERS_RE.search(entry.careers_url)
            slug = m.group(1) if m else None
        return _API.format(slug=slug) if slug else None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        base = self.board_url(entry)
        if base is None:
            raise FetchError(f"ashby: cannot resolve a board for {entry.name!r}")
        assert_allowed_url(base, _ALLOWED_HOSTS, self.name)

        payload = ctx.http.get_json(base, timeout=_TIMEOUT_S)
        raw_jobs = payload.get("jobs") if isinstance(payload, dict) else None
        if not isinstance(raw_jobs, list):
            return []

        jobs: list[Job] = []
        for j in raw_jobs:
            if not isinstance(j, dict):
                continue
            job_id = j.get("id")
            listing_url = j.get("jobUrl")
            title = clean(j.get("title"))
            if not job_id or not listing_url or not title:
                log.debug("ashby: skipping malformed posting %r", job_id)
                continue
            apply_raw = j.get("applyUrl")
            apply_url = apply_raw if apply_raw and apply_raw != listing_url else None
            location = _fold_locations(j)
            try:
                jobs.append(
                    Job(
                        job_id=f"ashby:{job_id}",
                        url=listing_url,
                        title=title,
                        company=entry.name,
                        source=self.name,
                        description=clean(j.get("descriptionPlain")),
                        location=location,
                        date_posted=iso_date(j.get("publishedAt")),
                        job_type=clean(j.get("employmentType")),
                        salary=_salary(j),
                        apply_url=apply_url,
                        work_mode=work_mode(j.get("workplaceType")),
                        **_salary_range(j),
                        is_remote=infer_remote(
                            f"{location or ''} {title}",
                            workplace_type=j.get("workplaceType"),
                        ),
                    )
                )
            except (ValueError, TypeError) as err:
                log.warning("ashby: skipping posting %s -- %s", job_id, err)
        return jobs
