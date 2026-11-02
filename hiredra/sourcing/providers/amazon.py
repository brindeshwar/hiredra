"""Amazon provider -- amazon.jobs' public search API, no key (build step 3, wave 2).

Ported from freehire's ``internal/ingest/sources/amazon.go`` (MIT, see THIRD_PARTY_NOTICES.md). One employer, so
freehire has no board; ours is a country, Amazon's own ``country`` facet (``IND``). The search stops at 10,000 hits
for any query (measured 2026-11-02: the whole catalogue answers ``hits: 10000``), so the catalogue can only be read
whole a country at a time; India is 2,100. Each page carries 100 postings with their text: no detail request.

One change from freehire: the text is the description AND the basic and preferred qualifications, the three
sections amazon.jobs shows. freehire keeps the description alone, and the qualifications are where Amazon states
the years: of India's 2,100 postings, 1,986 name no years in the description and 1,646 of those name them in the
qualifications, which the seniority gate must read.
"""

from __future__ import annotations

import re
from datetime import datetime

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import clean, infer_remote, log

SEARCH = "https://www.amazon.jobs/en/search.json"
_COUNTRY = re.compile(r"^[A-Z]{3}$")
_PAGE = 100
_CAP = 10_000
_TIMEOUT_S = 40.0
_SECTIONS = (("description", None), ("basic_qualifications", "Basic qualifications"),
             ("preferred_qualifications", "Preferred qualifications"))  # fmt: skip
_JOB_TYPE = {"full-time": "fulltime", "part-time": "parttime"}


def _country(entry: CompanyEntry) -> str:
    country = ((entry.extra or {}).get("slug") or "").strip().upper()
    if not _COUNTRY.match(country):
        raise FetchError(f"amazon: board {country!r} is not a country code (IND)")
    return country


def _posted(value: str | None) -> str | None:
    """``posted_date`` as Amazon writes it ("November  1, 2026") as ``YYYY-MM-DD``."""
    try:
        return datetime.strptime(" ".join((value or "").split()), "%B %d, %Y").date().isoformat()
    except ValueError:
        return None


class AmazonProvider:
    name = "amazon"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        try:
            return f"https://www.amazon.jobs/en/search?country={_country(entry)}"
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return False  # one employer: a board is named explicitly, never recognised from a link

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        country = _country(entry)
        jobs: dict[str, Job] = {}
        offset = 0
        while True:
            url = f"{SEARCH}?result_limit={_PAGE}&offset={offset}&sort=recent&country={country}"
            data = ctx.http.get_json(url, timeout=_TIMEOUT_S)
            page = data.get("jobs") if isinstance(data, dict) else None
            if not isinstance(page, list):
                raise FetchError(f"amazon: {url} has no jobs list", url=url)
            for posting in page:
                if isinstance(posting, dict) and (job := to_job(entry.name, posting)) is not None:
                    jobs.setdefault(job.job_id, job)
            hits = data.get("hits") or 0
            offset += _PAGE
            if not page or offset >= hits:
                if hits >= _CAP:
                    log.warning("amazon %s: %d hits, the search's cap: postings past it are not read", country, hits)
                    ctx.incomplete(f"{hits} hits: the search's cap")
                return list(jobs.values())


def to_job(company: str, posting: dict) -> Job | None:
    title, pid, path = clean(posting.get("title")), clean(posting.get("id_icims")), clean(posting.get("job_path"))
    if not title or not pid or not path:
        return None
    parts = []
    for field, heading in _SECTIONS:
        text = html_to_text(posting.get(field))
        if text:
            parts.append(f"{heading}\n{text}" if heading else text)
    location = clean(posting.get("normalized_location")) or clean(posting.get("location"))
    job_type = (
        "intern" if posting.get("is_intern") is True else _JOB_TYPE.get(clean(posting.get("job_schedule_type")) or "")
    )
    return Job(
        job_id=f"amazon:{pid}",
        url=f"https://www.amazon.jobs{path}",
        title=title,
        company=company,
        source="amazon",
        description="\n\n".join(parts) or None,
        location=location,
        date_posted=_posted(posting.get("posted_date")),
        job_type=job_type,
        is_remote=infer_remote(location or ""),
    )
