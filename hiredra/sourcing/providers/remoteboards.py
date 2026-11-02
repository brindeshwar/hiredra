"""Remote-job boards -- RemoteOK, Remotive, Jobicy, Himalayas, Working Nomads and We Work Remotely (build step 3,
wave 3).

Ported from freehire's ``remoteok.go``, ``remotive.go``, ``jobicy.go``, ``himalayas.go``, ``workingnomads.go`` and
``weworkremotely.go`` (MIT, see THIRD_PARTY_NOTICES.md). Each is one public feed with every posting's text inline (no
detail request) and many employers: each job carries its own company, and the board is the feed, stored with no
company of its own. A feed is the board's recent window, not its whole history, so a posting missing from it is not
closed. Each feed's terms ask for a link back and a modest pace: the job URL is the board's own page, and a run reads
a feed once. Every board lists remote jobs only; the location is where a candidate may live, as the board writes it.

Two changes from freehire, both measured 2026-11-02:
- Jobicy: the documented ``/api/jobs`` (100 a page, a cursor through the 7-day window), not the older
  ``/api/v2/remote-jobs``, which freehire reads once at 50.
- Himalayas: the board is a country, the documented search (``country=IN`` includes worldwide roles open to India:
  6,726), not the whole browse feed (116,708), of which freehire reads the newest 2,400 a run.

These are JSON or RSS feeds, not HTML pages, each allowed by its robots.txt but one: Remotive's disallows ``/api/*``
while its API README grants developers access (one feed, at most 4 reads a day, a link back). The owner chose to read
it (2026-11-02); a run reads it once.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import quote, urlsplit

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import clean, iso_date, log, rss_date, rss_items, salary_range

_TIMEOUT_S = 40.0
_JOB_TYPE = {"fulltime": "fulltime", "parttime": "parttime", "contract": "contract", "contractor": "contract",
             "temporary": "contract", "freelance": "contract", "intern": "intern", "internship": "intern"}  # fmt: skip


def job_type(value: object) -> str | None:
    """A board's employment type ("Full-Time", "Full Time", ["Full-Time"]) as ours; the first one named wins."""
    for item in value if isinstance(value, list) else [value]:
        if found := _JOB_TYPE.get(re.sub(r"[\s_-]", "", str(item or "")).lower()):
            return found
    return None


def _id(value: object) -> str:
    """A posting id as a string: RemoteOK sends ``"1137611"``, Jobicy ``158470``."""
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return clean(value) or ""


def _remote_job(source: str, key: str, url: str | None, title: object, company: object, **fields) -> Job | None:
    """A posting from a remote-only board, or ``None`` without an id, link, title or company (freehire skips the
    same: a posting without its employer cannot be placed)."""
    title, company, url = clean(title), clean(company), clean(url)
    if not key or not url or not title or not company:
        return None
    return Job(
        job_id=f"{source}:{key}",
        url=url,
        title=title,
        company=company,
        source=source,
        is_remote=True,
        work_mode="remote",
        **fields,
    )


def once(jobs: Iterable[Job]) -> list[Job]:
    """Each posting once, the first sighting kept: a feed can list one posting twice (We Work Remotely, 2 of 100)."""
    first: dict[str, Job] = {}
    for job in jobs:
        first.setdefault(job.job_id, job)
    return list(first.values())


class _Feed:
    """One feed per site: the board is the site, named explicitly, never recognised from a link."""

    name = ""
    site = ""
    many_companies = True  # the board is stored with no company (``boards import``): each job names its own

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        return f"https://{self.site}/" if self._board(entry) == self.site else None

    def detect(self, entry: CompanyEntry) -> bool:
        return False

    def _board(self, entry: CompanyEntry) -> str:
        return ((entry.extra or {}).get("slug") or "").strip().lower()

    def _check(self, entry: CompanyEntry) -> None:
        if self._board(entry) != self.site:
            raise FetchError(f"{self.name}: board {self._board(entry)!r} is not {self.site}")


class RemoteOKProvider(_Feed):
    """``/api``: the newest ~100 postings, after a legal notice (the first element, no id)."""

    name = "remoteok"
    site = "remoteok.com"

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        self._check(entry)
        feed = ctx.http.get_json("https://remoteok.com/api", timeout=_TIMEOUT_S)
        if not isinstance(feed, list):
            raise FetchError("remoteok: the feed is not a list")
        jobs = (
            _remote_job(
                self.name, _id(p.get("id")), p.get("url"), p.get("position"), p.get("company"),
                description=html_to_text(p.get("description")) or None, location=clean(p.get("location")),
                date_posted=iso_date(p.get("date")),
            )
            for p in feed if isinstance(p, dict)
        )  # fmt: skip
        return once(j for j in jobs if j)


class RemotiveProvider(_Feed):
    """``/api/remote-jobs``: every active posting, 24 hours delayed, in one response (Remotive asks for at most 4
    reads a day)."""

    name = "remotive"
    site = "remotive.com"

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        self._check(entry)
        data = ctx.http.get_json("https://remotive.com/api/remote-jobs", timeout=_TIMEOUT_S)
        feed = data.get("jobs") if isinstance(data, dict) else None
        if not isinstance(feed, list):
            raise FetchError("remotive: the feed has no jobs list")
        jobs = (
            _remote_job(
                self.name, _id(p.get("id")), p.get("url"), p.get("title"), p.get("company_name"),
                description=html_to_text(p.get("description")) or None,
                location=clean(p.get("candidate_required_location")), date_posted=iso_date(p.get("publication_date")),
                job_type=job_type(p.get("job_type")), salary=clean(p.get("salary")),
            )
            for p in feed if isinstance(p, dict)
        )  # fmt: skip
        return once(j for j in jobs if j)


class JobicyProvider(_Feed):
    """``/api/jobs``: 100 a page, the cursor followed to the end of the 7-day window (Jobicy's documented walk)."""

    name = "jobicy"
    site = "jobicy.com"
    max_pages = 50  # 5,000 postings: far above a week of Jobicy (a page past it means the walk never ends)

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        self._check(entry)
        jobs: dict[str, Job] = {}
        cursor = None
        for _ in range(self.max_pages):
            url = "https://jobicy.com/api/jobs?count=100" + (f"&cursor={quote(cursor, safe='')}" if cursor else "")
            data = ctx.http.get_json(url, timeout=_TIMEOUT_S)
            page = data.get("jobs") if isinstance(data, dict) else None
            if not isinstance(page, list):
                raise FetchError(f"jobicy: {url} has no jobs list", url=url)
            for p in page:
                if isinstance(p, dict) and (job := self._job(p)):
                    jobs.setdefault(job.job_id, job)
            cursor = clean(data.get("nextCursor"))
            if not cursor or not page:
                return list(jobs.values())
        raise FetchError(f"jobicy: no end after {self.max_pages} pages")

    def _job(self, p: dict) -> Job | None:
        return _remote_job(
            self.name, _id(p.get("id")), p.get("url"), p.get("jobTitle"), p.get("companyName"),
            description=html_to_text(p.get("jobDescription")) or None, location=clean(p.get("jobGeo")),
            date_posted=iso_date(p.get("pubDate")), job_type=job_type(p.get("jobType")),
        )  # fmt: skip


_COUNTRY = re.compile(r"^[A-Z]{2}$")
_PROMO = re.compile(r"\s*Originally posted on Himalayas\s*$", re.I)


class HimalayasProvider(_Feed):
    """``/jobs/api/search?country=IN``: 20 a page, read newest first until a page comes back empty (Himalayas'
    documented walk; its ``totalCount`` is an estimate). A run reads at most ``max_pages`` pages; a page failing after
    the first ends the run's walk with what it read, as freehire does (the public allowance answers 429 to a long
    walk), and the next run starts from the newest again."""

    name = "himalayas"
    site = "himalayas.app"
    max_pages = 100

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        country = self._board(entry).upper()
        return f"https://himalayas.app/jobs/api/search?country={country}" if _COUNTRY.match(country) else None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        country = self._board(entry).upper()
        if not _COUNTRY.match(country):
            raise FetchError(f"himalayas: board {country!r} is not a country code (IN)")
        jobs: dict[str, Job] = {}
        for page in range(1, self.max_pages + 1):
            url = f"https://himalayas.app/jobs/api/search?country={country}&sort=recent&page={page}"
            try:
                data = ctx.http.get_json(url, timeout=_TIMEOUT_S)
            except FetchError as err:
                if page == 1:
                    raise
                log.warning("himalayas %s: page %d failed (%s): keeping the %d read", country, page, err, len(jobs))
                return list(jobs.values())
            found = data.get("jobs") if isinstance(data, dict) else None
            if not isinstance(found, list):
                raise FetchError(f"himalayas: {url} has no jobs list", url=url)
            if not found:
                return list(jobs.values())
            for p in found:
                if isinstance(p, dict) and (job := self._job(p)):
                    jobs.setdefault(job.job_id, job)  # results can shift between pages: one job, once
        log.info("himalayas %s: stopped at %d pages (%d jobs); older ones are not read", country, page, len(jobs))
        return list(jobs.values())

    def _job(self, p: dict) -> Job | None:
        guid = clean(p.get("guid")) or ""
        parts = urlsplit(guid)
        key = parts.path.removeprefix("/companies/") if parts.hostname == self.site else ""
        company = clean(p.get("companyName"))
        if (company or "").lower() == "name":  # a template field the feed leaves unresolved (freehire)
            company = clean(p.get("companySlug"))
        text = _PROMO.sub("", html_to_text(p.get("description")))
        places = [clean(x) for x in p.get("locationRestrictions") or [] if clean(x)]
        return _remote_job(
            self.name, key, guid, p.get("title"), company, description=text or None,
            location="; ".join(places) or None, date_posted=iso_date((p.get("pubDate") or 0) * 1000 or None),
            job_type=job_type(p.get("employmentType")),
            **salary_range(p.get("minSalary"), p.get("maxSalary"), p.get("currency"), p.get("salaryPeriod") or "year"),
        )  # fmt: skip


_WN_ID = re.compile(r"/(\d+)/?$")


class WorkingNomadsProvider(_Feed):
    """``/api/exposed_jobs/``: one flat list, the recent window; the id is the job link's number."""

    name = "workingnomads"
    site = "workingnomads.com"

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        self._check(entry)
        feed = ctx.http.get_json("https://www.workingnomads.com/api/exposed_jobs/", timeout=_TIMEOUT_S)
        if not isinstance(feed, list):
            raise FetchError("workingnomads: the feed is not a list")
        jobs: list[Job] = []
        for p in feed:
            m = _WN_ID.search(clean((p or {}).get("url")) or "") if isinstance(p, dict) else None
            job = m and _remote_job(
                self.name, m.group(1), p.get("url"), p.get("title"), p.get("company_name"),
                description=html_to_text(p.get("description")) or None, location=clean(p.get("location")),
                date_posted=iso_date(p.get("pub_date")),
            )  # fmt: skip
            if job:
                jobs.append(job)
        return once(jobs)


class WeWorkRemotelyProvider(_Feed):
    """``/remote-jobs.rss``: the newest 100; an item's title is "Company: Role", its id the link's last segment."""

    name = "weworkremotely"
    site = "weworkremotely.com"

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        self._check(entry)
        items = rss_items(
            ctx.http.get_text("https://weworkremotely.com/remote-jobs.rss", timeout=_TIMEOUT_S), self.name
        )
        jobs: list[Job] = []
        for item in items:
            link = clean(item.findtext("link")) or ""
            company, _, title = (clean(item.findtext("title")) or "").partition(": ")
            job = _remote_job(
                self.name, link.rstrip("/").rsplit("/", 1)[-1], link, title, company,
                description=html_to_text(item.findtext("description")) or None,
                location=clean(item.findtext("region")), date_posted=rss_date(clean(item.findtext("pubDate"))),
                job_type=job_type(item.findtext("type")),
            )  # fmt: skip
            if job:
                jobs.append(job)
        return once(jobs)
