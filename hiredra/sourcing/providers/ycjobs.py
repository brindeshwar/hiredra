"""YC startup jobs -- ycombinator.com's public job pages (build step 3, wave 3; Hiredra's own, not a freehire port).

Work at a Startup, YC's job board, is readable only from a signed-in session (an access control we do not work
around, slice 3.10). ycombinator.com publishes a sample of the same postings on public pages that server-render their
rows (``WaasJobListingsPage``): ``/jobs/location/india`` (24 postings, the same whatever role or page is asked),
``/jobs/location/remote`` (34) and each role's remote page (``/jobs/role/software-engineer/remote``: 37), measured
2026-11-02. A run reads those pages, then each new posting's own page, whose schema.org JobPosting carries the text
(a hydrating source, D3.4). These are HTML pages, so robots.txt is obeyed (D4.2; ``/companies?*`` is disallowed,
``/companies/<slug>/jobs/<id>`` is not). Each row names its own employer: the board is the site, with no company.

A row states the experience it asks for (``minExperience``: "6+ years", "Any (new grads ok)"): it opens the text as an
"Experience: ..." line, so the seniority gate reads it, as Keka's field does. The row's apply link is YC's sign-up
flow, not the employer's, so none is stored.
"""

from __future__ import annotations

import html
import json
import re
from urllib.parse import urlsplit

from hiredra.sourcing._html import html_to_text, ld_job_posting
from hiredra.sourcing.http import FetchError, HttpConfig
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import DETAIL_BUDGET, clean, hydrate, infer_remote, iso_date, Robots, work_mode
from hiredra.sourcing.providers.remoteboards import job_type

SITE = "www.ycombinator.com"
BOARD = "ycombinator.com"
ROLES = ("software-engineer", "designer", "product-manager", "recruiting-hr", "sales-manager", "marketing", "support",
         "operations", "science")  # fmt: skip
PAGES = ("jobs/location/india", "jobs/location/remote", *(f"jobs/role/{r}/remote" for r in ROLES))
_DATA_PAGE = re.compile(r"""\bdata-page=(["'])(.*?)\1""", re.S)
_JOB_PATH = re.compile(r"^/companies/[a-z0-9-]+/jobs/[A-Za-z0-9_-]+$")
_TIMEOUT_S = 30.0


def rows(page: str) -> list[dict]:
    """The postings a listing page server-renders, or a ``FetchError`` when it carries none."""
    m = _DATA_PAGE.search(page)
    try:
        data = json.loads(html.unescape(m.group(2))) if m else None
    except json.JSONDecodeError as err:
        raise FetchError(f"ycjobs: the page's data is not JSON: {err}") from err
    found = ((data or {}).get("props") or {}).get("jobPostings") if isinstance(data, dict) else None
    if not isinstance(found, list):
        raise FetchError("ycjobs: the page carries no job postings")
    return [r for r in found if isinstance(r, dict)]


class YCJobsProvider:
    name = "ycjobs"
    many_companies = True  # the board is stored with no company (``boards import``): each row names its own

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        return f"https://{SITE}/jobs/location/india" if _board(entry) == BOARD else None

    def detect(self, entry: CompanyEntry) -> bool:
        return False

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        if _board(entry) != BOARD:
            raise FetchError(f"ycjobs: board {_board(entry)!r} is not {BOARD}")

        def get(url: str) -> str:
            if urlsplit(url).scheme != "https" or urlsplit(url).hostname != SITE:
                raise FetchError(f"ycjobs: refusing {url!r} off {SITE}", url=url)
            return ctx.http.get_text(url, timeout=_TIMEOUT_S)

        def robots_txt(url: str) -> str | None:
            try:
                return get(url)
            except FetchError:
                return None

        config = getattr(ctx.http, "config", None)
        robots = Robots(robots_txt, getattr(config, "user_agent", HttpConfig().user_agent))
        found: dict[str, dict] = {}
        for path in PAGES:  # a page that fails fails the board: the rest of a run would close nothing it lists
            url = f"https://{SITE}/{path}"
            if not robots.allowed(url):
                raise FetchError(f"ycjobs: robots.txt disallows {url}", url=url)
            for row in rows(get(url)):
                if isinstance(row.get("id"), int) and _JOB_PATH.match(clean(row.get("url")) or ""):
                    found.setdefault(str(row["id"]), row)

        def detail(pid: str) -> Job | None:
            url = f"https://{SITE}{found[pid]['url']}"
            return to_job(found[pid], get(url)) if robots.allowed(url) else None

        return hydrate(
            found,
            key=lambda pid: f"ycjobs:{pid}",
            seen=ctx.seen,
            listing=lambda pid: to_job(found[pid], None),
            detail=detail,
            budget=DETAIL_BUDGET,
            label="ycjobs",
        )


def _board(entry: CompanyEntry) -> str:
    return ((entry.extra or {}).get("slug") or "").strip().lower()


def to_job(row: dict, page: str | None) -> Job | None:
    """A row as a job: with its page, the JobPosting's text and date; without (a stored posting re-sighted), the row
    alone. ``None`` without a title or employer, or when the page carries no JobPosting (read again next run)."""
    title, company = clean(row.get("title")), clean(row.get("companyName"))
    if not title or not company:
        return None
    posting = ld_job_posting(page) if page is not None else {}
    if posting is None:
        return None
    experience = clean(row.get("minExperience"))
    text = "\n\n".join(p for p in (f"Experience: {experience}" if experience else "",
                                   html_to_text(posting.get("description"))) if p)  # fmt: skip
    location = clean(row.get("location"))
    workplace = "remote" if posting.get("jobLocationType") == "TELECOMMUTE" else None
    return Job(
        job_id=f"ycjobs:{row['id']}",
        url=f"https://{SITE}{row['url']}",
        title=title,
        company=company,
        source="ycjobs",
        description=text if page is not None and text else None,
        location=location,
        date_posted=iso_date(posting.get("datePosted")),
        job_type=job_type(row.get("type")),
        work_mode=work_mode(workplace),
        is_remote=infer_remote(location or "", workplace_type=workplace),
    )
