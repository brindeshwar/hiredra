"""EPAM provider -- the careers site's sitemap, then each new vacancy's page (parity step, 2026-11-03).

Ported from freehire's ``internal/ingest/sources/epam.go`` (MIT, see THIRD_PARTY_NOTICES.md). One employer; the board
is the site, ``careers.epam.com``. Its job-search API is disallowed by robots.txt (``/api``), and a country's jobs
page (``/en/jobs/india``: 263 jobs) renders only its first 20, paging by ``?page=``, disallowed too. The sitemap that
robots.txt names, ``/sitemap.xml.gz``, lists every vacancy (4,094 in English, measured 2026-11-03), and each vacancy
page server-renders a schema.org JobPosting and the page's own data (``__NEXT_DATA__``): a hydrating source (D3.4).
These are HTML pages, so robots.txt is obeyed (D4.2).

A vacancy is read once, under its English link (``/en/vacancy/<slug>-<uid>_en``). The sitemap dates each one, and a
run reads the newest unseen first, so a first crawl of the whole site (about 21 runs at the detail budget) brings in
the newest vacancies first.

The text is built from the page data's sections (intro, Responsibilities, Requirements, Nice to have), as freehire
does: the JobPosting runs them into one paragraph, and skill extraction reads the headings to tell required skills
from optional ones. The JobPosting's text is the fallback. Benefits are left out (freehire keeps them): they say
nothing about the role. EPAM names no city; the place is the countries the JobPosting takes applicants from, written
``Remote (A, B)`` for a remote vacancy (``jobLocationType: TELECOMMUTE``) and ``A; B`` otherwise, the shapes our
location filter reads (a remote vacancy open to India passes, one restricted elsewhere drops).
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

from hiredra.sourcing._html import html_to_text, ld_job_posting, next_data
from hiredra.sourcing.http import FetchError, HttpConfig
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import (
    DETAIL_BUDGET,
    Robots,
    clean,
    hydrate,
    infer_remote,
    iso_date,
    sitemap_locs,
)

SITE = "careers.epam.com"
#: An English vacancy link and its uid; the listing, the language roots and the other localisations do not match.
_VACANCY = re.compile(r"/en/vacancy/([^/?#]*)-(blt[a-z0-9]+)_en(?:[/?#]|$)")
_SECTIONS = (
    ("responsibilities", "Responsibilities"),
    ("requirements", "Requirements"),
    ("nice_to_have", "Nice to have"),
)
#: Section headings EPAM's data sometimes leaks as the last item of the list before (freehire's ``epamSectionLabels``).
_LABELS = frozenset({"responsibilities", "requirements", "nice to have", "benefits"})
_TIMEOUT_S = 30.0


def vacancy(loc: str) -> tuple[str, str] | None:
    """``(title slug, uid)`` of an English vacancy link, else ``None``."""
    m = _VACANCY.search(urlsplit(loc).path)
    return (m.group(1), m.group(2)) if m else None


def _board(entry: CompanyEntry) -> str:
    return ((entry.extra or {}).get("slug") or "").strip().lower()


class EPAMProvider:
    name = "epam"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        return f"https://{SITE}/en/jobs" if _board(entry) == SITE else None

    def detect(self, entry: CompanyEntry) -> bool:
        return False  # one employer: a board is named explicitly, never recognised from a link

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        if _board(entry) != SITE:
            raise FetchError(f"epam: board {_board(entry)!r} is not {SITE}")

        def get(url: str) -> str:
            if urlsplit(url).scheme != "https" or urlsplit(url).hostname != SITE:
                raise FetchError(f"epam: refusing {url!r} off {SITE}", url=url)
            return ctx.http.get_text(url, timeout=_TIMEOUT_S)

        def robots_txt(url: str) -> str | None:
            try:
                return get(url)
            except FetchError:
                return None

        config = getattr(ctx.http, "config", None)
        robots = Robots(robots_txt, getattr(config, "user_agent", HttpConfig().user_agent))
        sitemap = f"https://{SITE}/sitemap.xml.gz"
        if not robots.allowed(sitemap):
            raise FetchError(f"epam: robots.txt disallows {sitemap}", url=sitemap)
        pages: dict[str, tuple[str, str | None]] = {}  # uid -> (link, lastmod)
        for loc, lastmod in sitemap_locs(get(sitemap))[0]:
            if found := vacancy(loc):
                pages.setdefault(found[1], (loc, lastmod))
        if not pages:  # the site always has vacancies: none listed is a broken answer, never a reason to close them
            raise FetchError(f"epam: {sitemap} lists no vacancies", url=sitemap)
        newest_first = sorted(pages, key=lambda uid: pages[uid][1] or "", reverse=True)

        def detail(uid: str) -> Job | None:
            loc, lastmod = pages[uid]
            return to_job(entry.name, loc, lastmod, get(loc)) if robots.allowed(loc) else None

        return hydrate(
            newest_first,
            key=lambda uid: f"epam:{uid}",
            seen=ctx.seen,
            listing=lambda uid: listing_job(entry.name, *pages[uid]),
            detail=detail,
            budget=DETAIL_BUDGET,
            label="epam",
        )


def listing_job(company: str, loc: str, lastmod: str | None) -> Job:
    """A stored vacancy re-sighted in the sitemap, which carries only its link: the title is the link's slug (the
    store keeps the one the page gave, D3.4)."""
    slug, uid = vacancy(loc) or ("", "")
    return Job(
        job_id=f"epam:{uid}",
        url=loc,
        title=unquote(slug).replace("-", " ").strip() or f"EPAM vacancy {uid}",
        company=company,
        source="epam",
        date_posted=iso_date(lastmod),
    )


def sections(job: dict) -> str:
    """The page data's intro and its Responsibilities / Requirements / Nice to have lists as text; ``""`` when it
    has no list (the JobPosting's text is used then)."""
    category = job.get("category") if isinstance(job.get("category"), dict) else {}
    parts = []
    for key, heading in _SECTIONS:
        raw = category.get(key) if isinstance(category.get(key), list) else []
        items = [t for t in (html_to_text(i) for i in raw if isinstance(i, str)) if t and t.lower() not in _LABELS]
        if items:
            parts.append(heading + "\n" + "\n".join(f"- {t}" for t in items))
    if not parts:
        return ""
    return "\n\n".join(p for p in (html_to_text(job.get("description")), *parts) if p)


def _listed(value: object) -> list[dict]:
    return [v for v in (value if isinstance(value, list) else [value]) if isinstance(v, dict)]


def location(posting: dict) -> str | None:
    """The vacancy's countries: ``Remote (A, B)`` for a remote vacancy, else ``A; B``. A remote vacancy names the
    countries it takes applicants from (``applicantLocationRequirements``), an office one its office's
    (``jobLocation``: 13 of the newest 200, measured 2026-11-03; freehire reads only the first)."""
    named = [clean(c.get("name")) for c in _listed(posting.get("applicantLocationRequirements"))]
    named += [clean((p.get("address") or {}).get("addressCountry")) for p in _listed(posting.get("jobLocation"))
              if isinstance(p.get("address"), dict)]  # fmt: skip
    countries = list(dict.fromkeys(n for n in named if n))
    if posting.get("jobLocationType") == "TELECOMMUTE":
        return f"Remote ({', '.join(countries)})" if countries else "Remote"
    return "; ".join(countries) or None


def to_job(company: str, loc: str, lastmod: str | None, page: str) -> Job | None:
    """A vacancy from its page; ``None`` when the page carries no JobPosting or no title (skipped, read again next
    run)."""
    posting = ld_job_posting(page)
    if posting is None:
        return None
    job = ((((next_data(page) or {}).get("props") or {}).get("pageProps") or {}).get("job")) or {}
    title = clean(posting.get("title")) or clean(job.get("name") if isinstance(job, dict) else None)
    if not title:
        return None
    remote = posting.get("jobLocationType") == "TELECOMMUTE"
    place = location(posting)
    return Job(
        job_id=f"epam:{(vacancy(loc) or ('', ''))[1]}",
        url=loc,
        title=title,
        company=company,
        source="epam",
        description=(sections(job) if isinstance(job, dict) else "")
        or html_to_text(posting.get("description"))
        or None,
        location=place,
        date_posted=iso_date(posting.get("datePosted")) or iso_date(lastmod),
        is_remote=True if remote else infer_remote(f"{place or ''} {title}"),
        work_mode="remote" if remote else None,
    )
