"""The Muse -- themuse.com's public jobs API, no key (build step 3, wave 3).

Ported from freehire's ``internal/ingest/sources/themuse.go`` (MIT, see THIRD_PARTY_NOTICES.md). Many employers (each
posting names its own), big brands, US-heavy; every posting carries its text inline (no detail request). freehire
walks the unfiltered list from page 1 until the API refuses a page (past ~99, whatever ``page_count`` says) and keeps
what it read: some 2,000 of 413,000, in the API's own "trendiness" order, so which ones is arbitrary.

Ours is a board for India, read whole, within the API's limits as measured 2026-11-02:
- a ``location`` filter also returns every "Flexible / Remote" posting (6,210): India's 24 cities with postings
  are about 7,000, so India and remote together are 12,823, 641 pages;
- a query stops at page 99 (1,980 postings) and the API allows 500 requests an hour.
So the board reads the technical categories only (software, data, IT, science and engineering: 4,800 of the 12,823,
the rest being sales, finance, healthcare and the like, which the role-type filter would drop anyway), each split by
level so that no query reaches the page limit: about 275 requests a run. A request failing after the first ends the
run's walk with what it read (the hourly allowance), as freehire does.
"""

from __future__ import annotations


from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import clean, infer_remote, iso_date, log

API = "https://www.themuse.com/api/public/jobs"
#: The Indian locations with postings (2026-11-02, postings beyond the remote ones each adds): Bangalore 2,644,
#: Hyderabad 1,173, Pune 900, Chennai 878, Mumbai 637, Gurgaon 451, Noida 283, Vadodara 173, the rest under 50.
#: Muse names a place one way ("Bangalore"; "Bengaluru" has 14), and "India" alone names nothing.
INDIA = (
    "Bangalore, India", "Bengaluru, India", "Hyderabad, India", "Pune, India", "Chennai, India", "Mumbai, India",
    "Gurgaon, India", "Noida, India", "Vadodara, India", "Coimbatore, India", "Cochin, India", "New Delhi, India",
    "Calcutta, India", "Greater Noida, India", "Ahmedabad, India", "Jaipur, India", "Mysore, India", "Lucknow, India",
    "Chandigarh, India", "Faridabad, India", "Thiruvananthapuram, India", "Guwahati, India", "Indore, India",
    "Vijayawada, India",
)  # fmt: skip
CATEGORIES = ("Software Engineering", "Software Engineer", "Data and Analytics", "Data Science", "Computer and IT", "IT",
              "Science and Engineering")  # fmt: skip
LEVELS = ("Internship", "Entry Level", "Mid Level", "Senior Level", "management")
_MAX_PAGE = 99  # the API answers 400 past it, whatever page_count says
_TIMEOUT_S = 40.0


class TheMuseProvider:
    name = "themuse"
    many_companies = True  # the board is stored with no company (``boards import``): each posting names its own

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        return "https://www.themuse.com/search/location/india" if _board(entry) == "IN" else None

    def detect(self, entry: CompanyEntry) -> bool:
        return False

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        if _board(entry) != "IN":
            raise FetchError(f"themuse: board {_board(entry)!r} is not IN (the only one with its city list)")
        jobs: dict[str, Job] = {}
        asked = 0

        def get(category: str, level: str, cities: tuple[str, ...], page: int) -> dict:
            nonlocal asked
            params = [("page", page), ("category", category), ("level", level)] + [("location", c) for c in cities]
            try:
                data = ctx.http.get_json(API, params=params, timeout=_TIMEOUT_S)
            except FetchError as err:
                if not asked:
                    raise
                raise _Stop(f"{category} / {level} page {page}: {err}") from err
            asked += 1
            if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                raise FetchError(f"themuse: {category} / {level} page {page} has no results list", url=API)
            return data

        def walk(category: str, level: str, cities: tuple[str, ...]) -> None:
            """One query read to its end; one past the page limit is read again as two halves of its cities (each
            half carries the remote postings again: read once)."""
            data = get(category, level, cities, 1)
            pages = data.get("page_count") or 0
            if pages > _MAX_PAGE:
                if len(cities) > 1:
                    half = len(cities) // 2
                    walk(category, level, cities[:half])
                    walk(category, level, cities[half:])
                    return
                log.warning("themuse: %s / %s in %s has %d pages: past %d are not read", category, level, cities[0],
                            pages, _MAX_PAGE)  # fmt: skip
            page = 1
            while True:
                for p in data["results"]:
                    if isinstance(p, dict) and (job := to_job(p)):
                        jobs.setdefault(job.job_id, job)
                if not data["results"] or page >= min(pages, _MAX_PAGE):
                    return
                page += 1
                data = get(category, level, cities, page)

        try:
            for category in CATEGORIES:
                for level in LEVELS:
                    walk(category, level, INDIA)
        except _Stop as stop:  # the hourly allowance, most likely: what was read is kept, as freehire does
            log.warning("themuse: %s failed: keeping the %d read", stop, len(jobs))
        return list(jobs.values())


class _Stop(Exception):
    """A request failing after the first: the run's walk ends there."""


def _board(entry: CompanyEntry) -> str:
    return ((entry.extra or {}).get("slug") or "").strip().upper()


def to_job(posting: dict) -> Job | None:
    pid, title = posting.get("id"), clean(posting.get("name"))
    company = clean((posting.get("company") or {}).get("name"))
    url = clean((posting.get("refs") or {}).get("landing_page"))
    if not isinstance(pid, int) or isinstance(pid, bool) or not title or not company or not url:
        return None
    places = [clean(p.get("name")) for p in posting.get("locations") or [] if isinstance(p, dict)]
    location = "; ".join(dict.fromkeys(p for p in places if p)) or None  # freehire keeps the first only
    levels = {clean(lv.get("name")) for lv in posting.get("levels") or [] if isinstance(lv, dict)}
    return Job(
        job_id=f"themuse:{pid}",
        url=url,
        title=title,
        company=company,
        source="themuse",
        description=html_to_text(posting.get("contents")) or None,
        location=location,
        date_posted=iso_date(posting.get("publication_date")),
        job_type="intern" if "Internship" in levels else None,
        is_remote=infer_remote(location or ""),
    )
