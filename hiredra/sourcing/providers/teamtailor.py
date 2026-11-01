"""Teamtailor provider -- the career site's RSS feed (build step 3, wave 1).

Ported from freehire's ``internal/ingest/sources/teamtailor.go`` (MIT, see THIRD_PARTY_NOTICES.md). The board is the
career-site host (``clearroute.teamtailor.com``, ``teoco.na.teamtailor.com``, or an employer's own domain).

One change from freehire, measured 2026-11-01: freehire walks the HTML listing and reads each posting's page, and
uses ``jobs.rss`` only for the remote status. The feed alone carries everything a posting needs (title, HTML text,
date, link, ``remoteStatus`` and every location), and listed the same postings as the HTML listing on all six of our
boards. So the board is read from the feed: one request per 500 postings, no page per posting (the volume that had
Teamtailor turn freehire away). The feed is paged with ``offset``; a board that has not ended within ``_MAX_PAGES``
pages is an error, never a partial board.
"""

from __future__ import annotations

import re
from xml.etree import ElementTree

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import clean, infer_remote, rss_date, rss_items

_HOST = re.compile(r"^[a-z0-9][a-z0-9-]*(?:\.[a-z0-9][a-z0-9-]*)+$")
_JOB_ID = re.compile(r"/jobs/(\d+)")
_NS = "{https://teamtailor.com/locations}"
_PAGE = 500
_MAX_PAGES = 20
#: The feed's ``remoteStatus`` as a workplace type: "none" is the default nobody set, and "temporary" (remote for
#: now) does not state the arrangement the job hires into, so both stay unknown (freehire's reading).
_WORKPLACE = {"fully": "remote", "hybrid": "hybrid", "onsite": "onsite"}
_TIMEOUT_S = 30.0


def site(entry: CompanyEntry) -> str:
    host = ((entry.extra or {}).get("slug") or "").strip().lower()
    if not _HOST.match(host):
        raise FetchError(f"teamtailor: board {host!r} is not a host")
    return host


class TeamtailorProvider:
    name = "teamtailor"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        try:
            return f"https://{site(entry)}/jobs"
        except FetchError:
            return None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        host = site(entry)
        jobs: dict[str, Job] = {}
        for page in range(_MAX_PAGES):
            url = f"https://{host}/jobs.rss?offset={page * _PAGE}&per_page={_PAGE}"
            found = rss_items(ctx.http.get_text(url, timeout=_TIMEOUT_S), self.name)
            for item in found:
                if (job := to_job(entry.name, host, item)) is not None:
                    jobs.setdefault(job.job_id, job)
            if len(found) < _PAGE:
                return list(jobs.values())
        raise FetchError(f"teamtailor: {host} has not ended within {_MAX_PAGES} feed pages", url=url)


def to_job(company: str, host: str, item: ElementTree.Element) -> Job | None:
    title, link = clean(item.findtext("title")), clean(item.findtext("link"))
    m = _JOB_ID.search(link or "")
    if not title or not m:
        return None
    places = []
    for place in item.iterfind(f"{_NS}locations/{_NS}location"):
        city = clean(place.findtext(f"{_NS}city")) or clean(place.findtext(f"{_NS}name"))
        places.append(", ".join(p for p in (city, clean(place.findtext(f"{_NS}country"))) if p))
    location = "; ".join(dict.fromkeys(p for p in places if p)) or None
    workplace = _WORKPLACE.get((clean(item.findtext("remoteStatus")) or "").lower())
    return Job(
        job_id=f"teamtailor:{host}:{m.group(1)}",
        url=link,
        title=title,
        company=company,
        source="teamtailor",
        description=html_to_text(item.findtext("description")) or None,
        location=location,
        date_posted=rss_date(clean(item.findtext("pubDate"))),
        # the stated arrangement first; the location only as a fallback, never the title ("Remote ..." role names)
        is_remote=infer_remote(location or "", workplace_type=workplace),
    )
