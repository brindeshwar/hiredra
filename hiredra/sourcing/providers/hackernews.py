"""Hacker News "Who is hiring?" -- the monthly threads, read through the Algolia HN API, no key (build step 3,
wave 3).

Ported from freehire's ``internal/ingest/sources/hackernews.go`` (MIT, see THIRD_PARTY_NOTICES.md). The
``whoishiring`` account posts one hiring thread a month; a run reads the two newest (this month's, still filling, and
last month's, still live), each whole, in one request: every top-level comment is one employer's post, replies are
discussion. The thread search filters by that account's author tag, not free text, and the title must say "who is
hiring" (the account also posts "Who wants to be hired?"). A thread that cannot be read fails the board.

A post's first paragraph is its header by the thread's own convention, ``Company | Role | Location | ...``: the first
segment is the employer, the second the role, the location the first later segment that is not a commitment word, a
link or a salary. A post with fewer than two segments, or whose first segment is a role ("Senior Software Engineer")
rather than an employer, is skipped: filing it would invent an employer (freehire's rules). The whole post is the
text. The board is the site, stored with no company: each post names its own.

Two changes from freehire: the job's URL is the post itself (its permalink: stable, unique, and where the text lives),
and the post's first link, usually the employer's careers page, is the apply link (freehire makes that link the URL, so
two posts naming one careers page share it); and the header is the first paragraph's first LINE (50 of 509 posts run
on into their prose with a line break, no <p>, which freehire reads as part of the last segment: "sf bay area only
competitive salary, ..." as a location).
"""

from __future__ import annotations

import re

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import clean, infer_remote, iso_date, log

SITE = "news.ycombinator.com"
SEARCH = "https://hn.algolia.com/api/v1/search_by_date?tags=story,author_whoishiring&hitsPerPage=10"
ITEM = "https://hn.algolia.com/api/v1/items/{}"
_THREADS = 2
_TIMEOUT_S = 60.0
_HIRING = re.compile(r"who is hiring", re.I)
_PARAGRAPH = re.compile(r"<p\b[^>]*>", re.I)  # HN writes an unclosed <p> between paragraphs, none before the first
_URL = re.compile(r"https?://[^\s)\]}>,]+")
_HREF = re.compile(r"""<a\b[^>]*\bhref=["'](https?:[^"']+)["']""", re.I)
_COMMITMENT = re.compile(r"^(full[ -]?time|part[ -]?time|contract(or)?|intern(ship)?s?|permanent|freelance)$", re.I)
#: A first segment that opens with a seniority word AND names a role noun is a role, not an employer ("Lead Bank" and
#: "Chief Industries" are employers: only the pair is evidence).
_SENIORITY = re.compile(r"^(sr|jr|senior|junior|staff|principal|lead|founding|head|director|vp|chief)\b", re.I)
_ROLE_NOUN = re.compile(
    r"\b(engineer|engineering|developer|designer|scientist|analyst|architect|manager|programmer|devops|sre|researcher)\b",
    re.I,
)


def paragraphs(post: str) -> list[str]:
    """A post's paragraphs as plain text."""
    return [p for p in (html_to_text(part) for part in _PARAGRAPH.split(post)) if p]


def header(post: str) -> tuple[str, str, str | None, bool | None] | None:
    """``(company, role, location, remote)`` from a post's first line, or ``None`` when it names no employer and
    role."""
    first = (paragraphs(post) or [""])[0].split("\n", 1)[0]  # the line: 50 of 509 run on into prose without a <p>
    parts = [p.strip() for p in first.split("|")]
    if len(parts) < 2 or not parts[0]:
        return None
    if _SENIORITY.match(parts[0]) and _ROLE_NOUN.search(parts[0]):
        return None
    company, role = _URL.sub("", parts[0]).strip(), _URL.sub("", parts[1]).strip()
    if not company or not role:
        return None
    location = None
    for seg in parts[2:]:
        seg = _URL.sub("", seg).strip()
        if seg and not _COMMITMENT.match(seg) and seg[0] not in "$€£":
            location = seg
            break
    return company, role, location, infer_remote(" | ".join(parts[1:]))


class HackerNewsProvider:
    name = "hackernews"
    many_companies = True  # the board is stored with no company (``boards import``): each post names its own

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        return f"https://{SITE}/submitted?id=whoishiring" if _board(entry) == SITE else None

    def detect(self, entry: CompanyEntry) -> bool:
        return False

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        if _board(entry) != SITE:
            raise FetchError(f"hackernews: board {_board(entry)!r} is not {SITE}")
        found = ctx.http.get_json(SEARCH, timeout=_TIMEOUT_S)
        hits = found.get("hits") if isinstance(found, dict) else None
        if not isinstance(hits, list):
            raise FetchError("hackernews: the thread search has no hits list", url=SEARCH)
        threads = [h["objectID"] for h in hits if isinstance(h, dict) and _HIRING.search(str(h.get("title")))
                   and str(h.get("objectID", "")).isdigit()][:_THREADS]  # fmt: skip
        if not threads:
            raise FetchError(f"hackernews: no 'Who is hiring?' thread among the newest {len(hits)} stories")
        if len(threads) < _THREADS:
            log.warning("hackernews: only %d of %d hiring threads found", len(threads), _THREADS)
        jobs: dict[str, Job] = {}
        for thread in threads:
            item = ctx.http.get_json(ITEM.format(thread), timeout=_TIMEOUT_S)
            posts = item.get("children") if isinstance(item, dict) else None
            if not isinstance(posts, list):
                raise FetchError(f"hackernews: thread {thread} has no comments list", url=ITEM.format(thread))
            for post in posts:
                if isinstance(post, dict) and (job := to_job(post)):
                    jobs.setdefault(job.job_id, job)
        return list(jobs.values())


def _board(entry: CompanyEntry) -> str:
    return ((entry.extra or {}).get("slug") or "").strip().lower()


def to_job(post: dict) -> Job | None:
    """One top-level comment as a job; ``None`` for a deleted one or one whose header names no employer and role."""
    pid, text = post.get("id"), clean(post.get("text"))
    if not isinstance(pid, int) or isinstance(pid, bool) or not text or not (head := header(text)):
        return None
    company, role, location, remote = head
    link = _HREF.search(text)
    return Job(
        job_id=f"hackernews:{pid}",
        url=f"https://{SITE}/item?id={pid}",
        title=role,
        company=company,
        source="hackernews",
        description="\n\n".join(paragraphs(text)) or None,
        location=location,
        date_posted=iso_date(post.get("created_at")),
        apply_url=html_to_text(link.group(1)) if link else None,
        is_remote=remote,
    )
