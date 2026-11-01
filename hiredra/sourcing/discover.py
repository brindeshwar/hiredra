"""Finding a company's board by guessing its slug and asking the ATS (decision D2.2; build slices 2.4 and 2.6).
Ported in idea from freehire's ``cmd/harvest-boards`` (MIT, see THIRD_PARTY_NOTICES.md): a candidate slug is valid
only if the live ATS answers for it.

Greenhouse, Lever and Ashby all answer an unknown board with a 404 and a real one with 200 (measured 2026-10-31),
so a probe is one ordinary fetch through the provider a run uses. The jobs it returns are handed back, so the caller
can measure a board's worth without fetching it twice.

The YC directory (slice 2.4) is read from yc-oss's public JSON, which states no licence: it is used only on the
owner's machine as a list of facts (name, website, hiring regions) and is never committed or republished
(owner decision, 2026-10-31).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from html import unescape
from urllib.parse import parse_qs, urljoin, urlsplit

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.identity import company_slug
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext, Provider

#: Probe order: YC startups use Ashby most, then Greenhouse, then Lever. The first board found wins.
PROBE_ORDER = ("ashby", "greenhouse", "lever")
#: The YC regions that can hold an India- or remote-eligible job.
YC_REGIONS = frozenset({"India", "South Asia", "Remote", "Fully Remote", "Partly Remote"})


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    slugs: tuple[str, ...]  # guesses, best first
    website: str | None = None
    team_size: int | None = None


@dataclass(slots=True)
class Probe:
    candidate: Candidate
    provider: str | None = None  # None: no board found
    slug: str | None = None
    jobs: list[Job] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # anything but a 404, per guess


def slug_guesses(name: str, hints: Iterable[str] = ()) -> tuple[str, ...]:
    """The board slugs to try for a company: the hints first (e.g. YC's own slug), then the company key squashed
    ("Together AI" -> ``togetherai``) and hyphenated (``together-ai``). Lowercase, no duplicates."""
    key = company_slug(name)
    out = [h.strip().lower() for h in hints if h and h.strip()]
    out += [key.replace("-", ""), key]
    return tuple(dict.fromkeys(s for s in out if s))


def yc_candidates(path: Path | str, known: set[str]) -> list[Candidate]:
    """yc-oss ``companies/all.json`` entries that are hiring in an India- or remote-eligible region, minus companies
    whose squashed key is in ``known`` (they already have a board)."""
    out = []
    for c in json.loads(Path(path).read_text(encoding="utf-8")):
        if not c.get("isHiring") or not YC_REGIONS & set(c.get("regions") or ()):
            continue
        name = (c.get("name") or "").strip()
        if not name:
            continue
        if {company_slug(name).replace("-", ""), (c.get("slug") or "").replace("-", "")} & known:
            continue
        out.append(
            Candidate(
                name=name,
                slugs=slug_guesses(name, [c.get("slug") or ""]),
                website=c.get("website"),
                team_size=c.get("team_size") or None,
            )
        )
    return out


#: Below this many jobs a board can't out-hire a startup convincingly, so the team-size guard stays off (freehire's
#: ``collisionJobFloor`` idea in ``cmd/import-yc``).
HOMONYM_JOB_FLOOR = 5


def _host(url: str | None) -> str | None:
    host = urlsplit(url or "").hostname or ""
    host = host.removeprefix("www.")
    return host or None


def confirm(result: Probe) -> str:
    """Is the board found really this company's? A guessed slug can belong to a namesake ("Apollo").

    ``domain``  the company's own web host appears in a job's URL or text: the board is theirs.
    ``name``    only the company's name appears, and the board lists no more jobs than the company has people
                (freehire's homonym guard): probably theirs.
    ``none``    an empty board (nothing to check), a board out-hiring the startup, or no mention at all.
    """
    jobs = result.jobs
    if not jobs:
        return "none"
    cand = result.candidate
    texts = [f"{j.url} {getattr(j, 'apply_url', '') or ''} {j.description or ''}".lower() for j in jobs]
    host = _host(cand.website)
    if host and any(host in t for t in texts):
        return "domain"
    size = cand.team_size or 0
    if size and len(jobs) > max(size, HOMONYM_JOB_FLOOR):
        return "none"
    name = cand.name.lower()
    return "name" if any(name in t for t in texts) else "none"


def probe(
    candidate: Candidate,
    providers: Sequence[Provider],
    ctx: FetchContext,
    *,
    sleep: Callable[[float], None] = lambda s: None,
    pause: float = 0.0,
) -> Probe:
    """Try each guess on each provider in ``PROBE_ORDER`` until one answers 200. A 404 means "not here"; anything
    else is recorded and the search goes on (a failed guess must not hide a board under the next one)."""
    result = Probe(candidate)
    rank = {name: i for i, name in enumerate(PROBE_ORDER)}
    for provider in sorted(providers, key=lambda p: rank.get(p.name, len(rank))):
        name = provider.name
        for slug in candidate.slugs:
            sleep(pause)
            entry = CompanyEntry(name=candidate.name, provider=name, extra={"slug": slug})
            try:
                jobs = provider.fetch(entry, ctx)
            except HTTPError as err:
                if err.status != 404:
                    result.errors.append(f"{name}/{slug}: HTTP {err.status}")
                continue
            except FetchError as err:
                result.errors.append(f"{name}/{slug}: {err}")
                continue
            result.provider, result.slug, result.jobs = name, slug, list(jobs)
            return result
    return result


def known_folds(rows: Iterable[Any]) -> set[str]:
    """Squashed company keys that already have a board (``slug`` per row)."""
    return {str(r[0]).replace("-", "") for r in rows}


# ── slice 2.5: the board link on a company's own careers page ──────────────
# Ported from freehire's ``internal/ingest/atsdetect`` (``Detect``) and ``cmd/harvest-ats`` (``resolve``), MIT,
# for the three ATSs we have adapters for. A board linked from the company's own site is the company's: no
# namesake question, unlike a guessed slug (2.4). The other ATSs' URL shapes come with their adapters (step 3).
_BOARD_LINKS = (
    ("greenhouse", re.compile(r"(?:^|[^A-Za-z0-9])(?:boards|job-boards)\.greenhouse\.io/embed/job_board(?:/js)?\?for=([a-z0-9][a-z0-9-]*)")),
    ("greenhouse", re.compile(r"(?:^|[^A-Za-z0-9])(?:boards|job-boards)(?:\.eu)?\.greenhouse\.io/([a-z0-9][a-z0-9-]*)")),
    ("lever", re.compile(r"(?:^|[^A-Za-z0-9])jobs\.(?:eu\.)?lever\.co/([a-z0-9][a-z0-9-]*)")),
    ("ashby", re.compile(r"(?:^|[^A-Za-z0-9])jobs\.ashbyhq\.com/([a-z0-9][a-z0-9-]*)")),
)  # fmt: skip
#: Workday: ``tenant.wdN.myworkdayjobs.com/[locale/]Site/...`` -> ``host/Site``; on the shared host,
#: ``wdN.myworkdaysite.com/[locale/]recruiting/tenant/site`` -> ``host/tenant/site``. The site keeps its case.
#: Oracle: ``host.oraclecloud.com/hcmUI/CandidateExperience/<lang>/sites/CX_1`` -> ``host/CX_1``.
#: SmartRecruiters: ``jobs.`` or ``careers.smartrecruiters.com/<CompanyId>`` -> ``CompanyId``.
_HOST_SITE_LINKS = (
    ("workday", re.compile(
        r"(?i)\b([a-z0-9-]+\.wd\d+\.myworkdayjobs\.com)/(?:[a-z]{2}(?:-[a-z]{2})?/(?=[A-Za-z0-9_-]+/))?"
        r"(?!job/|details/|wday/)([A-Za-z0-9_-]+)")),
    ("workday", re.compile(
        r"(?i)\b(wd\d+\.myworkdaysite\.com)/(?:[a-z]{2}-[A-Z]{2}/)?recruiting/([A-Za-z0-9_-]+/[A-Za-z0-9_-]+)")),
    ("oracle", re.compile(
        r"(?i)\b([a-z0-9.-]+\.oraclecloud\.com)/hcmUI/CandidateExperience/[a-z-]+/sites/([A-Za-z0-9_]+)")),
)  # fmt: skip
_SMARTRECRUITERS_LINK = re.compile(
    r"(?i)\b(?:jobs|careers)\.smartrecruiters\.com/(?!oneclick|sr-jobs/)([A-Za-z0-9_.-]+)"
)
#: Workable: ``apply.workable.com/<account>``; ``/j/<code>`` is a short link with no account (followed instead), and
#: ``/oops`` is where the short link of a closed job points.
_WORKABLE_LINK = re.compile(
    r"(?i)(?<![A-Za-z0-9.-])apply\.workable\.com/(?!j/|api/|oops(?![a-z0-9_-]))([a-z0-9][a-z0-9_-]*)"
)
#: iCIMS: a posting on ``<label>.icims.com/jobs/``. A classic ``careers-<slug>`` label is the board ``slug``; any
#: other label keeps the whole host (freehire's two board forms).
_ICIMS_LINK = re.compile(r"(?i)(?<![A-Za-z0-9.-])([a-z0-9][a-z0-9-]*)\.icims\.com/jobs/")
#: SuccessFactors sites on SAP's own domains: the board is the host, one label under ``jobs2web.com`` or
#: ``jobs.hr.cloud.sap`` (``stage``/``www`` are the vendor's). The legacy ``careerNN.sapsf.*`` sections serve no
#: sitemap and are not read.
_SUCCESSFACTORS_LINK = re.compile(
    r"(?i)(?<![A-Za-z0-9.-])((?!stage\.|www\.)[a-z0-9][a-z0-9-]*\.(?:jobs2web\.com|jobs\.hr\.cloud\.sap))(?![A-Za-z0-9.-])"
)
#: Taleo Business Edition: any ``tbe.taleo.net`` link with ``org`` and ``cws`` parameters -> ``ORG:cws``.
_TALEOBE_LINK = re.compile(r"""(?i)(?<![A-Za-z0-9.-])(?:[a-z0-9-]+\.)?tbe\.taleo\.net/[^\s"'<>]*""")
#: Keka: the board is the subdomain of ``<board>.keka.com/careers`` (the vendor's own hosts are freehire's
#: ``platformLabels``).
_KEKA_LINK = re.compile(
    r"(?i)(?<![A-Za-z0-9.-])((?!(?:www|app|dashboard|admin|api|support|help|blog|docs)\.)[a-z0-9][a-z0-9-]*)"
    r"\.keka\.com/careers"
)
#: Zoho Recruit: a tenant host on any regional domain (``<tenant>.zohorecruit.com``/``.in``/``.eu``) is the board.
_ZOHO_LINK = re.compile(
    r"(?i)(?<![A-Za-z0-9.-])((?!www\.)[a-z0-9][a-z0-9-]*\.zohorecruit(?:\.[a-z]{2,3}){1,2})/(?:jobs|recruit)/"
)
_RESERVED = frozenset({"embed", "demo"})  # "demo": an embed placeholder seen on a real careers page
#: Where a company keeps its careers page, tried after its homepage (freehire's careerPaths and careerHostPrefixes).
CAREER_PATHS = ("/careers", "/jobs", "/career", "/vacancies", "/employment", "/about/careers", "/join-us")
CAREER_HOSTS = ("careers", "jobs")
_ANCHOR = re.compile(r"""(?is)<a\b[^>]*\bhref=["']([^"']*)["'][^>]*>(.*?)</a>""")
_TAG = re.compile(r"<[^>]*>")
_DEEPER = re.compile(r"(?i)(vacan|opening|apply|position|current-|search.?job|job.?search|opportunit)")


def detect_board(html: str) -> tuple[str, str] | None:
    """The first Greenhouse, Lever or Ashby board a page links to or embeds, else a Workday, Oracle,
    SmartRecruiters, Workable, iCIMS, SuccessFactors, Taleo Business Edition, Keka or Zoho Recruit one, as
    ``(provider, slug)``."""
    for provider, pattern in _BOARD_LINKS:
        for m in pattern.finditer(html):
            slug = m.group(1).lower()
            if slug not in _RESERVED:
                return provider, slug
    for provider, pattern in _HOST_SITE_LINKS:
        if m := pattern.search(html):
            return provider, f"{m.group(1).lower()}/{m.group(2)}"
    if m := _SMARTRECRUITERS_LINK.search(html):
        return "smartrecruiters", m.group(1)
    if m := _WORKABLE_LINK.search(html):
        return "workable", m.group(1).lower()
    if m := _ICIMS_LINK.search(html):
        label = m.group(1).lower()
        classic = label.startswith("careers-") and len(label) > len("careers-")
        return "icims", label.removeprefix("careers-") if classic else f"{label}.icims.com"
    if m := _SUCCESSFACTORS_LINK.search(html):
        return "successfactors", m.group(1).lower()
    for m in _TALEOBE_LINK.finditer(html):
        query = {k.lower(): v for k, v in parse_qs(urlsplit(unescape(m.group(0))).query).items()}
        org, cws = (query.get("org") or [""])[0].upper(), (query.get("cws") or [""])[0]
        if re.fullmatch(r"[A-Z0-9_]+", org) and cws.isdigit():
            return "taleobe", f"{org}:{cws}"
    if m := _KEKA_LINK.search(html):
        return "keka", m.group(1).lower()
    if m := _ZOHO_LINK.search(html):
        return "zohorecruit", m.group(1).lower()
    return None


def slug_names(company: str, slug: str) -> bool:
    """Does a board slug carry the company's name? One squashed form contains the other ("Bolna AI" ~ ``bolna``,
    "Nourish" ~ ``usenourish``), the shorter one at least 4 characters. Measured 2026-11-01: all 20 boards that
    were really the company's passed, and all 5 that were not (a careers page listing customers' jobs, a site
    redirected to its acquirer, an embed placeholder) failed."""
    key, board = company_slug(company).replace("-", ""), slug.replace("-", "")
    short, long = sorted((key, board), key=len)
    return len(short) >= 4 and short in long


def _link(html: str, base: str, wanted: Callable[[str], bool]) -> str | None:
    for href, text in _ANCHOR.findall(html):
        if wanted(href.lower()) or wanted(_TAG.sub("", text).lower()):
            link = urljoin(base, href.strip())
            if urlsplit(link).scheme in ("http", "https"):
                return link
    return None


def careers_walk(website: str, fetch: Callable[[str], str | None]) -> tuple[str, str, str] | None:
    """Find the board a company's site links to: the homepage, then its common careers paths and hosts, then the
    homepage's careers link; from each careers page, one link deeper ("open positions"). ``fetch`` returns a page's
    HTML or ``None``. Returns ``(provider, slug, page_url)``: the page the link was found on."""
    base = website.strip().rstrip("/")
    if not urlsplit(base).hostname:
        return None
    home = fetch(base)
    if home is None:
        return None
    if found := detect_board(home):
        return (*found, base)
    host = urlsplit(base).hostname.removeprefix("www.")
    scheme = urlsplit(base).scheme or "https"
    pages = [base + path for path in CAREER_PATHS] + [f"{scheme}://{h}.{host}" for h in CAREER_HOSTS]
    if link := _link(home, base, lambda s: "career" in s or "job" in s):
        pages.append(link)
    seen = {base}
    for page in pages:
        if page in seen:
            continue
        seen.add(page)
        html = fetch(page)
        if html is None:
            continue
        if found := detect_board(html):
            return (*found, page)
        deeper = _link(html, page, lambda s: bool(_DEEPER.search(s)))
        if deeper and deeper not in seen:
            seen.add(deeper)
            if (html := fetch(deeper)) is not None and (found := detect_board(html)):
                return (*found, deeper)
    return None


# ── slice 2.6: companies seen only on an aggregator (JobSpy) ────────────────
# Ported in idea from freehire's ``cmd/harvest-orphans`` (MIT): the worklist is the companies the store holds only
# through aggregators, each proposing board ids from its name alone, a candidate two employers propose dropped.
# Our additions, from slices 2.4-2.5: an aggregator job's own apply link is read first (it names the board, no
# guess), and a guessed board is kept only when it lists a job title we saw on the aggregator for that company.

#: Apply links that show a company uses an ATS other than Greenhouse/Lever/Ashby. Its board comes from the link when
#: ``detect_board`` reads one; otherwise it is counted, never guessed on our three name-keyed ATSs, where it could
#: only meet a namesake.
OTHER_ATS = (
    ("workday", re.compile(r"\.myworkday(?:jobs|site)\.com")),
    ("oracle", re.compile(r"\.oraclecloud\.com")),
    ("smartrecruiters", re.compile(r"smartrecruiters\.com")),
    ("icims", re.compile(r"\.icims\.com")),
    ("workable", re.compile(r"workable\.com")),
    ("taleo", re.compile(r"\.taleo\.net")),
    ("successfactors", re.compile(r"successfactors\.|\.sapsf\.|\.jobs2web\.com|\.jobs\.hr\.cloud\.sap")),
    ("keka", re.compile(r"\.keka\.com")),
    ("zohorecruit", re.compile(r"\.zohorecruit\.")),
)
#: Short links that redirect to the board: Greenhouse's ``grnh.se`` (to the board, or a careers page embedding it)
#: and Workable's account-less ``apply.workable.com/j/<code>`` (to ``/<account>/j/<code>``).
SHORT_LINKS = re.compile(r"(?i)^https?://(?:grnh\.se/|apply\.workable\.com/j/)")
#: Words a board id commonly drops that the company key keeps (freehire's ``boardNameTails``).
_BOARD_NAME_TAILS = re.compile(r"(?i)[\s,]+(?:group|s\.?p\.?a\.?)$")
#: The shortest guess worth probing (freehire's ``minCandidate``): shorter ids belong to someone else.
MIN_GUESS = 3


@dataclass(slots=True)
class Orphan:
    company_id: int
    name: str
    titles: set[str] = field(default_factory=set)  # normalised, as seen on the aggregator
    links: list[str] = field(default_factory=list)  # the jobs' apply and listing URLs
    board: tuple[str, str] | None = None  # (provider, slug) an apply link names
    other_ats: str | None = None
    short_links: list[str] = field(default_factory=list)
    slugs: tuple[str, ...] = ()


def norm_title(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (title or "").lower()))


def orphans(rows: Iterable[Sequence[Any]]) -> list[Orphan]:
    """Rows of ``(company_id, company name, job title, url, apply_url)``, one per aggregator job, grouped per company
    and sorted by name. Each orphan's apply links are read: a Greenhouse/Lever/Ashby board, another ATS, or a
    short link to follow. The rest get guessed slugs, a guess two employers propose dropped (freehire's rule)."""
    by_id: dict[int, Orphan] = {}
    for cid, name, title, *urls in rows:
        o = by_id.setdefault(cid, Orphan(cid, name))
        o.titles.add(norm_title(title))
        o.links += [u for u in urls if u]
    out = sorted(by_id.values(), key=lambda o: o.name.lower())
    for o in out:
        for link in o.links:
            if o.board is None and (found := detect_board(link)):
                o.board = found
            if o.other_ats is None:
                o.other_ats = next((ats for ats, pattern in OTHER_ATS if pattern.search(link)), None)
            if SHORT_LINKS.match(link) and link not in o.short_links:
                o.short_links.append(link)
    for o in out:
        tail_off = company_slug(_BOARD_NAME_TAILS.sub("", o.name))
        o.slugs = tuple(g for g in slug_guesses(o.name, [tail_off, tail_off.replace("-", "")]) if len(g) >= MIN_GUESS)
    claims: dict[str, str] = {}  # guess -> the squashed key of the first employer proposing it
    contested: set[str] = set()
    for o in out:
        key = company_slug(o.name).replace("-", "")
        contested.update(g for g in o.slugs if claims.setdefault(g, key) != key)
    for o in out:
        o.slugs = tuple(g for g in o.slugs if g not in contested)
    return out


def same_opening(jobs: Iterable[Job], titles: set[str]) -> bool:
    """Does the board list a job with a title we saw on the aggregator for this company? The same opening on both
    is the employer's evidence a namesake can't give (2.4 found three namesakes that a name match let through)."""
    return any(norm_title(j.title) in titles for j in jobs)
