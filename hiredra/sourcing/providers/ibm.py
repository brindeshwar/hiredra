"""IBM provider -- the site-search API ibm.com/careers itself calls, no key (build step 3, wave 2).

Ported from freehire's ``internal/ingest/sources/ibm.go`` (MIT, see THIRD_PARTY_NOTICES.md). One employer: the board
is fixed, ``careers.ibm.com``. IBM's careers site (Avature) answers every crawler with a WAF challenge, but the
search page's own API (an Elasticsearch front door at www-api.ibm.com) holds every open posting with its whole
text. Pages of 100 are read in a total order (``id`` ascending: the page's own sort ties on an empty query and would
skip and repeat postings), and the walk must reach the exact count the index states, or it fails (2,218 postings,
measured 2026-11-02).

The index stores each page's text on one line with the posting's facet values appended; freehire's repairs are
kept: the suffix is cut, the one mojibake IBM's bodies carry is fixed, inline bullets go back on their own lines.
"""

from __future__ import annotations

import re

from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import clean, infer_remote, iso_date, work_mode

SEARCH = "https://www-api.ibm.com/search/api/v2"
BOARD = "careers.ibm.com"
_PAGE = 100  # the API's ceiling: 101 is a 400
_MAX_PAGES = 99  # the API refuses from >= 9,990
_TIMEOUT_S = 40.0
#: The fields asked for: the API returns no ``_source`` unless each is named.
_FIELDS = (
    "field_text_01",  # job id
    "title",
    "body",  # the page's text, flattened, the facets appended
    "dcdate",  # YYYY-MM-DD
    "field_keyword_05",  # country name
    "field_keyword_08",  # area of work
    "field_keyword_17",  # "Hybrid", "Remote" or "" (on-site is never stated)
    "field_keyword_18",  # "Professional", "Entry Level" or "Internship"
    "field_keyword_19",  # "Bangalore, IN" or "Multiple Cities"
    "field_keyword_20",  # hiring entity, "(0063) IBM India Private Limited"
)
_FACETS = ("field_keyword_05", "field_keyword_08", "field_keyword_17", "field_keyword_18", "field_keyword_19",
           "field_keyword_20")  # fmt: skip
_CITY_CODE = re.compile(r"^(.+),\s*([A-Za-z]{2})$")
_BULLET = re.compile(r"[\s  -   　]+[•·*●▪][\s  -   　]+")


def _body(start: int) -> dict:
    return {
        "appId": "careers", "scopes": ["careers2"], "query": {"bool": {"must": []}},
        "post_filter": {"bool": {"must": []}}, "size": _PAGE, "from": start, "sort": [{"id": "asc"}], "lang": "zz",
        "localeSelector": {}, "sm": {"query": "", "lang": "zz"}, "_source": list(_FIELDS),
    }  # fmt: skip


class IBMProvider:
    name = "ibm"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        return "https://www.ibm.com/careers/search" if (entry.extra or {}).get("slug") == BOARD else None

    def detect(self, entry: CompanyEntry) -> bool:
        return False  # one employer: a board is named explicitly, never recognised from a link

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        if (entry.extra or {}).get("slug") != BOARD:
            raise FetchError(f"ibm: the board is {BOARD}, not {(entry.extra or {}).get('slug')!r}")
        postings: dict[int, dict] = {}
        for page in range(_MAX_PAGES):
            data = ctx.http.request("POST", SEARCH, json=_body(page * _PAGE), timeout=_TIMEOUT_S).json()
            hits = (data.get("hits") or {}) if isinstance(data, dict) else {}
            total = hits.get("total") or {}
            if total.get("relation") != "eq" or not isinstance(hits.get("hits"), list):
                raise FetchError(f"ibm: page {page} states no exact total ({total})")
            for hit in hits["hits"]:
                source = hit.get("_source") if isinstance(hit, dict) else None
                if isinstance(source, dict) and isinstance(source.get("field_text_01"), int):
                    postings.setdefault(source["field_text_01"], source)
            if len(hits["hits"]) < _PAGE:
                if len(postings) != total.get("value"):
                    raise FetchError(
                        f"ibm: read {len(postings)} distinct postings, the index states {total.get('value')}"
                    )
                jobs = (to_job(entry.name, p) for p in postings.values())
                return [j for j in jobs if j is not None]
        raise FetchError(f"ibm: no end of the listing within {_MAX_PAGES} pages")


def description(posting: dict) -> str:
    body = (posting.get("body") or "").strip()
    suffix = " ".join(f for f in (clean(posting.get(k)) for k in _FACETS) if f)
    if suffix and body.endswith(suffix):
        body = body[: -len(suffix)].strip()
    body = body.replace("â€™", "’")  # the one mojibake in IBM's bodies (freehire: 1,351 of them, nothing else)
    return _BULLET.sub("\n• ", body).strip()


def location(city: str | None, country: str | None) -> str | None:
    """``<city>, <country>`` from the city facet ("BANGALORE, IN") and the country's name; "Multiple Cities" names no
    place, so the country alone. A country name with a comma ("Korea, Republic of") is not spliced in."""
    city, country = clean(city), clean(country)
    if not city or city.lower() == "multiple cities":
        return country
    m = _CITY_CODE.match(city)
    return f"{m.group(1).strip()}, {country}" if m and country and "," not in country else city


def to_job(company: str, posting: dict) -> Job | None:
    title, text = " ".join((posting.get("title") or "").split()), description(posting)
    if not title or not text:
        return None
    pid = posting["field_text_01"]
    place = location(posting.get("field_keyword_19"), posting.get("field_keyword_05"))
    mode = (clean(posting.get("field_keyword_17")) or "").lower() or None
    return Job(
        job_id=f"ibm:{pid}",
        url=f"https://careers.ibm.com/careers/JobDetail?jobId={pid}",
        title=title,
        company=company,
        source="ibm",
        description=text,
        location=place,
        date_posted=iso_date(posting.get("dcdate")),
        job_type="intern" if clean(posting.get("field_keyword_18")) == "Internship" else None,
        work_mode=work_mode(mode),
        is_remote=infer_remote(place or "", workplace_type=mode),
    )
