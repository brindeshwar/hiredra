"""EPAM provider (parity step): cases ported from freehire's ``epam_test.go`` (MIT), plus the office place, the
newest-first order and robots.txt. Made-up pages, faked transport, no network."""

from __future__ import annotations

import json

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import epam

SITE = "https://careers.epam.com"


def _sitemap(*locs):
    urls = "".join(
        f"<url><loc>{loc}</loc><lastmod>{lastmod}</lastmod></url>" if lastmod else f"<url><loc>{loc}</loc></url>"
        for loc, lastmod in (loc if isinstance(loc, tuple) else (loc, None) for loc in locs)
    )
    return f'<?xml version="1.0" encoding="UTF-8"?><urlset>{urls}</urlset>'


def _page(title="Role", description="<p>Flat text.</p>", remote=True, countries=("Poland",), office=None, job=None):
    posting = {"@context": "https://schema.org/", "@type": "JobPosting", "title": title, "description": description,
               "datePosted": "2026-06-18", "applicantLocationRequirements": [{"@type": "Country", "name": c}
                                                                             for c in countries]}  # fmt: skip
    if remote:
        posting["jobLocationType"] = "TELECOMMUTE"
    if office:
        posting["jobLocation"] = [{"@type": "Place", "address": {"@type": "PostalAddress", "addressCountry": office}}]
    script = f'<script type="application/ld+json">{json.dumps(posting)}</script>'
    if job is not None:
        data = json.dumps({"props": {"pageProps": {"job": job}}})
        script += f'<script id="__NEXT_DATA__" type="application/json">{data}</script>'
    return f"<html><head>{script}</head><body></body></html>"


class _Http:
    def __init__(self, pages, robots=None):
        self.pages, self.robots = pages, robots
        self.asked: list[str] = []

    def get_text(self, url, *, timeout=None):
        self.asked.append(url)
        if url.endswith("/robots.txt"):
            if self.robots is None:
                raise HTTPError("none", status=404, url=url)
            return self.robots
        page = self.pages.get(url) or self.pages.get(url.removeprefix(SITE))
        if page is None:
            raise HTTPError("not found", status=404, url=url)
        return page


def _fetch(http, seen=None, board="careers.epam.com"):
    entry = CompanyEntry(name="EPAM Systems", provider="epam", extra={"slug": board})
    return epam.EPAMProvider().fetch(entry, FetchContext(http=http, seen=seen or (lambda key: False)))


@pytest.mark.parametrize(
    ("loc", "want"),
    [
        (
            f"{SITE}/en/vacancy/senior-platform-engineer-bltu03h0s7z1ufrl6d9_en",
            ("senior-platform-engineer", "bltu03h0s7z1ufrl6d9"),
        ),
        (f"{SITE}/en/vacancy/role-blt111aaa_en/", ("role", "blt111aaa")),
        (f"{SITE}/uk/vacancy/role-blt222bbb_uk", None),  # a localisation: the English link is the one read
        (f"{SITE}/en/vacancy/role-blt333ccc_de", None),
        (f"{SITE}/en/jobs", None),
        (f"{SITE}/en", None),
    ],
)
def test_vacancy_reads_the_uid_from_an_english_link_only(loc, want):
    assert epam.vacancy(loc) == want


def test_the_sitemap_then_each_vacancys_page():
    en = f"{SITE}/en/vacancy/backend-developer-blt111aaa_en"
    http = _Http({
        "/sitemap.xml.gz": _sitemap(f"{SITE}/en", f"{SITE}/en/jobs", f"{SITE}/uk/vacancy/role-blt222bbb_uk", en),
        en: _page("Backend Developer", countries=("India", "Poland")),
    })  # fmt: skip
    (job,) = _fetch(http)
    assert (job.job_id, job.url, job.title, job.company, job.source) == (
        "epam:blt111aaa",
        en,
        "Backend Developer",
        "EPAM Systems",
        "epam",
    )
    assert (job.location, job.is_remote, job.work_mode, job.date_posted) == (
        "Remote (India, Poland)",
        True,
        "remote",
        "2026-06-18",
    )
    assert job.description == "Flat text."  # no page data: the JobPosting's text
    assert http.asked.count(en) == 1


def test_an_office_vacancy_is_placed_by_its_office_country():
    en = f"{SITE}/en/vacancy/lead-data-modeler-blt1office_en"
    http = _Http({"/sitemap.xml.gz": _sitemap(en), en: _page(remote=False, countries=(), office="India")})
    (job,) = _fetch(http)
    assert (job.location, job.work_mode) == ("India", None)


@pytest.mark.parametrize(
    ("posting", "want"),
    [
        ({"applicantLocationRequirements": [{"name": "Colombia"}, {"name": "Mexico"}]}, "Colombia; Mexico"),
        ({"jobLocationType": "TELECOMMUTE", "applicantLocationRequirements": {"name": "Mexico"}}, "Remote (Mexico)"),
        ({"jobLocationType": "TELECOMMUTE"}, "Remote"),
        ({"jobLocation": [{"address": {"addressCountry": "UK"}}, {"address": {"addressCountry": "UK"}}]}, "UK"),
        ({}, None),
    ],
)
def test_location(posting, want):
    assert epam.location(posting) == want


def test_the_page_datas_sections_are_the_text():
    en = f"{SITE}/en/vacancy/role-blt111aaa_en"
    job = {"description": "<p>Join the AI pod.</p>", "benefits": [{"content": "<ul><li>Insurance</li></ul>"}],
           "category": {"responsibilities": ["Build services", "Requirements"],  # a leaked heading, dropped
                        "requirements": ["3+ years of Python", " ", "Nice to have"],
                        "nice_to_have": ["LangGraph"]}}  # fmt: skip
    http = _Http(
        {"/sitemap.xml.gz": _sitemap(en), en: _page(description="flattened Requirements Requirements", job=job)}
    )
    (got,) = _fetch(http)
    assert got.description == (
        "Join the AI pod.\n\nResponsibilities\n- Build services\n\nRequirements\n- 3+ years of Python\n\n"
        "Nice to have\n- LangGraph"
    )  # benefits left out


def test_page_data_with_no_section_falls_back_to_the_jobposting_text():
    en = f"{SITE}/en/vacancy/role-blt111aaa_en"
    http = _Http({"/sitemap.xml.gz": _sitemap(en), en: _page(description="<p>Flat.</p>", job={"category": {}})})
    assert _fetch(http)[0].description == "Flat."


def test_a_failed_or_empty_page_drops_only_that_vacancy():
    kept, failed, empty = (f"{SITE}/en/vacancy/{s}-blt{s}_en" for s in ("kept", "fail", "empty"))
    http = _Http({"/sitemap.xml.gz": _sitemap(kept, failed, empty), kept: _page("Kept"), empty: "<html></html>"})
    assert [j.job_id for j in _fetch(http)] == ["epam:bltkept"]


def test_the_newest_unseen_are_read_first_and_a_stored_one_costs_nothing():
    old, new, stored = (f"{SITE}/en/vacancy/{s}-blt{s}_en" for s in ("old", "new", "stored"))
    http = _Http({
        "/sitemap.xml.gz": _sitemap((old, "2026-09-01"), (stored, "2026-11-01"), (new, "2026-11-02")),
        old: _page("Old"), new: _page("New"),
    })  # fmt: skip
    jobs = _fetch(http, seen=lambda key: key == "epam:bltstored")
    assert [j.title for j in jobs] == ["New", "stored", "Old"]  # the re-sighted one titled from its link
    assert stored not in http.asked
    assert [u for u in http.asked if "/vacancy/" in u] == [new, old]


def test_a_sitemap_with_no_vacancy_fails_the_board():
    with pytest.raises(FetchError, match="lists no vacancies"):
        _fetch(_Http({"/sitemap.xml.gz": _sitemap(f"{SITE}/en", f"{SITE}/en/jobs")}))


def test_robots_txt_is_obeyed():
    en, blocked = f"{SITE}/en/vacancy/role-blt111aaa_en", f"{SITE}/en/vacancy/hidden-blt999zzz_en"
    robots = "User-agent: *\nDisallow: /en/vacancy/hidden\n"
    http = _Http({"/sitemap.xml.gz": _sitemap(en, blocked), en: _page()}, robots=robots)
    assert [j.job_id for j in _fetch(http)] == ["epam:blt111aaa"]
    assert blocked not in http.asked
    with pytest.raises(FetchError, match="disallows"):
        _fetch(_Http({}, robots="User-agent: *\nDisallow: /sitemap\n"))


def test_only_the_epam_site_is_a_board():
    provider = epam.EPAMProvider()
    assert provider.board_url(CompanyEntry(name="EPAM", provider="epam", extra={"slug": "careers.epam.com"})) == (
        f"{SITE}/en/jobs"
    )
    assert provider.board_url(CompanyEntry(name="X", provider="epam", extra={"slug": "example.com"})) is None
    assert not provider.detect(CompanyEntry(name="EPAM", careers_url=f"{SITE}/en/jobs"))
    with pytest.raises(FetchError, match="is not careers.epam.com"):
        _fetch(_Http({}), board="example.com")
