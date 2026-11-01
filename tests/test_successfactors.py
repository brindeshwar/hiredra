"""SAP SuccessFactors provider (build step 3): cases ported from freehire's ``successfactors_test.go`` (MIT), plus the
localized redirect and robots.txt. Faked transport, no network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import successfactors as sf

DETAIL = """<html><head>
<meta property="og:title" content="Fallback Title"/>
</head><body>
<div itemscope itemtype="http://schema.org/JobPosting">
  <span itemprop="title">Commissioning Engineer</span>
  <div itemprop="description">   </div>
  <div itemprop="description"><h2>Role</h2><div><p>Build it.</p></div><script>alert(1)</script></div>
  <div itemprop="description"> </div>
</div></body></html>"""


def _sitemap(*locs):
    urls = "".join(f"<url><loc>{loc}</loc><lastmod>2026-06-06</lastmod></url>" for loc in locs)
    return f'<?xml version="1.0"?><urlset>{urls}</urlset>'


class _Resp:
    def __init__(self, url, text):
        self.url, self.text = url, text


class _Http:
    """Answers by the first route the URL ends with; ``(final_url, body)`` stands for a redirect. Else a 404."""

    def __init__(self, *routes):
        self.routes = routes
        self.asked: list[str] = []

    def request(self, method, url, *, follow_redirects=None, timeout=None):
        self.asked.append(url)
        for part, body in self.routes:
            if url.endswith(part):
                if isinstance(body, Exception):
                    raise body
                final, text = body if isinstance(body, tuple) else (url, body)
                return _Resp(final if follow_redirects else url, text)
        raise HTTPError("not found", status=404)


def _fetch(http, seen=lambda key: False):
    entry = CompanyEntry(name="Tetra Pak", provider="successfactors", extra={"slug": "jobs.tetrapak.com"})
    return sf.SuccessFactorsProvider().fetch(entry, FetchContext(http=http, seen=seen))


@pytest.mark.parametrize(
    ("loc", "want"),
    [
        ("https://jobs.tetrapak.com/job/Munich-Engineer/12345/", "12345"),
        ("https://jobs.tetrapak.com/job/Commissioning-Engineer/98012-en_GB", "98012"),
        ("https://jobs.tetrapak.com/job/Slug/883999301", "883999301"),
        ("https://jobs.tetrapak.com/job/Slug/", None),
    ],
)
def test_job_id(loc, want):
    assert sf.job_id(loc) == want


def test_sitemap_then_the_localized_page_and_maps():
    loc = "https://jobs.tetrapak.com/job/Munich-Engineer/12345/"
    http = _Http(
        ("/job_sitemap.xml", _sitemap(loc, "https://jobs.tetrapak.com/job/no-id/")),
        ("/job/Munich-Engineer/12345/", ("https://jobs.tetrapak.com/job/Munich-Engineer/101-en_GB/", DETAIL)),
    )
    [job] = _fetch(http)
    assert (job.job_id, job.url, job.title, job.company) == (
        "successfactors:jobs.tetrapak.com:12345", loc, "Commissioning Engineer", "Tetra Pak"
    )  # fmt: skip
    assert job.description == "Role\nBuild it."  # the richest description region, no script
    assert (job.location, job.date_posted) == (None, "2026-06-06")


def test_the_title_falls_back_to_og_title():
    page = '<head><meta property="og:title" content="OG &amp; Only"/></head><div itemprop="description">b</div>'
    http = _Http(("/job_sitemap.xml", _sitemap("https://jobs.tetrapak.com/job/X/777/")), ("/job/X/777/", page))
    assert _fetch(http)[0].title == "OG & Only"


def test_an_unreadable_page_waits_for_the_next_run_and_a_stored_posting_costs_nothing():
    kept, lost, empty = (f"https://jobs.tetrapak.com/job/{s}/{n}/" for s, n in (("Kept", 1), ("Lost", 2), ("Empty", 3)))
    stored = "https://jobs.tetrapak.com/job/Pune-Data-Engineer/4/"
    http = _Http(
        ("/job_sitemap.xml", _sitemap(kept, lost, empty, stored)),
        ("/job/Kept/1/", DETAIL),
        ("/job/Lost/2/", FetchError("timed out")),
        ("/job/Empty/3/", "<html>maintenance</html>"),
    )
    jobs = _fetch(http, seen=lambda key: key.endswith(":4"))
    assert [j.job_id.rsplit(":", 1)[1] for j in jobs] == ["1", "4"]
    assert jobs[1].title == "Pune Data Engineer" and jobs[1].date_posted == "2026-06-06"
    assert not any("/4/" in url for url in http.asked)


def test_a_page_that_redirects_off_the_site_is_refused():
    loc = "https://jobs.tetrapak.com/job/X/9/"
    http = _Http(("/job_sitemap.xml", _sitemap(loc)), ("/job/X/9/", ("https://evil.example/x", DETAIL)))
    assert _fetch(http) == []


def test_robots_txt_is_obeyed():
    loc = "https://jobs.tetrapak.com/job/X/9/"
    closed = _Http(("/robots.txt", "User-agent: *\nDisallow: /\n"), ("/job_sitemap.xml", _sitemap(loc)))
    with pytest.raises(FetchError, match="robots"):
        _fetch(closed)
    pages_closed = _Http(
        ("/robots.txt", "User-agent: *\nDisallow: /job/\n"), ("/job_sitemap.xml", _sitemap(loc)), ("/job/X/9/", DETAIL)
    )
    assert _fetch(pages_closed) == []


def test_an_empty_sitemap_is_an_empty_board_and_a_failed_one_an_error():
    assert _fetch(_Http(("/job_sitemap.xml", _sitemap()))) == []
    with pytest.raises(FetchError):
        _fetch(_Http())


def test_board_must_be_a_host():
    provider = sf.SuccessFactorsProvider()
    assert provider.board_url(CompanyEntry(name="X", provider="successfactors", extra={"slug": "jobs.x.com"}))
    assert provider.board_url(CompanyEntry(name="X", provider="successfactors", extra={"slug": "x/y"})) is None
