"""iCIMS provider (build step 3): cases ported from freehire's ``icims_test.go`` (MIT), plus the page URL chosen by
the posting link's shape and robots.txt. Faked transport, no network."""

from __future__ import annotations

import json

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import icims

#: A ``?in_iframe=1`` page: its only payload we read is the JobPosting ld+json. The description's script is written
#: ``<\/script>`` so the block isn't cut short; the text keeps none of it. ``UNAVAILABLE`` never reaches the location.
DETAIL = r"""<html><head></head><body>
<script type="application/ld+json">
{"@context":"http://schema.org","@type":"JobPosting",
"title":"Mobile Medical Assistant",
"description":"<h2>Overview</h2><p>Assist physicians.</p><script>alert(1)<\/script>",
"datePosted":"2026-04-15T00:00:00.000Z",
"hiringOrganization":{"@type":"Organization","name":"360care Inc"},
"jobLocation":[{"@type":"Place","address":{"@type":"PostalAddress",
"streetAddress":"UNAVAILABLE","postalCode":"UNAVAILABLE",
"addressLocality":"Kalamazoo","addressRegion":"MI","addressCountry":"US"}}]}
</script>
</body></html>"""


def _sitemap(*locs):
    urls = "".join(f"<url><loc>{loc}</loc><lastmod>2026-06-16T15:51:06-04:00</lastmod></url>" for loc in locs)
    return f'<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>'


def _index(*locs):
    subs = "".join(f"<sitemap><loc>{loc}</loc></sitemap>" for loc in locs)
    return (
        f'<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{subs}</sitemapindex>'
    )


def _posting(title="Role", **fields):
    body = {"@type": "JobPosting", "title": title, "datePosted": "2026-01-01T00:00:00.000Z", **fields}
    return f'<html><body><script type="application/ld+json">{json.dumps(body)}</script></body></html>'


class _Http:
    """Answers by the first route whose path-and-query ends the URL; anything else is a 404."""

    def __init__(self, *routes):
        self.routes = routes
        self.asked: list[str] = []

    def get_text(self, url, *, timeout=None):
        self.asked.append(url)
        for part, body in self.routes:
            if url.endswith(part):
                if isinstance(body, Exception):
                    raise body
                return body
        raise HTTPError("not found", status=404)


def _fetch(http, board="360care", seen=lambda key: False):
    entry = CompanyEntry(name="360care", provider="icims", extra={"slug": board})
    return icims.ICIMSProvider().fetch(entry, FetchContext(http=http, seen=seen))


def test_host_and_board_forms():
    assert icims.host("360care") == "careers-360care.icims.com"
    assert icims.host("globalcareers-msci.icims.com") == "globalcareers-msci.icims.com"
    assert icims.host("careers.docusign.com") == "careers.docusign.com"
    with pytest.raises(FetchError):
        icims.host("bad/board")


@pytest.mark.parametrize(
    ("loc", "want"),
    [
        ("https://careers-360care.icims.com/jobs/3787/mobile-medical-assistant/job", "3787"),
        ("https://careers-acme.icims.com/jobs/12345/some-role/job?in_iframe=1", "12345"),
        ("https://careers-acme.icims.com/jobs/search", None),
        ("https://careers-acme.icims.com/jobs/intro", None),
        ("https://careers.docusign.com/jobs/27897?lang=en-us", "27897"),
        ("https://careers.docusign.com/jobs/29339", "29339"),
    ],
)
def test_job_id(loc, want):
    assert icims.job_id(loc) == want


def test_sitemap_then_page_and_maps():
    loc = "https://careers-360care.icims.com/jobs/3787/mobile-medical-assistant/job"
    http = _Http(
        ("/sitemap.xml", _sitemap("https://careers-360care.icims.com/jobs/search", loc)),
        ("/jobs/3787/mobile-medical-assistant/job?in_iframe=1", DETAIL),
    )
    [job] = _fetch(http)
    assert job.job_id == "icims:careers-360care.icims.com:3787"
    assert job.url == loc  # the canonical link, not the iframe page
    assert (job.title, job.company, job.location) == ("Mobile Medical Assistant", "360care", "Kalamazoo, MI, US")
    assert job.description == "Overview\nAssist physicians."
    assert job.date_posted == "2026-04-15"
    assert job.is_remote is None


def test_a_prefixed_icims_host_reads_the_link_plus_iframe():
    """``globalcareers-msci.icims.com`` is a whole-host board, but its pages are classic links: freehire asks
    ``/careers-home/jobs/<id>`` there, which is a 404 (measured 2026-11-01)."""
    loc = "https://globalcareers-msci.icims.com/jobs/5708/hris-specialist/job"
    http = _Http(("/sitemap.xml", _sitemap(loc)), ("/jobs/5708/hris-specialist/job?in_iframe=1", _posting("HRIS")))
    [job] = _fetch(http, board="globalcareers-msci.icims.com")
    assert job.title == "HRIS"


def test_a_careers_home_site_follows_its_index_and_reads_the_careers_home_page():
    loc = "https://careers.docusign.com/jobs/27897?lang=en-us"
    http = _Http(
        ("/sitemap.xml", _index("https://careers.docusign.com/sitemap1.xml")),
        ("/sitemap1.xml", _sitemap(loc)),
        ("/careers-home/jobs/27897?in_iframe=1", DETAIL),
    )
    [job] = _fetch(http, board="careers.docusign.com")
    assert (job.job_id, job.url, job.title) == ("icims:careers.docusign.com:27897", loc, "Mobile Medical Assistant")


def test_a_failed_sub_sitemap_fails_the_board():
    http = _Http(
        (
            "/sitemap.xml",
            _index("https://careers.docusign.com/sitemap1.xml", "https://careers.docusign.com/sitemap2.xml"),
        ),
        ("/sitemap1.xml", _sitemap("https://careers.docusign.com/jobs/1?lang=en-us")),
    )
    with pytest.raises(FetchError):
        _fetch(http, board="careers.docusign.com")


def test_a_single_object_location_and_the_placeholder():
    loc = "https://careers-acme.icims.com/jobs/555/solo-role/job"
    one = {"address": {"addressLocality": "Seattle", "addressRegion": "Washington", "addressCountry": "US"}}
    http = _Http(("/sitemap.xml", _sitemap(loc)), ("?in_iframe=1", _posting(jobLocation=one)))
    assert _fetch(http, board="acme")[0].location == "Seattle, Washington, US"

    gap = [{"address": {"addressLocality": "Austin", "addressRegion": "UNAVAILABLE", "addressCountry": "US"}}]
    http = _Http(("/sitemap.xml", _sitemap(loc)), ("?in_iframe=1", _posting(jobLocation=gap)))
    assert _fetch(http, board="acme")[0].location == "Austin, US"


def test_telecommute_is_remote():
    loc = "https://careers-acme.icims.com/jobs/9/remote-role/job"
    http = _Http(("/sitemap.xml", _sitemap(loc)), ("?in_iframe=1", _posting(jobLocationType="TELECOMMUTE")))
    assert _fetch(http, board="acme")[0].is_remote is True


def test_an_unreadable_page_is_skipped_for_the_next_run_and_a_stored_posting_costs_nothing():
    kept = "https://careers-acme.icims.com/jobs/111/kept/job"
    lost = "https://careers-acme.icims.com/jobs/222/lost/job"
    empty = "https://careers-acme.icims.com/jobs/333/empty/job"
    stored = "https://careers-acme.icims.com/jobs/444/data-engineer%2c-ii/job"
    http = _Http(
        ("/sitemap.xml", _sitemap(kept, lost, empty, stored)),
        ("/jobs/111/kept/job?in_iframe=1", DETAIL),
        ("/jobs/222/lost/job?in_iframe=1", FetchError("timed out")),
        ("/jobs/333/empty/job?in_iframe=1", "<html>no posting here</html>"),
    )
    jobs = _fetch(http, board="acme", seen=lambda key: key.endswith(":444"))
    assert [j.job_id.rsplit(":", 1)[1] for j in jobs] == ["111", "444"]
    assert jobs[1].title == "data engineer, ii"  # re-sighted from its link; the store keeps the page's title
    assert not any("/444/" in url for url in http.asked)


def test_robots_txt_is_obeyed():
    loc = "https://careers-acme.icims.com/jobs/9/role/job"
    closed = _Http(("/robots.txt", "User-agent: *\nDisallow: /\n"), ("/sitemap.xml", _sitemap(loc)))
    with pytest.raises(FetchError, match="robots"):
        _fetch(closed, board="acme")
    assert not any(url.endswith("/sitemap.xml") for url in closed.asked)

    pages_closed = _Http(
        ("/robots.txt", "User-agent: *\nDisallow: /jobs/9/\n"),
        ("/sitemap.xml", _sitemap(loc)),
        ("?in_iframe=1", DETAIL),
    )
    assert _fetch(pages_closed, board="acme") == []
    assert not any("in_iframe" in url for url in pages_closed.asked)


def test_an_empty_sitemap_is_an_empty_board_and_a_non_sitemap_an_error():
    assert _fetch(_Http(("/sitemap.xml", _sitemap())), board="acme") == []
    with pytest.raises(FetchError):
        _fetch(_Http(("/sitemap.xml", "<html>maintenance</html>")), board="acme")


def test_a_link_off_the_board_site_is_refused():
    off = "https://evil.example/jobs/1/x/job"
    http = _Http(("/sitemap.xml", _sitemap(off)), ("?in_iframe=1", DETAIL))
    assert _fetch(http, board="acme") == []
    assert not any("evil.example" in url for url in http.asked)


def test_board_url_and_detect():
    provider = icims.ICIMSProvider()
    entry = CompanyEntry(name="Acme", provider="icims", extra={"slug": "acme"})
    assert provider.board_url(entry) == "https://careers-acme.icims.com/jobs/search"
    assert provider.detect(entry)
    assert not provider.detect(CompanyEntry(name="Acme", provider="lever", extra={"slug": "acme"}))
