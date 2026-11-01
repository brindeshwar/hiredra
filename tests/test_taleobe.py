"""Taleo Business Edition provider (build step 3): cases ported from freehire's ``taleobe_test.go`` (MIT), on a small
made-up feed of the same shape. Faked transport, no network."""

from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import taleobe

FEED = """<?xml version="1.0" encoding="UTF-8" ?>
<rss xmlns:taleo="urn:TBERss" version="2.0"><channel><title>Acme Job Feed</title>
<item>
<title>Data Engineer</title>
<link>http://phf.tbe.taleo.net/phf01/ats/careers/requisition.jsp?org=ACME&amp;cws=37&amp;rid=14016</link>
<pubDate>Tue, 20 Oct 2026 13:12:18 GMT</pubDate>
<description>Plain text body</description>
<taleo:html-description>&lt;p&gt;Build &lt;b&gt;pipelines&lt;/b&gt; in C++.&lt;/p&gt;</taleo:html-description>
<taleo:reqId>14016</taleo:reqId>
<taleo:location>Kochi, India-Corporate</taleo:location>
<taleo:locationCountry>IN</taleo:locationCountry>
</item>
<item>
<title>Support Specialist</title>
<link>https://phf.tbe.taleo.net/phf01/ats/careers/requisition.jsp?org=ACME&amp;cws=37&amp;rid=14020</link>
<description>Text only</description>
<taleo:locationCity>Remote</taleo:locationCity><taleo:locationState>TX</taleo:locationState>
</item>
<item><title>No id</title><link>https://phf.tbe.taleo.net/x</link></item>
</channel></rss>"""


class _Resp:
    def __init__(self, url, text):
        self.url, self.text = url, text


class _Http:
    def __init__(self, final, text):
        self.final, self.text = final, text
        self.asked: list[str] = []

    def request(self, method, url, *, follow_redirects=None, timeout=None):
        self.asked.append(url)
        return _Resp(self.final, self.text)


def _fetch(http, board="ACME:37"):
    entry = CompanyEntry(name="Acme", provider="taleobe", extra={"slug": board})
    return taleobe.TaleoBEProvider().fetch(entry, FetchContext(http=http))


def test_feed_url_goes_through_the_dispatcher():
    url = urlsplit(taleobe.feed_url("AMERINC:52"))
    assert (url.hostname, url.path) == ("tbe.taleo.net", "/dispatcher/servlet/DispatcherServlet")
    query = {k: v[0] for k, v in parse_qs(url.query).items()}
    assert query == {"org": "AMERINC", "act": "redirectOrg",
                     "redirectUrl": "https://tbe.taleo.net/ats/servlet/Rss?org=AMERINC&cws=52"}  # fmt: skip


@pytest.mark.parametrize("bad", ["", "AMERINC", "AMERINC:", ":52", "AMERINC:x", "amerinc:52", "phh02/AMERINC:52"])
def test_a_malformed_board_is_refused(bad):
    with pytest.raises(FetchError):
        taleobe.feed_url(bad)


def test_one_feed_lists_the_board_with_its_text():
    http = _Http("https://phf.tbe.taleo.net/phf01/ats/servlet/Rss?org=ACME&cws=37", FEED)
    data, support = _fetch(http)
    assert http.asked == [taleobe.feed_url("ACME:37")]
    assert (data.job_id, data.title, data.company) == ("taleobe:ACME:14016", "Data Engineer", "Acme")
    assert data.url == "https://phf.tbe.taleo.net/phf01/ats/careers/requisition.jsp?org=ACME&cws=37&rid=14016"
    assert (data.location, data.date_posted, data.is_remote) == ("Kochi, India-Corporate", "2026-10-20", None)
    assert data.description == "Build pipelines in C++."  # the HTML body, not the plain one
    assert (support.job_id, support.location, support.is_remote) == ("taleobe:ACME:14020", "Remote, TX", True)
    assert support.description == "Text only" and support.date_posted is None


@pytest.mark.parametrize(
    "final",
    [
        "https://phf.tbe.taleo.net/dispatcher/login.jsp?loginErrorMsg=x",
        "https://tbe.taleo.net/errorpages/page_not_found_404.html.en",
        "https://phf.tbe.taleo.net/phf02/ats/rssErrMsg.jsp?org=ACME&orgCode=ACME",
    ],
)
def test_a_gone_board_is_a_404(final):
    with pytest.raises(HTTPError) as err:
        _fetch(_Http(final, "<!DOCTYPE html><html></html>"))
    assert err.value.status == 404


def test_a_feed_that_is_not_rss_or_leaves_taleo_is_an_error():
    with pytest.raises(FetchError):
        _fetch(_Http("https://phf.tbe.taleo.net/phf01/ats/servlet/Rss", "<html"))
    with pytest.raises(FetchError):
        _fetch(_Http("https://evil.example/rss", FEED))
    empty = '<rss xmlns:taleo="urn:TBERss" version="2.0"><channel><title>x</title></channel></rss>'
    assert _fetch(_Http("https://phf.tbe.taleo.net/rss", empty)) == []
