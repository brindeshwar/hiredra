"""Zoho Recruit provider (build step 3): cases ported from freehire's ``zoho_test.go`` (MIT), plus robots.txt, a gone
board and the stored-posting re-sighting. Faked transport, no network."""

from __future__ import annotations

import html
import json
from datetime import date, timedelta

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import zohorecruit as zoho

SITE = "https://acme.zohorecruit.com"


def _listing(openings, quote='"'):
    value = html.escape(json.dumps(openings), quote=True) if quote == '"' else json.dumps(openings)
    return f'<html><body><input type="hidden" value={quote}{value}{quote} id="jobs"><input id="x"></body></html>'


def _detail(description, opened=None):
    record = r"{\x22id\x22:\x221\x22,\x22Job_Description\x22:\x22" + description
    if opened is not None:
        record += r"\x22,\x22Date_Opened\x22:\x22" + opened
    return f'<html><body><script>var rec = "{record}\\x22,\\x22Country\\x22:null}}";</script></body></html>'


class _Http:
    """Answers by the URL; anything unrouted is a 404."""

    def __init__(self, routes):
        self.routes = routes
        self.asked: list[str] = []

    def get_text(self, url, *, timeout=None):
        self.asked.append(url)
        body = self.routes.get(url)
        if body is None:
            raise HTTPError("not found", status=404, url=url)
        if isinstance(body, Exception):
            raise body
        return body


def _fetch(http, seen=lambda job_id: False, board="acme.zohorecruit.com"):
    entry = CompanyEntry(name="Acme", provider="zohorecruit", extra={"slug": board})
    return zoho.ZohoRecruitProvider().fetch(entry, FetchContext(http=http, seen=seen))


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        (r"\x3Cp\x3EHello\x3C/p\x3E", "<p>Hello</p>"),
        (r"a\x22b\x22c", 'a"b"c'),
        (r"line\nbreak", "line\nbreak"),
        (r"path\/to", "path/to"),
        ("plain", "plain"),
        (r"<\\\/span>", "</span>"),  # escaped twice: one pass would leave a stray backslash
        (r"<\\\/p>", "</p>"),
        (r"\\u2022", "•"),
        (r"\\\x22q\\\x22", '"q"'),
        (r"margin\-top", "margin-top"),  # Zoho's stray backslash-hyphen
    ],
)
def test_js_unescape(raw, want):
    assert zoho.js_unescape(raw) == want


def test_the_opening_list_is_read_off_the_hidden_input_in_either_quoting():
    rows = [{"id": "1", "Posting_Title": "Engineer – Backend"}]
    assert zoho.openings(_listing(rows)) == rows
    assert zoho.openings(_listing(rows, quote="'")) == rows
    assert zoho.openings('<input id="other" value="[1]">') is None
    with pytest.raises(FetchError):
        zoho.openings('<input id="jobs" value="[{">')


def test_listing_then_each_new_posting_s_page():
    rows = [
        {"id": "100", "Posting_Title": "Backend Engineer", "City": "Lisbon", "Country": "Portugal", "Remote_Job": False,
         "Publish": True, "Job_Type": "Full time"},
        {"id": "200", "Posting_Title": "Remote Designer", "City": None, "Country": None, "Remote_Job": True,
         "Publish": True, "Job_Type": "Internship"},
        {"id": "300", "Posting_Title": "Draft Role", "Publish": False},
    ]  # fmt: skip
    http = _Http({
        f"{SITE}/jobs/Careers": _listing(rows),
        f"{SITE}/jobs/Careers/100": _detail(r"\x3Cp\x3EBuild things\x3C/p\x3E", r"2024\-11\-11"),
        f"{SITE}/jobs/Careers/200": _detail(r"\x3Cp\x3EDesign things\x3C/p\x3E"),
    })  # fmt: skip
    backend, designer = _fetch(http)
    assert http.asked == [f"{SITE}/robots.txt", f"{SITE}/jobs/Careers", f"{SITE}/jobs/Careers/100",
                          f"{SITE}/jobs/Careers/200"]  # the unpublished 300 is never asked  # fmt: skip
    assert (backend.job_id, backend.title, backend.company) == ("zohorecruit:acme.zohorecruit.com:100",
                                                                "Backend Engineer", "Acme")  # fmt: skip
    assert (backend.url, backend.location) == (f"{SITE}/jobs/Careers/100", "Lisbon, Portugal")
    assert (backend.description, backend.date_posted, backend.job_type) == ("Build things", "2024-11-11", "fulltime")
    assert backend.is_remote is None
    assert (designer.location, designer.is_remote, designer.date_posted) == (None, True, None)
    assert designer.job_type == "intern"


def test_a_double_escaped_description_leaks_no_markup():
    page = _detail(r"<p style=\\\x22margin\-top:0px\\\x22>Duties\\u2022audit files.<br\/><\\\/p>")
    [job] = _fetch(_Http({
        f"{SITE}/jobs/Careers": _listing([{"id": "1", "Posting_Title": "Auditor", "Publish": True}]),
        f"{SITE}/jobs/Careers/1": page,
    }))  # fmt: skip
    assert job.description == "Duties•audit files."


@pytest.mark.parametrize(
    ("opened", "want"),
    [(r"2024\-11\-11", "2024-11-11"), (None, None), (r"2024\-13\-45", None),
     ((date.today() + timedelta(days=30)).isoformat().replace("-", r"\-"), None)],
)  # fmt: skip
def test_the_posting_date_is_the_record_s_date_opened_never_a_future_one(opened, want):
    [job] = _fetch(_Http({
        f"{SITE}/jobs/Careers": _listing([{"id": "1", "Posting_Title": "X", "Publish": True}]),
        f"{SITE}/jobs/Careers/1": _detail("Body", opened),
    }))  # fmt: skip
    assert job.date_posted == want


def test_a_stored_posting_is_re_sighted_without_its_page_and_one_without_text_waits():
    rows = [{"id": "1", "Posting_Title": "Stored", "Publish": True},
            {"id": "2", "Posting_Title": "Empty page", "Publish": True}]  # fmt: skip
    http = _Http({f"{SITE}/jobs/Careers": _listing(rows), f"{SITE}/jobs/Careers/2": "<html>no record</html>"})
    [stored] = _fetch(http, seen=lambda job_id: job_id.endswith(":1"))
    assert (stored.title, stored.description) == ("Stored", None)
    assert f"{SITE}/jobs/Careers/1" not in http.asked


def test_robots_txt_is_obeyed():
    http = _Http({f"{SITE}/robots.txt": "User-agent: *\nDisallow: /", f"{SITE}/jobs/Careers": _listing([])})
    with pytest.raises(FetchError):
        _fetch(http)


def test_a_host_that_is_no_tenant_is_a_404_and_a_page_without_the_list_an_error():
    gone = "<html><head><title>Page does not exist</title></head><body>x does not exist.</body></html>"
    with pytest.raises(HTTPError) as err:
        _fetch(_Http({f"{SITE}/jobs/Careers": gone}))
    assert err.value.status == 404
    with pytest.raises(FetchError) as err:
        _fetch(_Http({f"{SITE}/jobs/Careers": "<html>maintenance</html>"}))
    assert not isinstance(err.value, HTTPError)


@pytest.mark.parametrize("bad", ["", "acme", "https://acme.zohorecruit.com", "acme.zohorecruit.com/jobs"])
def test_a_malformed_board_is_refused(bad):
    with pytest.raises(FetchError):
        _fetch(_Http({}), board=bad)
