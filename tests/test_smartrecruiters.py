"""SmartRecruiters provider (build step 3): cases ported from freehire's ``smartrecruiters_test.go`` (MIT). Faked
transport, no network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import smartrecruiters as sr


class _Http:
    def __init__(self, *routes):
        self.routes = routes
        self.asked: list[str] = []

    def get_json(self, url, *, timeout=None):
        self.asked.append(url)
        for part, body in self.routes:
            if part in url:
                return body
        raise HTTPError("not found", status=404)


def _detail(pid, **sections):
    sections = sections or {
        "companyDescription": {"text": "<p>Company boilerplate.</p>"},
        "jobDescription": {"text": "<p>Come do the job.</p>"},
        "qualifications": {"text": "<ul><li>Go</li></ul>"},
        "additionalInformation": {"text": "<p>EEO notice.</p>"},
    }
    return {"postingUrl": f"https://jobs.smartrecruiters.com/Acme/{pid}", "typeOfEmployment": {"id": "permanent"},
            "jobAd": {"sections": sections}}  # fmt: skip


def _fetch(http, seen=lambda key: False):
    entry = CompanyEntry(name="Acme", provider="smartrecruiters", extra={"slug": "Acme"})
    return sr.SmartRecruitersProvider().fetch(entry, FetchContext(http=http, seen=seen))


def test_pages_and_fetches_each_detail():
    http = _Http(
        ("offset=0", {"totalFound": 3, "content": [
            {"id": "P1", "name": "Backend Engineer", "releasedDate": "2024-06-11T15:19:46.134Z",
             "location": {"city": "Berlin", "region": "", "country": "de", "remote": True}},
            {"id": "P2", "name": "Frontend Engineer", "location": {"city": "Pune", "country": "in", "hybrid": True}},
        ]}),
        ("offset=2", {"totalFound": 3, "content": [{"id": "P3", "name": "Data Engineer", "location": {"city": "NYC"}}]}),
        *((f"/postings/{p}", _detail(p)) for p in ("P1", "P2", "P3")),
    )  # fmt: skip
    p1, p2, p3 = _fetch(http)
    assert p1.url == "https://jobs.smartrecruiters.com/Acme/P1" and p1.location == "Berlin, de"
    assert p1.is_remote is True and p2.is_remote is False and p3.is_remote is None
    assert (
        all(t in p1.description for t in ("do the job.", "Go", "EEO notice.")) and "boilerplate" not in p1.description
    )
    assert p1.date_posted == "2024-06-11" and p1.job_type == "fulltime" and p1.job_id == "smartrecruiters:P1"


def test_the_company_section_is_the_ad_only_when_the_role_sections_are_empty():
    only_company = _detail("P", companyDescription={"text": "<p>The whole ad.</p>"}, jobDescription={"text": " "})
    assert sr.description(only_company) == "The whole ad."


def test_a_stored_posting_costs_no_detail_and_an_empty_answer_is_skipped():
    listing = {"totalFound": 2, "content": [{"id": "A", "name": "A"}, {"id": "B", "name": "B"}]}
    http = _Http(("offset=0", listing), ("/postings/B", {"postingUrl": ""}))
    jobs = _fetch(http, seen=lambda key: key == "smartrecruiters:A")
    assert [(j.title, j.description) for j in jobs] == [("A", None)]  # B's detail answered nothing usable
    assert not any(u.endswith("/postings/A") for u in http.asked)


@pytest.mark.parametrize("board", ["", "acme/../x", "acme?x=1"])
def test_a_malformed_board_is_refused(board):
    entry = CompanyEntry(name="X", provider="smartrecruiters", extra={"slug": board})
    with pytest.raises(FetchError):
        sr.SmartRecruitersProvider().fetch(entry, FetchContext(http=_Http()))
