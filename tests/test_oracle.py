"""Oracle Recruiting Cloud provider (build step 3): cases ported from freehire's ``oracle_test.go`` (MIT). Faked
transport, no network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import oracle

BOARD = "fa-test.fa.ocs.oraclecloud.com/CX_1"


class _Http:
    """Answers by the first route whose substring is in the URL."""

    def __init__(self, *routes):
        self.routes = routes
        self.asked: list[str] = []

    def get_json(self, url, *, timeout=None):
        self.asked.append(url)
        for part, body in self.routes:
            if part in url:
                return body
        raise HTTPError("not found", status=404)


def _req(rid, title="Engineer", **kw):
    return {"Id": rid, "Title": title, "PostedDate": "2026-06-16", **kw}


def _list(*reqs, total=None):
    return {"items": [{"TotalJobsCount": len(reqs) if total is None else total, "requisitionList": list(reqs)}]}


def _fetch(http, seen=lambda key: False):
    entry = CompanyEntry(name="Acme", provider="oracle", extra={"slug": BOARD})
    return oracle.OracleProvider().fetch(entry, FetchContext(http=http, seen=seen))


def test_lists_and_fetches_each_detail():
    http = _Http(
        ("findReqs", _list(
            _req("30607", "Backend Engineer", PrimaryLocation="Berlin, Germany", WorkplaceTypeCode="ORA_ON_SITE"),
            _req("30610", "Data Engineer", PrimaryLocation="United States", WorkplaceTypeCode="ORA_REMOTE"),
        )),
        ("30607", {"items": [{
            "ExternalDescriptionStr": "<p>Build the backend.</p>",
            "ExternalResponsibilitiesStr": "<ul><li>Own services</li></ul>",
            "ExternalQualificationsStr": "<p>Go experience.</p>", "JobSchedule": "Full time",
        }]}),
        ("30610", {"items": [{"ExternalDescriptionStr": "<p>Crunch data.</p>"}]}),
    )  # fmt: skip
    backend, data = _fetch(http)
    assert backend.url == "https://fa-test.fa.ocs.oraclecloud.com/hcmUI/CandidateExperience/en/sites/CX_1/job/30607"
    assert all(t in backend.description for t in ("Build the backend.", "Own services", "Go experience."))
    assert backend.is_remote is False and backend.job_type == "fulltime" and backend.date_posted == "2026-06-16"
    assert data.is_remote is True and backend.job_id == "oracle:fa-test.fa.ocs.oraclecloud.com:30607"


def test_blank_external_sections_fall_back_to_the_corporate_text_but_escaped_text_is_text():
    blank = {"ExternalDescriptionStr": "<br>", "CorporateDescriptionStr": "<p>About the program.</p>",
             "OrganizationDescriptionStr": "<p>About the division.</p>"}  # fmt: skip
    assert all(t in oracle.description(blank) for t in ("About the program.", "About the division."))
    literal = {"ExternalDescriptionStr": "<p>&lt;tag&gt;</p>", "CorporateDescriptionStr": "<p>Fallback</p>"}
    assert "<tag>" in oracle.description(literal)


def test_the_offset_sits_inside_the_finder_and_paging_stops_at_the_total():
    http = _Http(
        (",offset=0,", _list(_req("1"), _req("2"), total=3)),
        (",offset=2,", _list(_req("3"), total=3)),
        ("ById", {"items": [{"ExternalDescriptionStr": "<p>desc</p>"}]}),
    )
    assert [j.job_id.rsplit(":", 1)[1] for j in _fetch(http)] == ["1", "2", "3"]
    assert sum("findReqs" in u for u in http.asked) == 2


def test_a_stored_requisition_costs_no_detail_and_an_empty_detail_is_skipped():
    http = _Http(("findReqs", _list(_req("1"), _req("2"))), ("Id=%222%22", {"items": []}))
    jobs = _fetch(http, seen=lambda key: key.endswith(":1"))
    assert [(j.title, j.description) for j in jobs] == [("Engineer", None)]  # 1 re-sighted; 2 answered nothing
    assert not any("Id=%221%22" in u for u in http.asked)


@pytest.mark.parametrize("board", ["", "acme.oraclecloud.com", "evil.example.com/CX_1", "x.oraclecloud.com/CX 1"])
def test_a_malformed_or_foreign_board_is_refused(board):
    with pytest.raises(FetchError):
        oracle.parse_board(board)


def test_a_list_that_stops_short_of_its_total_is_reported_incomplete():
    http = _Http(
        (",offset=0,", _list(_req("1"), _req("2"), total=5)),
        (",offset=2,", _list(total=5)),  # the site stops serving before its total
        ("ById", {"items": [{"ExternalDescriptionStr": "<p>desc</p>"}]}),
    )
    notes: list[str] = []
    entry = CompanyEntry(name="Acme", provider="oracle", extra={"slug": BOARD})
    jobs = oracle.OracleProvider().fetch(entry, FetchContext(http=http, incomplete=notes.append))
    assert len(jobs) == 2 and notes == ["2 of 5 requisitions served"]


def test_a_big_list_is_planned_as_chunks_of_fifty_pages_after_its_first():
    reqs = [{"Id": str(i), "Title": "X"} for i in range(100)]
    got = oracle.read_chunk(lambda url: {"items": [{"requisitionList": reqs, "TotalJobsCount": 12000}]},
                            oracle.parse_board(BOARD), None)  # fmt: skip
    assert got.info == {"total": 12000} and got.served == 100
    assert [(c["offset"], c["end"]) for c in got.more] == [(100, 5100), (5100, 10100), (10100, 12000)]
