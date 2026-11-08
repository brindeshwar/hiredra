"""Workday provider (build step 3): cases ported from freehire's ``workday_test.go`` (MIT). Faked transport, no
network."""

from __future__ import annotations

import json as jsonlib

import pytest

from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import workday


class _Resp:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class _Http:
    """Listing answers by its ``appliedFacets`` (or a page list), details by a URL substring."""

    def __init__(self, listings=None, pages=None, details=None):
        self.listings = {jsonlib.dumps(k, sort_keys=True): v for k, v in (listings or [])}
        self.pages = list(pages or [])
        self.details = details or {}
        self.posted: list[str] = []

    def request(self, method, url, *, json=None, timeout=None):  # noqa: A002
        assert method == "POST" and url.endswith("/jobs")
        self.posted.append(url)
        if self.pages:
            return _Resp(self.pages[json["offset"] // workday.PAGE])
        key = jsonlib.dumps(json["appliedFacets"], sort_keys=True)
        if key not in self.listings:
            raise HTTPError(f"unrouted facets {key}", status=400)
        return _Resp(self.listings[key])

    def get_json(self, url, *, timeout=None):
        for part, body in self.details.items():
            if part in url:
                return body
        raise HTTPError("not found", status=404)


def _fetch(http, board="acme.wd1.myworkdayjobs.com/Careers"):
    entry = CompanyEntry(name="Acme", provider="workday", extra={"slug": board})
    return workday.WorkdayProvider().fetch(entry, FetchContext(http=http))


def test_lists_and_fetches_each_detail():
    http = _Http(
        listings=[({}, {"total": 2, "jobPostings": [
            {"title": "Backend Engineer", "externalPath": "/job/Berlin/Backend_JR-1", "locationsText": "Berlin, Germany"},
            {"title": "Data Engineer", "externalPath": "/job/Remote/Data_JR-2", "locationsText": "Remote, US"},
        ]})],
        details={
            "Backend_JR-1": {"jobPostingInfo": {
                "title": "Backend Engineer", "jobDescription": "<p>Build the backend.</p>", "location": "Berlin, Germany",
                "startDate": "2024-06-11", "remoteType": "On-site", "timeType": "Full time",
                "externalUrl": "https://acme.wd1.myworkdayjobs.com/en-US/Careers/job/Berlin/Backend_JR-1",
            }},
            "Data_JR-2": {"jobPostingInfo": {
                "title": "Data Engineer", "jobDescription": "<p>Crunch data.</p>", "location": "Remote, US",
                "startDate": "2024-06-12", "remoteType": "Remote",
            }},
        },
    )  # fmt: skip
    backend, data = _fetch(http)
    assert backend.url == "https://acme.wd1.myworkdayjobs.com/en-US/Careers/job/Berlin/Backend_JR-1"
    assert "Build the backend." in backend.description and backend.location == "Berlin, Germany"
    assert backend.date_posted == "2024-06-11" and backend.job_type == "fulltime" and not backend.is_remote
    assert data.is_remote is True
    assert data.url == "https://acme.wd1.myworkdayjobs.com/Careers/job/Remote/Data_JR-2"  # built when absent
    assert backend.job_id == "workday:acme:/job/Berlin/Backend_JR-1"


def test_a_failed_detail_skips_only_that_posting():
    http = _Http(
        listings=[({}, {"total": 2, "jobPostings": [{"title": "A", "externalPath": "/job/X/A"}, {"title": "B", "externalPath": "/job/X/B"}]})],
        details={"/job/X/A": {"jobPostingInfo": {"title": "A"}}},
    )  # fmt: skip
    assert [j.title for j in _fetch(http)] == ["A"]


def test_pages_by_the_first_pages_total():
    page = lambda total, *ids: {
        "total": total,
        "jobPostings": [{"title": i, "externalPath": f"/job/X/{i}"} for i in ids],
    }  # noqa: E731
    pages = [page(4, *(f"P{i}" for i in range(20))), page(0, "Q1", "Q2"), page(0, "R")]
    pages[0]["total"] = 22
    http = _Http(pages=pages, details={"/job/X/": {"jobPostingInfo": {}}})
    jobs = _fetch(http)  # later pages say total 0; the walk goes on to 22, and no further
    assert len(jobs) == 22 and "R" not in {j.title for j in jobs}


def test_a_shared_host_board_spells_its_tenant_out():
    board = workday.parse_board("wd1.myworkdaysite.com/snapchat/snap")
    assert board == workday.Board("wd1.myworkdaysite.com", "snapchat", "snap", "recruiting/snapchat/snap")
    assert board.api == "https://wd1.myworkdaysite.com/wday/cxs/snapchat/snap"
    http = _Http(
        listings=[({}, {"total": 1, "jobPostings": [{"title": "Engineer", "externalPath": "/job/X/JR-1"}]})],
        details={"/job/X/JR-1": {"jobPostingInfo": {"title": "Engineer"}}},
    )
    [job] = _fetch(http, "wd1.myworkdaysite.com/snapchat/snap")
    assert job.url == "https://wd1.myworkdaysite.com/recruiting/snapchat/snap/job/X/JR-1"
    assert http.posted == ["https://wd1.myworkdaysite.com/wday/cxs/snapchat/snap/jobs"]


@pytest.mark.parametrize(
    "board", ["", "no-slash-host", "/onlysite", "acme.wd1.myworkdayjobs.com", "evil.example.com/site"]
)
def test_a_malformed_or_foreign_board_is_refused(board):
    with pytest.raises(FetchError):
        workday.parse_board(board)


def _one(path):
    return {"total": 1, "jobPostings": [{"title": path, "externalPath": path}]}


def _postings(listings):
    http = _Http(listings=listings)
    return {
        p["externalPath"] for p in workday.list_postings(lambda body: http.request("POST", "u/jobs", json=body).json())
    }


def test_a_capped_board_is_listed_slice_by_slice():
    capped = {"total": 2000, "jobPostings": [], "facets": [
        {"facetParameter": "jobFamilyGroup", "values": [{"id": "eng", "count": 1200}, {"id": "sales", "count": 800}]},
    ]}  # fmt: skip
    listings = [({}, capped), ({"jobFamilyGroup": ["eng"]}, _one("/E")), ({"jobFamilyGroup": ["sales"]}, _one("/S"))]
    assert _postings(listings) == {"/E", "/S"}


def test_the_split_uses_a_nested_location_facet_and_never_one_that_does_not_cover_the_board():
    capped = {"total": 2000, "jobPostings": [], "facets": [
        {"facetParameter": "distance", "values": [{"id": "d10", "count": 1}, {"id": "d25", "count": 1}]},
        {"facetParameter": "jobFamilyGroup", "values": [{"id": "f1", "count": 3000}, {"id": "f2", "count": 1300}]},
        {"facetParameter": "locationMainGroup", "values": [
            {"facetParameter": "locationCountry", "values": [{"id": "us", "count": 4100}]},
            {"facetParameter": "locationRegionStateProvince", "values": [{"id": "tx", "count": 1400}, {"id": "ca", "count": 1300}]},
        ]},
    ]}  # fmt: skip
    listings = [
        ({}, capped),
        ({"locationRegionStateProvince": ["tx"]}, _one("/TX")),
        ({"locationRegionStateProvince": ["ca"]}, _one("/CA")),
    ]
    assert _postings(listings) == {"/TX", "/CA"}


def test_a_slice_still_capped_splits_again_and_overlaps_count_once():
    capped = lambda *facets: {"total": 2000, "jobPostings": [], "facets": list(facets)}  # noqa: E731
    fam = {"facetParameter": "fam", "values": [{"id": "a", "count": 2500}, {"id": "b", "count": 10}]}
    loc = {"facetParameter": "loc", "values": [{"id": "x", "count": 2000}, {"id": "y", "count": 600}]}
    listings = [
        ({}, capped(fam, loc)),  # loc's largest value is smaller: split on it first
        ({"loc": ["x"]}, capped(fam)),
        ({"loc": ["x"], "fam": ["a"]}, _one("/A")),
        ({"loc": ["x"], "fam": ["b"]}, {"total": 2, "jobPostings": [{"title": "", "externalPath": "/A"}, {"title": "", "externalPath": "/B"}]}),
        ({"loc": ["y"]}, _one("/Y")),
    ]  # fmt: skip
    assert _postings(listings) == {"/A", "/B", "/Y"}


def test_a_board_that_cannot_be_listed_completely_fails():
    capped = {
        "total": 2000,
        "jobPostings": [],
        "facets": [{"facetParameter": "d", "values": [{"id": "x", "count": 3}]}],
    }
    with pytest.raises(FetchError, match="capped"):
        _postings([({}, capped)])


class _Refusing(_Http):
    """Every listing page answers one HTTP status, as a tenant gone from its host does (422)."""

    def __init__(self, status):
        super().__init__()
        self.status = status

    def request(self, method, url, *, json=None, timeout=None):  # noqa: A002
        raise HTTPError(f"{url}: HTTP {self.status}", status=self.status, url=url)


def test_a_tenant_gone_from_its_host_is_a_gone_board():
    # crawl 1 (2026-11-05): Comcast moved from wd5 to wd115; its old host answers 422, as a made-up tenant does
    with pytest.raises(HTTPError, match="not on this host") as err:
        _fetch(_Refusing(422))
    assert err.value.status == 404  # the orchestrator's mark of a gone board
    with pytest.raises(HTTPError) as err:  # any other refusal stays what it was
        _fetch(_Refusing(401))
    assert err.value.status == 401


def test_only_new_postings_cost_a_detail_and_a_run_fetches_at_most_the_budget(monkeypatch):
    monkeypatch.setattr(workday, "DETAIL_BUDGET", 2)
    listing = {"total": 4, "jobPostings": [{"title": t, "externalPath": f"/job/X/{t}"} for t in "ABCD"]}
    asked = []

    class Http(_Http):
        def get_json(self, url, *, timeout=None):
            asked.append(url.rsplit("/", 1)[1])
            return {"jobPostingInfo": {"jobDescription": "<p>new</p>"}}

    entry = CompanyEntry(name="Acme", provider="workday", extra={"slug": "acme.wd1.myworkdayjobs.com/Careers"})
    ctx = FetchContext(http=Http(listings=[({}, listing)]), seen=lambda key: key.endswith("/A"))
    jobs = workday.WorkdayProvider().fetch(entry, ctx)
    assert asked == ["B", "C"]  # A is stored, D waits for the next run
    assert [(j.title, j.description) for j in jobs] == [("A", None), ("B", "new"), ("C", "new")]


def test_a_big_board_is_planned_as_chunks_of_fifty_pages_after_its_first():
    first = {"total": 2500, "jobPostings": [{"title": "A", "externalPath": f"/job/{i}"} for i in range(20)]}
    got = workday.read_chunk(lambda body: first, None)
    assert len(got.items) == 20 and got.served == 20
    assert [(c["offset"], c["end"]) for c in got.more] == [(20, 1020), (1020, 2020), (2020, 2500)]


def test_a_tenant_in_maintenance_is_a_refusal_that_can_wait_not_a_board_that_cant_be_done():
    # Saturday 2026-11-07, Workday's weekly window: every call is a 303 to community.workday.com/maintenance-page
    from hiredra.sourcing.http import Blocked, CooldownRegistry
    from hiredra.sourcing.orchestrator import can_wait

    class Maintenance(_Http):
        def __init__(self):
            super().__init__()
            self.cooldowns = CooldownRegistry(now=lambda: 1000.0)

        def request(self, method, url, *, json=None, timeout=None):  # noqa: A002
            raise HTTPError(f"{url}: unexpected redirect (HTTP 303)", status=303, url=url,
                            location="https://community.workday.com/maintenance-page")  # fmt: skip

    http = Maintenance()
    with pytest.raises(Blocked, match="maintenance") as err:
        _fetch(http)
    assert can_wait(err.value) and err.value.cooldown_key == "acme.wd1.myworkdayjobs.com"
    assert http.cooldowns.active() == {"acme.wd1.myworkdayjobs.com": 1000.0 + workday.MAINTENANCE_PAUSE_S}
    with pytest.raises(HTTPError):  # any other redirect stays what it was

        class Elsewhere(Maintenance):
            def request(self, method, url, *, json=None, timeout=None):  # noqa: A002
                raise HTTPError("redirect", status=302, url=url, location="https://acme.example/careers")

        _fetch(Elsewhere())


def test_a_board_that_states_no_total_is_never_paged_past_the_cap_where_it_repeats_page_one():
    # past offset 2,000 some boards answer page one again, forever (adderleydata, glitchbound): stop at the cap
    first = {"total": 0, "jobPostings": [{"title": "A", "externalPath": f"/job/{i}"} for i in range(20)]}
    asked: list[int] = []

    def post(body):
        asked.append(body["offset"])
        return first  # every offset: page one again, never an empty page

    jobs = workday.list_postings(post)
    assert len(jobs) == 20 and max(asked) < workday.CAP


def test_a_boards_data_centre_is_read_from_its_host():
    def dc(slug):
        return workday.WorkdayProvider().data_centre(CompanyEntry(name="A", provider="workday", extra={"slug": slug}))

    assert dc("nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite") == "wd5"
    assert dc("acme.wd103.myworkdayjobs.com/Careers") == "wd103"
    assert dc("wd1.myworkdaysite.com/acme/site") == "wd1"  # the shared host spells it first
    assert dc("not-a-board") is None
