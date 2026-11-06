"""Eightfold provider (build step 3): cases ported from freehire's ``eightfold_test.go`` (MIT) -- both list
generations, paging by the stated count, the canonical-URL fallback -- plus our per-generation cooldown keys, the
domain read off the careers page and hydration. Faked transport, no network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import Blocked, CooldownRegistry, FetchError, HTTPError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import eightfold

HOST = "acme.eightfold.ai"
PCSX = f"https://{HOST}/api/pcsx/search?domain=acme.com&query=&start={{}}&num=10&sort_by=relevance"
V2 = f"https://{HOST}/api/apply/v2/jobs?domain=acme.com&query=&start={{}}&num=10&sort_by=relevance"


def _detail(pid, text="<p>Build it.</p>", canonical=None):
    return f"https://{HOST}/api/apply/v2/jobs/{pid}?domain=acme.com", {
        "job_description": text, "canonicalPositionUrl": canonical
    }  # fmt: skip


class _Http:
    """Answers by URL: a body, or an exception to raise. Records each URL and the cooldown key it was asked under."""

    def __init__(self, routes):
        self.routes = dict(routes)
        self.asked: list[tuple[str, str | None]] = []

    def _get(self, url, cooldown_key=None):
        self.asked.append((url, cooldown_key))
        body = self.routes.get(url)
        if body is None:
            raise HTTPError("not found", status=404, url=url)
        if isinstance(body, Exception):
            raise body
        return body

    def get_json(self, url, *, timeout=None, cooldown_key=None):
        return self._get(url, cooldown_key)

    def get_text(self, url, *, timeout=None):
        return self._get(url)


def _blocked():
    return Blocked("403", cooldown_key="k", until=0.0)


def _fetch(http, board=f"{HOST}/acme.com", seen=lambda job_id: False):
    entry = CompanyEntry(name="Acme", provider="eightfold", extra={"slug": board})
    return eightfold.EightfoldProvider(interval=0).fetch(entry, FetchContext(http=http, seen=seen))


def test_pcsx_pages_by_the_stated_count_and_details_give_the_text():
    first = [{"id": i, "name": f"Engineer {i}", "locations": ["Bengaluru, India"], "postedTs": 1784545200,
              "workLocationOption": "hybrid"} for i in range(1, 11)]  # fmt: skip
    second = [{"id": 11, "name": "AI Engineer", "locations": ["Pune, India", "Remote, India"],
               "workLocationOption": "remote"}]  # fmt: skip
    routes = {PCSX.format(0): {"data": {"positions": first, "count": 11}},
              PCSX.format(10): {"data": {"positions": second, "count": 11}}}  # fmt: skip
    routes.update(dict(_detail(i) for i in range(1, 11)))
    routes.update([_detail(11, canonical="https://careers.acme.com/job/11")])
    jobs = _fetch(_Http(routes))
    assert len(jobs) == 11
    one, ai = jobs[0], jobs[-1]
    assert (one.job_id, one.url, one.description) == ("eightfold:acme.com:1", f"https://{HOST}/careers/job/1",
                                                      "Build it.")  # fmt: skip
    assert (one.location, one.is_remote, one.date_posted) == ("Bengaluru, India", False, "2026-07-20")
    assert (ai.url, ai.location, ai.is_remote) == ("https://careers.acme.com/job/11", "Pune, India; Remote, India",
                                                   True)  # fmt: skip


def test_a_legacy_tenant_is_read_from_v2_under_its_own_cooldown_key():
    legacy = [{"id": 5, "name": "Product Owner", "location": "Bangalore,India", "locations": ["Bangalore,India"],
               "t_create": 1792997640, "work_location_option": "onsite",
               "canonicalPositionUrl": f"https://{HOST}/careers/job/5"}]  # fmt: skip
    http = _Http({PCSX.format(0): _blocked(), V2.format(0): {"positions": legacy, "count": 1}, **dict([_detail(5)])})
    [job] = _fetch(http)
    assert http.asked[:2] == [(PCSX.format(0), f"{HOST} pcsx"), (V2.format(0), f"{HOST} v2")]
    assert (job.location, job.date_posted, job.is_remote) == ("Bangalore,India", "2026-10-26", False)


def test_a_failure_past_the_first_page_fails_the_board():
    first = [{"id": i, "name": "X"} for i in range(10)]
    with pytest.raises(FetchError):
        _fetch(_Http({PCSX.format(0): {"data": {"positions": first, "count": 30}}, PCSX.format(10): _blocked()}))
    with pytest.raises(FetchError):  # neither generation answers
        _fetch(_Http({PCSX.format(0): _blocked(), V2.format(0): HTTPError("403", status=403)}))


def test_the_domain_is_read_off_the_careers_page_when_the_board_has_none():
    page = '<script>window.config = {"domain": "acme.com", "x": 1, "branding": {"domain":"acme.com"}}</script>'
    http = _Http({f"https://{HOST}/careers": page, PCSX.format(0): {"data": {"positions": [], "count": 0}}})
    assert _fetch(http, board=HOST) == []
    assert http.asked[0][0] == f"https://{HOST}/careers"
    with pytest.raises(FetchError):
        _fetch(_Http({f"https://{HOST}/careers": "<html></html>"}), board=HOST)


def test_an_html_escaped_domain_on_the_careers_page_is_read():
    # Worley's page (2026-11-03): the config sits HTML-escaped in an attribute, next to a OneTrust domain script id
    page = ('<script data-domain-script="0199c901-6074-7a1e"></script><div data-x="{&#34;domain&#34;: '
            '&#34;acme.com&#34;, &#34;configPath&#34;: &#34;PCS&#34;}"></div>')  # fmt: skip
    http = _Http({f"https://{HOST}/careers": page, PCSX.format(0): {"data": {"positions": [], "count": 0}}})
    assert _fetch(http, board=HOST) == []
    assert "domain=acme.com" in http.asked[1][0]


def test_a_stored_position_is_re_sighted_and_one_without_text_waits():
    positions = [{"id": 1, "name": "Stored", "locations": ["Pune"]}, {"id": 2, "name": "Empty detail"}]
    http = _Http({PCSX.format(0): {"data": {"positions": positions, "count": 2}}, **dict([_detail(2, text="")])})
    [stored] = _fetch(http, seen=lambda job_id: job_id.endswith(":1"))
    assert (stored.title, stored.description, stored.location) == ("Stored", None, "Pune")
    assert not any("/jobs/1?" in url for url, _ in http.asked)


@pytest.mark.parametrize("bad", ["", "acme", "acme.eightfold.ai/acme", "https://acme.eightfold.ai"])
def test_a_malformed_board_is_refused(bad):
    with pytest.raises(FetchError):
        _fetch(_Http({}), board=bad)


def test_a_walk_that_repeats_positions_is_joined_with_one_by_date():
    rows = [{"id": i, "name": f"Job {i}"} for i in range(1, 16)]
    relevance = {0: rows[:10], 10: rows[:5]}  # the second page re-serves the first five: 11-15 never come
    by_date = {0: rows[5:15], 10: rows[:5]}
    routes = {PCSX.format(s): {"data": {"positions": page, "count": 15}} for s, page in relevance.items()}
    date_url = PCSX.replace("relevance", "timestamp")
    routes.update({date_url.format(s): {"data": {"positions": page, "count": 15}} for s, page in by_date.items()})
    jobs = _fetch(_Http(routes), seen=lambda job_id: True)
    assert sorted(int(j.job_id.rsplit(":", 1)[1]) for j in jobs) == list(range(1, 16))


def test_a_walk_still_short_after_both_orders_is_reported_incomplete():
    rows = [{"id": i, "name": f"Job {i}"} for i in range(1, 11)]
    routes = {PCSX.format(0): {"data": {"positions": rows, "count": 12}}, PCSX.format(10): {"data": {"positions": [],
              "count": 12}}}  # fmt: skip
    date_url = PCSX.replace("relevance", "timestamp")
    routes.update({date_url.format(0): {"data": {"positions": rows, "count": 12}},
                   date_url.format(10): {"data": {"positions": [], "count": 12}}})  # fmt: skip
    notes: list[str] = []
    entry = CompanyEntry(name="Acme", provider="eightfold", extra={"slug": f"{HOST}/acme.com"})
    eightfold.EightfoldProvider(interval=0).fetch(entry, FetchContext(http=_Http(routes), seen=lambda j: True,
                                                            incomplete=notes.append))  # fmt: skip
    assert notes == ["10 of 12 positions served"]


def test_a_405_parks_every_tenant_and_the_next_board_waits_without_a_request():
    # crawl 1 (2026-11-05): 11 minutes into reading 81 tenants, every Eightfold request answered 405
    http = _Http({PCSX.format(0): HTTPError("405", status=405, url=PCSX.format(0))})
    http.cooldowns = CooldownRegistry(now=lambda: 1000.0)
    with pytest.raises(Blocked, match="throttle") as err:
        _fetch(http)
    assert err.value.cooldown_key == eightfold.THROTTLE_KEY
    asked = len(http.asked)
    with pytest.raises(Blocked, match="cooldown"):  # another tenant, same platform: nothing is sent
        _fetch(http, board="other.eightfold.ai/other.com")
    assert len(http.asked) == asked


def test_a_405_on_a_detail_leaves_the_boards_other_new_positions_for_the_next_run():
    positions = [{"id": i, "name": f"Job {i}"} for i in (1, 2, 3)]
    http = _Http({PCSX.format(0): {"data": {"positions": positions, "count": 3}}, **dict([_detail(1)]),
                  _detail(2)[0]: HTTPError("405", status=405), **dict([_detail(3)])})  # fmt: skip
    http.cooldowns = CooldownRegistry(now=lambda: 1000.0)
    jobs = _fetch(http)
    assert [j.job_id for j in jobs] == ["eightfold:acme.com:1"]  # 2 throttled, 3 never asked: both new next run
    assert not any("/jobs/3?" in url for url, _ in http.asked)


def test_every_request_waits_on_the_one_pace():
    provider = eightfold.EightfoldProvider()
    waited: list[str] = []
    provider.pace.wait = lambda: waited.append("x")
    http = _Http({PCSX.format(0): {"data": {"positions": [{"id": 1, "name": "X"}], "count": 1}}, **dict([_detail(1)])})
    provider.fetch(CompanyEntry(name="Acme", provider="eightfold", extra={"slug": f"{HOST}/acme.com"}),
                   FetchContext(http=http, seen=lambda j: False))  # fmt: skip
    assert len(waited) == len(http.asked) == 2


def test_a_date_walk_refused_on_its_first_page_keeps_the_relevance_walk_and_reports_it_short():
    rows = [{"id": i, "name": f"Job {i}"} for i in range(1, 11)]
    date_url = PCSX.replace("relevance", "timestamp")
    routes = {PCSX.format(0): {"data": {"positions": rows, "count": 600}}, PCSX.format(10): {"data": {"positions": [],
              "count": 600}}, PCSX.format(510): {"data": {"positions": [], "count": 600}},
              date_url.format(0): _blocked()}  # fmt: skip  # each later chunk of the walk asks once, finds it empty
    notes: list[str] = []
    entry = CompanyEntry(name="Acme", provider="eightfold", extra={"slug": f"{HOST}/acme.com"})
    http = _Http(routes)
    jobs = eightfold.EightfoldProvider(interval=0).fetch(entry, FetchContext(http=http, seen=lambda j: True,
                                                                   incomplete=notes.append))  # fmt: skip
    assert len(jobs) == 10 and notes == ["10 of 600 positions served"]
    assert [u for u, _ in http.asked if "timestamp" in u] == [date_url.format(0)]  # nothing past the refusal


def test_a_big_walk_is_planned_as_chunks_of_fifty_pages():
    provider = eightfold.EightfoldProvider(interval=0)
    entry = CompanyEntry(name="Acme", provider="eightfold", extra={"slug": f"{HOST}/acme.com"})
    http = _Http({PCSX.format(0): {"data": {"positions": [{"id": i, "name": "X"} for i in range(10)], "count": 21754}}})
    first = provider.list_chunk(entry, FetchContext(http=http), None)
    assert [(c["start"], c["end"]) for c in first.more][:2] == [(10, 510), (510, 1010)]
    assert first.more[-1]["end"] == 21754 and len(first.more) == 44  # Starbucks' size: read to its end
