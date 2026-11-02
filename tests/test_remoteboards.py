"""Remote-job boards (build step 3, wave 3): cases ported from freehire's ``remoteok_test.go``, ``jobicy_test.go``,
``himalayas_test.go``, ``workingnomads_test.go`` and ``weworkremotely_test.go`` (MIT), on small made-up feeds, plus the
Jobicy cursor, the Himalayas country board and its page cap. Faked transport, no network."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import Blocked, FetchError
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers import remoteboards as rb


class _Http:
    """Answers each URL from ``pages`` (a value, or an exception to raise); records what was asked."""

    def __init__(self, pages):
        self.pages = pages
        self.asked: list[str] = []

    def _get(self, url):
        self.asked.append(url)
        page = self.pages[url]
        if isinstance(page, Exception):
            raise page
        return page

    def get_json(self, url, *, timeout=None):
        return self._get(url)

    def get_text(self, url, *, timeout=None):
        return self._get(url)


def _fetch(provider, http, board):
    entry = CompanyEntry(name=board, provider=provider.name, extra={"slug": board})
    return provider.fetch(entry, FetchContext(http=http))


def test_remoteok_skips_the_legal_notice_and_postings_without_a_company():
    feed = [
        {"legal": "API Terms of Service: please link back"},
        {"id": "1137611", "company": "Acme", "position": "Backend Engineer", "location": "Worldwide",
         "date": "2026-11-01T08:00:12+00:00", "url": "https://remoteOK.com/remote-jobs/1137611",
         "description": "<p>Build <b>APIs</b>.</p>"},
        {"id": "2", "company": "", "position": "Ghost", "url": "https://remoteOK.com/remote-jobs/2"},
    ]  # fmt: skip
    [job] = _fetch(rb.RemoteOKProvider(), _Http({"https://remoteok.com/api": feed}), "remoteok.com")
    assert (job.job_id, job.url, job.title, job.company, job.source) == (
        "remoteok:1137611", "https://remoteOK.com/remote-jobs/1137611", "Backend Engineer", "Acme", "remoteok"
    )  # fmt: skip
    assert (job.description, job.location, job.date_posted, job.is_remote) == (
        "Build APIs.", "Worldwide", "2026-11-01", True
    )  # fmt: skip


def _jobicy(i, **over):
    return {"id": i, "url": f"https://jobicy.com/jobs/{i}-x", "jobTitle": f"Job {i}", "companyName": "Binance",
            "jobGeo": "APAC", "jobDescription": "<p>Do it.</p>", "pubDate": "2026-11-02T04:33:31+00:00",
            "jobType": ["Full-Time"], **over}  # fmt: skip


def test_jobicy_follows_the_cursor_to_the_end_of_the_window():
    first = "https://jobicy.com/api/jobs?count=100"
    http = _Http({
        first: {"jobs": [_jobicy(1), _jobicy(2)], "nextCursor": "Ag+A/="},
        f"{first}&cursor=Ag%2BA%2F%3D": {"jobs": [_jobicy(2), _jobicy(3, companyName="")], "nextCursor": None},
    })  # fmt: skip
    jobs = _fetch(rb.JobicyProvider(), http, "jobicy.com")
    assert [j.job_id for j in jobs] == ["jobicy:1", "jobicy:2"]  # a numeric id; a repeat once; no company: skipped
    assert (jobs[0].location, jobs[0].job_type, jobs[0].date_posted) == ("APAC", "fulltime", "2026-11-02")


def test_jobicy_without_an_end_fails():
    provider = rb.JobicyProvider()
    provider.max_pages = 2
    first = "https://jobicy.com/api/jobs?count=100"
    http = _Http({first: {"jobs": [_jobicy(1)], "nextCursor": "a"}, f"{first}&cursor=a": {"jobs": [_jobicy(2)],
                  "nextCursor": "a"}})  # fmt: skip
    with pytest.raises(FetchError):
        _fetch(provider, http, "jobicy.com")


def _himalayas(slug, **over):
    return {
        "title": "Backend Engineer", "companyName": "Acme", "companySlug": "acme",
        "guid": f"https://himalayas.app/companies/acme/jobs/{slug}",
        "applicationLink": f"https://himalayas.app/companies/acme/jobs/{slug}?utm_source=api",
        "locationRestrictions": ["India", "Philippines"], "employmentType": "Intern", "pubDate": 1793603781,
        "description": "<p>Build it.</p><p>Originally posted on <a href='https://himalayas.app/'>Himalayas</a></p>",
        **over,
    }  # fmt: skip


def _him_url(page):
    return f"https://himalayas.app/jobs/api/search?country=IN&sort=recent&page={page}"


def test_himalayas_reads_the_country_search_until_an_empty_page():
    http = _Http({
        _him_url(1): {"jobs": [_himalayas("a"), _himalayas("b", companyName="name", companySlug="talentcross")]},
        _him_url(2): {"jobs": [_himalayas("b"), _himalayas("c", guid="https://elsewhere.com/x")]},
        _him_url(3): {"jobs": []},
    })  # fmt: skip
    a, b = _fetch(rb.HimalayasProvider(), http, "in")
    assert (a.job_id, a.url, a.company, a.location, a.job_type) == (
        "himalayas:acme/jobs/a", "https://himalayas.app/companies/acme/jobs/a", "Acme", "India; Philippines", "intern"
    )  # fmt: skip
    assert (a.description, a.date_posted, a.is_remote) == ("Build it.", "2026-11-02", True)  # the trailer cut
    assert b.company == "talentcross"  # the unresolved "name" falls back to the slug; the repeat on page 2 is dropped


def test_himalayas_keeps_what_it_read_when_a_later_page_is_refused_and_stops_at_its_cap():
    blocked = _Http(
        {_him_url(1): {"jobs": [_himalayas("a")]}, _him_url(2): Blocked("429", cooldown_key="himalayas.app", until=0.0)}
    )
    assert [j.job_id for j in _fetch(rb.HimalayasProvider(), blocked, "IN")] == ["himalayas:acme/jobs/a"]
    with pytest.raises(FetchError):
        _fetch(
            rb.HimalayasProvider(), _Http({_him_url(1): Blocked("429", cooldown_key="himalayas.app", until=0.0)}), "IN"
        )
    capped = rb.HimalayasProvider()
    capped.max_pages = 2
    http = _Http({_him_url(n): {"jobs": [_himalayas(str(n))]} for n in (1, 2, 3)})
    assert len(_fetch(capped, http, "IN")) == 2 and http.asked == [_him_url(1), _him_url(2)]


def test_workingnomads_takes_the_id_from_the_link():
    feed = [
        {"url": "https://www.workingnomads.com/job/go/1981017/", "title": "Senior Magento Developer",
         "company_name": "Lemon.io", "location": "Europe, LATAM, APAC", "pub_date": "2026-10-29T09:50:51-04:00",
         "description": "<p>PHP.</p>"},
        {"url": "https://www.workingnomads.com/job/go/no-id/", "title": "X", "company_name": "Y"},
    ]  # fmt: skip
    [job] = _fetch(rb.WorkingNomadsProvider(), _Http({"https://www.workingnomads.com/api/exposed_jobs/": feed}),
                   "workingnomads.com")  # fmt: skip
    assert (job.job_id, job.company, job.location, job.date_posted) == (
        "workingnomads:1981017", "Lemon.io", "Europe, LATAM, APAC", "2026-10-29"
    )  # fmt: skip


def test_weworkremotely_splits_company_and_role_and_lists_a_posting_once():
    item = """<item><title>Smartsheet: Senior Channel Account Manager</title><region>Anywhere in the World</region>
      <type>Full-Time</type><link>https://weworkremotely.com/remote-jobs/smartsheet-senior-channel-account-manager</link>
      <pubDate>Mon, 02 Nov 2026 08:00:00 +0000</pubDate><description>&lt;p&gt;Sell.&lt;/p&gt;</description></item>"""
    no_company = "<item><title>No colon here</title><link>https://weworkremotely.com/remote-jobs/x</link></item>"
    feed = f"<rss><channel>{item}{item}{no_company}</channel></rss>"
    [job] = _fetch(rb.WeWorkRemotelyProvider(), _Http({"https://weworkremotely.com/remote-jobs.rss": feed}),
                   "weworkremotely.com")  # fmt: skip
    assert (job.job_id, job.company, job.title, job.location) == (
        "weworkremotely:smartsheet-senior-channel-account-manager", "Smartsheet", "Senior Channel Account Manager",
        "Anywhere in the World",
    )  # fmt: skip
    assert (job.description, job.date_posted, job.job_type) == ("Sell.", "2026-11-02", "fulltime")


@pytest.mark.parametrize(
    ("provider", "good", "bad"),
    [(rb.RemoteOKProvider(), "remoteok.com", "remotive.com"), (rb.JobicyProvider(), "jobicy.com", ""),
     (rb.WorkingNomadsProvider(), "workingnomads.com", "x"), (rb.WeWorkRemotelyProvider(), "weworkremotely.com", "x"),
     (rb.HimalayasProvider(), "IN", "India")],
)  # fmt: skip
def test_each_board_is_named_explicitly(provider, good, bad):
    with pytest.raises(FetchError):
        _fetch(provider, _Http({}), bad)
    assert provider.board_url(CompanyEntry(name="x", provider=provider.name, extra={"slug": good}))
    assert provider.board_url(CompanyEntry(name="x", provider=provider.name, extra={"slug": bad})) is None
    assert not provider.detect(CompanyEntry(name="x", provider=provider.name, extra={"slug": good}))


@pytest.mark.parametrize(("value", "want"), [("Full Time", "fulltime"), (["Part-Time", "Contract"], "parttime"),
                                             ("Contractor", "contract"), ("Volunteer", None), (None, None)])  # fmt: skip
def test_job_type(value, want):
    assert rb.job_type(value) == want
