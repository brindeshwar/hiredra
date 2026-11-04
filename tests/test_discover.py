"""Board discovery by guessed slug (build slice 2.4): the guesses, the YC filter, the live check and the
namesake guard. Providers are faked; no network."""

from __future__ import annotations

import json

import pytest

from hiredra.sourcing import discover
from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.job import Job
from hiredra.sourcing.providers._common import Robots


def _job(desc="", url="https://jobs.ashbyhq.com/x/1"):
    return Job(url=url, title="Engineer", company="X", source="ashby", description=desc)


class _Fake:
    def __init__(self, name, boards, errors=()):
        self.name = name
        self.boards = boards  # slug -> jobs
        self.errors = set(errors)
        self.asked: list[str] = []

    def fetch(self, entry, ctx):
        slug = entry.extra["slug"]
        self.asked.append(slug)
        if slug in self.errors:
            raise FetchError("timed out")
        if slug not in self.boards:
            raise HTTPError("not found", status=404)
        return self.boards[slug]


def test_slug_guesses():
    assert discover.slug_guesses("Together AI", ["together-ai"]) == ("together-ai", "togetherai")
    assert discover.slug_guesses("Ramp Pvt Ltd") == ("ramp",)
    assert discover.slug_guesses("Ramp", ["", "  "]) == ("ramp",)


def test_yc_candidates_keep_hiring_india_or_remote_without_a_board(tmp_path):
    rows = [
        {"name": "Acme AI", "slug": "acme-ai", "isHiring": True, "regions": ["Remote"], "website": "https://acme.ai",
         "team_size": 12},
        {"name": "Busy", "slug": "busy", "isHiring": False, "regions": ["India"]},
        {"name": "Stateside", "slug": "stateside", "isHiring": True, "regions": ["United States of America"]},
        {"name": "Known Co", "slug": "known-co", "isHiring": True, "regions": ["India"]},
    ]  # fmt: skip
    path = tmp_path / "all.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    [cand] = discover.yc_candidates(path, known={"knownco"})
    assert cand == discover.Candidate("Acme AI", ("acme-ai", "acmeai"), "https://acme.ai", 12)


def test_probe_walks_providers_in_order_and_404_means_not_here():
    cand = discover.Candidate("Acme", ("acme", "acmeai"))
    ashby = _Fake("ashby", {})
    gh = _Fake("greenhouse", {"acmeai": [_job()]}, errors={"acme"})
    lever = _Fake("lever", {"acme": [_job()]})
    result = discover.probe(cand, [lever, gh, ashby], ctx=None)
    assert (result.provider, result.slug) == ("greenhouse", "acmeai")  # ashby first, then greenhouse; lever unasked
    assert ashby.asked == ["acme", "acmeai"] and lever.asked == []
    assert result.errors == ["greenhouse/acme: timed out"]  # an error never hides the next guess

    none = discover.probe(discover.Candidate("Nobody", ("nobody",)), [ashby], ctx=None)
    assert none.provider is None and none.errors == []


def test_confirm_tells_the_company_from_a_namesake():
    cand = discover.Candidate("Apollo", ("apollo",), website="https://www.apollo.io", team_size=8)

    def result(jobs):
        return discover.Probe(cand, provider="greenhouse", slug="apollo", jobs=jobs)

    assert discover.confirm(result([])) == "none"  # an empty board can't be checked
    assert discover.confirm(result([_job("Apply at apollo.io/careers")])) == "domain"
    assert discover.confirm(result([_job("Apollo is hiring")])) == "name"
    assert discover.confirm(result([_job("Apollo is hiring")] * 40)) == "none"  # out-hires an 8-person startup
    assert discover.confirm(result([_job("We build boats")])) == "none"


# ── slice 2.5: the careers site ─────────────────────────────────────────────
@pytest.mark.parametrize(
    ("html", "want"),
    [
        ('<script src="https://boards.greenhouse.io/embed/job_board/js?for=acme"></script>', ("greenhouse", "acme")),
        ('<a href="https://job-boards.greenhouse.io/razorpay/jobs/1">', ("greenhouse", "razorpay")),
        ('<a href="https://jobs.lever.co/acme-co">Jobs</a>', ("lever", "acme-co")),
        ('<a href="https://jobs.eu.lever.co/acme">Jobs</a>', ("lever", "acme")),
        ('<iframe src="https://jobs.ashbyhq.com/apollo-graphql/embed"></iframe>', ("ashby", "apollo-graphql")),
        (
            "https://abb.wd3.myworkdayjobs.com/en-US/External_Career_Page/job/Bangalore/X_JR1",
            ("workday", "abb.wd3.myworkdayjobs.com/External_Career_Page"),
        ),  # the site keeps its case
        ("https://Autodesk.wd1.myworkdayjobs.com/Ext/details/Y_1", ("workday", "autodesk.wd1.myworkdayjobs.com/Ext")),
        (
            "https://wd1.myworkdaysite.com/en-US/recruiting/snapchat/snap/job/Z",
            ("workday", "wd1.myworkdaysite.com/snapchat/snap"),
        ),
        ("https://acme.wd5.myworkdayjobs.com/job/Pune/X", None),  # no site in the link
        (
            "https://broadcom.wd1.myworkdayjobs.com/es/External_Career/job/IND-Hyderabad/R_1",
            ("workday", "broadcom.wd1.myworkdayjobs.com/External_Career"),
        ),  # a bare language segment
        ("https://acme.wd1.myworkdayjobs.com/Ext", ("workday", "acme.wd1.myworkdayjobs.com/Ext")),
        (
            "https://eeho.fa.us2.oraclecloud.com/hcmUI/CandidateExperience/en/sites/jobsearch/job/123",
            ("oracle", "eeho.fa.us2.oraclecloud.com/jobsearch"),
        ),
        ("https://jobs.smartrecruiters.com/Visa/744000067-engineer", ("smartrecruiters", "Visa")),
        ("https://jobs.smartrecruiters.com/oneclick-ui/company/x", None),
        ("https://apply.workable.com/capgemini-insurance/j/C15CE8BF0F", ("workable", "capgemini-insurance")),
        ("https://apply.workable.com/j/C15CE8BF0F", None),  # a short link: followed, not read
        ("https://apply.workable.com/oops", None),  # where a closed job's short link points
        ("https://apply.workable.com/oopsla/", ("workable", "oopsla")),
        ("https://careers-shure.icims.com/jobs/5039/job?utm_source=x", ("icims", "shure")),
        (
            "https://globalcareers-msci.icims.com/jobs/5929/job?utm_source=x",
            ("icims", "globalcareers-msci.icims.com"),
        ),  # not the classic careers-<slug> shape: the whole host
        ("https://www.icims.com/products/", None),
        ("https://assaabloy.jobs2web.com/job/Pune-Engineer/1234/", ("successfactors", "assaabloy.jobs2web.com")),
        ("https://acme.jobs.hr.cloud.sap/job/x/1/", ("successfactors", "acme.jobs.hr.cloud.sap")),
        ("https://acme.stage.jobs2web.com/job/x/1/", None),  # a staging copy, one label deeper
        ("https://career55.sapsf.eu/career?company=HCLPRD&career_job_req_id=1", None),  # legacy: no sitemap
        (
            "https://phf.tbe.taleo.net/phf01/ats/careers/v2/viewRequisition?org=COVESTIC2&cws=37&rid=14016",
            ("taleobe", "COVESTIC2:37"),
        ),
        (
            '<a href="https://chp.tbe.taleo.net/chp02/ats/careers/requisition.jsp?cws=4&amp;org=acme_co">',
            ("taleobe", "ACME_CO:4"),
        ),
        ("https://phf.tbe.taleo.net/phf01/ats/careers/searchResults.jsp?org=ACME", None),  # no career site
        ("https://uhg.taleo.net/careersection/10780/jobdetail.ftl?job=1", None),  # Taleo Enterprise: no adapter
        ("https://anblicks.keka.com/careers/jobdetails/164414?source=indeed%20job%20posts", ("keka", "anblicks")),
        ("https://cloudsufi.keka.com/careers//jobdetails/91133", ("keka", "cloudsufi")),
        ("https://app.keka.com/careers/x", None),  # the vendor's own host
        ("https://www.keka.com/careers", None),
        (
            "https://telradsol.zohorecruit.in/jobs/TRS/102249000007241027/Junior-AI-Engineer?source=CareerSite",
            ("zohorecruit", "telradsol.zohorecruit.in"),
        ),  # another portal, the same board
        ("https://Kanerika.zohorecruit.com/jobs/Careers/7/x", ("zohorecruit", "kanerika.zohorecruit.com")),
        ("https://acme.zohorecruit.com.au/recruit/Portal.na?digest=x", ("zohorecruit", "acme.zohorecruit.com.au")),
        ("https://www.zohorecruit.com/jobs/", None),
        (
            "https://clearroute.teamtailor.com/jobs/8509129-ai-platform-support-engineer",
            ("teamtailor", "clearroute.teamtailor.com"),
        ),
        ("https://teoco.na.teamtailor.com/jobs/727050-data-engineer", ("teamtailor", "teoco.na.teamtailor.com")),
        ("https://tt.teamtailor.com/jobs/x", None),  # the vendor's own tracking host
        ("https://ats.rippling.com/oneorigin/jobs/1e3cce74?jobSite=Indeed", ("rippling", "oneorigin")),
        ("https://ats.rippling.com/api/v1/board/x/jobs", None),
        (
            "https://fortive.eightfold.ai/careers/job/893398346266?domain=fortive.com",
            ("eightfold", "fortive.eightfold.ai/fortive.com"),
        ),
        ("https://paypal.eightfold.ai/careers/job/274922789363", ("eightfold", "paypal.eightfold.ai")),
        ("https://nab.eightfold.ai/careers?microsite=bnz&start=0&pid=563980770683036", None),  # a sub-brand's view
        ("https://latentview.darwinbox.in/ms/candidatev2/main/careers/jobDetails/a1", None),  # no adapter (D4.3)
        ('<a href="https://boards.greenhouse.io/embed/">x</a>', None),  # "embed" is never a board
        ('<a href="https://notjobs.lever.cool/x">', None),
        ("<p>We are hiring!</p>", None),
    ],
)
def test_detect_board(html, want):
    assert discover.detect_board(html) == want


@pytest.mark.parametrize(
    ("company", "slug", "want"),
    [
        ("Razorpay", "razorpaysoftwareprivatelimited", True),
        ("Nourish", "usenourish", True),
        ("Bolna AI", "bolna", True),
        ("Dyneti Technologies", "dyneti", True),
        ("Kivo Health", "kivo-health", True),
        ("Cargo", "checkr", False),  # a careers page listing customers' jobs
        ("Minimum", "novisto78", False),  # the site redirects to its acquirer
        ("Flock Safety", "demo", False),
        ("AI Co", "ai", False),  # too short to mean anything
        ("Arista Networks", "AristaNetworks", True),  # SmartRecruiters' case
        ("The Eplane Company", "eplane.zohorecruit.in", True),  # a host-form board: judged by its parts
        ("Slack", "salesforce.wd12.myworkdayjobs.com/Slack", True),  # its own site on its parent's tenant
        ("Frame.io", "adobe.wd5.myworkdayjobs.com/external_experienced", False),  # the parent's board (Getro, 3.13)
        ("OPOWER", "eeho.fa.us2.oraclecloud.com/jobearch", False),
    ],
)
def test_slug_names(company, slug, want):
    assert discover.slug_names(company, slug) is want


def test_careers_walk_follows_paths_links_and_one_level_deeper():
    site = {
        "https://acme.com": '<a href="/about">About</a><a href="/team/join">Join our team</a>',
        "https://acme.com/careers": '<a href="/careers/open-positions">See open positions</a>',
        "https://acme.com/careers/open-positions": '<a href="https://jobs.ashbyhq.com/acme">Apply</a>',
    }
    asked = []

    def fetch(url):
        asked.append(url)
        return site.get(url)

    assert discover.careers_walk("https://acme.com/", fetch) == (
        "ashby", "acme", "https://acme.com/careers/open-positions"
    )  # fmt: skip
    assert asked == ["https://acme.com", "https://acme.com/careers", "https://acme.com/careers/open-positions"]


def test_careers_walk_uses_the_homepage_link_and_gives_up_cleanly():
    site = {
        "https://acme.com": '<a href="https://work.acme.com/">Careers</a>',
        "https://work.acme.com/": '<a href="https://jobs.lever.co/acme">x</a>',
    }
    assert discover.careers_walk("https://acme.com", site.get) == ("lever", "acme", "https://work.acme.com/")
    assert discover.careers_walk("https://quiet.example", lambda u: None) is None  # homepage unreachable
    assert discover.careers_walk("not a url", lambda u: "") is None


def test_robots_is_read_once_per_host_and_obeyed():
    asked = []

    def fetch(url):
        asked.append(url)
        return "User-agent: *\nDisallow: /careers\n" if "acme" in url else None

    robots = Robots(fetch, "Hiredra/0.1")
    assert not robots.allowed("https://acme.com/careers")
    assert robots.allowed("https://acme.com/jobs")
    assert robots.allowed("https://other.com/careers")  # no robots.txt: everything allowed
    assert asked == ["https://acme.com/robots.txt", "https://other.com/robots.txt"]


# ── slice 2.6: companies seen only on JobSpy ────────────────────────────────
def test_orphans_read_apply_links_and_guess_the_rest():
    rows = [
        (1, "Velaura", "Backend Engineer", "https://jobs.lever.co/velaura/abc", "https://in.indeed.com/x"),
        (1, "Velaura", "ML Engineer", None, None),
        (
            2,
            "Infosys Limited",
            "Java Developer",
            "https://career.infosys.taleo.net/careersection/2/jobdetail.ftl?job=1",
            None,
        ),
        (3, "Genea", "SRE", "https://grnh.se/abc123", None),
        (5, "Capgemini", "QA", "https://apply.workable.com/j/C15CE8BF0F", None),
        (4, "Acme Group", "Data Engineer", None, "https://in.indeed.com/y"),
    ]
    by_name = {o.name: o for o in discover.orphans(rows)}
    velaura = by_name["Velaura"]
    assert velaura.board == ("lever", "velaura") and velaura.titles == {"backend engineer", "ml engineer"}
    assert by_name["Infosys Limited"].other_ats == "taleo" and by_name["Infosys Limited"].board is None
    assert by_name["Genea"].short_links == ["https://grnh.se/abc123"]
    assert by_name["Capgemini"].short_links == ["https://apply.workable.com/j/C15CE8BF0F"]
    assert by_name["Capgemini"].board is None and by_name["Capgemini"].other_ats == "workable"
    assert by_name["Acme Group"].slugs == ("acme", "acmegroup", "acme-group")  # the board-name tail off first


def test_orphans_drop_a_guess_two_employers_propose():
    rows = [(1, "Nova Labs", "x", None, None), (2, "Nova-Labs Inc", "y", None, None), (3, "NovaLabs", "z", None, None)]
    rows += [(4, "Bolt", "x", None, None), (5, "Bolt Group", "y", None, None)]
    by_name = {o.name: o for o in discover.orphans(rows)}
    assert by_name["NovaLabs"].slugs == ("novalabs",)  # one employer under three spellings keeps its guesses
    assert by_name["Bolt"].slugs == () and by_name["Bolt Group"].slugs == ("boltgroup", "bolt-group")
    assert discover.orphans([(1, "AI", "x", None, None)])[0].slugs == ()  # too short to be anyone's


def test_same_opening_needs_a_shared_title():
    titles = {discover.norm_title("Backend Engineer (Python) - Bengaluru")}
    assert discover.same_opening(
        [Job(url="u", title="Backend Engineer (Python) – Bengaluru", company="X", source="lever")], titles
    )
    assert not discover.same_opening([Job(url="u", title="Backend Engineer", company="X", source="lever")], titles)


# ── slice 3.13: the boards behind a Getro network ───────────────────────────
def test_getro_job_pages_group_a_sitemaps_job_pages_by_company():
    locs = [
        "https://jobs.3one4capital.com/companies",
        "https://jobs.3one4capital.com/companies/ayr-energy-2",
        "https://jobs.3one4capital.com/companies/ayr-energy-2/jobs/93878274-engineer",
        "https://jobs.3one4capital.com/companies/lokal/jobs/1-sde",
        "https://jobs.3one4capital.com/companies/ayr-energy-2/jobs/93878275-analyst/",
    ]
    assert discover.getro_job_pages(locs) == {
        "ayr-energy-2": [locs[2], locs[4]],
        "lokal": [locs[3]],
    }


def test_getro_job_reads_the_employer_and_the_jobs_source_link():
    data = {"props": {"pageProps": {"initialState": {"jobs": {"currentJob": {
        "title": "Engineer", "url": "https://ats.rippling.com/ayr-careers/jobs/abc",
        "organization": {"name": "Ayr Energy"}}}}}}}  # fmt: skip
    page = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(data)}</script></html>'
    assert discover.getro_job(page) == ("Ayr Energy", "https://ats.rippling.com/ayr-careers/jobs/abc")
    assert discover.detect_board(discover.getro_job(page)[1]) == ("rippling", "ayr-careers")
    assert discover.getro_job("<html>no data</html>") is None
    assert discover.getro_job(page.replace("Ayr Energy", "")) is None  # no employer: nothing to file it under


# ── the step-3 ATS company lists: the board read from each row's URL ─────────
@pytest.mark.parametrize(
    ("provider", "url", "domain", "want"),
    [
        ("workday", "https://23andme.wd5.myworkdayjobs.com/23", None, "23andme.wd5.myworkdayjobs.com/23"),
        ("oracle", "https://ebwg.fa.us2.oraclecloud.com/hcmUI/CandidateExperience/en/sites/CX", None,
         "ebwg.fa.us2.oraclecloud.com/CX"),
        ("keka", "https://abcoffee.keka.com/careers", None, "abcoffee"),
        ("eightfold", "https://albemarle.eightfold.ai/careers", "albemarle.com", "albemarle.eightfold.ai/albemarle.com"),
        ("icims", "https://careers-360care.icims.com", None, "360care"),  # a bare host: the classic label's slug
        ("icims", "https://globalcareers-msci.icims.com", None, "globalcareers-msci.icims.com"),
        ("teamtailor", "https://123pousse.teamtailor.com", None, "123pousse.teamtailor.com"),
        ("successfactors", "https://apply.iav.com", None, "apply.iav.com"),  # its own domain: the host
        ("workday", "https://jobs.lever.co/acme", None, None),  # another platform's board: not this list's
        ("smartrecruiters", "not a url", None, None),
    ],
)  # fmt: skip
def test_board_from_url(provider, url, domain, want):
    assert discover.board_from_url(provider, url, domain) == want


def test_phenom_boards_group_a_sites_jobs_by_the_board_they_apply_through():
    rows = [
        {"applyUrl": "https://lilly.wd115.myworkdayjobs.com/LLY/job/Bengaluru/Engineer_R-1/apply"},
        {"applyUrl": "https://lilly.wd115.myworkdayjobs.com/LLY/job/Pune/Analyst_R-2/apply"},
        {"applyUrl": "https://dpdhlgroup.avature.net/en_US/careers/JobDetail?jobId=7"},  # no adapter of ours
        {"applyUrl": ""},  # the site's own form
        {"applyUrl": "https://jobs.lever.co/acme/1234"},  # an adapter not offered here
    ]
    groups = discover.phenom_boards(rows, ["workday"])
    assert {k: len(v) for k, v in groups.items()} == {("workday", "lilly.wd115.myworkdayjobs.com/LLY"): 2, None: 3}
