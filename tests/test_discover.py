"""Board discovery by guessed slug (build slice 2.4): the guesses, the YC filter, the live check and the
namesake guard. Providers are faked; no network."""

from __future__ import annotations

import json

import pytest

from hiredra.sourcing import discover
from hiredra.sourcing.http import FetchError, HTTPError
from hiredra.sourcing.job import Job


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

    robots = discover.Robots(fetch, "Hiredra/0.1")
    assert not robots.allowed("https://acme.com/careers")
    assert robots.allowed("https://acme.com/jobs")
    assert robots.allowed("https://other.com/careers")  # no robots.txt: everything allowed
    assert asked == ["https://acme.com/robots.txt", "https://other.com/robots.txt"]
