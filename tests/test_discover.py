"""Board discovery by guessed slug (build slice 2.4): the guesses, the YC filter, the live check and the
namesake guard. Providers are faked; no network."""

from __future__ import annotations

import json

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
