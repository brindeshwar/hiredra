"""Tests for the one piece of genuinely new logic in discovery/harvest.py --
resolve_ats_slug(). Everything else in this program calls already-tested
hiredra code. Kept out of tests/ deliberately: this is a separate, off-to-
the-side program, not part of the mainline suite (`pytest discovery/` to run
these directly)."""

from __future__ import annotations

from discovery.harvest import resolve_ats_slug


def test_greenhouse_board_url():
    assert resolve_ats_slug("https://job-boards.greenhouse.io/stripe") == ("greenhouse", "stripe")


def test_greenhouse_legacy_boards_host():
    assert resolve_ats_slug("https://boards.greenhouse.io/acme") == ("greenhouse", "acme")


def test_lever_url():
    assert resolve_ats_slug("https://jobs.lever.co/leverdemo") == ("lever", "leverdemo")


def test_lever_eu_url():
    assert resolve_ats_slug("https://jobs.eu.lever.co/someco") == ("lever", "someco")


def test_ashby_url():
    assert resolve_ats_slug("https://jobs.ashbyhq.com/ramp") == ("ashby", "ramp")


def test_unrecognized_url_returns_none():
    assert resolve_ats_slug("https://careers.example.com/jobs") is None


def test_workday_url_returns_none():
    # a real ats_links entry from companies_v2.json, not one of our 3 providers
    assert resolve_ats_slug("https://acme.wd1.myworkdayjobs.com/careers") is None
