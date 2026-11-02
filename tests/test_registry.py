"""Tests for the provider registry + router (build sub-slice 2a.5)."""

from __future__ import annotations

import logging

import pytest

from hiredra.sourcing.provider import CompanyEntry, Provider
from hiredra.sourcing.providers import AshbyProvider, GreenhouseProvider, LeverProvider
from hiredra.sourcing.registry import REGISTRY, route


@pytest.mark.parametrize(
    "careers_url, cls",
    [
        ("https://job-boards.greenhouse.io/acme", GreenhouseProvider),
        ("https://boards.greenhouse.io/acme", GreenhouseProvider),
        ("https://jobs.lever.co/acme", LeverProvider),
        ("https://jobs.eu.lever.co/acme", LeverProvider),
        ("https://jobs.ashbyhq.com/acme", AshbyProvider),
    ],
)
def test_route_by_careers_url(careers_url, cls):
    assert isinstance(route(CompanyEntry(name="Acme", careers_url=careers_url)), cls)


def test_unroutable_returns_none():
    assert route(CompanyEntry(name="A", careers_url="https://acme.com/careers")) is None
    assert route(CompanyEntry(name="A")) is None


def test_explicit_provider_name_skips_detect():
    entry = CompanyEntry(name="A", provider="lever")  # no careers_url at all
    assert isinstance(route(entry), LeverProvider)


def test_explicit_unknown_provider_name_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="hiredra.sourcing.registry"):
        assert route(CompanyEntry(name="A", provider="bogus")) is None
    assert "unknown provider" in caplog.text


def test_detect_exception_is_skipped_not_fatal(caplog):
    class Boom:
        name = "boom"

        def detect(self, entry):
            raise RuntimeError("kaboom")

        def fetch(self, entry, ctx):
            return []

    class Always:
        name = "always"

        def detect(self, entry):
            return True

        def fetch(self, entry, ctx):
            return []

    always = Always()
    with caplog.at_level(logging.ERROR, logger="hiredra.sourcing.registry"):
        result = route(CompanyEntry(name="A"), registry=(Boom(), always))
    assert result is always
    assert "kaboom" in caplog.text


def test_registry_contents():
    assert [p.name for p in REGISTRY] == [
        "greenhouse", "lever", "ashby", "workday", "oracle", "smartrecruiters", "workable", "icims",
        "successfactors", "taleobe", "keka", "zohorecruit", "teamtailor", "rippling", "eightfold", "amazon", "ibm",
        "apple", "remoteok", "remotive", "jobicy", "himalayas", "workingnomads", "weworkremotely",
        "hackernews",
    ]  # fmt: skip
    assert all(isinstance(p, Provider) for p in REGISTRY)
