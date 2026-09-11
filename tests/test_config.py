"""Config loading (build sub-slice 2a.8a)."""

from __future__ import annotations

import textwrap

import pytest

from hiredra.sourcing.config import ConfigError, load_companies, load_run_config
from hiredra.sourcing.jobspy_source import JobSpyConfig, SiteQuery


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    return p


def _write_raw(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


# ── companies.yml ──────────────────────────────────────────────────────────
def test_load_companies_maps_fields(tmp_path):
    p = _write(tmp_path, "c.yml", """
        companies:
          - name: Stripe
            careers_url: https://stripe.com/jobs
          - name: Ramp
            slug: ramp
            provider: ashby
          - name: Disabled Co
            careers_url: https://x
            enabled: false
    """)
    entries = load_companies(p)
    assert [e.name for e in entries] == ["Stripe", "Ramp"]          # disabled one dropped
    assert entries[1].provider == "ashby" and entries[1].extra == {"slug": "ramp"}


@pytest.mark.parametrize("body", [
    "companies:\n  - careers_url: https://x\n",                     # missing name
    "companies:\n  - name: X\n    wat: 1\n",                        # unknown key
    "companies: not-a-list\n",
])
def test_load_companies_rejects_bad_input(tmp_path, body):
    with pytest.raises(ConfigError):
        load_companies(_write(tmp_path, "c.yml", body))


# ── run config: strict, nested ─────────────────────────────────────────────
_GOOD = textwrap.dedent("""
    kind: scoring
    companies_path: config/companies.yml
    pacing: {between_companies: 2.0, jitter: 1.0}
    http: {timeout: 20.0}
    jobspy:
      sites: [indeed, linkedin]
      country_indeed: India
      default_query:
        search_terms: [AI Engineer, Backend Engineer]
        hours_old: 200
      linkedin_query:
        search_terms: [AI Engineer]
        job_type: fulltime
    path2: {enabled: false}
""").lstrip("\n")


def test_load_run_config_builds_everything(tmp_path):
    rc = load_run_config(_write(tmp_path, "scoring.yml", _GOOD))
    assert rc.kind == "scoring" and rc.config_name == "scoring.yml"
    assert rc.pacing.between_companies == 2.0 and rc.pacing.jitter == 1.0
    assert rc.http.timeout == 20.0 and rc.http.cooldown_seconds == 900.0  # default fills in
    assert rc.jobspy.sites == ("indeed", "linkedin")
    assert rc.jobspy.default_query.search_terms == ("AI Engineer", "Backend Engineer")
    assert rc.jobspy.default_query.hours_old == 200
    assert rc.jobspy.linkedin_query.job_type == "fulltime"


def test_jobspy_block_round_trips_into_jobspyconfig(tmp_path):
    rc = load_run_config(_write(tmp_path, "s.yml", _GOOD))
    expected = JobSpyConfig(
        sites=("indeed", "linkedin"),
        country_indeed="India",
        default_query=SiteQuery(
            search_terms=("AI Engineer", "Backend Engineer"), hours_old=200
        ),
        linkedin_query=SiteQuery(search_terms=("AI Engineer",), job_type="fulltime"),
    )
    assert rc.jobspy == expected


def test_run_config_rejects_bad_kind(tmp_path):
    body = _GOOD.replace("kind: scoring", "kind: backfilll")
    with pytest.raises(ConfigError):
        load_run_config(_write(tmp_path, "s.yml", body))


def test_run_config_rejects_unknown_top_level_key(tmp_path):
    body = _GOOD + "\nwat: 1\n"
    with pytest.raises(ConfigError) as ei:
        load_run_config(_write(tmp_path, "s.yml", body))
    assert "unknown top-level" in str(ei.value)


def test_run_config_rejects_unknown_key_in_nested_default_query(tmp_path):
    body = _GOOD.replace("search_terms: [AI Engineer, Backend Engineer]",
                         "serch_terms: [AI Engineer]")
    with pytest.raises(ConfigError) as ei:
        load_run_config(_write(tmp_path, "s.yml", body))
    assert "jobspy.default_query" in str(ei.value) and "serch_terms" in str(ei.value)


def test_run_config_rejects_unknown_key_in_pacing_and_http(tmp_path):
    for section in ("pacing: {between_companies: 2.0, jitter: 1.0}",
                    "http: {timeout: 20.0}"):
        bad = section.replace("}", ", bogus: 1}")
        body = _GOOD.replace(section, bad)
        with pytest.raises(ConfigError):
            load_run_config(_write(tmp_path, "s.yml", body))


def test_run_config_missing_required_search_terms(tmp_path):
    body = _GOOD.replace("    search_terms: [AI Engineer, Backend Engineer]\n", "")
    with pytest.raises(ConfigError):
        load_run_config(_write(tmp_path, "s.yml", body))
