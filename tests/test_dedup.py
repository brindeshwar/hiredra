"""Tests for dedup (build sub-slice 2a.7)."""

from __future__ import annotations

import pytest

from hiredra.sourcing.dedup import (
    _norm_company,
    _norm_role,
    _norm_url,
    dedup,
)
from hiredra.sourcing.job import Job
from hiredra.sourcing.persist import persist_jobs
from hiredra.storage import db


@pytest.fixture()
def conn():
    c = db.connect(":memory:")
    db.init_db(c)
    yield c
    c.close()


def _job(**over) -> Job:
    base = {
        "url": "https://job-boards.greenhouse.io/acme/jobs/1",
        "title": "Backend Engineer",
        "company": "Acme",
        "source": "greenhouse",
        "description": "Build APIs.",
        "date_posted": "2026-09-01",
    }
    base.update(over)
    return Job(**base)


# ── _norm_url ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "a, b",
    [
        ("https://www.linkedin.com/jobs/view/123", "http://linkedin.com/jobs/view/123/"),
        ("https://job-boards.greenhouse.io/acme/jobs/5", "https://boards.greenhouse.io/acme/jobs/5"),
        ("https://x.com/j?utm_source=li&jk=abc", "https://x.com/j?jk=abc"),
        ("https://x.com/j?b=2&a=1", "https://x.com/j?a=1&b=2"),
        ("https://x.com/j#apply-now", "https://x.com/j"),
        ("https://in.indeed.com/viewjob?jk=x", "https://indeed.com/viewjob?jk=x"),
    ],
)
def test_norm_url_equivalences(a, b):
    assert _norm_url(a) == _norm_url(b) is not None


def test_norm_url_keeps_unknown_params():
    # 'from' is Indeed navigation context -- not definitely tracking -> keep it
    assert _norm_url("https://in.indeed.com/viewjob?jk=x&from=serp") != _norm_url("https://in.indeed.com/viewjob?jk=x")


@pytest.mark.parametrize("value", [None, "", "   ", 123])
def test_norm_url_none_ish(value):
    assert _norm_url(value) is None


# ── _norm_company ──────────────────────────────────────────────────────────
def test_norm_company_strips_legal_suffixes():
    forms = ["Acme", "Acme Inc.", "ACME  Ltd", "Acme Pvt. Ltd.", "Acme, LLC", "Acme Private Limited"]
    assert len({_norm_company(f) for f in forms}) == 1


def test_norm_company_keeps_descriptive_words():
    assert _norm_company("Acme Labs") != _norm_company("Acme")
    assert _norm_company("Scale AI") != _norm_company("Scale")


# ── _norm_role ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "a, b",
    [
        ("AI Engineer (Remote)", "AI Engineer"),
        ("Backend Engineer - Bengaluru, India", "Backend Engineer"),
        ("Backend Engineer, Remote", "backend engineer"),
        ("Data Engineer (Full-time)", "Data Engineer"),
        ("ML Engineer | Bangalore", "ml engineer"),
    ],
)
def test_norm_role_equivalences(a, b):
    assert _norm_role(a) == _norm_role(b)


@pytest.mark.parametrize(
    "a, b",
    [
        ("Backend Engineer - Payments", "Backend Engineer"),  # team, not location
        ("Backend Engineer - London Markets", "Backend Engineer"),  # "Markets" -> not stripped
        ("Senior Backend Engineer", "Backend Engineer"),  # seniority kept
        ("Backend Engineer, Core Platform", "Backend Engineer"),
    ],
)
def test_norm_role_does_not_over_merge(a, b):
    assert _norm_role(a) != _norm_role(b)


# ── intra-run ──────────────────────────────────────────────────────────────
def test_exact_job_id_dup_collapses(conn):
    r = dedup(conn, [_job(job_id="greenhouse:1"), _job(job_id="greenhouse:1")])
    assert len(r.new) == 1 and r.collapsed_in_run == 1


def test_url_dup_collapses_ats_wins(conn):
    ats = _job(job_id="greenhouse:1", url="https://boards.greenhouse.io/acme/jobs/1")
    agg = _job(
        job_id="jobspy-linkedin:x",
        url="https://job-boards.greenhouse.io/acme/jobs/1/",
        source="jobspy-linkedin",
        description="richer agg text",
        salary="X",
    )
    r = dedup(conn, [agg, ats])
    assert len(r.new) == 1
    assert r.new[0].source == "greenhouse"  # ATS wins even though agg had more fields
    assert r.merges[0].on == "url" and r.merges[0].kept.source == "greenhouse"


def test_company_role_dup_collapses_when_url_differs(conn):
    ats = _job(job_id="greenhouse:1", company="Acme Inc.", url="https://acme.com/careers/1")
    agg = _job(
        job_id="jobspy-indeed:x", company="Acme", url="https://in.indeed.com/viewjob?jk=z", source="jobspy-indeed"
    )
    r = dedup(conn, [agg, ats])
    assert len(r.new) == 1 and r.new[0].source == "greenhouse"
    assert r.merges[0].on == "company+role"


def test_two_aggregators_more_complete_wins(conn):
    a = _job(
        job_id="jobspy-indeed:1",
        source="jobspy-indeed",
        url="u://a",
        description=None,
        salary=None,
        is_remote=None,
        location=None,
    )
    b = _job(
        job_id="jobspy-linkedin:1",
        source="jobspy-linkedin",
        url="u://b",
        description="full jd",
        salary="₹20L",
        is_remote=True,
        location="Remote",
    )
    r = dedup(conn, [a, b])  # same company+role, different url
    assert len(r.new) == 1 and r.new[0].job_id == "jobspy-linkedin:1"


def test_distinct_roles_not_merged(conn):
    r = dedup(
        conn,
        [
            _job(job_id="greenhouse:1", title="Backend Engineer", url="u://1"),
            _job(job_id="greenhouse:2", title="Backend Engineer, Payments", url="u://2"),
        ],
    )
    assert len(r.new) == 2 and r.collapsed_in_run == 0


# (company, role) is not identity on its own (2026-10-27): measured on 675 boards that post in India it merged
# 24% of 49,906 jobs and threw away 624 India-eligible postings for a foreign copy of the same title.
def test_one_boards_postings_of_a_title_in_two_countries_are_both_kept(conn):
    r = dedup(
        conn,
        [
            _job(job_id="greenhouse:1", title="Software Engineer II", location="San Jose", url="u://1"),
            _job(
                job_id="greenhouse:2",
                title="Software Engineer II (Bangalore, India)",
                location="Bengaluru",
                url="u://2",
            ),
        ],
    )
    assert len(r.new) == 2 and r.collapsed_in_run == 0


def test_one_boards_postings_of_a_title_are_both_kept_even_in_one_country(conn):
    # two ids on one ATS board are two postings by the board's own word; a true duplicate shares a url
    r = dedup(
        conn,
        [
            _job(job_id="greenhouse:1", title="Founding Engineer", location="San Francisco", url="u://1"),
            _job(job_id="greenhouse:2", title="Founding Engineer", location="San Francisco", url="u://2"),
        ],
    )
    assert len(r.new) == 2


def test_an_aggregator_copy_in_another_country_is_not_merged(conn):
    ats = _job(job_id="greenhouse:1", title="Forward Deployment Strategist", location="San Mateo", url="u://1")
    agg = _job(
        job_id="jobspy-indeed:x",
        title="Forward Deployment Strategist - India",
        location="Bengaluru, Karnataka, India",
        source="jobspy-indeed",
        url="https://in.indeed.com/viewjob?jk=z",
    )
    r = dedup(conn, [ats, agg])
    assert len(r.new) == 2


def test_an_aggregator_copy_in_the_same_country_still_merges(conn):
    ats = _job(job_id="greenhouse:1", location="Bangalore", url="https://acme.com/careers/1")
    agg = _job(
        job_id="jobspy-indeed:x",
        location="Bengaluru, Karnataka, India",
        source="jobspy-indeed",
        url="https://in.indeed.com/viewjob?jk=z",
    )
    r = dedup(conn, [agg, ats])
    assert len(r.new) == 1 and r.new[0].source == "greenhouse"


def test_cross_run_a_new_city_is_not_matched_to_a_stored_one(conn):
    persist_jobs(conn, [_job(job_id="greenhouse:1", location="London", url="u://1")])
    db.set_status(conn, "greenhouse:1", "dropped", detail="location: restricted to GB")
    r = dedup(conn, [_job(job_id="greenhouse:2", location="Bengaluru, India", url="u://2")])
    assert [j.job_id for j in r.new] == ["greenhouse:2"] and r.seen == []


# ── cross-run ──────────────────────────────────────────────────────────────
def test_cross_run_match_by_job_id(conn):
    persist_jobs(conn, [_job(job_id="greenhouse:1")])
    r = dedup(
        conn,
        [
            _job(job_id="greenhouse:1"),  # -> seen (same job)
            _job(job_id="greenhouse:2", title="Data Engineer", url="u://new"),  # -> new (distinct role)
        ],
    )
    assert r.seen == ["greenhouse:1"]
    assert [j.job_id for j in r.new] == ["greenhouse:2"]
    assert r.matched_in_store == 1


def test_cross_run_match_by_url_including_apply_url(conn):
    persist_jobs(conn, [_job(job_id="greenhouse:1", url="https://job-boards.greenhouse.io/acme/jobs/1")])
    # a jobspy row whose apply_url (job_url_direct) resolves to the stored ATS url
    agg = _job(
        job_id="jobspy-linkedin:x",
        source="jobspy-linkedin",
        url="https://www.linkedin.com/jobs/view/999",
        apply_url="https://boards.greenhouse.io/acme/jobs/1",
    )
    r = dedup(conn, [agg])
    assert r.new == [] and r.seen == ["greenhouse:1"]
    assert r.store_matches[0][1] == "greenhouse:1"


def test_cross_run_match_by_company_role(conn):
    persist_jobs(conn, [_job(job_id="greenhouse:1", company="Acme Inc.")])
    agg = _job(
        job_id="jobspy-indeed:x", company="Acme", source="jobspy-indeed", url="https://in.indeed.com/viewjob?jk=z"
    )
    r = dedup(conn, [agg])
    assert r.new == [] and r.seen == ["greenhouse:1"]


def test_cross_run_matches_a_dropped_job(conn):
    persist_jobs(conn, [_job(job_id="greenhouse:1")])
    db.set_status(conn, "greenhouse:1", "dropped", detail="seniority")
    r = dedup(conn, [_job(job_id="greenhouse:1")])
    assert r.seen == ["greenhouse:1"] and r.new == []
    assert db.get_job(conn, "greenhouse:1")["status"] == "dropped"  # unchanged


def test_empty_input(conn):
    r = dedup(conn, [])
    assert r.new == [] and r.seen == [] and r.collapsed_in_run == 0


# ── touch_job + full flow ──────────────────────────────────────────────────
def test_touch_job(conn):
    persist_jobs(conn, [_job(job_id="greenhouse:1")])
    before = db.get_job(conn, "greenhouse:1")["last_seen_at"]
    import time as _t

    _t.sleep(0.01)  # ensure a later timestamp is possible
    db.touch_job(conn, "greenhouse:1")
    after = db.get_job(conn, "greenhouse:1")
    assert after["last_seen_at"] >= before
    kinds = [e["kind"] for e in db.get_events(conn, "greenhouse:1")]
    assert kinds.count("seen") == 2  # first-sight + the touch
    with pytest.raises(KeyError):
        db.touch_job(conn, "nope")


def test_full_flow_persist_new_touch_seen(conn):
    persist_jobs(conn, [_job(job_id="greenhouse:1")])

    pile = [
        _job(job_id="greenhouse:1"),  # -> seen
        _job(job_id="greenhouse:2", title="Data Engineer", url="u://2"),  # -> new
        _job(
            job_id="jobspy-indeed:2", title="Data Engineer", url="u://3", source="jobspy-indeed"
        ),  # -> merges into greenhouse:2
    ]
    r = dedup(conn, pile)
    assert [j.job_id for j in r.new] == ["greenhouse:2"]
    assert r.seen == ["greenhouse:1"]

    persist_jobs(conn, r.new)
    for jid in r.seen:
        db.touch_job(conn, jid)

    assert db.status_counts(conn)["fetched"] == 2
    assert [e["kind"] for e in db.get_events(conn, "greenhouse:1")].count("seen") == 2


def test_two_copies_from_one_aggregator_in_one_country_merge(conn):
    # aggregators repost one role under new ids; unlike an ATS board their ids are not postings
    a = _job(job_id="jobspy-indeed:1", source="jobspy-indeed", url="u://a", location="Pune, Maharashtra, India")
    b = _job(job_id="jobspy-indeed:2", source="jobspy-indeed", url="u://b", location="Pune, India")
    r = dedup(conn, [a, b])
    assert len(r.new) == 1
