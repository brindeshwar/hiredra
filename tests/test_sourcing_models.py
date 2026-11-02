"""Tests for the Job dataclass and Provider protocol (build sub-slice 2a.2)."""

from __future__ import annotations

import dataclasses

import pytest

from hiredra.sourcing import CompanyEntry, FetchContext, Job, Provider
from hiredra.storage import db


def _kwargs(**over):
    base = {
        "url": "https://boards.greenhouse.io/acme/jobs/1",
        "title": "Backend Engineer",
        "company": "Acme",
        "source": "greenhouse",
        "description": "Build things.",
        "years_required": 0,
        "location": "Remote",
        "date_posted": "2026-09-01",
        "job_type": "fulltime",
        "is_remote": True,
    }
    base.update(over)
    return base


# ── Job: construction & identity ────────────────────────────────────────────
def test_job_id_falls_back_to_url():
    assert Job(**_kwargs()).job_id == _kwargs()["url"]


def test_explicit_job_id_kept():
    assert Job(**_kwargs(job_id="greenhouse:gh-42")).job_id == "greenhouse:gh-42"


def test_job_is_frozen():
    job = Job(**_kwargs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        job.title = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("field", ["url", "title", "company", "source"])
@pytest.mark.parametrize("bad", ["", "   ", None])
def test_required_fields_must_be_nonempty(field, bad):
    with pytest.raises(ValueError):
        Job(**_kwargs(**{field: bad}))


def test_optional_fields_default_to_none():
    job = Job(url="u://x", title="t", company="c", source="s")
    assert job.description is None
    assert job.years_required is None
    assert job.salary is None
    assert job.is_remote is None


# ── Job: years_required 0 vs None vs bad ────────────────────────────────────
def test_years_required_zero_kept():
    assert Job(**_kwargs(years_required=0)).years_required == 0


def test_years_required_none_kept():
    assert Job(**_kwargs(years_required=None)).years_required is None


@pytest.mark.parametrize("bad", ["unknown", 0.0, True, "3"])
def test_years_required_non_int_rejected(bad):
    with pytest.raises(TypeError):
        Job(**_kwargs(years_required=bad))


@pytest.mark.parametrize("bad", [1, 0, "yes"])
def test_is_remote_non_bool_rejected(bad):
    with pytest.raises(TypeError):
        Job(**_kwargs(is_remote=bad))


# ── Job <-> storage round-trip ─────────────────────────────────────────────
def test_to_dict_maps_is_remote_to_int():
    assert Job(**_kwargs(is_remote=True)).to_dict()["is_remote"] == 1
    assert Job(**_kwargs(is_remote=False)).to_dict()["is_remote"] == 0
    assert Job(**_kwargs(is_remote=None)).to_dict()["is_remote"] is None


def test_job_survives_a_full_storage_round_trip(pg_url):
    conn = db.connect(pg_url)
    db.init_db(conn)
    original = Job(**_kwargs(job_id="greenhouse:gh-1", years_required=0, is_remote=True))

    db.upsert_job(conn, original.to_dict())
    rebuilt = Job.from_row(db.get_job(conn, "greenhouse:gh-1"))

    assert rebuilt == original
    assert rebuilt.years_required == 0
    assert rebuilt.is_remote is True
    conn.close()


def test_from_row_ignores_non_job_columns(pg_url):
    conn = db.connect(pg_url)
    db.init_db(conn)
    db.upsert_job(conn, Job(**_kwargs(job_id="greenhouse:gh-1")).to_dict())
    db.set_status(conn, "greenhouse:gh-1", "scored", detail="s=0.7")

    row = db.get_job(conn, "greenhouse:gh-1")
    assert "status" in row and "first_seen_at" in row  # wider than Job
    job = Job.from_row(row)  # must not choke on the extra columns
    assert job.job_id == "greenhouse:gh-1"
    conn.close()


# ── structured fields (03-adapters.md §1) ───────────────────────────────────
def test_the_structured_fields_survive_a_storage_round_trip(pg_url):
    conn = db.connect(pg_url)
    db.init_db(conn)
    original = Job(**_kwargs(job_id="greenhouse:gh-2", work_mode="hybrid", salary_min=8, salary_max=10.5,
                             salary_currency="INR", salary_period="year"))  # fmt: skip
    db.upsert_job(conn, original.to_dict())
    rebuilt = Job.from_row(db.get_job(conn, "greenhouse:gh-2"))
    assert rebuilt == original  # a numeric comes back as a Decimal: read as a float
    assert (rebuilt.employment_type, rebuilt.salary_max) == ("full_time", 10.5)  # from job_type "fulltime"
    conn.close()


@pytest.mark.parametrize(
    ("job_type", "want"),
    [("fulltime", "full_time"), ("Full-time Employment", "full_time"), ("Part Time", "part_time"),
     ("intern", "internship"), ("Contract Full time", "contract"), ("Freelance-Remote", "contract"),
     ("Full time or Part time", None), ("On- roll", None), ("Remote", None), (None, None)],
)  # fmt: skip
def test_employment_type_is_read_from_the_platforms_job_type(job_type, want):
    assert Job(**_kwargs(job_type=job_type)).employment_type == want
    assert Job(**_kwargs(job_type="intern", employment_type="contract")).employment_type == "contract"  # stated wins


@pytest.mark.parametrize(
    "over",
    [{"work_mode": "Remote"}, {"employment_type": "fulltime"}, {"salary_period": "annual"}, {"salary_min": -1},
     {"salary_max": "10"}, {"salary_min": True}],
)  # fmt: skip
def test_a_structured_field_outside_its_vocabulary_is_refused(over):
    with pytest.raises((TypeError, ValueError)):
        Job(**_kwargs(**over))


# ── Provider protocol ──────────────────────────────────────────────────────
class _FakeProvider:
    name = "fake"

    def detect(self, entry: CompanyEntry) -> bool:
        return entry.provider == "fake" or bool(entry.careers_url)

    def board_url(self, entry: CompanyEntry) -> str | None:
        return entry.careers_url

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        return [Job(url="u://1", title="Role", company=entry.name, source=self.name)]


def test_fake_provider_satisfies_protocol():
    assert isinstance(_FakeProvider(), Provider)


def test_provider_detect_and_fetch():
    p = _FakeProvider()
    assert p.detect(CompanyEntry(name="Acme", careers_url="https://acme.com/careers"))
    assert not p.detect(CompanyEntry(name="Acme"))

    jobs = p.fetch(CompanyEntry(name="Acme"), ctx=object())  # type: ignore[arg-type]
    assert len(jobs) == 1 and jobs[0].company == "Acme" and jobs[0].source == "fake"


def test_non_provider_fails_isinstance():
    class NotAProvider:
        name = "x"
        # no detect / fetch

    assert not isinstance(NotAProvider(), Provider)


# ── CompanyEntry ───────────────────────────────────────────────────────────
def test_company_entry_defaults():
    e = CompanyEntry(name="Acme")
    assert e.careers_url is None and e.api is None and e.provider is None
    assert e.extra == {}
