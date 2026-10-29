"""Shared test fixtures."""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

from hiredra.sourcing.http import HttpClient, HttpConfig


@pytest.fixture()
def http_factory():
    """Build HttpClient(s) wired to a MockTransport handler and a no-op sleep."""
    made: list[HttpClient] = []

    def _make(handler, **cfg) -> HttpClient:
        client = HttpClient(
            HttpConfig(**cfg),
            transport=httpx.MockTransport(handler),
            sleep=lambda _seconds: None,
        )
        made.append(client)
        return client

    yield _make
    for client in made:
        client.close()


@pytest.fixture()
def greenhouse_payload() -> dict:
    """A realistic ?content=true list response (bodies double-encoded, like the real API)."""
    return {
        "jobs": [
            {
                "id": 12345,
                "title": "Backend Engineer",
                "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/12345",
                "location": {"name": "San Francisco, CA"},
                "content": "&lt;p&gt;Build APIs. 0-2 years experience.&lt;/p&gt;",
                "first_published": "2026-08-15T09:30:00.000Z",
                "updated_at": "2026-09-01T10:00:00.000Z",
            },
            {
                "id": 67890,
                "title": "AI Engineer (Remote)",
                "absolute_url": "https://job-boards.greenhouse.io/acme/jobs/67890",
                "location": {"name": "Remote - US"},
                "content": "&lt;p&gt;LLMs, agents, RAG.&lt;/p&gt;",
                "updated_at": "2026-09-05T00:00:00.000Z",
            },
            {
                "id": 999,
                "title": "Malformed (no absolute_url)",
                "location": {"name": "Nowhere"},
            },
        ]
    }


@pytest.fixture()
def lever_payload() -> list:
    """A realistic Lever v0 postings response -- a bare array."""
    return [
        {
            "id": "aaaa-1111",
            "text": "Senior Backend Engineer",
            "hostedUrl": "https://jobs.lever.co/acme/aaaa-1111",
            "applyUrl": "https://jobs.lever.co/acme/aaaa-1111/apply",
            "categories": {
                "location": "Berlin",
                "allLocations": ["Berlin", "Remote - EU"],
                "commitment": "Full-time",
            },
            "descriptionPlain": "Build backend systems.",
            "createdAt": 1_726_000_000_000,
            "workplaceType": "hybrid",
        },
        {
            "id": "bbbb-2222",
            "text": "AI Engineer",
            "hostedUrl": "https://jobs.lever.co/acme/bbbb-2222",
            "categories": {"location": "New York", "commitment": "Full-time"},
            "descriptionPlain": "LLMs and agents.",
            "createdAt": 1_726_500_000_000,
            "workplaceType": "remote",
        },
        {"id": "cccc-3333", "text": "Malformed (no hostedUrl)"},
    ]


@pytest.fixture()
def ashby_payload() -> dict:
    """A realistic Ashby posting-api response -- {jobs: [...]}."""
    return {
        "jobs": [
            {
                "id": "job_001",
                "title": "Backend Engineer",
                "jobUrl": "https://jobs.ashbyhq.com/acme/job_001",
                "applyUrl": "https://jobs.ashbyhq.com/acme/job_001/application",
                "location": "New York",
                "secondaryLocations": [
                    {"location": "Remote (US)"},
                    {
                        "address": {
                            "postalAddress": {
                                "addressLocality": "Austin",
                                "addressCountry": "USA",
                            }
                        }
                    },
                ],
                "descriptionPlain": "Backend work.",
                "publishedAt": "2026-08-20T12:00:00.000Z",
                "employmentType": "FullTime",
                "workplaceType": "Onsite",
                "compensation": {"compensationTierSummary": "$180K - $220K"},
            },
            {
                "id": "job_002",
                "title": "ML Engineer",
                "jobUrl": "https://jobs.ashbyhq.com/acme/job_002",
                "location": "Remote",
                "descriptionPlain": "ML systems.",
                "publishedAt": "2026-09-01T00:00:00Z",
                "workplaceType": "Remote",
            },
            {"id": "job_003", "title": "Malformed (no jobUrl)"},
        ]
    }


@pytest.fixture(autouse=True)
def _no_real_nvidia_key(monkeypatch):
    """No test may reach NVIDIA's real API: the key is blanked for every test (python-dotenv never
    overrides a variable that is already set), so a real NvidiaChain can't be built by accident.
    Tests that need one pass a fake client."""
    monkeypatch.setenv("NVIDIA_API_KEY", "")


@pytest.fixture()
def db_url():
    """A fresh, empty Postgres database for one test, dropped afterwards (D1.12). Skips unless
    HIREDRA_TEST_DATABASE_URL points at a server the tests may create and drop databases on."""
    admin_url = os.environ.get("HIREDRA_TEST_DATABASE_URL")
    if not admin_url:
        pytest.skip("HIREDRA_TEST_DATABASE_URL not set (needs a Postgres server)")
    psycopg = pytest.importorskip("psycopg")
    name = f"hiredra_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    yield psycopg.conninfo.make_conninfo(admin_url, dbname=name)
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')


@pytest.fixture(scope="session")
def _pg_session_url():
    """One migrated scratch database for the whole test session, dropped at the end (D1.12)."""
    admin_url = os.environ.get("HIREDRA_TEST_DATABASE_URL")
    if not admin_url:
        pytest.skip("HIREDRA_TEST_DATABASE_URL not set (needs a Postgres server)")
    psycopg = pytest.importorskip("psycopg")
    from hiredra import pg

    name = f"hiredra_test_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    url = psycopg.conninfo.make_conninfo(admin_url, dbname=name)
    with pg.connect(url) as conn:
        pg.migrate(conn)
    yield url
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE "{name}" WITH (FORCE)')


@pytest.fixture()
def pg_url(_pg_session_url):
    """The session database, emptied first: every table but schema_migrations truncated, identities restarted.
    Use it wherever a test needs a store (``hiredra.storage.connect(pg_url)``, or ``--db pg_url`` for a CLI)."""
    import psycopg

    with psycopg.connect(_pg_session_url, autocommit=True) as conn:
        tables = [r[0] for r in conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename <> 'schema_migrations'"
        )]  # fmt: skip
        conn.execute(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE")
    return _pg_session_url
