"""Shared test fixtures."""

from __future__ import annotations

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
