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
