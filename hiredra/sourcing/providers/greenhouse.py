"""Greenhouse provider -- public boards-api JSON, no key.

Ported from career-ops ``providers/greenhouse.mjs``.

* One request per board with ``?content=true`` -- the list response embeds each
  posting's body, so there are no per-job fetches (docs/sourcing.md sec. 3).
* SSRF: HTTPS only + a fixed host allowlist, re-checked on the exact URL that
  goes on the wire. Redirects are not followed (career-ops ``redirect: 'error'``);
  an unexpected 3xx surfaces as an ``HTTPError`` from the http layer.
"""

from __future__ import annotations

import logging
import re

from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import (
    assert_allowed_url,
    clean,
    infer_remote,
    iso_date,
)

log = logging.getLogger("hiredra.sourcing.greenhouse")

_ALLOWED_HOSTS = frozenset(
    {
        "boards-api.greenhouse.io",
        "boards.greenhouse.io",
        "job-boards.greenhouse.io",
        "job-boards.eu.greenhouse.io",
    }
)
_SLUG_RE = re.compile(r"(?:job-boards(?:\.eu)?|boards)\.greenhouse\.io/([^/?#]+)")
_API = "https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"


class GreenhouseProvider:
    """Satisfies ``hiredra.sourcing.Provider``. Instantiate once and reuse."""

    name = "greenhouse"

    def _board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        if entry.api:
            try:
                assert_allowed_url(entry.api, _ALLOWED_HOSTS, self.name)
            except FetchError:
                return None
            return entry.api
        slug = entry.extra.get("slug") if entry.extra else None
        if not slug and entry.careers_url:
            m = _SLUG_RE.search(entry.careers_url)
            slug = m.group(1) if m else None
        return _API.format(slug=slug) if slug else None

    def detect(self, entry: CompanyEntry) -> bool:
        return self._board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        base = self._board_url(entry)
        if base is None:
            raise FetchError(f"greenhouse: cannot resolve a board for {entry.name!r}")
        url = f"{base}{'&' if '?' in base else '?'}content=true"
        assert_allowed_url(url, _ALLOWED_HOSTS, self.name)

        payload = ctx.http.get_json(url)  # follow_redirects=False (the client default)
        raw_jobs = payload.get("jobs") if isinstance(payload, dict) else None
        if not isinstance(raw_jobs, list):
            return []

        jobs: list[Job] = []
        for j in raw_jobs:
            if not isinstance(j, dict):
                continue
            job_id, apply_url = j.get("id"), j.get("absolute_url")
            title = clean(j.get("title"))
            if job_id is None or not apply_url or not title:
                log.debug("greenhouse: skipping malformed posting %r", j.get("id"))
                continue
            location = clean((j.get("location") or {}).get("name"))
            try:
                jobs.append(
                    Job(
                        job_id=f"greenhouse:{job_id}",
                        url=apply_url,
                        title=title,
                        company=entry.name,
                        source=self.name,
                        description=html_to_text(j.get("content")) or None,
                        location=location,
                        date_posted=iso_date(j.get("first_published") or j.get("updated_at")),
                        is_remote=infer_remote(f"{location or ''} {title}"),
                    )
                )
            except (ValueError, TypeError) as err:
                log.warning("greenhouse: skipping posting %s -- %s", job_id, err)
        return jobs
