"""Lever provider -- public v0 postings JSON, no key.

Ported from career-ops ``providers/lever.mjs``. The list endpoint ships the full
``descriptionPlain`` for free, so there are no per-job fetches. The response is a
bare JSON array (not ``{jobs: [...]}``).
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlsplit

from hiredra.sourcing.http import FetchError
from hiredra.sourcing.job import Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext
from hiredra.sourcing.providers._common import (
    assert_allowed_url,
    clean,
    infer_remote,
    iso_date,
)

log = logging.getLogger("hiredra.sourcing.lever")

_ALLOWED_HOSTS = frozenset({"api.lever.co", "api.eu.lever.co"})
_CAREERS_RE = re.compile(r"^jobs\.((?:eu\.)?lever\.co)$", re.I)
_API = "https://{host}/v0/postings/{slug}"


def _fold_locations(categories: dict) -> str | None:
    """``categories.location`` + ``categories.allLocations[]``, deduped, ``"; "``."""
    candidates = [categories.get("location")]
    extra = categories.get("allLocations")
    if isinstance(extra, list):
        candidates.extend(extra)
    parts: list[str] = []
    for value in candidates:
        c = clean(value)
        if c and c.lower() not in {p.lower() for p in parts}:
            parts.append(c)
    return "; ".join(parts) or None


class LeverProvider:
    name = "lever"

    def board_url(self, entry: CompanyEntry) -> str | None:
        if entry.provider and entry.provider != self.name:
            return None
        if entry.api:
            try:
                assert_allowed_url(entry.api, _ALLOWED_HOSTS, self.name)
            except FetchError:
                return None
            return entry.api
        slug = entry.extra.get("slug") if entry.extra else None
        host = "api.lever.co"
        if entry.careers_url:
            parts = urlsplit(entry.careers_url)
            m = _CAREERS_RE.match(parts.hostname or "")
            if m:
                host = "api." + m.group(1).lower()
                segments = [s for s in parts.path.split("/") if s]
                slug = slug or (segments[0] if segments else None)
        return _API.format(host=host, slug=slug) if slug else None

    def detect(self, entry: CompanyEntry) -> bool:
        return self.board_url(entry) is not None

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        base = self.board_url(entry)
        if base is None:
            raise FetchError(f"lever: cannot resolve a board for {entry.name!r}")
        assert_allowed_url(base, _ALLOWED_HOSTS, self.name)

        payload = ctx.http.get_json(base)  # a bare JSON array
        if not isinstance(payload, list):
            return []

        jobs: list[Job] = []
        for j in payload:
            if not isinstance(j, dict):
                continue
            job_id = j.get("id")
            listing_url = j.get("hostedUrl")
            title = clean(j.get("text"))
            if not job_id or not listing_url or not title:
                log.debug("lever: skipping malformed posting %r", job_id)
                continue
            apply_raw = j.get("applyUrl")
            apply_url = apply_raw if apply_raw and apply_raw != listing_url else None
            categories = j.get("categories")
            categories = categories if isinstance(categories, dict) else {}
            location = _fold_locations(categories)
            try:
                jobs.append(
                    Job(
                        job_id=f"lever:{job_id}",
                        url=listing_url,
                        title=title,
                        company=entry.name,
                        source=self.name,
                        description=clean(j.get("descriptionPlain")),
                        location=location,
                        date_posted=iso_date(j.get("createdAt")),
                        job_type=clean(categories.get("commitment")),
                        apply_url=apply_url,
                        is_remote=infer_remote(
                            f"{location or ''} {title}",
                            workplace_type=j.get("workplaceType"),
                        ),
                    )
                )
            except (ValueError, TypeError) as err:
                log.warning("lever: skipping posting %s -- %s", job_id, err)
        return jobs
