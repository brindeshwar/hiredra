"""Helpers shared by the ATS providers.

Kept here so the third-and-later provider doesn't re-implement URL guarding,
timestamp normalization, or the remote heuristic. Providers stay tiny.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from hiredra.sourcing.http import FetchError


def assert_allowed_url(url: str, allowed_hosts: frozenset[str], provider: str) -> None:
    """HTTPS + host in ``allowed_hosts``, else raise ``FetchError`` (SSRF guard)."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise FetchError(f"{provider}: refusing non-HTTPS url {url!r}", url=url)
    if parts.hostname not in allowed_hosts:
        raise FetchError(
            f"{provider}: refusing untrusted host {parts.hostname!r} "
            f"(allowed: {', '.join(sorted(allowed_hosts))})",
            url=url,
        )


def clean(value: Any) -> str | None:
    """``value.strip()``, or ``None`` if it isn't a non-empty string."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def iso_date(value: Any) -> str | None:
    """Normalize a source timestamp to ``YYYY-MM-DD``.

    Accepts epoch-milliseconds (Lever ``createdAt``), an ISO-8601 string
    (Ashby ``publishedAt``, Greenhouse ``first_published``), or a
    ``date``/``datetime``/``pandas.Timestamp`` (JobSpy ``date_posted``).
    ``None`` for anything absent or unparseable.
    """
    if value is None or value == "" or isinstance(value, bool):
        return None
    if str(value) in ("NaT", "nan", "NaN", "None"):  # pandas null sentinels
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc).date().isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        raw = value.strip()
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            try:
                return datetime.strptime(raw[:10], "%Y-%m-%d").date().isoformat()
            except ValueError:
                return None
    return None


_REMOTE_RE = re.compile(r"\bremote\b|\banywhere\b|\bwork from home\b|\bwfh\b", re.I)
_WORKPLACE_REMOTE = frozenset({"remote"})
_WORKPLACE_NOT_REMOTE = frozenset(
    {"hybrid", "on-site", "onsite", "on site", "in-office", "in office"}
)


def infer_remote(text: str, *, workplace_type: str | None = None) -> bool | None:
    """``True`` / ``False`` only on a clear signal; ``None`` on absence.

    A structured ``workplace_type`` (Lever/Ashby) is authoritative when it is a
    recognized value -- it can yield ``True`` or ``False``, because an explicit
    "hybrid" / "on-site" is a *present* signal, not an absence. Otherwise a
    keyword match on ``text`` (location + title) yields only ``True``; a missing
    keyword stays ``None``, never ``False``.
    """
    if workplace_type:
        wt = workplace_type.strip().lower()
        if wt in _WORKPLACE_REMOTE:
            return True
        if wt in _WORKPLACE_NOT_REMOTE:
            return False
    return True if _REMOTE_RE.search(text) else None
