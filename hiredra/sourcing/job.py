"""The ``Job`` object -- the normalization target for every fetch path.

Every sourcing path (each ATS provider, JobSpy, a future Path-4 provider) maps
its messy native shape into exactly this dataclass (docs/sourcing.md sec. 2).

A ``Job`` holds ONLY fetched facts -- nothing computed (no score / fit /
legitimacy), no extracted skills. When a source lacks a field, it stays ``None``
-- never invented.

Zero dependencies: this module imports only the stdlib.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_REQUIRED = ("url", "title", "company", "source")

#: The field names that belong to a Job, in schema order. Used by ``from_row``
#: to pick Job fields out of a wider storage row.
FIELDS: tuple[str, ...] = (
    "job_id",
    "url",
    "title",
    "company",
    "description",
    "years_required",
    "source",
    "location",
    "date_posted",
    "job_type",
    "salary",
    "apply_url",
    "is_remote",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Job:
    """A single normalized job posting.

    Required (a provider must supply these): ``url``, ``title``, ``company``,
    ``source``. Everything else is best-effort and defaults to ``None``.

    ``job_id`` falls back to ``url`` when not given -- providers pass the
    source's stable id, or leave it blank.

    ``years_required`` is ``None`` when unknown and ``0`` only for an explicitly
    zero-experience role; the two are never conflated. ``is_remote`` is a real
    ``bool`` here (``to_dict`` maps it to ``0/1/None`` for SQLite).
    """

    # required
    url: str
    title: str
    company: str
    source: str  # which path/board produced it, e.g. "greenhouse", "jobspy-linkedin"

    # stable identity (falls back to url)
    job_id: str = ""

    # core, best-effort
    description: str | None = None
    years_required: int | None = None

    # needed by downstream stages
    location: str | None = None
    date_posted: str | None = None  # ISO-8601 date string as given by the source

    # optional-if-free
    job_type: str | None = None  # fulltime / intern / contract
    salary: str | None = None  # raw salary string as given
    apply_url: str | None = None  # when different from url
    is_remote: bool | None = None

    def __post_init__(self) -> None:
        for name in _REQUIRED:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Job.{name} is required and must be a non-empty string")

        if not self.job_id:
            object.__setattr__(self, "job_id", self.url)

        yr = self.years_required
        if yr is not None and (isinstance(yr, bool) or not isinstance(yr, int)):
            raise TypeError(
                "Job.years_required must be an int or None (0 = explicitly zero-experience role; unknown = None)"
            )

        if self.is_remote is not None and not isinstance(self.is_remote, bool):
            raise TypeError("Job.is_remote must be a bool or None")

    # ─────────────────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        """Plain dict for ``hiredra.storage.upsert_job``."""
        return dataclasses.asdict(self)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Job:
        """Rebuild a ``Job`` from a storage row (e.g. ``storage.get_job``).

        Extra columns (``status``, timestamps, ``score`` ...) are ignored.
        ``date_posted`` (a ``date`` in the store) comes back as its ISO string.
        """
        data = {k: row[k] for k in FIELDS if k in row}
        if data.get("date_posted") is not None and not isinstance(data["date_posted"], str):
            data["date_posted"] = data["date_posted"].isoformat()
        return cls(**data)
