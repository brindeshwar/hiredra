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
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_REQUIRED = ("url", "title", "company", "source")

WORK_MODES = frozenset({"remote", "hybrid", "onsite"})
EMPLOYMENT_TYPES = frozenset({"full_time", "part_time", "internship", "contract"})
SALARY_PERIODS = frozenset({"year", "month", "week", "day", "hour"})
_INTERN = re.compile(r"\b(?:intern(?:ship)?s?|trainee|apprentice(?:ship)?)\b")
_CONTRACT = re.compile(r"\b(?:contract(?:or|ual)?|freelance|temporary|temp|fixed term|seasonal)\b")
_FULL = re.compile(r"\bfull ?time\b")
_PART = re.compile(r"\bpart ?time\b")


def employment_type_of(job_type: str | None) -> str | None:
    """The employment type a platform's own job-type text states, or ``None`` when it states none or two. Lever's is
    free text ("Contract Full time", "Full-time Employment", "Full time or Part time", "On- roll"): an internship or
    contract word wins, a full-and-part one says nothing, and anything else is left unread rather than guessed."""
    text = re.sub(r"[-_/]+", " ", (job_type or "").lower())
    if _INTERN.search(text):
        return "internship"
    if _CONTRACT.search(text):
        return "contract"
    full, part = bool(_FULL.search(text)), bool(_PART.search(text))
    if full != part:
        return "full_time" if full else "part_time"
    return None


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
    "work_mode",
    "employment_type",
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_period",
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
    ``bool`` (or ``None``), stored as a Postgres boolean.
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

    # structured, only where the platform states them (03-adapters.md §1: "adopt freehire's richer Job model")
    work_mode: str | None = None  # remote | hybrid | onsite: the platform's own workplace field, never a text guess
    employment_type: str | None = None  # full_time | part_time | internship | contract; from job_type when not given
    salary_min: float | None = None  # as the platform states it: an Indian "8-10 per year" is 8-10 LPA
    salary_max: float | None = None
    salary_currency: str | None = None  # ISO 4217, upper-case
    salary_period: str | None = None  # year | month | week | day | hour

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

        if self.employment_type is None:
            object.__setattr__(self, "employment_type", employment_type_of(self.job_type))
        for name, allowed in (("work_mode", WORK_MODES), ("employment_type", EMPLOYMENT_TYPES),
                              ("salary_period", SALARY_PERIODS)):  # fmt: skip
            value = getattr(self, name)
            if value is not None and value not in allowed:
                raise ValueError(f"Job.{name} must be one of {sorted(allowed)} or None, not {value!r}")
        for name in ("salary_min", "salary_max"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
                raise TypeError(f"Job.{name} must be a non-negative number or None")

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
        for name in ("salary_min", "salary_max"):  # a Postgres numeric comes back as a Decimal
            if data.get(name) is not None:
                data[name] = float(data[name])
        return cls(**data)
