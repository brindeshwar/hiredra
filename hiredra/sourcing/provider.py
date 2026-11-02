"""The provider contract (docs/sourcing.md sec. 3).

Each ATS provider implements the ``Provider`` protocol below (ported from
career-ops' ``{id, detect(), fetch()}`` shape) and is routed by the registry.

The JobSpy source is deliberately NOT a ``Provider``: it is company-agnostic
(a search, not a per-company board), takes a search config rather than a
``CompanyEntry``, and is not in the registry -- see
``hiredra/sourcing/jobspy_source.py`` and docs/sourcing.md sec. 7.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

from hiredra.sourcing.http import HttpClient
from hiredra.sourcing.job import Job


@dataclass(frozen=True, slots=True)
class CompanyEntry:
    """One company to fetch from -- what the registry routes (sub-slice 5).

    ``api`` (an explicit API URL) wins over ``careers_url``. ``provider`` (an
    explicit provider name) skips ``detect()`` entirely. ``extra`` carries
    source-specific fields a provider needs (e.g. a Workday tenant + site id);
    keep provider-agnostic code out of it.
    """

    name: str
    careers_url: str | None = None
    api: str | None = None
    provider: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FetchContext:
    """Shared context handed to every ``Provider.fetch`` call.

    Carries the http client (retry / backoff / cooldown / proxy) and ``seen``: is a ``job_id`` already stored? A
    hydrating provider (one detail request per posting, D3.4) fetches details only for postings ``seen`` says are
    new; the CLI answers it from the database, so the orchestrator stays DB-free. Default: nothing is stored.

    ``incomplete(reason)``: a provider calls it when this fetch did not see the board's whole listing (a search cap
    reached, a walk short of its stated count), so the crawl closes nothing (D5.3). The orchestrator gives each board
    its own; a provider whose listing is a window by design says so once instead (``full_listing = False``).
    """

    http: HttpClient
    seen: Callable[[str], bool] = field(default=lambda job_id: False)
    incomplete: Callable[[str], None] = field(default=lambda reason: None)


@runtime_checkable
class Provider(Protocol):
    """A fetch source. Implementations are plain objects, one per board/path.

    ``name`` fills ``Job.source``. ``fetch`` returns a uniform ``list[Job]`` so
    everything downstream is path-agnostic; one company's board is small, so a
    provider returns a list and the orchestrator handles volume across providers.
    """

    name: str

    def detect(self, entry: CompanyEntry) -> bool:
        """Does THIS provider handle ``entry``? (cheap, no network)"""
        ...

    def board_url(self, entry: CompanyEntry) -> str | None:
        """The board's API URL for ``entry``, or ``None`` if it can't be resolved (no network).

        ``detect`` and ``fetch`` use it, and so does anything that must find the same board
        again later from stored facts (liveness, ``company_slug``).
        """
        ...

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        """Hit the source, map raw records to ``Job`` objects, return them."""
        ...


#: The path segment that comes right before the slug in each provider's board URL:
#: greenhouse ``/v1/boards/{slug}/jobs``, lever ``/v0/postings/{slug}``,
#: ashby ``/posting-api/job-board/{slug}``.
_SLUG_MARKERS = ("boards", "postings", "job-board")


def company_slug(provider: Provider, entry: CompanyEntry) -> str | None:
    """The board slug for ``entry``: ``extra["slug"]`` when the list gives one, else read
    from ``provider.board_url(entry)``. ``None`` if neither works (the caller stores NULL)."""
    slug = entry.extra.get("slug") if entry.extra else None
    if slug:
        return str(slug)
    board = provider.board_url(entry)
    if not board:
        return None
    segments = [s for s in urlsplit(board).path.split("/") if s]
    for marker in _SLUG_MARKERS:
        if marker in segments[:-1]:
            return segments[segments.index(marker) + 1]
    return None
