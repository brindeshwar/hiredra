"""The provider contract (docs/sourcing.md sec. 3).

Each ATS provider implements the ``Provider`` protocol below (ported from
career-ops' ``{id, detect(), fetch()}`` shape) and is routed by the registry.

The JobSpy source is deliberately NOT a ``Provider``: it is company-agnostic
(a search, not a per-company board), takes a search config rather than a
``CompanyEntry``, and is not in the registry -- see
``hiredra/sourcing/jobspy_source.py`` and docs/sourcing.md sec. 7.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

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

    Carries the http client (retry / backoff / cooldown / proxy). Results-per-run,
    concurrency and delay knobs are added here when a provider or the orchestrator
    needs them (sub-slice 4+).
    """

    http: HttpClient


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

    def fetch(self, entry: CompanyEntry, ctx: FetchContext) -> list[Job]:
        """Hit the source, map raw records to ``Job`` objects, return them."""
        ...
