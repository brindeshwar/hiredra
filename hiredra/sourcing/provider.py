"""The provider contract (docs/sourcing.md sec. 3).

Every fetch source -- each ATS provider, the JobSpy source, a future Path-4
provider -- implements the ``Provider`` protocol. Ported from career-ops'
``{id, detect(), fetch()}`` shape.

``CompanyEntry`` and ``FetchContext`` are intentionally thin here: their full
shape firms up when the first real provider (Greenhouse, sub-slice 4) and the
shared http layer (sub-slice 3) are built.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

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


class FetchContext(Protocol):
    """Shared context handed to every ``Provider.fetch`` call.

    Carries the machinery providers must not each re-implement: the http client
    (retry / backoff / cooldown / proxy) and config knobs (results-per-run,
    concurrency, delays). Fleshed out in sub-slice 3 -- for now only the http
    client is named, loosely typed.
    """

    http: Any


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
