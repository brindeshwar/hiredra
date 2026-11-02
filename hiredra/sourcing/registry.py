"""The provider registry: an explicit, ordered list + a router.

Explicit, not auto-discovered (docs/decisions.md #24). Each provider is added
here by hand when it is built. JobSpy is NOT in the registry -- it is a
company-agnostic parallel source, not a per-company board (docs/sourcing.md
sec. 7).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from hiredra.sourcing.provider import CompanyEntry, Provider
from hiredra.sourcing.providers import (
    AmazonProvider,
    AppleProvider,
    AshbyProvider,
    EightfoldProvider,
    GreenhouseProvider,
    IBMProvider,
    ICIMSProvider,
    KekaProvider,
    LeverProvider,
    OracleProvider,
    RipplingProvider,
    SmartRecruitersProvider,
    SuccessFactorsProvider,
    TaleoBEProvider,
    TeamtailorProvider,
    WorkableProvider,
    WorkdayProvider,
    ZohoRecruitProvider,
)

log = logging.getLogger("hiredra.sourcing.registry")

# detect() runs top-to-bottom, first hit wins. Host patterns don't overlap
# today, so order isn't yet significant -- but the list is explicit and ordered.
REGISTRY: tuple[Provider, ...] = (
    GreenhouseProvider(),
    LeverProvider(),
    AshbyProvider(),
    WorkdayProvider(),
    OracleProvider(),
    SmartRecruitersProvider(),
    WorkableProvider(),
    ICIMSProvider(),
    SuccessFactorsProvider(),
    TaleoBEProvider(),
    KekaProvider(),
    ZohoRecruitProvider(),
    TeamtailorProvider(),
    RipplingProvider(),
    EightfoldProvider(),
    AmazonProvider(),
    IBMProvider(),
    AppleProvider(),
)


def route(entry: CompanyEntry, registry: Sequence[Provider] = REGISTRY) -> Provider | None:
    """The provider that handles ``entry``, or ``None`` (caller logs & skips).

    1. An explicit ``entry.provider`` name resolves directly (``detect()`` is
       skipped); an unknown name yields ``None`` with a warning.
    2. Otherwise each provider's ``detect()`` runs in order; first ``True`` wins.
       A ``detect()`` that raises is logged and skipped -- routing never crashes
       (docs/sourcing.md sec. 7).
    """
    if entry.provider:
        for provider in registry:
            if provider.name == entry.provider:
                return provider
        log.warning("registry: %r names unknown provider %r", entry.name, entry.provider)
        return None

    for provider in registry:
        try:
            if provider.detect(entry):
                return provider
        except Exception:
            log.exception(
                "registry: %s.detect() raised for %r -- skipping",
                provider.name,
                entry.name,
            )
    log.debug("registry: no provider for %r", entry.name)
    return None
