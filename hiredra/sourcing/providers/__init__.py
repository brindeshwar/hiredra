"""ATS providers. The registry (build sub-slice 5) assembles these into a router."""

from hiredra.sourcing.providers.greenhouse import GreenhouseProvider

__all__ = ["GreenhouseProvider"]
