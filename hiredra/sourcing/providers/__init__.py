"""ATS providers. The registry (``hiredra.sourcing.registry``) assembles these."""

from hiredra.sourcing.providers.ashby import AshbyProvider
from hiredra.sourcing.providers.greenhouse import GreenhouseProvider
from hiredra.sourcing.providers.lever import LeverProvider
from hiredra.sourcing.providers.workday import WorkdayProvider

__all__ = ["AshbyProvider", "GreenhouseProvider", "LeverProvider", "WorkdayProvider"]
