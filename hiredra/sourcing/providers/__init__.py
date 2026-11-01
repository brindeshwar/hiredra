"""ATS providers. The registry (``hiredra.sourcing.registry``) assembles these."""

from hiredra.sourcing.providers.ashby import AshbyProvider
from hiredra.sourcing.providers.greenhouse import GreenhouseProvider
from hiredra.sourcing.providers.icims import ICIMSProvider
from hiredra.sourcing.providers.keka import KekaProvider
from hiredra.sourcing.providers.lever import LeverProvider
from hiredra.sourcing.providers.oracle import OracleProvider
from hiredra.sourcing.providers.smartrecruiters import SmartRecruitersProvider
from hiredra.sourcing.providers.successfactors import SuccessFactorsProvider
from hiredra.sourcing.providers.taleobe import TaleoBEProvider
from hiredra.sourcing.providers.workable import WorkableProvider
from hiredra.sourcing.providers.workday import WorkdayProvider
from hiredra.sourcing.providers.zohorecruit import ZohoRecruitProvider

__all__ = [
    "AshbyProvider",
    "GreenhouseProvider",
    "ICIMSProvider",
    "KekaProvider",
    "LeverProvider",
    "OracleProvider",
    "SmartRecruitersProvider",
    "SuccessFactorsProvider",
    "TaleoBEProvider",
    "WorkableProvider",
    "WorkdayProvider",
    "ZohoRecruitProvider",
]
