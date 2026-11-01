"""ATS providers. The registry (``hiredra.sourcing.registry``) assembles these."""

from hiredra.sourcing.providers.amazon import AmazonProvider
from hiredra.sourcing.providers.ashby import AshbyProvider
from hiredra.sourcing.providers.eightfold import EightfoldProvider
from hiredra.sourcing.providers.greenhouse import GreenhouseProvider
from hiredra.sourcing.providers.ibm import IBMProvider
from hiredra.sourcing.providers.icims import ICIMSProvider
from hiredra.sourcing.providers.keka import KekaProvider
from hiredra.sourcing.providers.lever import LeverProvider
from hiredra.sourcing.providers.oracle import OracleProvider
from hiredra.sourcing.providers.rippling import RipplingProvider
from hiredra.sourcing.providers.smartrecruiters import SmartRecruitersProvider
from hiredra.sourcing.providers.successfactors import SuccessFactorsProvider
from hiredra.sourcing.providers.taleobe import TaleoBEProvider
from hiredra.sourcing.providers.teamtailor import TeamtailorProvider
from hiredra.sourcing.providers.workable import WorkableProvider
from hiredra.sourcing.providers.workday import WorkdayProvider
from hiredra.sourcing.providers.zohorecruit import ZohoRecruitProvider

__all__ = [
    "AmazonProvider",
    "AshbyProvider",
    "EightfoldProvider",
    "GreenhouseProvider",
    "IBMProvider",
    "ICIMSProvider",
    "KekaProvider",
    "LeverProvider",
    "OracleProvider",
    "RipplingProvider",
    "SmartRecruitersProvider",
    "SuccessFactorsProvider",
    "TaleoBEProvider",
    "TeamtailorProvider",
    "WorkableProvider",
    "WorkdayProvider",
    "ZohoRecruitProvider",
]
