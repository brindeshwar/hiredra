"""Hiredra sourcing layer -- fetches and normalizes raw jobs.

Knows nothing about filtering or scoring. See ``docs/sourcing.md``.
"""

from hiredra.sourcing.http import (
    Blocked,
    FetchError,
    HTTPError,
    HttpClient,
    HttpConfig,
    TransientError,
)
from hiredra.sourcing.job import FIELDS, Job
from hiredra.sourcing.provider import CompanyEntry, FetchContext, Provider

__all__ = [
    # models
    "FIELDS",
    "Job",
    "CompanyEntry",
    "FetchContext",
    "Provider",
    # http layer
    "HttpClient",
    "HttpConfig",
    "FetchError",
    "HTTPError",
    "TransientError",
    "Blocked",
]
