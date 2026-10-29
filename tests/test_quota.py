"""Tests for the failure classes in hiredra.quota (plan unit M0.1): an expected stop vs a bug, and when a Google
429 really means the day's quota is gone. These ran through the Gemini embedder until it was removed (after DB4);
extraction and the skill-description writer still use them. No key, no network."""

from __future__ import annotations

import httpx
import pytest
from google.genai.errors import ClientError, ServerError

from hiredra.quota import (
    DailyBudgetReached,
    DailyQuotaExceeded,
    ProviderError,
    _is_daily_quota,
    _quota_values,
    _retry_after_seconds,
    classify_failure,
)


def _wrapped(cause: BaseException) -> Exception:
    """How langchain-google-genai surfaces an SDK error: wrapped, cause chained."""
    try:
        raise RuntimeError("Error calling the model") from cause
    except RuntimeError as e:
        return e


def _quota_error(quota_id: str, retry_delay: str | None = "57s") -> dict:
    details = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {
                    "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                    "quotaId": quota_id,
                    "quotaValue": "100",
                }
            ],
        }
    ]
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    return {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": details}}


_PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
_PER_DAY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
_RETRY_INFO_429 = {
    "error": {
        "code": 429,
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.Help", "links": []},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "57s"},
        ],
    }
}


@pytest.mark.parametrize(
    "exc,expected",
    [
        (DailyQuotaExceeded("daily quota"), "daily_quota"),
        (DailyBudgetReached("local budget"), "daily_quota"),
        (_wrapped(ClientError(429, _quota_error(_PER_DAY, retry_delay="41000s"))), "daily_quota"),
        (_wrapped(ClientError(429, _quota_error(_PER_MINUTE))), "transient"),
        (_wrapped(ServerError(503, {"error": {"message": "unavailable"}})), "transient"),
        (_wrapped(httpx.ConnectError("connection refused")), "transient"),
        (_wrapped(httpx.ReadTimeout("timed out")), "transient"),
        (_wrapped(ConnectionResetError("reset by peer")), "transient"),
        (_wrapped(TimeoutError("timed out")), "transient"),
        (_wrapped(ClientError(400, {"error": {"message": "bad request"}})), "api_error"),
        (_wrapped(ClientError(403, {"error": {"message": "API key not valid"}})), "api_error"),
        (ProviderError("expected 768-dim vectors, got 3"), "bad_response"),
        (KeyError("bug"), None),
        (TypeError("bug"), None),
        (RuntimeError("plain runtime error"), None),
        (ValueError("bug"), None),
    ],
)
def test_classify_failure(exc, expected):
    assert classify_failure(exc) == expected


def test_a_daily_quota_is_a_provider_error_so_generic_handlers_still_catch_it():
    assert issubclass(DailyQuotaExceeded, ProviderError)


def test_only_429s_can_be_a_daily_quota():
    assert not _is_daily_quota(_wrapped(ServerError(503, _quota_error(_PER_DAY))))
    assert not _is_daily_quota(RuntimeError("no api error at all"))


# --- a "PerDay" label is a claim, not a fact (lesson from spoin_bundle's gemini-saga) ----
@pytest.mark.parametrize(
    ("retry_delay", "daily"),
    [("57s", False), ("20s", False), ("599s", False), ("600s", True), ("41000s", True), (None, True)],
)
def test_a_per_day_label_needs_a_long_or_missing_retry_delay(retry_delay, daily):
    err = _wrapped(ClientError(429, _quota_error(_PER_DAY, retry_delay=retry_delay)))
    assert _is_daily_quota(err) is daily


def test_retry_after_parsing_tolerates_missing_or_malformed_info():
    assert _retry_after_seconds(_wrapped(ClientError(429, _RETRY_INFO_429))) == 57.0
    assert _retry_after_seconds(_wrapped(ClientError(429, {"error": {"message": "x"}}))) is None
    bad = {"error": {"details": [{"@type": "x.RetryInfo", "retryDelay": "soon"}]}}
    assert _retry_after_seconds(_wrapped(ClientError(429, bad))) is None
    assert _retry_after_seconds(RuntimeError("no api error at all")) is None


def test_quota_values_read_the_limit_and_tolerate_errors_without_details():
    assert _quota_values(_wrapped(ClientError(429, _quota_error(_PER_DAY)))) == ["100"]
    assert _quota_values(_wrapped(ClientError(429, {"error": {"message": "x"}}))) == []
    assert _quota_values(RuntimeError("no api error")) == []
