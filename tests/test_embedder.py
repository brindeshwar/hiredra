"""Tests for hiredra.graph.embedder (build sub-slice 4.1). No key, no network."""

from __future__ import annotations

import numpy as np
import pytest
from google.genai.errors import ClientError, ServerError

from hiredra.graph.config import GraphConfig
from hiredra.graph.embedder import EmbeddingError, build_embedder, l2_normalize

DIM = 8


def _cfg(**over) -> GraphConfig:
    base = dict(
        model="gemini-embedding-2",
        dimensions=DIM,
        prefix="",
        inputs_per_request=3,
        requests_per_minute=200,  # x0.9 headroom = 180/min, so a 3-input request occupies 1s
        dedup_threshold=0.9,
        have=0.9,
        adj=0.7,
    )
    base.update(over)
    return GraphConfig(**base)


class FakeClient:
    """Stands in for GoogleGenerativeAIEmbeddings; records every request."""

    def __init__(self, dim: int = DIM, fail_with: list[BaseException] | None = None):
        self.dim = dim
        self.calls: list[list[str]] = []
        self.fail_with = list(fail_with or [])

    def embed_documents(self, texts, batch_size=100):
        self.calls.append(list(texts))
        if self.fail_with:
            raise self.fail_with.pop(0)
        out = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            out.append(list(rng.normal(size=self.dim) * 5.0))  # deliberately NOT unit length
        return out


class FakeTime:
    """Clock + sleep pair where sleeping advances the clock."""

    def __init__(self):
        self.now = 1000.0
        self.slept: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def _embedder(cfg=None, client=None, t=None):
    t = t or FakeTime()
    client = client or FakeClient()
    return build_embedder(cfg or _cfg(), client=client, sleep=t.sleep, clock=t.clock), client, t


def _wrapped(cause: BaseException) -> Exception:
    """How langchain-google-genai surfaces an SDK error: wrapped, cause chained."""
    try:
        raise RuntimeError("Error embedding content") from cause
    except RuntimeError as e:
        return e


def test_returns_one_unit_length_float32_row_per_text():
    embed, _, _ = _embedder()
    out = embed(["fastapi", "kafka", "docker", "redis", "react"])
    assert out.shape == (5, DIM)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_dot_product_of_normalized_rows_is_cosine():
    embed, _, _ = _embedder()
    a, b = embed(["fastapi", "kafka"])
    raw = FakeClient()
    ra, rb = np.array(raw.embed_documents(["fastapi", "kafka"]))
    cosine = float(ra @ rb / (np.linalg.norm(ra) * np.linalg.norm(rb)))
    assert float(a @ b) == pytest.approx(cosine, abs=1e-5)


def test_chunks_by_inputs_per_request_and_preserves_order():
    embed, client, _ = _embedder()
    texts = [f"skill-{i}" for i in range(7)]
    out = embed(texts)
    assert [len(c) for c in client.calls] == [3, 3, 1]
    assert [t for c in client.calls for t in c] == texts
    assert out.shape[0] == 7


def test_prefix_is_applied_to_every_text_sent():
    embed, client, _ = _embedder(_cfg(prefix="task: sentence similarity | query: "))
    embed(["fastapi", "kafka"])
    assert client.calls == [["task: sentence similarity | query: fastapi", "task: sentence similarity | query: kafka"]]


def test_empty_input_makes_no_request():
    embed, client, _ = _embedder()
    out = embed([])
    assert out.shape == (0, DIM)
    assert client.calls == []


def test_paces_by_items_not_requests():
    embed, _, t = _embedder()  # 180 items/min -> 3 strings occupy 1s, 1 string 1/3s
    embed([f"s{i}" for i in range(7)])  # chunks of 3, 3, 1
    # first request immediate; a 3-string request then blocks the next for 1s
    assert t.slept == pytest.approx([1.0, 1.0])
    embed(["one-more"])  # previous chunk was 1 string -> only 1/3s budget owed
    assert t.slept[-1] == pytest.approx(1 / 3)


def test_smaller_request_frees_the_budget_sooner_than_a_larger_one():
    embed, _, t = _embedder(_cfg(inputs_per_request=100))
    embed(["a", "b", "c"])  # 3 strings -> next request owes 1s
    embed(["d"])
    assert t.slept == pytest.approx([1.0])


def test_retries_a_wrapped_429_then_succeeds():
    err = _wrapped(ClientError(429, {"error": {"message": "quota"}}))
    embed, client, t = _embedder(client=FakeClient(fail_with=[err]))
    out = embed(["fastapi"])
    assert out.shape == (1, DIM)
    assert len(client.calls) == 2  # one failure + one success
    assert any(1.0 <= s < 2.0 for s in t.slept)  # first retry: 1 s + up to 1 s jitter (owner rule)


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


def test_a_429_with_retry_info_waits_the_servers_delay_not_the_short_backoff():
    err = _wrapped(ClientError(429, _RETRY_INFO_429))
    embed, client, t = _embedder(client=FakeClient(fail_with=[err]))
    embed(["fastapi"])
    assert len(client.calls) == 2
    assert any(58.0 <= s < 59.0 for s in t.slept)  # 57s hint + 1s margin + up to 1s jitter


def test_retry_after_parsing_tolerates_missing_or_malformed_info():
    from hiredra.graph.embedder import _retry_after_seconds

    assert _retry_after_seconds(_wrapped(ClientError(429, _RETRY_INFO_429))) == 57.0
    assert _retry_after_seconds(_wrapped(ClientError(429, {"error": {"message": "x"}}))) is None
    bad = {"error": {"details": [{"@type": "x.RetryInfo", "retryDelay": "soon"}]}}
    assert _retry_after_seconds(_wrapped(ClientError(429, bad))) is None
    assert _retry_after_seconds(RuntimeError("no api error at all")) is None


def test_retries_a_wrapped_5xx():
    err = _wrapped(ServerError(503, {"error": {"message": "unavailable"}}))
    embed, client, _ = _embedder(client=FakeClient(fail_with=[err]))
    embed(["fastapi"])
    assert len(client.calls) == 2


def test_gives_up_after_four_transient_failures_and_reraises():
    errs = [_wrapped(ClientError(429, {"error": {}})) for _ in range(4)]
    embed, client, _ = _embedder(client=FakeClient(fail_with=errs))
    with pytest.raises(RuntimeError, match="Error embedding content"):
        embed(["fastapi"])
    assert len(client.calls) == 4


def test_does_not_retry_a_non_transient_error():
    err = _wrapped(ClientError(400, {"error": {"message": "bad request"}}))
    embed, client, _ = _embedder(client=FakeClient(fail_with=[err]))
    with pytest.raises(RuntimeError):
        embed(["fastapi"])
    assert len(client.calls) == 1


def test_wrong_vector_count_raises():
    class Aggregating(FakeClient):
        def embed_documents(self, texts, batch_size=100):
            super().embed_documents(texts, batch_size)
            return [[1.0] * DIM]  # the raw-SDK failure mode: many in, one out

    embed, _, _ = _embedder(client=Aggregating())
    with pytest.raises(EmbeddingError, match="asked for 2 embeddings, got 1"):
        embed(["a", "b"])


def test_wrong_dimension_raises():
    embed, _, _ = _embedder(client=FakeClient(dim=DIM + 1))
    with pytest.raises(EmbeddingError, match="expected 8-dim"):
        embed(["a"])


def test_zero_vector_raises():
    class Zeros(FakeClient):
        def embed_documents(self, texts, batch_size=100):
            return [[0.0] * DIM for _ in texts]

    embed, _, _ = _embedder(client=Zeros())
    with pytest.raises(EmbeddingError, match="zero or non-finite"):
        embed(["a"])


def test_l2_normalize_rejects_nan():
    with pytest.raises(EmbeddingError):
        l2_normalize(np.array([[1.0, float("nan")]]))


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("hiredra.graph.embedder.load_dotenv", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        build_embedder(_cfg())


# ── daily quota (build 4.3) ──────────────────────────────────────────────────
def _quota_error(quota_id: str, retry_delay: str | None = "57s") -> dict:
    details = [
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [
                {
                    "quotaMetric": "generativelanguage.googleapis.com/embed_content_free_tier_requests",
                    "quotaId": quota_id,
                    "quotaValue": "100",
                }
            ],
        }
    ]
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    return {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": details}}


_PER_MINUTE = "EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier"
_PER_DAY = "EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier"


def test_a_per_minute_quota_429_is_still_retried():
    err = _wrapped(ClientError(429, _quota_error(_PER_MINUTE)))
    embed, client, _ = _embedder(client=FakeClient(fail_with=[err]))
    embed(["fastapi"])
    assert len(client.calls) == 2


def test_a_per_day_quota_429_is_raised_at_once_and_never_retried():
    from hiredra.graph.embedder import DailyQuotaExceeded

    err = _wrapped(ClientError(429, _quota_error(_PER_DAY, retry_delay="41000s")))
    embed, client, t = _embedder(client=FakeClient(fail_with=[err]))
    with pytest.raises(DailyQuotaExceeded, match="PerDay"):
        embed(["fastapi"])
    assert len(client.calls) == 1  # no retry burned on a cap that will not clear
    assert t.slept == []  # ...and no waiting


def test_a_per_day_id_is_recognised_even_with_no_retry_delay():
    from hiredra.graph.embedder import DailyQuotaExceeded

    err = _wrapped(ClientError(429, _quota_error(_PER_DAY, retry_delay=None)))
    embed, client, _ = _embedder(client=FakeClient(fail_with=[err]))
    with pytest.raises(DailyQuotaExceeded):
        embed(["fastapi"])
    assert len(client.calls) == 1


def test_an_unrecognised_429_with_a_very_long_retry_delay_is_treated_as_a_daily_cap():
    from hiredra.graph.embedder import DailyQuotaExceeded

    err = _wrapped(ClientError(429, _quota_error("SomeNewQuotaName", retry_delay="3600s")))
    embed, client, _ = _embedder(client=FakeClient(fail_with=[err]))
    with pytest.raises(DailyQuotaExceeded):
        embed(["fastapi"])
    assert len(client.calls) == 1


def test_daily_quota_is_an_embedding_error_so_generic_handlers_still_catch_it():
    from hiredra.graph.embedder import DailyQuotaExceeded

    assert issubclass(DailyQuotaExceeded, EmbeddingError)


def test_only_429s_can_be_a_daily_quota():
    from hiredra.graph.embedder import _is_daily_quota

    assert not _is_daily_quota(_wrapped(ServerError(503, _quota_error(_PER_DAY))))
    assert not _is_daily_quota(RuntimeError("no api error at all"))


def test_the_daily_quota_error_records_the_limit_value_not_just_the_quota_id():
    from hiredra.graph.embedder import DailyQuotaExceeded

    err = _wrapped(ClientError(429, _quota_error(_PER_DAY, retry_delay="41000s")))
    embed, _, _ = _embedder(client=FakeClient(fail_with=[err]))
    with pytest.raises(DailyQuotaExceeded) as info:
        embed(["fastapi"])
    assert "PerDay" in str(info.value) and "limit 100" in str(info.value)  # helper's quotaValue is "100"


def test_quota_values_tolerates_errors_without_details():
    from hiredra.graph.embedder import _quota_values

    assert _quota_values(_wrapped(ClientError(429, {"error": {"message": "x"}}))) == []
    assert _quota_values(RuntimeError("no api error")) == []


# ── requests/min + tokens/min pacing and the daily counter (build 4.3 fix) ───
from datetime import datetime, timezone  # noqa: E402

from hiredra.graph.embedder import estimate_tokens  # noqa: E402
from hiredra.graph.usage import DailyBudgetReached, DailyUsage  # noqa: E402


def _usage(tmp_path, budget=900):
    moment = datetime(2026, 9, 21, 14, 0, tzinfo=timezone.utc)
    return DailyUsage(budget, tmp_path / "usage.json", now=lambda: moment)


def test_estimate_tokens_is_pessimistic_and_never_zero():
    assert estimate_tokens("") == 1 and estimate_tokens("a") == 1
    assert estimate_tokens("abc") == 1 and estimate_tokens("abcd") == 2
    assert estimate_tokens("x" * 300) == 100  # ~3 chars per token


def test_one_http_call_per_chunk_not_one_per_skill():
    """The real batching guarantee at our layer: N skills -> ceil(N / inputs_per_request) calls."""
    embed, client, _ = _embedder(_cfg(inputs_per_request=50, requests_per_minute=1000))
    embed([f"skill-{i}" for i in range(120)])
    assert [len(c) for c in client.calls] == [50, 50, 20]


def test_tokens_per_minute_can_be_the_binding_limit():
    # 3 inputs of 300 chars = ~300 tokens/chunk. tpm budget 540 -> the chunk occupies 33.3s,
    # far more than its requests share (1s), so the token limit sets the pace.
    embed, _, t = _embedder(_cfg(tokens_per_minute=600))
    embed(["x" * 300] * 6)
    assert t.slept == pytest.approx([60.0 * 300 / 540])


def test_requests_per_minute_is_the_binding_limit_for_short_names():
    embed, _, t = _embedder()  # default tokens/min is huge relative to tiny skill names
    embed([f"s{i}" for i in range(6)])
    assert t.slept == pytest.approx([1.0])  # 3 requests at 180/min = 1s


def test_a_chunk_bigger_than_the_whole_token_budget_is_refused_before_any_call():
    embed, client, _ = _embedder(_cfg(tokens_per_minute=100))  # budget 90 tokens
    with pytest.raises(EmbeddingError, match="above the whole per-minute token budget"):
        embed(["x" * 300])
    assert client.calls == []


def test_usage_counts_each_input_as_one_request_after_google_accepts(tmp_path):
    usage = _usage(tmp_path)
    embed, client, _ = _embedder()
    embed = build_embedder(_cfg(), client=client, usage=usage, sleep=lambda s: None, clock=lambda: 0.0)
    embed([f"s{i}" for i in range(7)])  # chunks of 3, 3, 1
    assert usage.used() == 7 and len(client.calls) == 3


def test_the_budget_stops_the_run_before_the_call_that_would_pass_it(tmp_path):
    usage = _usage(tmp_path, budget=5)
    client = FakeClient()
    embed = build_embedder(_cfg(), client=client, usage=usage, sleep=lambda s: None, clock=lambda: 0.0)
    with pytest.raises(DailyBudgetReached):
        embed([f"s{i}" for i in range(7)])  # chunk 1 (3) fits; chunk 2 (3 more = 6 > 5) must not be sent
    assert len(client.calls) == 1  # the second request was never made
    assert usage.used() == 3


def test_every_attempt_of_a_retried_chunk_is_counted(tmp_path):
    usage = _usage(tmp_path)
    err = _wrapped(ClientError(429, {"error": {"message": "per-minute"}}))
    client = FakeClient(fail_with=[err])
    embed = build_embedder(_cfg(), client=client, usage=usage, sleep=lambda s: None, clock=lambda: 0.0)
    embed(["a", "b"])
    # 2 inputs x 2 attempts: the rejected attempt counts too (Google may count it; ahead is safe)
    assert len(client.calls) == 2 and usage.used() == 4


def test_a_failed_request_is_still_counted(tmp_path):
    usage = _usage(tmp_path)
    err = _wrapped(ClientError(400, {"error": {"message": "bad request"}}))
    client = FakeClient(fail_with=[err])
    embed = build_embedder(_cfg(), client=client, usage=usage, sleep=lambda s: None, clock=lambda: 0.0)
    with pytest.raises(RuntimeError):
        embed(["a"])
    assert usage.used() == 1


def test_a_server_side_daily_quota_marks_the_day_exhausted_and_later_calls_make_no_request(tmp_path):
    from hiredra.graph.embedder import DailyQuotaExceeded

    usage = _usage(tmp_path)
    err = _wrapped(ClientError(429, _quota_error(_PER_DAY, retry_delay="41000s")))
    client = FakeClient(fail_with=[err])
    embed = build_embedder(_cfg(), client=client, usage=usage, sleep=lambda s: None, clock=lambda: 0.0)
    with pytest.raises(DailyQuotaExceeded):
        embed(["a"])
    assert usage.exhausted()
    calls_before = len(client.calls)
    with pytest.raises(DailyQuotaExceeded, match="already reported exhausted"):
        embed(["b"])  # refused locally...
    assert len(client.calls) == calls_before  # ...with NO request to Google


def test_without_a_usage_counter_nothing_is_tracked(tmp_path):
    embed, client, _ = _embedder()
    embed(["a", "b", "c", "d"])
    assert not list(tmp_path.iterdir())  # no stray state files


# ── failure classification: expected stop vs bug (build 4.3 hardening) ───────
import httpx  # noqa: E402

from hiredra.graph.embedder import DailyQuotaExceeded as _DQE  # noqa: E402
from hiredra.graph.embedder import classify_failure  # noqa: E402


@pytest.mark.parametrize(
    "exc,expected",
    [
        (_DQE("daily quota"), "daily_quota"),
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
        (EmbeddingError("expected 768-dim vectors, got 3"), "bad_response"),
        (KeyError("bug"), None),
        (TypeError("bug"), None),
        (RuntimeError("plain runtime error"), None),
        (ValueError("bug"), None),
    ],
)
def test_classify_failure(exc, expected):
    assert classify_failure(exc) == expected


def test_a_network_error_is_retried_like_a_5xx_and_then_succeeds():
    err = _wrapped(httpx.ConnectError("connection refused"))
    embed, client, t = _embedder(client=FakeClient(fail_with=[err]))
    out = embed(["fastapi"])
    assert out.shape == (1, DIM) and len(client.calls) == 2 and any(1.0 <= s < 2.0 for s in t.slept)


def test_a_network_error_that_never_clears_gives_up_after_four_attempts():
    errs = [_wrapped(httpx.ConnectError("down")) for _ in range(4)]
    embed, client, _ = _embedder(client=FakeClient(fail_with=errs))
    with pytest.raises(RuntimeError, match="Error embedding content"):
        embed(["fastapi"])
    assert len(client.calls) == 4


def test_a_bug_shaped_error_is_never_retried():
    embed, client, _ = _embedder(client=FakeClient(fail_with=[KeyError("bug")]))
    with pytest.raises(KeyError):
        embed(["fastapi"])
    assert len(client.calls) == 1


# --- a "PerDay" label is a claim, not a fact (lesson from spoin_bundle's gemini-saga) ----
@pytest.mark.parametrize(
    ("retry_delay", "daily"),
    [("57s", False), ("20s", False), ("599s", False), ("600s", True), ("41000s", True), (None, True)],
)
def test_a_per_day_label_needs_a_long_or_missing_retry_delay(retry_delay, daily):
    from hiredra.graph.embedder import _is_daily_quota

    err = _wrapped(ClientError(429, _quota_error(_PER_DAY, retry_delay=retry_delay)))
    assert _is_daily_quota(err) is daily


def test_a_mislabelled_short_throttle_is_retried_not_treated_as_the_day_gone(tmp_path):
    usage = _usage(tmp_path)
    err = _wrapped(ClientError(429, _quota_error(_PER_DAY, retry_delay="20s")))
    client = FakeClient(fail_with=[err])
    embed = build_embedder(_cfg(), client=client, usage=usage, sleep=lambda s: None, clock=lambda: 0.0)
    embed(["a"])
    assert len(client.calls) == 2 and not usage.exhausted()
