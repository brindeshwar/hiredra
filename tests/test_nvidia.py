"""Tests for hiredra.nvidia.NvidiaChain and the extraction fallback to it. No network: a mock
HTTP transport stands in for NVIDIA, and conftest blanks the real key for every test."""

from __future__ import annotations

import json
import random

import httpx
import pytest
from pydantic import ValidationError

from hiredra import extraction
from hiredra.nvidia import NvidiaChain
from hiredra.quota import HTTPProviderRefused, HTTPProviderTransient, classify_failure, jittered_wait
from hiredra.storage import connect, get_job, init_db, set_status, upsert_job

MODEL = "nvidia/test-model"
GOOD = {
    "skills": [{"name": "python", "requirement": "required"}],
    "domain_bucket": "backend",
    "legitimacy": "legitimate",
    "extraction_confidence": "high",
}


def _chain(handler) -> tuple[NvidiaChain, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(record), headers={"Authorization": "Bearer test-key"})
    return NvidiaChain(MODEL, extraction.ExtractionResult, client=client), seen


def _answer(content: str, status: int = 200, headers: dict | None = None):
    def handler(request):
        if status != 200:
            return httpx.Response(status, text="nope", headers=headers or {})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return handler


# ── the request ─────────────────────────────────────────────────────────────
def test_request_forces_the_schema_turns_thinking_off_and_names_the_model():
    chain, seen = _chain(_answer(json.dumps(GOOD)))
    chain.invoke("read this")
    body = json.loads(seen[0].content)
    assert seen[0].url == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert body["model"] == MODEL and body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == extraction.ExtractionResult.model_json_schema()
    assert body["messages"] == [{"role": "user", "content": "read this"}]


# ── answers ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "content",
    [
        json.dumps(GOOD),
        "<think>let me see</think>" + json.dumps(GOOD),
        "Here you go:\n```json\n" + json.dumps(GOOD) + "\n```",
    ],
    ids=["plain", "think-tag", "fenced"],
)
def test_a_good_answer_parses(content):
    chain, _ = _chain(_answer(content))
    result = chain.invoke("x")
    assert result.domain_bucket == "backend" and result.skills[0].name == "python"


def test_an_empty_answer_is_none():
    chain, _ = _chain(_answer("<think>only thoughts</think>"))
    assert chain.invoke("x") is None


def test_a_wrong_shape_is_a_validation_error():
    chain, _ = _chain(_answer(json.dumps({"skills": [], "domain_bucket": "frontend"})))
    with pytest.raises(ValidationError):
        chain.invoke("x")


# ── failures map onto the shared classes ───────────────────────────────────
@pytest.mark.parametrize("status", [429, 500, 503])
def test_429_and_5xx_are_transient_and_carry_retry_after(status):
    chain, _ = _chain(_answer("", status=status, headers={"retry-after": "7"}))
    with pytest.raises(HTTPProviderTransient) as info:
        chain.invoke("x")
    assert classify_failure(info.value) == "transient" and info.value.retry_after == 7.0


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_other_4xx_are_api_errors(status):
    chain, _ = _chain(_answer("", status=status))
    with pytest.raises(HTTPProviderRefused) as info:
        chain.invoke("x")
    assert classify_failure(info.value) == "api_error"


def test_a_timeout_is_transient_and_never_shows_the_key():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    chain, _ = _chain(handler)
    with pytest.raises(HTTPProviderTransient) as info:
        chain.invoke("x")
    assert classify_failure(info.value) == "transient" and "test-key" not in str(info.value)


def test_no_key_means_no_chain():
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY is not set"):
        NvidiaChain(MODEL, extraction.ExtractionResult)  # conftest blanks the real key


# ── the owner's retry rule ──────────────────────────────────────────────────
class _State:
    def __init__(self, attempt, exc):
        self.attempt_number = attempt
        self.outcome = type("O", (), {"exception": lambda self_: exc})()


@pytest.mark.parametrize(("attempt", "base"), [(1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0)])
def test_jittered_wait_doubles_with_jitter(attempt, base):
    exc = HTTPProviderTransient("x")
    assert jittered_wait(_State(attempt, exc), rng=lambda: 0.0) == base
    assert jittered_wait(_State(attempt, exc), rng=lambda: 0.999) == pytest.approx(base * 1.999)
    waits = {jittered_wait(_State(attempt, exc), rng=random.Random(seed).random) for seed in range(20)}
    assert len(waits) > 1 and all(base <= w < 2 * base for w in waits)  # jitter spreads retries apart


def test_jittered_wait_honours_the_servers_delay():
    exc = HTTPProviderTransient("x", retry_after=30)
    assert jittered_wait(_State(1, exc), rng=lambda: 0.5) == 31.5
    assert jittered_wait(_State(1, HTTPProviderTransient("x", retry_after=500)), rng=lambda: 0.0) == 120.0


# ── the extraction fallback ─────────────────────────────────────────────────
class _FakeNvidia:
    instances: list = []

    def __init__(self, model, schema, **kw):
        self.model, self.calls = model, 0
        _FakeNvidia.instances.append(self)

    def invoke(self, prompt):
        self.calls += 1
        return extraction.ExtractionResult(**GOOD)


def _db(tmp_path, n=3):
    path = tmp_path / "h.db"
    conn = connect(path)
    init_db(conn)
    for i in range(n):
        upsert_job(conn, {"job_id": f"j{i}", "url": f"u://{i}", "title": "Engineer", "company": "Acme",
                          "source": "greenhouse", "description": "We write Python services daily. " * 10})  # fmt: skip
        set_status(conn, f"j{i}", "filtered")
    conn.close()
    return path


def _args(tmp_path, path, *extra):
    return [
        "--db",
        str(path),
        "--usage-path",
        str(tmp_path / "gem.json"),
        "--lock-path",
        str(tmp_path / "llm.lock"),
        *extra,
    ]


@pytest.fixture()
def fakes(monkeypatch):
    _FakeNvidia.instances = []
    monkeypatch.setattr(extraction, "NvidiaChain", _FakeNvidia)
    monkeypatch.setattr(extraction._Pacer, "wait", lambda self: None)


class _GeminiThenSpent:
    """Answers once, then our own daily budget is reached."""

    def __init__(self, usage_budget_hit_after=1):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return extraction.ExtractionResult(**GOOD)


def test_fallback_takes_over_when_gemini_budget_is_spent(tmp_path, monkeypatch, fakes, capsys):
    path = _db(tmp_path, n=3)
    from hiredra.quota import DailyUsage

    gemini = DailyUsage(1, tmp_path / "gem.json")  # budget of ONE call
    monkeypatch.setattr(extraction, "build_extraction_chain", lambda **kw: _GeminiThenSpent())
    monkeypatch.setattr(extraction.ExtractionConfig, "daily_budget", property(lambda self: 1))

    assert extraction.main(_args(tmp_path, path)) == 0
    out = capsys.readouterr().out
    assert "gemini STOPPED (daily_quota)" in out and "nvidia: seen=2 extracted=2" in out
    conn = connect(path)
    stamps = [get_job(conn, f"j{i}")["extraction_version"] for i in range(3)]
    conn.close()
    cfg = extraction.load_extraction_config()
    assert stamps[0] == extraction.extraction_version(cfg)
    assert stamps[1] == stamps[2] == extraction.extraction_version(cfg, cfg.fallback_model)
    assert gemini.used() == 1


def test_provider_nvidia_skips_gemini(tmp_path, monkeypatch, fakes, capsys):
    path = _db(tmp_path, n=2)
    monkeypatch.setattr(extraction, "build_extraction_chain", lambda **kw: pytest.fail("gemini used"))
    assert extraction.main(_args(tmp_path, path, "--provider", "nvidia")) == 0
    assert "nvidia: seen=2 extracted=2" in capsys.readouterr().out
    assert (tmp_path / "nvidia_nemotron-3.5-super-vl-120b-a12b.json").exists()  # its own visible counter


def test_a_non_quota_stop_does_not_hand_over(tmp_path, monkeypatch, fakes, capsys):
    path = _db(tmp_path, n=2)
    from google.genai.errors import ClientError

    class Refused:
        def invoke(self, prompt):
            raise ClientError(403, {"error": {"message": "key not valid"}})

    monkeypatch.setattr(extraction, "build_extraction_chain", lambda **kw: Refused())
    assert extraction.main(_args(tmp_path, path)) == 2
    out = capsys.readouterr().out
    assert "gemini STOPPED (api_error)" in out and "nvidia:" not in out and _FakeNvidia.instances == []


def test_without_a_key_the_fallback_is_skipped_cleanly(tmp_path, monkeypatch, capsys):
    path = _db(tmp_path, n=1)
    monkeypatch.setattr(extraction, "build_extraction_chain", lambda **kw: pytest.fail("gemini used"))
    assert extraction.main(_args(tmp_path, path, "--provider", "nvidia")) == 0
    assert "nvidia NOT RUN: NVIDIA_API_KEY is not set" in capsys.readouterr().out


def test_stale_counts_both_providers_stamps_as_current(tmp_path):
    cfg = extraction.load_extraction_config()
    conn = connect(":memory:")
    init_db(conn)
    for i, stamp in enumerate(
        [extraction.extraction_version(cfg), extraction.extraction_version(cfg, cfg.fallback_model), "old:stamp"]
    ):
        upsert_job(conn, {"job_id": f"s{i}", "url": f"u://s{i}", "title": "E", "company": "A", "source": "greenhouse"})
        set_status(conn, f"s{i}", "filtered")
        set_status(conn, f"s{i}", "extracted")
        conn.execute("UPDATE jobs SET extraction_version = ? WHERE job_id = ?", (stamp, f"s{i}"))
    conn.commit()
    assert list(extraction.stale_jobs(conn, cfg)) == ["old:stamp"]


def test_config_rejects_an_unknown_fallback(tmp_path):
    p = tmp_path / "e.yml"
    p.write_text("fallback:\n  provider: groq\n  model: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fallback.provider must be nvidia or null"):
        extraction.load_extraction_config(p)
    p.write_text("fallback:\n  provider: nvidia\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fallback.model is required"):
        extraction.load_extraction_config(p)
