"""Tests for the shared http layer (build sub-slice 2a.3).

All requests go through ``httpx.MockTransport``; the clock and the retry-sleep
are injected, so nothing here actually waits.
"""

from __future__ import annotations

import httpx
import pytest

from hiredra.sourcing.http import (
    BROWSER_UA,
    Blocked,
    CooldownRegistry,
    FetchError,
    HTTPError,
    HttpClient,
    HttpConfig,
    TransientError,
)


class Clock:
    """A hand-cranked monotonic clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_client(handler, clock: Clock | None = None, **cfg):
    """HttpClient wired to a MockTransport handler, a fake clock, and a no-op sleep."""
    clock = clock or Clock()
    slept: list[float] = []
    client = HttpClient(
        HttpConfig(**cfg),
        transport=httpx.MockTransport(handler),
        sleep=slept.append,
        now=clock,
    )
    return client, clock, slept


# ── who we say we are (D4.2) ───────────────────────────────────────────────
def test_we_identify_ourselves_except_to_a_host_that_refused_our_name():
    seen = {}

    def handler(req):
        seen[req.url.host] = req.headers["user-agent"]
        return httpx.Response(200, text="ok")

    client, _, _ = make_client(handler, browser_ua_hosts=frozenset({"refuses.example"}))
    for url in ("https://api.example.com/x", "https://jobs.refuses.example/x", "https://notrefuses.example/x"):
        client.get_text(url)
    assert seen["api.example.com"].startswith("Hiredra/") and seen["notrefuses.example"].startswith("Hiredra/")
    assert seen["jobs.refuses.example"] == BROWSER_UA


def test_we_offer_gzip_and_deflate_only():
    # httpx offers zstd when zstandard is installed, and its decoder fails on zstandard 0.25 (Amazon, 2026-11-02)
    seen = {}

    def handler(req):
        seen["accept-encoding"] = req.headers["accept-encoding"]
        return httpx.Response(200, text="ok")

    client, _, _ = make_client(handler)
    client.get_text("https://api.example.com/x")
    assert seen["accept-encoding"] == "gzip, deflate"


# ── happy paths ────────────────────────────────────────────────────────────
def test_get_json_ok():
    client, _, _ = make_client(lambda req: httpx.Response(200, json={"jobs": [1, 2]}))
    assert client.get_json("https://api.example.com/v1/jobs") == {"jobs": [1, 2]}


def test_get_text_ok():
    client, _, _ = make_client(lambda req: httpx.Response(200, text="hello"))
    assert client.get_text("https://api.example.com/x") == "hello"


def test_non_json_body_raises_fetcherror_from_get_json():
    client, _, _ = make_client(lambda req: httpx.Response(200, text="not json"))
    with pytest.raises(FetchError):
        client.get_json("https://api.example.com/x")


# ── retry / transient ─────────────────────────────────────────────────────
def test_transient_then_success_is_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(200, json={"ok": True}) if calls["n"] == 3 else httpx.Response(500)

    client, _, slept = make_client(handler, attempts=3)
    assert client.get_json("https://api.example.com/x") == {"ok": True}
    assert calls["n"] == 3
    assert len(slept) == 2  # backed off between the 3 attempts


def test_transient_exhausted_raises_transienterror():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(502)

    client, _, _ = make_client(handler, attempts=3)
    with pytest.raises(TransientError):
        client.get_json("https://api.example.com/x")
    assert calls["n"] == 3


def test_timeout_is_transient_and_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectTimeout("boom", request=req)
        return httpx.Response(200, json={"ok": 1})

    client, _, _ = make_client(handler, attempts=3)
    assert client.get_json("https://api.example.com/x") == {"ok": 1}
    assert calls["n"] == 2


# ── non-retryable HTTP ────────────────────────────────────────────────────
def test_404_is_httperror_and_not_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(404)

    client, _, _ = make_client(handler, attempts=3)
    with pytest.raises(HTTPError) as ei:
        client.get_json("https://api.example.com/gone")
    assert ei.value.status == 404
    assert calls["n"] == 1


def test_redirect_not_followed_by_default_is_httperror():
    def handler(req):
        return httpx.Response(302, headers={"location": "https://elsewhere.example.com/"})

    client, _, _ = make_client(handler)
    with pytest.raises(HTTPError) as ei:
        client.get_json("https://api.example.com/x")
    assert (ei.value.status, ei.value.location) == (302, "https://elsewhere.example.com/")


def test_a_relative_redirect_location_is_made_absolute():
    client, _, _ = make_client(lambda req: httpx.Response(301, headers={"location": "/acme/j/C1"}))
    with pytest.raises(HTTPError) as ei:
        client.get_text("https://apply.workable.com/j/C1")
    assert ei.value.location == "https://apply.workable.com/acme/j/C1"


def test_redirect_followed_when_provider_opts_in():
    def handler(req):
        if req.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://api.example.com/final"})
        return httpx.Response(200, json={"ok": 1})

    client, _, _ = make_client(handler)
    assert client.get_json("https://api.example.com/start", follow_redirects=True) == {"ok": 1}


# ── blocks + cooldown ─────────────────────────────────────────────────────
def test_429_blocks_and_cools_the_host():
    hits: list[str] = []

    def handler(req):
        hits.append(req.url.host)
        return httpx.Response(429) if req.url.host == "blocked.example.com" else httpx.Response(200, json={"ok": 1})

    client, _, _ = make_client(handler)

    with pytest.raises(Blocked) as ei:
        client.get_json("https://blocked.example.com/x")
    assert ei.value.cooldown_key == "blocked.example.com"

    # second request to the same host short-circuits -- transport NOT hit again
    with pytest.raises(Blocked):
        client.get_json("https://blocked.example.com/y")
    assert hits == ["blocked.example.com"]

    # a different host is unaffected
    assert client.get_json("https://ok.example.com/z") == {"ok": 1}


@pytest.mark.parametrize("status", [403, 406, 429, 503])
def test_all_block_statuses_cool_down(status):
    client, _, _ = make_client(lambda req: httpx.Response(status))
    with pytest.raises(Blocked):
        client.get_json("https://b.example.com/x")


def test_block_is_not_retried():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(429)

    client, _, _ = make_client(handler, attempts=3)
    with pytest.raises(Blocked):
        client.get_json("https://b.example.com/x")
    assert calls["n"] == 1


def test_retry_after_seconds_sets_cooldown_length():
    client, clock, _ = make_client(lambda req: httpx.Response(429, headers={"retry-after": "50"}))
    with pytest.raises(Blocked) as ei:
        client.get_json("https://b.example.com/x")
    assert ei.value.until == pytest.approx(clock.t + 50)


def test_retry_after_is_clamped_to_max_cooldown():
    client, clock, _ = make_client(
        lambda req: httpx.Response(429, headers={"retry-after": "999999"}),
        max_cooldown=3600.0,
    )
    with pytest.raises(Blocked) as ei:
        client.get_json("https://b.example.com/x")
    assert ei.value.until == pytest.approx(clock.t + 3600.0)


def test_cooldown_expires_after_clock_advances():
    state = {"blocked": True}

    def handler(req):
        return httpx.Response(429) if state["blocked"] else httpx.Response(200, json={"ok": 1})

    client, clock, _ = make_client(handler, cooldown_seconds=100.0)
    with pytest.raises(Blocked):
        client.get_json("https://b.example.com/x")

    clock.advance(50)
    with pytest.raises(Blocked):
        client.get_json("https://b.example.com/x")  # still cooling

    state["blocked"] = False
    clock.advance(60)  # now past the 100s cooldown
    assert client.get_json("https://b.example.com/x") == {"ok": 1}


def test_explicit_cooldown_key_overrides_host():
    client, _, _ = make_client(lambda req: httpx.Response(429))
    with pytest.raises(Blocked) as ei:
        client.get_json("https://linkedin.example.com/x", cooldown_key="jobspy-linkedin")
    assert ei.value.cooldown_key == "jobspy-linkedin"
    assert set(client.cooldowns.active()) == {"jobspy-linkedin"}


# ── challenge detection: must never trip on a real API response ────────────
_CHALLENGE_HTML = "<html><head><title>Just a moment...</title></head><body>/cdn-cgi/challenge-platform/</body></html>"


def test_challenge_html_body_is_blocked():
    client, _, _ = make_client(
        lambda req: httpx.Response(200, headers={"content-type": "text/html"}, text=_CHALLENGE_HTML)
    )
    with pytest.raises(Blocked):
        client.get_text("https://b.example.com/x")


def test_challenge_not_detected_when_sniffing_disabled():
    client, _, _ = make_client(
        lambda req: httpx.Response(200, headers={"content-type": "text/html"}, text=_CHALLENGE_HTML),
        sniff_challenges=False,
    )
    assert "Just a moment" in client.get_text("https://b.example.com/x")


def test_json_response_is_never_sniffed_even_if_it_mentions_a_challenge():
    payload = {
        "title": "Just a moment... before you apply",
        "description": "checking your browser before accessing our benefits page",
        "note": "/cdn-cgi/challenge-platform/ is a great band name",
    }
    client, _, _ = make_client(lambda req: httpx.Response(200, json=payload))
    assert client.get_json("https://api.greenhouse.io/v1/boards/acme/jobs") == payload


def test_plain_json_string_body_with_wrong_content_type_still_safe():
    # some ATS return JSON as text/plain; a JSON array can't match a challenge regex
    client, _, _ = make_client(
        lambda req: httpx.Response(200, headers={"content-type": "text/plain"}, text='[{"title":"Engineer"}]')
    )
    assert client.get_json("https://api.example.com/x") == [{"title": "Engineer"}]


# ── misc ──────────────────────────────────────────────────────────────────
def test_context_manager_closes_client():
    with make_client(lambda req: httpx.Response(200, json={}))[0] as client:
        client.get_json("https://api.example.com/x")
    assert client._client.is_closed


def test_proxy_config_constructs_without_error():
    client = HttpClient(HttpConfig(proxy="http://127.0.0.1:9"))
    try:
        assert client.config.proxy == "http://127.0.0.1:9"
    finally:
        client.close()


def test_cooldown_registry_extends_never_shortens():
    clock = Clock()
    reg = CooldownRegistry(now=clock)
    long_until = reg.trip("k", 100)
    short_until = reg.trip("k", 10)  # must not shorten
    assert short_until == long_until


# ── SSRF guard (D4.4; freehire platform/safehttp) ─────────────────────────
import ipaddress  # noqa: E402
import socket  # noqa: E402

import httpcore  # noqa: E402

from hiredra.sourcing import http as http_mod  # noqa: E402


@pytest.mark.parametrize(
    ("ip", "public"),
    [
        ("127.0.0.1", False),
        ("10.1.2.3", False),
        ("172.16.0.1", False),
        ("192.168.1.1", False),
        ("169.254.169.254", False),  # cloud metadata
        ("100.64.0.1", False),  # CGNAT
        ("192.0.2.1", False),  # TEST-NET-1
        ("198.18.0.1", False),  # benchmarking
        ("240.0.0.1", False),  # reserved
        ("224.0.0.1", False),  # multicast
        ("0.0.0.0", False),
        ("::1", False),
        ("fd00::1", False),
        ("::ffff:127.0.0.1", False),  # IPv4-mapped loopback
        ("8.8.8.8", True),
        ("2606:4700:4700::1111", True),
    ],
)
def test_public_address_classification(ip, public):
    assert http_mod._public(ipaddress.ip_address(ip)) is public


def _fake_dns(monkeypatch, *ips):
    calls = []

    def getaddrinfo(host, port, *a, **k):
        calls.append(host)
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)) for ip in ips]

    monkeypatch.setattr(http_mod.socket, "getaddrinfo", getaddrinfo)
    return calls


def test_a_host_resolving_to_a_private_address_is_refused_and_not_retried(monkeypatch):
    calls = _fake_dns(monkeypatch, "127.0.0.1")
    with HttpClient(HttpConfig(attempts=3), sleep=lambda s: None) as client:
        with pytest.raises(http_mod.NonPublicAddress):
            client.get_text("http://looks-public.example/")
    assert calls == ["looks-public.example"]  # one lookup: a non-public address is not transient


def test_a_host_with_any_private_address_is_refused(monkeypatch):
    _fake_dns(monkeypatch, "93.184.216.34", "10.0.0.5")  # a rebinding-style mixed answer
    with HttpClient(HttpConfig(attempts=1)) as client, pytest.raises(http_mod.NonPublicAddress):
        client.get_text("http://mixed.example/")


def test_the_default_client_is_guarded():
    # httpx has no public hook for the network backend: this pins the private attribute the guard sets
    with HttpClient() as client:
        assert isinstance(client._client._transport._pool._network_backend, http_mod._PublicOnlyBackend)


def test_the_guard_tries_each_checked_address_in_turn(monkeypatch):
    _fake_dns(monkeypatch, "93.184.216.34", "93.184.216.35")
    backend = http_mod._PublicOnlyBackend()
    tried = []

    def connect_tcp(host, port, *a, **k):
        tried.append(host)
        if host == "93.184.216.34":
            raise httpcore.ConnectError("unreachable")
        return "stream"

    monkeypatch.setattr(backend._inner, "connect_tcp", connect_tcp)
    assert backend.connect_tcp("two.example", 443) == "stream"
    assert tried == ["93.184.216.34", "93.184.216.35"]
