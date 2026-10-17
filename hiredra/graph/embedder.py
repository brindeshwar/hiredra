"""Sub-slice 4.1: the embedder (build step 4, docs/graph.md sec 4).

``build_embedder`` returns one plain callable, ``embed(texts) -> ndarray``:
an ``(n, dimensions)`` float32 matrix, one L2-normalized row per input, so
nearest-skill is a dot product everywhere downstream. Everything else in the
graph package takes that callable as an argument, so tests hand in a fake and
never need a key or the network.

Client: LangChain's ``GoogleGenerativeAIEmbeddings`` (already pinned for Step
3). Two facts about ``gemini-embedding-2`` shaped this file (both verified live):

* It has no ``task_type`` parameter -- the task is a text prefix on the input.
  The wrapper still sends a ``task_type`` (default RETRIEVAL_DOCUMENT), which
  the model silently ignores: vectors are bit-for-bit the same with or without
  it. The configured ``prefix`` is therefore applied here, in code, to the text.
* A raw-SDK list of strings comes back as ONE aggregated vector. The wrapper
  sends one ``Content`` per string; the count check below guards a regression.

Batching -- what is and is not true (proven offline, see docs/graph.md sec 4b):
this embedder already sends TRUE batches. The SDK always calls
``:batchEmbedContents``; one chunk of N skills is ONE HTTP request carrying N
inputs and returning N separate vectors. But Google COUNTS every input inside a
batch as one "request" against its limits. So batching saves HTTP round trips
and cannot lower the counted requests. The limits (free tier, this model) are
100 requests/min, 30,000 tokens/min and 1,000 requests/day; pacing here follows
requests/min and tokens/min, and an optional ``DailyUsage`` counter guards the
daily cap BEFORE a request is sent. A 429 carries a ~1 minute
``RetryInfo.retryDelay`` that the retry wait honors (a 2-30s exponential backoff
would burn every attempt inside the same quota window).

The wrapper re-raises every failure as ``GoogleGenerativeAIError`` with the SDK
error as ``__cause__``, so the transient-error check walks the cause chain.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence

import httpx
import numpy as np
from dotenv import load_dotenv
from google.genai.errors import APIError
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

from hiredra.graph.config import GraphConfig

#: text list in, (n, dimensions) float32 L2-normalized matrix out.
Embedder = Callable[[Sequence[str]], np.ndarray]


class EmbeddingError(RuntimeError):
    """The provider returned something we can't use (wrong count/dim, zero vector)."""


class DailyQuotaExceeded(EmbeddingError):
    """A DAILY quota is exhausted. Unlike the per-minute window this will not
    clear in a retry or two, so it is raised at once and never retried: callers
    stop cleanly and resume from the DB-backed queue on a later day."""


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise unit-length copy, float32. Raises on a zero/non-finite row."""
    m = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise EmbeddingError("embedding has a zero or non-finite norm")
    return m / norms


def _api_errors(exc: BaseException):
    """Every SDK ``APIError`` in ``exc``'s cause/context chain (usually one)."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, APIError):
            yield exc
        exc = exc.__cause__ or exc.__context__


#: A retryDelay this long is not a per-minute window clearing.
_DAILY_RETRY_DELAY_S = 600.0


def _quota_ids(exc: BaseException) -> list[str]:
    """``QuotaFailure`` quota ids on the error, e.g.
    ``EmbedContentRequestsPerMinutePerUserPerProjectPerModel-FreeTier``."""
    ids: list[str] = []
    for e in _api_errors(exc):
        details = e.details if isinstance(e.details, dict) else {}
        body = details.get("error", details)
        for item in (body.get("details") or []) if isinstance(body, dict) else []:
            if isinstance(item, dict) and str(item.get("@type", "")).endswith("QuotaFailure"):
                ids += [str(v.get("quotaId", "")) for v in item.get("violations") or [] if isinstance(v, dict)]
    return ids


def _quota_values(exc: BaseException) -> list[str]:
    """The ``quotaValue`` (the limit itself) of each violated quota, so a daily
    stop records WHAT the cap is, not just which quota it was."""
    values: list[str] = []
    for e in _api_errors(exc):
        details = e.details if isinstance(e.details, dict) else {}
        body = details.get("error", details)
        for item in (body.get("details") or []) if isinstance(body, dict) else []:
            if isinstance(item, dict) and str(item.get("@type", "")).endswith("QuotaFailure"):
                values += [str(v.get("quotaValue", "")) for v in item.get("violations") or [] if isinstance(v, dict)]
    return values


def _is_daily_quota(exc: BaseException) -> bool:
    """A 429 whose quota id says per-DAY, or whose retryDelay is far too long to
    be a per-minute window (>= 10 min)."""
    if not any(e.code == 429 for e in _api_errors(exc)):
        return False
    if any("perday" in q.lower() for q in _quota_ids(exc)):
        return True
    hint = _retry_after_seconds(exc)
    return hint is not None and hint >= _DAILY_RETRY_DELAY_S


#: Network-level failures (no HTTP status at all): a dropped connection, a DNS
#: hiccup, a timeout. Worth a retry, and not a bug in our code.
_NETWORK_ERRORS = (httpx.TransportError, ConnectionError, TimeoutError)


def _cause_chain(exc: BaseException):
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _is_network_error(exc: BaseException) -> bool:
    return any(isinstance(e, _NETWORK_ERRORS) for e in _cause_chain(exc))


def _is_transient(exc: BaseException) -> bool:
    """429 / 5xx anywhere in the cause chain, or a network-level failure -- worth
    a retry. Anything else (a bad request, our own EmbeddingError) is not, and
    neither is a daily quota, which a retry cannot fix."""
    if _is_daily_quota(exc):
        return False
    return _is_network_error(exc) or any(e.code == 429 or e.code >= 500 for e in _api_errors(exc))


def classify_failure(exc: BaseException) -> str | None:
    """What kind of failure is this? Used to tell an expected stop from a bug.

    "daily_quota"  Google's daily cap (or our local budget): wait for the reset.
    "transient"    429 / 5xx / network trouble that outlasted the retries: re-run later.
    "api_error"    Google answered and said no (bad key, bad request, no access).
    "bad_response" Google answered with something unusable (wrong count/size/zero vector).
    None           anything else: an unknown exception is most likely a BUG, so callers
                   must re-raise it rather than turn it into a status line.
    """
    if isinstance(exc, DailyQuotaExceeded) or _is_daily_quota(exc):
        return "daily_quota"
    if _is_transient(exc):
        return "transient"
    if any(True for _ in _api_errors(exc)):
        return "api_error"
    if isinstance(exc, EmbeddingError):
        return "bad_response"
    return None


def _retry_after_seconds(exc: BaseException) -> float | None:
    """The server's own ``RetryInfo.retryDelay`` (e.g. ``"57s"``), if the error
    carries one. Free-tier 429s do."""
    for e in _api_errors(exc):
        details = e.details if isinstance(e.details, dict) else {}
        body = details.get("error", details)
        for item in (body.get("details") or []) if isinstance(body, dict) else []:
            if isinstance(item, dict) and str(item.get("@type", "")).endswith("RetryInfo"):
                delay = str(item.get("retryDelay", "")).rstrip("s")
                try:
                    return float(delay)
                except ValueError:
                    return None
    return None


def _default_client(cfg: GraphConfig, api_key: str | None):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    load_dotenv()
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (check .env) -- required to build the embedder.")
    return GoogleGenerativeAIEmbeddings(
        model=cfg.model,
        google_api_key=key,
        output_dimensionality=cfg.dimensions,
    )


def estimate_tokens(text: str) -> int:
    """A deliberately pessimistic token estimate (~3 characters per token; English
    skill names are nearer 4). Only used for pacing against tokens/minute, where a
    high guess is the safe direction. No tokenizer call, so no API call."""
    return max(1, -(-len(text) // 3))


def build_embedder(
    cfg: GraphConfig,
    *,
    client=None,
    api_key: str | None = None,
    usage=None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Embedder:
    """Build the ``embed(texts)`` callable for ``cfg``.

    ``client`` is anything with ``embed_documents(texts, batch_size=...)``;
    default is the LangChain wrapper. ``usage`` is an optional ``DailyUsage``: if
    given, every request is checked against today's budget BEFORE it is sent,
    counted after Google accepts it, and a server-side daily-quota error marks
    the day exhausted. ``sleep``/``clock`` are injectable so pacing and retry
    backoff are testable without real waiting.
    """
    client = client if client is not None else _default_client(cfg, api_key)
    next_allowed = [None]  # closure-held: earliest clock() the next request may start

    def _pace(n_requests: int, n_tokens: int) -> None:
        """One chunk of n inputs counts as n requests (Google's rule) and about
        n_tokens tokens. Space chunks so neither per-minute budget is exceeded:
        the chunk 'occupies' the larger of its two time shares."""
        now = clock()
        if next_allowed[0] is not None and now < next_allowed[0]:
            sleep(next_allowed[0] - now)
        gap = max(60.0 * n_requests / cfg.rpm_budget, 60.0 * n_tokens / cfg.tpm_budget)
        next_allowed[0] = clock() + gap

    def _request(chunk: list[str]) -> list[list[float]]:
        if usage is not None:
            usage.reserve(len(chunk))  # refuse BEFORE the call if today's budget can't cover it
        _pace(len(chunk), sum(estimate_tokens(t) for t in chunk))  # every attempt spends quota
        vectors = client.embed_documents(chunk, batch_size=len(chunk))
        if usage is not None:
            usage.record(len(chunk))  # counted only once Google accepted it
        return vectors

    backoff = wait_exponential(multiplier=1, min=2, max=30)

    def _wait(retry_state) -> float:
        hint = _retry_after_seconds(retry_state.outcome.exception())
        # server hint (+1s margin) when it gave one, else plain exponential backoff
        return backoff(retry_state) if hint is None else min(hint + 1.0, 120.0)

    retrying = Retrying(
        retry=retry_if_exception(_is_transient),
        stop=stop_after_attempt(4),
        wait=_wait,
        sleep=sleep,
        reraise=True,
    )

    def embed(texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, cfg.dimensions), dtype=np.float32)
        prefixed = [cfg.prefix + t for t in texts]
        rows: list[list[float]] = []
        for start in range(0, len(prefixed), cfg.inputs_per_request):
            chunk = prefixed[start : start + cfg.inputs_per_request]
            if sum(estimate_tokens(t) for t in chunk) > cfg.tpm_budget:
                raise EmbeddingError(
                    f"a chunk of {len(chunk)} inputs is estimated above the whole per-minute token budget "
                    f"({cfg.tpm_budget}); lower embedding.inputs_per_request"
                )
            try:
                vectors = retrying(_request, chunk)
            except Exception as exc:
                if _is_daily_quota(exc):
                    if usage is not None:
                        usage.mark_exhausted("reported by Google (server-side daily quota error)")
                    ids, limits = _quota_ids(exc), _quota_values(exc)
                    raise DailyQuotaExceeded(
                        f"daily embedding quota exhausted ({', '.join(ids) or 'long retryDelay'}"
                        + (f"; limit {', '.join(limits)}" if any(limits) else "")
                        + ")"
                    ) from exc
                raise
            if len(vectors) != len(chunk):
                raise EmbeddingError(f"asked for {len(chunk)} embeddings, got {len(vectors)}")
            for v in vectors:
                if len(v) != cfg.dimensions:
                    raise EmbeddingError(f"expected {cfg.dimensions}-dim vectors, got {len(v)}")
            rows.extend(vectors)
        return l2_normalize(np.array(rows, dtype=np.float32))

    return embed
