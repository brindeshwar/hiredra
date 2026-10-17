"""Live smoke test for hiredra.graph.embedder -- NOT a pytest test.

Kept outside tests/ for the same reason as smoke_test_extraction.py: it hits
the real Gemini API and needs GEMINI_API_KEY in .env.

WARNING -- IT SPENDS ABOUT 450 OF THE 1,000 DAILY REQUESTS. Google counts every
input inside a batch as one request, and this script sends ~13 + 13 + 3 + 3 +
100 + 100 + 50 inputs (plus deliberate over-limit probes) and does NOT use the
daily counter. It is not part of the reset-day runbook. Do not run it on a day
you need the budget for `sync`.

Run it manually:

    python scripts/smoke_test_embeddings.py

It confirms the unknowns from the Step 4 build plan against the live API:

1. The configured model string is accepted and vectors come back at the
   configured dimensionality, one per input, unit length after normalize.
2. Cosine separates a known related pair from an unrelated one, and the
   fastapi ~ "fast api" merge case clears the 0.90 dedup threshold.
3. Bare prefix vs the "task: sentence similarity" prefix (informational A/B --
   the evidence behind the bare default in config/graph.yml).
4. Determinism, which the version-gate canary (cosine >= 0.999 on re-embed)
   relies on.
5. The free-tier quota, measured rather than assumed: the paced embedder
   handles 100 inputs without tripping it, and a deliberate over-limit probe
   shows how the per-minute quota counts (each input in a batch = 1 request) and what the 429 says.
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hiredra.graph.config import load_graph_config  # noqa: E402
from hiredra.graph.embedder import (  # noqa: E402
    _default_client,
    _is_transient,
    _retry_after_seconds,
    build_embedder,
)

PROBE = [
    "fastapi", "fast api", "flask", "kubernetes", "docker", "kafka",
    "postgresql", "mysql", "langchain", "llamaindex", "react", "react native",
    "art market",
]  # fmt: skip

PAIRS = [
    ("fastapi", "fast api", "same"),
    ("fastapi", "flask", "related"),
    ("kubernetes", "docker", "related"),
    ("postgresql", "mysql", "related"),
    ("langchain", "llamaindex", "related"),
    ("react", "react native", "related"),
    ("kafka", "fastapi", "unrelated"),
    ("react", "kubernetes", "unrelated"),
    ("art market", "kubernetes", "unrelated"),
]  # fmt: skip

CANARY = ["python", "kubernetes", "postgresql"]

BATCH_100 = [
    "python", "java", "go", "rust", "typescript", "javascript", "c++", "c#", "scala", "kotlin",
    "swift", "ruby", "php", "sql", "bash", "react", "vue", "angular", "svelte", "next.js",
    "node.js", "express", "django", "flask", "fastapi", "spring boot", "rails", "graphql", "rest", "grpc",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra", "dynamodb", "snowflake", "bigquery", "clickhouse",
    "kafka", "rabbitmq", "spark", "airflow", "dbt", "flink", "hadoop", "hive", "presto", "databricks",
    "docker", "kubernetes", "terraform", "ansible", "helm", "jenkins", "github actions", "gitlab ci", "argo cd", "pulumi",
    "aws", "gcp", "azure", "lambda", "s3", "ec2", "eks", "cloud run", "cloudflare workers", "vercel",
    "pytorch", "tensorflow", "scikit-learn", "xgboost", "hugging face", "langchain", "llamaindex", "rag", "vector databases", "prompt engineering",
    "fine-tuning", "embeddings", "openai api", "mcp", "langgraph", "crewai", "pinecone", "chroma", "faiss", "weaviate",
    "prometheus", "grafana", "datadog", "opentelemetry", "sentry", "linux", "git", "ci/cd", "microservices", "distributed systems",
]  # fmt: skip
assert len(BATCH_100) == 100 and len(set(BATCH_100)) == 100

failures: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


def pair_table(vecs: np.ndarray, index: dict[str, int]) -> dict[tuple[str, str], float]:
    out = {}
    for a, b, kind in PAIRS:
        s = float(vecs[index[a]] @ vecs[index[b]])
        out[(a, b)] = s
        print(f"    {a:>12} ~ {b:<14} {kind:<10} {s:.3f}")
    return out


def main() -> int:
    cfg = load_graph_config()
    print(
        f"config: model={cfg.model!r} dim={cfg.dimensions} prefix={cfg.prefix!r} "
        f"inputs_per_request={cfg.inputs_per_request} req/min budget={cfg.rpm_budget}"
    )
    print(f"embed_tag = {cfg.embed_tag}\n")
    embed = build_embedder(cfg)

    print("1. shape / normalization")
    vecs = embed(PROBE)
    index = {s: i for i, s in enumerate(PROBE)}
    check(vecs.shape == (len(PROBE), cfg.dimensions), "one vector per input at configured dim", f"shape={vecs.shape}")
    check(bool(np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)), "rows are unit length")

    print("\n2. cosine separation (bare prefix = the configured default)")
    bare = pair_table(vecs, index)
    check(
        bare[("fastapi", "fast api")] >= cfg.dedup_threshold,
        f"fastapi ~ 'fast api' >= dedup threshold {cfg.dedup_threshold}",
        f"{bare[('fastapi', 'fast api')]:.3f}",
    )
    check(
        bare[("kubernetes", "docker")] > bare[("kafka", "fastapi")],
        "kubernetes ~ docker > kafka ~ fastapi (related beats unrelated)",
    )
    check(
        bare[("art market", "kubernetes")]
        < min(bare[("kubernetes", "docker")], bare[("postgresql", "mysql")], bare[("langchain", "llamaindex")]),
        "an off-topic skill scores below the related pairs",
    )
    react_pair = bare[("react", "react native")]
    print(
        f"  [INFO] react ~ 'react native' = {react_pair:.3f} "
        f"({'WOULD' if react_pair >= cfg.dedup_threshold else 'would NOT'} auto-fuse at "
        f"{cfg.dedup_threshold} -- the dry-run-review case)"
    )

    print("\n3. prefix A/B (informational; default stays bare)")
    sim_cfg = dataclasses.replace(cfg, prefix="task: sentence similarity | query: ")
    sim = pair_table(build_embedder(sim_cfg)(PROBE), index)
    for name, table in (("bare", bare), ("sim-prefix", sim)):
        unrel = [table[(a, b)] for a, b, k in PAIRS if k == "unrelated"]
        rel = [table[(a, b)] for a, b, k in PAIRS if k == "related"]
        print(
            f"    {name:<10} related mean {np.mean(rel):.3f} | unrelated mean {np.mean(unrel):.3f} "
            f"| gap {np.mean(rel) - np.mean(unrel):+.3f}"
        )

    print("\n4. determinism (the canary's premise)")
    a, b = embed(CANARY), embed(CANARY)
    cos = (a * b).sum(axis=1)
    check(bool(np.all(cos >= 0.999)), "re-embedding the same strings gives cosine >= 0.999", f"min={cos.min():.6f}")

    print("\n5a. paced embedder, 100 strings from a clean quota window")
    print("    waiting 65s for the rolling window to clear (sections 1-4 spent quota)...")
    time.sleep(65)
    client = _default_client(cfg, None)

    class Counting:
        """Proxy that counts real requests and records their outcomes."""

        def __init__(self):
            self.requests = 0
            self.failures: list[str] = []

        def embed_documents(self, texts, batch_size=100):
            self.requests += 1
            try:
                return client.embed_documents(texts, batch_size=batch_size)
            except Exception as exc:
                self.failures.append(f"transient={_is_transient(exc)} retry_after={_retry_after_seconds(exc)}")
                raise

    counting = Counting()
    t0 = time.monotonic()
    v100 = build_embedder(cfg, client=counting)(BATCH_100)
    elapsed = time.monotonic() - t0
    chunks = -(-100 // cfg.inputs_per_request)
    check(v100.shape == (100, cfg.dimensions), "100 strings -> 100 vectors", f"shape={v100.shape}")
    print(
        f"    {counting.requests} requests for {chunks} chunks in {elapsed:.0f}s "
        f"(~{100 / elapsed * 60:.0f} inputs/min); failures/retries: {counting.failures or 'none'}"
    )

    print("\n5b. quota probe (raw client, deliberately over the limit)")
    print("    waiting 65s for the rolling window to clear...")
    time.sleep(65)
    # Up to 4 full batches back to back: only a per-STRING quota can reject a
    # burst of 4 requests, so a 429 here proves strings (not requests) are counted.
    sent, hint, rejected = 0, None, False
    for i in range(4):
        try:
            client.embed_documents([f"{x} q{i}" for x in BATCH_100], batch_size=100)
            sent += 100
            print(f"    full batch {i + 1}: OK (cumulative {sent} strings in this minute)")
        except Exception as exc:
            hint = _retry_after_seconds(exc)
            rejected = _is_transient(exc)
            print(f"    full batch {i + 1}: 429 after {sent} accepted strings; retryDelay hint = {hint}s")
            break
    check(
        rejected,
        "quota rejects a burst of >100 inputs sent in only a few HTTP requests (each input in a batch counts as one request)",
        f"{sent} strings accepted before the 429",
    )
    if rejected and hint:
        print(f"    waiting the server's {hint:.0f}s hint + 1s, then retrying one batch...")
        time.sleep(hint + 1)
        try:
            client.embed_documents([f"{x} q-retry" for x in BATCH_100[:50]], batch_size=50)
            check(True, "the server's retryDelay hint is enough to recover")
        except Exception as exc:
            check(False, "the server's retryDelay hint is enough to recover", str(exc)[:120])

    print("\n" + ("SMOKE TEST PASSED" if not failures else f"SMOKE TEST FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
