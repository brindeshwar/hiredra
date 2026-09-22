"""Batch-vs-single consistency check (build 4.3 fix; run on reset day).

The question: does embedding a skill in a batch give the SAME vector as
embedding it alone, or in a different batch order? If it does not, the vectors
already stored (made in batches of 50) and the ones we are about to make would
not be comparable, and the index would be quietly inconsistent.

The check takes a small, evenly spread sample of nodes that already have a
current vector, embeds the SAME skills three ways, and compares each result with
the stored vector:

  batch       all sample skills together in one request
  one_by_one  each skill alone, one request per skill
  reversed    the same batch in the opposite order (different neighbours)

It passes only if every cosine is >= the canary threshold (0.999 in
``config/graph.yml``). Cost: 3 x sample counted requests (24 at the default 8),
and none at all in the tests, which use a fake embedder.

It never changes the database. If it fails, the CLI prints the fallback (re-embed
everything under one method); it does not do it for you.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hiredra.graph.config import GraphConfig
from hiredra.graph.embedder import Embedder, classify_failure
from hiredra.graph.index import blob_to_vector
from hiredra.storage import iter_skill_vocab


@dataclass(frozen=True)
class ConditionResult:
    name: str  # batch | one_by_one | reversed
    min_cosine: float
    worst: str  # the skill whose fresh vector is furthest from its stored one


@dataclass
class ConsistencyReport:
    sample: tuple[str, ...] = ()
    conditions: list[ConditionResult] = field(default_factory=list)
    threshold: float = 0.999
    consistent: bool | None = None  # None = could not tell (nothing to check, or the API call failed)
    requests_used: int = 0
    skipped_reason: str | None = None
    error: str | None = None


def pick_sample(names: list[str], k: int) -> list[str]:
    """``k`` names spread evenly over the sorted list (so early and late chunks of
    the original build are both represented). Deterministic."""
    names = sorted(names)
    if k >= len(names):
        return names
    picks = sorted({int(round(i)) for i in np.linspace(0, len(names) - 1, k)})
    return [names[i] for i in picks]


def check_batch_consistency(conn, cfg: GraphConfig, embed: Embedder, *, sample: int = 8) -> ConsistencyReport:
    stored = {
        r["name"]: blob_to_vector(r["vector"], cfg.dimensions)
        for r in iter_skill_vocab(conn)
        if r["vector"] is not None and r["embed_tag"] == cfg.embed_tag
    }
    report = ConsistencyReport(threshold=cfg.canary_min_cosine)
    if len(stored) < 2:
        report.skipped_reason = "fewer than 2 nodes have a current vector, nothing to compare against"
        return report

    names = pick_sample(list(stored), max(2, sample))
    report.sample = tuple(names)
    reference = np.stack([stored[n] for n in names])

    def one_by_one() -> np.ndarray:
        return np.stack([embed([n])[0] for n in names])

    runs = (
        ("batch", lambda: embed(names)),
        ("one_by_one", one_by_one),
        ("reversed", lambda: embed(names[::-1])[::-1]),
    )
    try:
        for label, run in runs:
            fresh = run()
            report.requests_used += len(names)  # Google counts each input as one request
            cos = (fresh * reference).sum(axis=1)  # both sides are unit length
            worst = int(np.argmin(cos))
            report.conditions.append(ConditionResult(label, float(cos[worst]), names[worst]))
    except Exception as exc:
        if classify_failure(exc) is None:
            raise  # an unknown exception is a bug, not "could not tell"
        report.error = f"{type(exc).__name__}: {str(exc)[:200]}"
        return report

    report.consistent = all(c.min_cosine >= report.threshold for c in report.conditions)
    return report
