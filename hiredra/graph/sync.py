"""Sub-slice 4.3: the between-runs sync step -- embed pending nodes and keep the
index version-consistent (docs/graph.md sec 4b, sec 6).

Everything here runs BETWEEN scoring runs, never during one (the freeze rule).
Three pieces:

* ``embed_pending`` -- drain the resumable embed queue. The queue is just
  "nodes with no vector, or a vector under another version tag"
  (``skills_needing_embedding``), and each chunk of vectors commits before the
  next chunk is requested. So there is no separate resume state: an interrupted
  run -- Ctrl-C, a crash, a daily quota -- leaves every finished chunk saved,
  and running it again continues from the DB. A failing chunk STOPS the run
  (it does not skip ahead or hammer the API); a daily-quota stop is reported as
  such so the caller can wait for the quota to reset.
* ``check_canary`` -- re-embed a few existing nodes and compare against their
  stored vectors. Embedding is deterministic, so a drop below the configured
  cosine means the provider changed the model under an unchanged id.
* ``ensure_index_current`` -- the version-consistency gate: canary first (drift
  invalidates every vector), then drain the queue. ``ready`` is True only when
  no node is un-embedded or stale, i.e. the whole index is comparable.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from hiredra.graph.config import GraphConfig
from hiredra.graph.embedder import DailyQuotaExceeded, Embedder, classify_failure
from hiredra.graph.usage import DailyBudgetReached
from hiredra.graph.index import blob_to_vector, save_vectors
from hiredra.storage import (
    get_skill_vocab,
    invalidate_skill_vectors,
    skill_index_status,
    skills_needing_embedding,
)


@dataclass
class EmbedReport:
    queue_size: int  # nodes needing an embedding when the run started
    attempted: int  # how many of them this run tried (queue_size, or `limit`)
    embedded: int = 0
    chunks: int = 0
    remaining: int = 0  # still needing an embedding afterwards
    elapsed_s: float = 0.0
    #: None (finished), or why it stopped: "daily_budget" (our local budget) | "daily_quota"
    #: (Google's cap) | "transient" (429/5xx/network outlasted the retries) | "api_error"
    #: (Google said no: key, request, access) | "bad_response" (unusable vectors).
    #: An UNKNOWN exception is never turned into a stop reason -- it is re-raised.
    stopped: str | None = None
    error: str | None = None


def embed_pending(
    conn,
    cfg: GraphConfig,
    embed: Embedder,
    *,
    limit: int | None = None,
    weights: Mapping[str, int] | None = None,
    daily_remaining: Callable[[], int] | None = None,
    log: Callable[[str], None] = lambda _msg: None,
    clock: Callable[[], float] = time.monotonic,
) -> EmbedReport:
    """Embed the queue in ``cfg.inputs_per_request`` chunks, committing each chunk.

    ``limit`` caps how many skills this call attempts (staged runs); ``weights``
    orders ties in the queue (see ``skills_needing_embedding``).
    ``daily_remaining`` (a callable, read once when the run starts) is how many
    more requests today's budget allows. Google counts each skill as one request,
    so the run only attempts that many and then stops with ``"daily_budget"`` --
    cleanly, before Google's own cap can reject anything.
    """
    queue = skills_needing_embedding(conn, cfg.embed_tag, weights=weights)
    planned = queue if limit is None else queue[:limit]
    budget = daily_remaining() if daily_remaining is not None else None
    target = planned if budget is None else planned[:budget]
    report = EmbedReport(queue_size=len(queue), attempted=len(target))
    started = clock()
    if budget is not None and len(target) < len(planned):
        report.stopped = "daily_budget"  # reported after the (possibly empty) run below
        report.error = f"today's request budget covers {len(target)} of {len(planned)} skills"

    for start in range(0, len(target), cfg.inputs_per_request):
        chunk = target[start : start + cfg.inputs_per_request]
        try:
            vectors = embed(chunk)
        except DailyBudgetReached as exc:  # safety net: the embedder's own check disagreed with our count
            report.stopped, report.error = "daily_budget", str(exc)
            break
        except DailyQuotaExceeded as exc:
            report.stopped, report.error = "daily_quota", str(exc)
            break
        except Exception as exc:  # retries already exhausted inside the embedder
            kind = classify_failure(exc)
            if kind is None:
                raise  # not a quota/network/API problem: most likely a bug, so it must be loud
            report.stopped, report.error = kind, f"{type(exc).__name__}: {str(exc)[:300]}"
            break
        save_vectors(conn, chunk, vectors, cfg.embed_tag)  # one transaction per chunk
        report.embedded += len(chunk)
        report.chunks += 1
        log(f"embedded {report.embedded}/{len(target)}  ({clock() - started:.0f}s)")

    report.remaining = len(queue) - report.embedded
    report.elapsed_s = clock() - started
    return report


@dataclass(frozen=True)
class CanaryResult:
    checked: tuple[str, ...] = ()
    min_cosine: float | None = None
    drifted: bool = False
    skipped_reason: str | None = None  # set when nothing could be checked
    error: str | None = None  # the canary embed itself failed (e.g. quota)


def check_canary(conn, cfg: GraphConfig, embed: Embedder) -> CanaryResult:
    """Re-embed the configured canary nodes that already have a CURRENT vector and
    compare to the stored ones. Nodes that don't exist / aren't embedded yet are
    skipped; if none qualify there is nothing to compare (a fresh index)."""
    present: list[str] = []
    stored: list[np.ndarray] = []
    for name in cfg.canary_skills:
        row = get_skill_vocab(conn, name)
        if row and row["vector"] is not None and row["embed_tag"] == cfg.embed_tag:
            present.append(name)
            stored.append(blob_to_vector(row["vector"], cfg.dimensions))
    if not present:
        return CanaryResult(skipped_reason="no canary skill has a current vector yet")
    try:
        fresh = embed(present)
    except Exception as exc:
        if classify_failure(exc) is None:
            raise  # an unknown exception is a bug, not "the canary couldn't run"
        return CanaryResult(checked=tuple(present), error=f"{type(exc).__name__}: {str(exc)[:200]}")
    cos = (fresh * np.stack(stored)).sum(axis=1)  # both sides are unit length
    low = float(cos.min())
    return CanaryResult(checked=tuple(present), min_cosine=low, drifted=low < cfg.canary_min_cosine)


@dataclass
class GateReport:
    before: dict[str, int]
    canary: CanaryResult
    invalidated: int
    embed: EmbedReport
    after: dict[str, int] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        """The whole index is comparable: every node embedded under the live tag,
        and the canary did not fail outright."""
        return self.after.get("unembedded", 1) == 0 and self.after.get("stale", 1) == 0 and self.canary.error is None


def ensure_index_current(
    conn,
    cfg: GraphConfig,
    embed: Embedder,
    *,
    limit: int | None = None,
    weights: Mapping[str, int] | None = None,
    daily_remaining: Callable[[], int] | None = None,
    log: Callable[[str], None] = lambda _msg: None,
) -> GateReport:
    """The version-consistency gate. Run at the START of a scoring run (before the
    graph is frozen), never mid-run:

    1. canary -- re-embed a few nodes; if they no longer match their stored
       vectors the provider changed weights under the same model id, so EVERY
       stored vector is invalidated;
    2. drain the embed queue (which now also covers any tag-stale vectors,
       e.g. after a model / dimension / prefix change in config).

    A tag change needs no special case: ``skills_needing_embedding`` already
    treats a different ``embed_tag`` as needing re-embedding.
    """
    before = skill_index_status(conn, cfg.embed_tag)
    canary = check_canary(conn, cfg, embed)
    invalidated = 0
    if canary.drifted:
        invalidated = invalidate_skill_vectors(conn)
        log(
            f"CANARY DRIFT (min cosine {canary.min_cosine:.4f} < {cfg.canary_min_cosine}) -- "
            f"invalidated {invalidated} vectors; the whole index will be re-embedded"
        )
    elif canary.skipped_reason:
        log(f"canary skipped: {canary.skipped_reason}")
    elif canary.error:
        log(f"canary could not run: {canary.error}")
    else:
        log(f"canary ok: {len(canary.checked)} nodes, min cosine {canary.min_cosine:.6f}")
    result = embed_pending(conn, cfg, embed, limit=limit, weights=weights, daily_remaining=daily_remaining, log=log)
    return GateReport(
        before=before,
        canary=canary,
        invalidated=invalidated,
        embed=result,
        after=skill_index_status(conn, cfg.embed_tag),
    )
