"""Sub-slice 4.4: the per-skill fit mechanic (docs/scoring.md sec 4).

For each JD skill: fold its name, look it up in an in-memory cache, and embed
fresh only on a miss. The graph itself is never written here -- scoring reads
it, it does not touch it (docs/graph.md's freeze rule: nodes are added/merged
between runs, never mid-run). Then find the single nearest ``reference`` skill
by cosine (a plain dot product -- every vector is already L2-normalized) and
map that similarity through the HAVE/ADJ thresholds.

``reference`` stands in for the owner's ``mine`` skills. The master-profile
overlay that produces real ``mine`` nodes is deferred past Step 4, so callers
(including the tests here) pass any named subset of market nodes instead --
the math does not care where ``reference`` came from.

Scope: this module returns the fit vector only. Coverage, gap_distance and the
score formula that consume it are Step 6 (docs/scoring.md secs 1-2).

PLACEHOLDER thresholds: ``cfg.have`` / ``cfg.adj`` (currently 0.90 / 0.70, see
``config/graph.yml``) are not calibrated. Step 5's F1 sweep sets the real
values for this embedding model -- nothing here changes when that happens,
only the numbers in the config.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hiredra.graph.config import GraphConfig
from hiredra.graph.embedder import Embedder
from hiredra.graph.index import Skill
from hiredra.vocab import normalize

#: float32 dot products of unit vectors can land a hair under an exact
#: threshold (see hiredra.graph.merge's own _EPS for the same reason).
_EPS = 1e-6


@dataclass(frozen=True)
class SkillFit:
    """One JD skill's fit against ``reference`` (docs/scoring.md sec 4).

    ``nearest``/``similarity`` are ``None``/``0.0`` when ``reference`` is
    empty -- there is nothing to compare against, so the skill is unmapped,
    not merely missing. ``skill`` is the folded (case/whitespace) canonical
    form of whatever was passed in, matching how every other node name in
    the graph is keyed.
    """

    skill: str
    nearest: str | None
    similarity: float
    fit: float


def fit_skills(
    jd_skills: Sequence[str],
    reference: Sequence[Skill],
    cfg: GraphConfig,
    embed: Embedder,
    *,
    cache: dict[str, np.ndarray] | None = None,
) -> list[SkillFit]:
    """Fit each of ``jd_skills`` against ``reference`` (the ``mine`` stand-in).

    ``cache`` is an optional folded-name -> unit-vector dict, shared across
    calls within one scoring run so the same JD skill (seen in an earlier job,
    or repeated within this one) is never embedded twice. Pass the same dict
    in across calls to get that reuse; omit it for a one-off call.
    """
    if cache is None:
        cache = {}

    ref_pairs = [(r.name, r.vector) for r in reference if r.vector is not None]
    ref_names = [name for name, _ in ref_pairs]
    ref_matrix = (
        np.array([vector for _, vector in ref_pairs], dtype=np.float32)
        if ref_pairs
        else np.zeros((0, cfg.dimensions), dtype=np.float32)
    )

    folded = [normalize(s) for s in jd_skills]
    misses = [name for name in dict.fromkeys(folded) if name not in cache]
    if misses:
        vectors = embed(misses)
        for name, vector in zip(misses, vectors):
            cache[name] = vector

    results: list[SkillFit] = []
    for name in folded:
        vector = cache[name]
        if ref_matrix.shape[0] == 0:
            nearest, similarity = None, 0.0
        else:
            sims = ref_matrix @ vector
            idx = int(np.argmax(sims))
            nearest, similarity = ref_names[idx], float(sims[idx])

        if similarity >= cfg.have - _EPS:
            fit = 1.0
        elif similarity >= cfg.adj - _EPS:
            fit = similarity
        else:
            fit = 0.0

        results.append(SkillFit(skill=name, nearest=nearest, similarity=similarity, fit=fit))
    return results
