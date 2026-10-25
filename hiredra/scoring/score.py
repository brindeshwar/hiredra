"""Step 6: the scoring formula as small pure functions, plus its strict config (plan unit S6.1,
docs/scoring.md sec 2).

    score = w_domain * domain_fit + w_skills * coverage - w_gap * gap_distance

* Each JD skill i carries ``fit_i`` (from ``hiredra.graph.fit.fit_skills``) and a weight
  ``w_i``: 1.0 when required, ``good_to_have_factor`` when good-to-have.
* ``coverage = sum(w_i * [fit_i >= cover_bar]) / sum(w_i)`` -- the weighted share of skills
  cleared.
* ``gap_distance = sqrt(sum over fit_i < cover_bar of (w_i * (1 - fit_i))^2)`` -- the size of
  the holes, so a near gap hurts less than a far one.
* ``domain_fit`` comes from the LLM's domain bucket through the owner's step table.

Code decides, the model never does (CLAUDE.md rule 3): the LLM only supplied the bucket and the
skill names. There is no legitimacy term (D2); ``review_reasons`` routes doubtful jobs to the
review list instead, including jobs with fewer than ``min_skills`` extracted skills (their
coverage would be trivially high: 1 of 1 cleared is 100%). Empty cases are defined (no skills -> coverage 0, gap 0; no bucket ->
domain_fit 0), so the score can never be NaN.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from hiredra.graph.config import GraphConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_PATH = _REPO_ROOT / "config" / "weights.yml"

#: The extraction schema's domain buckets (hiredra.extraction.ExtractionResult).
BUCKETS = ("agent_llm", "backend", "data", "ml", "generic_swe", "other")
#: Fixed by D15: the anchors of the domain scale.
FIXED_DOMAIN = {"agent_llm": 1.0, "other": 0.0}
REQUIREMENTS = ("required", "good_to_have")

_KEYS = {
    "calibration_id",
    "domain_fit",
    "w_domain",
    "w_skills",
    "w_gap",
    "cover_bar",
    "good_to_have_factor",
    "apply_line",
    "min_skills",
}

#: float32 cosines can land a hair under an exact bar (same tolerance as hiredra.graph.fit).
_EPS = 1e-6


@dataclass(frozen=True)
class Weights:
    calibration_id: str
    domain_fit: Mapping[str, float]
    w_domain: float
    w_skills: float
    w_gap: float
    cover_bar: float  # resolved: never None after loading
    good_to_have_factor: float
    apply_line: float | None
    min_skills: int  # fewer extracted skills than this -> review list (few_skills)
    score_version: str  # "<calibration_id>:<sha8>" over every value that changes a score


def _number(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"weights config: {where} must be a number, got {value!r}")
    return float(value)


def load_weights(graph_cfg: GraphConfig, path: Path | str = DEFAULT_WEIGHTS_PATH) -> Weights:
    """Read and validate ``config/weights.yml`` against the graph's HAVE/ADJ. Any violation
    stops the program here, before a single job is scored."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("weights config: the top level must be a mapping")
    unknown, missing = set(data) - _KEYS, _KEYS - set(data)
    if unknown:
        raise ValueError(f"weights config: unknown key(s) {sorted(unknown)}")
    if missing:
        raise ValueError(f"weights config: missing key(s) {sorted(missing)}")

    cid = data["calibration_id"]
    if not isinstance(cid, str) or not cid.strip():
        raise ValueError("weights config: calibration_id must be a non-empty string")

    table = data["domain_fit"]
    if not isinstance(table, dict) or set(table) != set(BUCKETS):
        got = sorted(table) if isinstance(table, dict) else table
        raise ValueError(f"weights config: domain_fit must have exactly the buckets {list(BUCKETS)}, got {got}")
    domain_fit = {b: _number(table[b], f"domain_fit.{b}") for b in BUCKETS}
    for bucket, value in domain_fit.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"weights config: domain_fit.{bucket} must be in [0, 1], got {value}")
    for bucket, fixed in FIXED_DOMAIN.items():
        if domain_fit[bucket] != fixed:
            raise ValueError(f"weights config: domain_fit.{bucket} is fixed at {fixed} (D15), got {domain_fit[bucket]}")

    w = {k: _number(data[k], k) for k in ("w_domain", "w_skills", "w_gap")}
    for k, value in w.items():
        if value < 0:
            raise ValueError(f"weights config: {k} must be >= 0, got {value}")

    x = _number(data["good_to_have_factor"], "good_to_have_factor")
    if not 0.0 < x <= 1.0:
        raise ValueError(f"weights config: good_to_have_factor must be in (0, 1], got {x}")

    adj, have = graph_cfg.adj, graph_cfg.have
    cover_bar = (adj + have) / 2 if data["cover_bar"] is None else _number(data["cover_bar"], "cover_bar")
    if not adj < cover_bar < have:
        raise ValueError(
            f"weights config: cover_bar must sit strictly between fit.adj and fit.have (D1): "
            f"need {adj} < cover_bar < {have}, got {cover_bar}"
        )

    apply_line = None if data["apply_line"] is None else _number(data["apply_line"], "apply_line")

    min_skills = data["min_skills"]
    if isinstance(min_skills, bool) or not isinstance(min_skills, int) or min_skills < 1:
        raise ValueError(f"weights config: min_skills must be a whole number >= 1, got {min_skills!r}")

    canonical = {
        "domain_fit": domain_fit,
        **w,
        "cover_bar": cover_bar,
        "good_to_have_factor": x,
        "apply_line": apply_line,
        "min_skills": min_skills,
        "fit.have": have,
        "fit.adj": adj,
    }
    sha8 = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()[:8]
    return Weights(
        calibration_id=cid,
        domain_fit=domain_fit,
        cover_bar=cover_bar,
        good_to_have_factor=x,
        apply_line=apply_line,
        min_skills=min_skills,
        score_version=f"{cid}:{sha8}",
        **w,
    )


# ─────────────────────────────────────────────────────────────────────────────
# the formula
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ScoreResult:
    score: float
    domain_fit: float
    coverage: float
    gap_distance: float


def skill_weight(requirement: str, weights: Weights) -> float:
    if requirement == "required":
        return 1.0
    if requirement == "good_to_have":
        return weights.good_to_have_factor
    raise ValueError(f"unknown requirement {requirement!r}; must be one of {REQUIREMENTS}")


def coverage(skills: Iterable[tuple[float, str]], weights: Weights) -> float:
    """Weighted share of skills whose fit clears ``cover_bar``. 0 when there are no skills."""
    total = cleared = 0.0
    for fit, requirement in skills:
        w = skill_weight(requirement, weights)
        total += w
        if fit >= weights.cover_bar - _EPS:
            cleared += w
    return cleared / total if total else 0.0


def gap_distance(skills: Iterable[tuple[float, str]], weights: Weights) -> float:
    """Euclidean size of the shortfalls (only skills under ``cover_bar``). 0 when there are none."""
    total = 0.0
    for fit, requirement in skills:
        if fit < weights.cover_bar - _EPS:
            total += (skill_weight(requirement, weights) * (1.0 - fit)) ** 2
    return math.sqrt(total)


def domain_value(bucket: str | None, weights: Weights) -> float:
    """The owner's step value for the bucket. ``None`` (a thin-JD skip) gives 0."""
    if bucket is None:
        return 0.0
    if bucket not in weights.domain_fit:
        raise ValueError(f"unknown domain bucket {bucket!r}; must be one of {BUCKETS}")
    return weights.domain_fit[bucket]


def score_job(skills: Iterable[tuple[float, str]], bucket: str | None, weights: Weights) -> ScoreResult:
    """Score one job from its ``(fit, requirement)`` pairs and its domain bucket."""
    skills = list(skills)
    d, c, g = domain_value(bucket, weights), coverage(skills, weights), gap_distance(skills, weights)
    score = weights.w_domain * d + weights.w_skills * c - weights.w_gap * g
    return ScoreResult(score=score, domain_fit=d, coverage=c, gap_distance=g)


def review_reasons(job: Mapping, n_skills: int, min_skills: int) -> list[str]:
    """Why a job goes to the review list instead of the ranked list (D2). Empty = no reason.
    ``few_skills``: with fewer than ``min_skills`` skills, coverage says too little to rank on."""
    reasons: list[str] = []
    if job.get("legitimacy") in ("suspect", "likely_ghost"):
        reasons.append(job["legitimacy"])
    if job.get("extraction_confidence") == "low":
        reasons.append("low_confidence")
    if n_skills < min_skills:
        reasons.append("few_skills")
    return reasons
