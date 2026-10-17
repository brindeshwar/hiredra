"""Tests for hiredra.scoring.score (plan unit S6.1): the worked example, the properties the
formula must keep, and every rule of the strict weights loader."""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest
import yaml

from hiredra.graph.config import GraphConfig
from hiredra.scoring import score as sc

REPO = Path(__file__).resolve().parents[1]


def _graph(have=0.90, adj=0.70) -> GraphConfig:
    return GraphConfig(
        model="gemini-embedding-2",
        dimensions=128,
        prefix="",
        inputs_per_request=50,
        requests_per_minute=100,
        dedup_threshold=0.9,
        have=have,
        adj=adj,
    )


_BASE = {
    "calibration_id": "placeholder",
    "domain_fit": {"agent_llm": 1.0, "backend": 0.5, "data": 0.5, "ml": 0.5, "generic_swe": 0.0, "other": 0.0},
    "w_domain": 1.0,
    "w_skills": 1.0,
    "w_gap": 0.5,
    "cover_bar": None,
    "good_to_have_factor": 0.4,
    "apply_line": None,
}


def _weights(tmp_path, graph=None, **over) -> sc.Weights:
    data = {**_BASE, **over}
    p = tmp_path / "weights.yml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return sc.load_weights(graph or _graph(), p)


# ── the worked example (plan S6.1 accept) ───────────────────────────────────
def test_worked_example(tmp_path):
    w = _weights(tmp_path, cover_bar=0.85)
    skills = [(1.0, "required"), (1.0, "required"), (1.0, "required"), (0.82, "required"), (0.0, "required")]
    assert sc.coverage(skills, w) == pytest.approx(0.60)
    assert sc.gap_distance(skills, w) == pytest.approx(math.sqrt(0.18**2 + 1.0**2))
    assert sc.gap_distance(skills, w) == pytest.approx(1.016, abs=1e-3)
    r = sc.score_job(skills, "backend", w)
    assert r.domain_fit == 0.5
    assert r.score == pytest.approx(1.0 * 0.5 + 1.0 * 0.60 - 0.5 * math.sqrt(0.18**2 + 1.0))


def test_good_to_have_weighs_x_in_coverage_and_gap(tmp_path):
    w = _weights(tmp_path, cover_bar=0.85, good_to_have_factor=0.4)
    skills = [(1.0, "required"), (0.0, "good_to_have")]
    assert sc.coverage(skills, w) == pytest.approx(1.0 / 1.4)
    assert sc.gap_distance(skills, w) == pytest.approx(0.4)
    # only good-to-have skills (total weight under 1): coverage is still a share, 1.0 when all clear
    assert sc.coverage([(1.0, "good_to_have"), (0.95, "good_to_have")], w) == 1.0
    assert sc.coverage([(1.0, "good_to_have"), (0.0, "good_to_have")], w) == pytest.approx(0.5)


def test_empty_cases_are_defined(tmp_path):
    w = _weights(tmp_path)
    r = sc.score_job([], None, w)
    assert (r.coverage, r.gap_distance, r.domain_fit, r.score) == (0.0, 0.0, 0.0, 0.0)


def test_a_fit_exactly_on_the_bar_is_covered_and_not_a_gap(tmp_path):
    w = _weights(tmp_path, cover_bar=0.85)
    assert sc.coverage([(0.85, "required")], w) == 1.0
    assert sc.gap_distance([(0.85, "required")], w) == 0.0
    assert sc.coverage([(0.8499999, "required")], w) == 1.0  # float32 noise under the bar still counts


def test_weights_apply_linearly(tmp_path):
    w = _weights(tmp_path, cover_bar=0.85, w_domain=2.0, w_skills=3.0, w_gap=0.25)
    r = sc.score_job([(1.0, "required"), (0.5, "required")], "agent_llm", w)
    assert r.score == pytest.approx(2.0 * 1.0 + 3.0 * 0.5 - 0.25 * 0.5)


def test_unknown_bucket_and_requirement_are_errors(tmp_path):
    w = _weights(tmp_path)
    with pytest.raises(ValueError, match="unknown domain bucket"):
        sc.score_job([], "frontend", w)
    with pytest.raises(ValueError, match="unknown requirement"):
        sc.score_job([(1.0, "nice")], "backend", w)


@pytest.mark.parametrize(
    ("job", "reasons"),
    [
        ({"legitimacy": "legitimate", "extraction_confidence": "high"}, []),
        ({"legitimacy": "suspect", "extraction_confidence": "high"}, ["suspect"]),
        ({"legitimacy": "likely_ghost", "extraction_confidence": "low"}, ["likely_ghost", "low_confidence"]),
        ({"legitimacy": None, "extraction_confidence": "low"}, ["low_confidence"]),
    ],
)
def test_review_reasons(job, reasons):
    assert sc.review_reasons(job) == reasons


def test_legitimacy_never_changes_the_score(tmp_path):
    w = _weights(tmp_path)
    skills = [(0.95, "required"), (0.3, "good_to_have")]
    assert sc.score_job(skills, "backend", w) == sc.score_job(list(skills), "backend", w)
    assert "legitimacy" not in sc.ScoreResult.__dataclass_fields__


# ── properties (randomized, seeded, so they are reproducible) ───────────────
def _random_skills(rng, n):
    return [
        (rng.choice([0.0, 1.0, rng.uniform(0.70, 0.90), rng.random()]), rng.choice(sc.REQUIREMENTS)) for _ in range(n)
    ]


@pytest.mark.parametrize("seed", range(25))
def test_raising_any_single_fit_never_lowers_the_score(tmp_path, seed):
    rng = random.Random(seed)
    w = _weights(tmp_path, w_domain=rng.uniform(0, 2), w_skills=rng.uniform(0, 2), w_gap=rng.uniform(0, 2))
    skills = _random_skills(rng, rng.randint(1, 12))
    bucket = rng.choice(sc.BUCKETS)
    base = sc.score_job(skills, bucket, w).score
    for i, (fit, req) in enumerate(skills):
        for higher in (fit + (1 - fit) * rng.random(), 1.0, min(1.0, fit + 0.01)):
            raised = skills[:i] + [(higher, req)] + skills[i + 1 :]
            assert sc.score_job(raised, bucket, w).score >= base - 1e-12


@pytest.mark.parametrize("seed", range(25))
def test_required_to_good_to_have_never_increases_that_skills_penalty(tmp_path, seed):
    rng = random.Random(seed)
    w = _weights(tmp_path, good_to_have_factor=rng.uniform(0.01, 1.0))
    skills = _random_skills(rng, rng.randint(1, 12))
    for i, (fit, req) in enumerate(skills):
        if req != "required":
            continue
        softened = skills[:i] + [(fit, "good_to_have")] + skills[i + 1 :]
        assert sc.gap_distance(softened, w) <= sc.gap_distance(skills, w) + 1e-12


def test_coverage_and_gap_stay_in_range(tmp_path):
    rng = random.Random(7)
    w = _weights(tmp_path)
    for _ in range(200):
        skills = _random_skills(rng, rng.randint(0, 15))
        assert 0.0 <= sc.coverage(skills, w) <= 1.0
        assert 0.0 <= sc.gap_distance(skills, w) <= math.sqrt(len(skills))


# ── the strict loader ───────────────────────────────────────────────────────
def test_the_shipped_config_loads_against_the_shipped_graph_config():
    from hiredra.graph.config import load_graph_config

    g = load_graph_config()
    w = sc.load_weights(g)
    assert w.cover_bar == pytest.approx((g.adj + g.have) / 2)
    assert w.apply_line is None and w.calibration_id == "placeholder"
    assert w.score_version.startswith("placeholder:") and len(w.score_version.split(":")[1]) == 8


def test_null_cover_bar_is_the_midpoint(tmp_path):
    assert _weights(tmp_path, graph=_graph(have=0.88, adj=0.66)).cover_bar == pytest.approx(0.77)


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"extra": 1}, "unknown key"),
        ({"calibration_id": ""}, "calibration_id must be a non-empty string"),
        ({"domain_fit": {"agent_llm": 1.0, "backend": 0.5}}, "exactly the buckets"),
        ({"domain_fit": {**_BASE["domain_fit"], "frontend": 0.5}}, "exactly the buckets"),
        ({"domain_fit": {**_BASE["domain_fit"], "backend": 1.5}}, r"domain_fit.backend must be in \[0, 1\]"),
        ({"domain_fit": {**_BASE["domain_fit"], "data": -0.1}}, r"domain_fit.data must be in \[0, 1\]"),
        ({"domain_fit": {**_BASE["domain_fit"], "agent_llm": 0.9}}, "domain_fit.agent_llm is fixed at 1.0"),
        ({"domain_fit": {**_BASE["domain_fit"], "other": 0.25}}, "domain_fit.other is fixed at 0.0"),
        ({"domain_fit": {**_BASE["domain_fit"], "ml": True}}, "domain_fit.ml must be a number"),
        ({"w_domain": -0.1}, "w_domain must be >= 0"),
        ({"w_skills": -1}, "w_skills must be >= 0"),
        ({"w_gap": "high"}, "w_gap must be a number"),
        ({"good_to_have_factor": 0}, r"good_to_have_factor must be in \(0, 1\]"),
        ({"good_to_have_factor": 1.2}, r"good_to_have_factor must be in \(0, 1\]"),
        ({"cover_bar": 0.70}, "strictly between"),
        ({"cover_bar": 0.90}, "strictly between"),
        ({"cover_bar": 0.65}, "strictly between"),
        ({"apply_line": "high"}, "apply_line must be a number"),
    ],
)
def test_loader_rejects(tmp_path, over, message):
    with pytest.raises(ValueError, match=message):
        _weights(tmp_path, **over)


def test_loader_rejects_a_missing_key_and_a_non_mapping(tmp_path):
    p = tmp_path / "w.yml"
    p.write_text(yaml.safe_dump({k: v for k, v in _BASE.items() if k != "w_gap"}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing key\(s\) \['w_gap'\]"):
        sc.load_weights(_graph(), p)
    p.write_text("- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        sc.load_weights(_graph(), p)


def test_edge_values_that_are_allowed(tmp_path):
    w = _weights(tmp_path, w_domain=0, w_skills=0, w_gap=0, good_to_have_factor=1.0, apply_line=0.25, cover_bar=0.71)
    assert (w.w_domain, w.good_to_have_factor, w.apply_line, w.cover_bar) == (0.0, 1.0, 0.25, 0.71)
    assert _weights(tmp_path, domain_fit={**_BASE["domain_fit"], "backend": 1.0}).domain_fit["backend"] == 1.0


def test_score_version_changes_with_any_value_and_with_have_adj(tmp_path):
    base = _weights(tmp_path).score_version
    assert _weights(tmp_path).score_version == base  # deterministic
    for over in ({"w_gap": 0.6}, {"good_to_have_factor": 0.5}, {"apply_line": 0.3}, {"cover_bar": 0.81}):
        assert _weights(tmp_path, **over).score_version != base, over
    assert _weights(tmp_path, domain_fit={**_BASE["domain_fit"], "ml": 0.25}).score_version != base
    assert _weights(tmp_path, graph=_graph(have=0.91)).score_version != base
    assert _weights(tmp_path, graph=_graph(adj=0.69)).score_version != base
    # with cover_bar pinned, HAVE and ADJ still change the version on their own
    pinned = _weights(tmp_path, cover_bar=0.80).score_version
    assert _weights(tmp_path, graph=_graph(have=0.91), cover_bar=0.80).score_version != pinned
    assert _weights(tmp_path, graph=_graph(adj=0.69), cover_bar=0.80).score_version != pinned
    assert _weights(tmp_path, calibration_id="step7-x").score_version.startswith("step7-x:")
