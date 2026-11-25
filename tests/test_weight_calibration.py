"""Tests for the Step 7 weight calibration (hiredra/scoring/calibrate.py, plan unit S7.3)."""

from __future__ import annotations

import csv
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from hiredra.graph.config import load_graph_config
from hiredra.scoring import calibrate as K
from hiredra.scoring.score import DEFAULT_WEIGHTS_PATH, Weights, load_weights, score_job
from hiredra.storage import connect, init_db, set_extraction, set_score, set_status, upsert_job

BASE = Weights(
    calibration_id="t",
    domain_fit={"agent_llm": 1.0, "backend": 0.5, "data": 0.5, "ml": 0.5, "generic_swe": 0.0, "other": 0.0},
    w_domain=1.0,
    w_skills=1.0,
    w_gap=0.5,
    cover_bar=0.68,
    good_to_have_factor=0.4,
    apply_line=None,
    min_skills=3,
    score_version="t:0",
)
HAVE, ADJ = 0.81, 0.55


# --- the metric -----------------------------------------------------------------------------------------------
def test_agreement_counts_concordant_discordant_and_tied_pairs():
    assert K.agreement([3, 2, 1], ["apply", "maybe", "skip"]) == (1.0, 3)
    assert K.agreement([1, 2, 3], ["apply", "maybe", "skip"]) == (0.0, 3)
    assert K.agreement([1, 1], ["apply", "skip"]) == (0.5, 1)  # the score cannot tell them apart
    assert K.agreement([5, 1, 3], ["maybe", "maybe", "skip"]) == (0.5, 2)  # the maybe/maybe pair is not counted
    assert K.agreement([1, 2], ["skip", "skip"]) == (None, 0)


def test_fit_follows_the_graph_rule():
    assert K.fit(0.81, HAVE, ADJ) == 1.0 and K.fit(0.9, HAVE, ADJ) == 1.0
    assert K.fit(0.6, HAVE, ADJ) == 0.6
    assert K.fit(0.5499, HAVE, ADJ) == 0.0
    assert K.fit(1.0, 1.0, 0.615) == 1.0 and K.fit(0.99, 1.0, 0.615) == 0.99  # full credit by name only


def test_cover_bars_sit_strictly_inside_with_the_midpoint_in_the_middle():
    bars = K.cover_bars(ADJ, HAVE)
    assert len(bars) == 5 and all(ADJ < b < HAVE for b in bars)
    assert bars[2] == pytest.approx((ADJ + HAVE) / 2)


def test_the_grid_is_the_planned_6300_and_holds_the_current_weights():
    combos = K.grid(ADJ, HAVE)
    assert len(combos) == 6300 == len(set(combos))
    assert K.default_combo(BASE) in combos


# --- the choice -----------------------------------------------------------------------------------------------
def test_choose_takes_the_smallest_move_within_a_hundredth_of_the_best():
    default = K.default_combo(BASE)
    near = K.Combo(1.0, 0.0, 0.68, 0.4, 0.5, 0.5, 0.0)  # one knob changed, 0.005 below the best
    far = K.Combo(2.0, 1.0, 0.68, 0.6, 1.0, 0.25, 0.25)  # five knobs changed, the best
    worse = K.Combo(1.0, 0.5, 0.68, 0.4, 0.5, 0.5, 0.25)  # one knob changed, 0.05 below
    assert K.choose([(far, 0.90), (near, 0.895), (worse, 0.85)], default) == (near, 0.895)
    assert K.choose([(far, 0.90), (worse, 0.85)], default) == (far, 0.90)


def test_apply_line_is_an_apply_score_and_ties_go_to_the_higher_line():
    assert K.apply_line([3.0, 2.0, 1.0, 0.5], ["apply", "apply", "skip", "skip"]) == (2.0, 1.0)
    # lines 1.0 and 3.0 both reach balanced accuracy 0.75: the higher one is kept
    assert K.apply_line([1.0, 3.0, 0.5, 2.0], ["apply", "apply", "skip", "skip"]) == (3.0, 0.75)
    assert K.apply_line([1.0, 2.0], ["apply", "maybe"]) == (None, None)


def test_bootstrap_is_seeded_and_bounded():
    scores, ratings = [5, 4, 3, 2, 1, 0], ["apply", "maybe", "apply", "skip", "maybe", "skip"]
    a = K.bootstrap(scores, ratings, draws=300)
    assert a == K.bootstrap(scores, ratings, draws=300)
    assert 0.0 <= a[0] <= a[1] <= 1.0


# --- the run --------------------------------------------------------------------------------------------------
def _job(jid, split, rating, bucket, sims):
    return K.RatedJob(jid, split, rating, "why", bucket, tuple((s, "required") for s in sims))


JOBS = [
    _job("a1", "tune", "apply", "agent_llm", [0.95, 0.9, 0.85, 0.7]),
    _job("a2", "tune", "apply", "backend", [0.95, 0.9, 0.9, 0.6]),
    _job("m1", "tune", "maybe", "backend", [0.9, 0.7, 0.5, 0.4]),
    _job("s1", "tune", "skip", "generic_swe", [0.5, 0.45, 0.6, 0.3]),
    _job("s2", "tune", "skip", "other", [0.4, 0.4, 0.4, 0.4]),
    _job("h1", "holdout", "apply", "agent_llm", [0.95, 0.9, 0.7]),
    _job("h2", "holdout", "skip", "data", [0.4, 0.5, 0.3]),
    _job("h3", "holdout", "maybe", "backend", [0.85, 0.6, 0.5]),
]


def test_calibrate_scores_with_the_real_formula_and_reports_the_holdout_once():
    cal = K.calibrate(JOBS, BASE, HAVE, ADJ)
    assert cal.chosen in K.grid(ADJ, HAVE)
    w = cal.chosen.weights(BASE)
    direct = [score_job([(K.fit(s, HAVE, ADJ), r) for s, r in j.skills], j.bucket, w).score for j in cal.holdout]
    assert cal.holdout_scores == direct
    assert cal.tune_a == 1.0 and cal.holdout_a == 1.0  # the toy data orders cleanly
    assert cal.apply_line is not None and cal.holdout_interval is not None
    assert "tune A" in K.summary(cal)


def test_calibrate_refuses_a_holdout_without_a_differing_pair():
    same = [j for j in JOBS if j.split == "tune"] + [_job("h", "holdout", "maybe", "ml", [0.9]) for _ in range(3)]
    with pytest.raises(K.CalibrationError):
        K.calibrate(same, BASE, HAVE, ADJ)


def test_write_weights_keeps_the_comments_and_loads_back(tmp_path):
    path = tmp_path / "weights.yml"
    shutil.copy(DEFAULT_WEIGHTS_PATH, path)
    before = path.read_text(encoding="utf-8")
    cal = K.calibrate(JOBS, BASE, HAVE, ADJ)
    cid = K.write_weights(cal, path)
    text = path.read_text(encoding="utf-8")
    assert cid.startswith("step7-") and f"calibration_id: {cid}" in text
    assert [ln for ln in before.splitlines() if ln.startswith("#")] == [
        ln for ln in text.splitlines() if ln.startswith("#")
    ]
    w = load_weights(replace(load_graph_config(), have=HAVE, adj=ADJ), path)
    assert w.calibration_id == cid and w.w_gap == cal.chosen.w_gap and w.apply_line == pytest.approx(cal.apply_line)
    assert w.domain_fit["backend"] == cal.chosen.backend and w.domain_fit["generic_swe"] == cal.chosen.generic


def test_load_rated_leaves_out_jobs_the_filters_have_since_dropped(pg_url, tmp_path):
    conn = connect(pg_url)
    init_db(conn)
    for i, status in enumerate(("scored", "scored", "dropped")):
        jid = f"greenhouse:{i}"
        upsert_job(conn, {"job_id": jid, "url": f"u{i}", "title": "Engineer", "company": "Acme", "source": "greenhouse",
                          "description": "x"})  # fmt: skip
        set_extraction(conn, jid, skills=[{"name": "python", "requirement": "required"}], domain_bucket="backend",
                       legitimacy="legitimate", extraction_confidence="high")  # fmt: skip
        set_score(conn, jid, score=1.0, features={"domain_bucket": "backend",
                  "skills": [{"skill": "python", "req": "required", "s": 0.9}]}, review_reason=None,
                  score_version="v", embed_tag="t", skills_hash="h", profile_hash="p")  # fmt: skip
        set_status(conn, jid, status, detail="language: description not in English" if status == "dropped" else None)
    ratings = tmp_path / "r.csv"
    with ratings.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["job_id", "split", "rating", "reason"])
        w.writerows([["greenhouse:0", "tune", "maybe", "old"], ["greenhouse:0", "tune", "apply", "new"],
                     ["greenhouse:1", "holdout", "skip", "no"], ["greenhouse:2", "tune", "skip", "german"]])  # fmt: skip
    jobs, left_out = K.load_rated(conn, ratings)
    assert [(j.job_id, j.rating) for j in jobs] == [("greenhouse:0", "apply"), ("greenhouse:1", "skip")]  # latest wins
    assert jobs[0].skills == ((0.9, "required"),) and jobs[0].bucket == "backend"
    assert len(left_out) == 1 and "greenhouse:2" in left_out[0] and "language" in left_out[0]
    conn.close()


# --- the frozen calibration (local only: the snapshot is private data) ------------------------------------------
@pytest.mark.skipif(not K.SNAPSHOT_PATH.exists(), reason="no Step 7 snapshot (private data, not in CI)")
def test_the_frozen_holdout_agreement_is_reproduced_from_the_snapshot():
    import json

    snap = json.loads(Path(K.SNAPSHOT_PATH).read_text(encoding="utf-8"))
    combo = K.Combo(**snap["chosen"])
    jobs = [K.RatedJob(f"h{i}", "holdout", h["rating"], "", h["bucket"], tuple(map(tuple, h["skills"])))
            for i, h in enumerate(snap["holdout"])]  # fmt: skip
    scores = K.score_all(jobs, combo, BASE, snap["have"], snap["adj"])
    assert K.agreement(scores, [j.rating for j in jobs])[0] == pytest.approx(snap["holdout_a"])
