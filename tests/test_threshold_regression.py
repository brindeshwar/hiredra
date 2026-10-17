"""Regression guard for the Step 5 calibration (plan unit S5.1).

Once ``calibrate --write`` has run, ``config/graph.yml`` carries a ``fit.calibration_id`` and
``calibration/`` holds the labeled pairs and the cosine snapshot. Re-running the sweep on that
snapshot must give back exactly the thresholds in the config: if the sweep code, the pair file
or the snapshot drifts, this fails. Before the first calibration there is nothing to guard.
"""

from __future__ import annotations

import pytest

from hiredra.graph import calibrate as cal
from hiredra.graph.config import load_graph_config


def test_config_thresholds_match_the_calibration_snapshot():
    cfg = load_graph_config()
    if cfg.calibration_id is None:
        pytest.skip("no Step 5 calibration yet (fit.calibration_id is unset)")
    assert cal.DEFAULT_PAIRS_PATH.exists(), "calibrated config but calibration/skill_pairs.csv is missing"
    assert cal.DEFAULT_COSINES_PATH.exists(), "calibrated config but calibration/step5_cosines.csv is missing"

    pairs = cal.load_pairs(cal.DEFAULT_PAIRS_PATH)
    cosines = cal._read_cached(cal.DEFAULT_COSINES_PATH, cfg.embed_tag)
    missing = [p for p in pairs if p.key not in cosines]
    assert not missing, f"{len(missing)} pair(s) have no cosine in the snapshot under {cfg.embed_tag}"

    result = cal.calibrate(pairs, cosines)
    assert (result.have, result.adj) == (cfg.have, cfg.adj)
    assert not result.broken
