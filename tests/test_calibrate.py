"""Tests for hiredra.graph.calibrate (plan unit S5.1): the pair file, the blind sampler, cosines,
the sweep and its gates, the config write-back, and the CLI. Vectors sit at exact cosines, so
every expected threshold is known in advance."""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from hiredra.graph import calibrate as cal
from hiredra.graph.__main__ import main as graph_main
from hiredra.graph.config import GraphConfig, load_graph_config
from hiredra.graph.index import save_vectors
from hiredra.quota import DailyQuotaExceeded
from hiredra.storage import add_market_skill, connect, init_db, set_extraction, set_status, upsert_job

DIM = 128
REPO = Path(__file__).resolve().parents[1]


def _cfg(**over) -> GraphConfig:
    base = dict(
        model="gemini-embedding-2",
        dimensions=DIM,
        prefix="",
        inputs_per_request=50,
        dedup_threshold=0.9,
        have=0.9,
        adj=0.7,
    )
    base.update(over)
    return GraphConfig(**base)


def _basis(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _tilted(i: int, cosine: float) -> np.ndarray:
    """Unit vector with exactly ``cosine`` against e_i, tilted toward e_(i+1)."""
    v = np.zeros(DIM, dtype=np.float32)
    v[i], v[i + 1] = cosine, math.sqrt(max(0.0, 1.0 - cosine**2))
    return v


@pytest.fixture()
def conn(pg_url):
    c = connect(pg_url)
    init_db(c)
    yield c
    c.close()


def _node(conn, cfg, name, vec, *, mine=False):
    add_market_skill(conn, name)
    save_vectors(conn, [name], np.asarray([vec], dtype=np.float32), cfg.embed_tag)
    if mine:
        conn.execute(
            "UPDATE skill_vocab SET mine = true, source = 'profile', level = 'production' WHERE name = %s", (name,)
        )
    conn.commit()


_HEAD = "# same = ...\n# related = ...\n# unrelated = ...\na,b,label,split,source,note\n"


def _write(tmp_path, body: str, name="pairs.csv") -> Path:
    p = tmp_path / name
    p.write_text(_HEAD + body, encoding="utf-8")
    return p


def _full_file(tmp_path, cosine_of=None) -> tuple[Path, dict]:
    """A file that passes the count gate: per label 10 tune + 5 holdout. Returns the path and the
    cosine each pair should get: same 0.92-0.96, related 0.75-0.85, unrelated 0.40-0.55."""
    bands = {"same": (0.92, 0.96), "related": (0.75, 0.85), "unrelated": (0.40, 0.55)}
    rows, cosines = [], {}
    for label, (lo, hi) in bands.items():
        for i in range(15):
            split = "tune" if i < 10 else "holdout"
            a, b = f"{label}-a{i}", f"{label}-b{i}"
            c = lo + (hi - lo) * i / 14
            cosines[(a, b)] = c if cosine_of is None else cosine_of(label, i)
            rows.append(f"{a},{b},{label},{split},{'trap' if i % 2 else 'real'},")
    return _write(tmp_path, "\n".join(rows) + "\n"), cosines


# ── the pair file ───────────────────────────────────────────────────────────
def test_load_pairs_reads_comments_header_and_folds_names(tmp_path):
    p = _write(tmp_path, "OAuth, oauth2 ,same,tune,trap,version variant\nsql,sql server,related,holdout,trap,\n")
    pairs = cal.load_pairs(p, enforce_counts=False)
    assert [(x.a, x.b, x.label, x.split, x.source, x.note, x.line) for x in pairs] == [
        ("oauth", "oauth2", "same", "tune", "trap", "version variant", 5),
        ("sql", "sql server", "related", "holdout", "trap", "", 6),
    ]


def test_load_pairs_lists_every_problem_with_its_line(tmp_path):
    p = _write(
        tmp_path,
        "a,b,same,,trap,\n"  # line 5: no split
        "a,c,friends,tune,trap,\n"  # line 6: bad label
        "React, react ,same,tune,real,\n"  # line 7: same after folding
        "x,y,same,tune,real,\n"  # line 8
        "Y,X,related,holdout,trap,\n"  # line 9: duplicate of 8, unordered
        "p,q,same,tune,guess,\n"  # line 10: bad source
        "too,few\n",  # line 11
    )
    with pytest.raises(cal.PairFileError) as info:
        cal.load_pairs(p, enforce_counts=False)
    msg = str(info.value)
    for expected in (
        "line 5: no split",
        "line 6: label 'friends'",
        "line 7: a and b are the same skill",
        "line 9: duplicate of line 8",
        "line 10: source 'guess'",
        "line 11: expected 5 or 6 columns",
    ):
        assert expected in msg


def test_load_pairs_rejects_a_wrong_header(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b,label,source,split,note\nx,y,same,trap,tune,\n", encoding="utf-8")
    with pytest.raises(cal.PairFileError, match="header must be"):
        cal.load_pairs(p, enforce_counts=False)


def test_the_count_gate_refuses_with_the_shortfall_per_label(tmp_path):
    rows = [f"s{i},t{i},same,tune,real," for i in range(10)] + [f"s{i},u{i},same,holdout,real," for i in range(5)]
    rows += [f"r{i},v{i},related,tune,real," for i in range(9)] + [f"r{i},w{i},related,holdout,real," for i in range(5)]
    rows += [f"n{i},x{i},unrelated,tune,real," for i in range(10)] + [
        f"n{i},y{i},unrelated,holdout,real," for i in range(4)
    ]
    p = _write(tmp_path, "\n".join(rows) + "\n")
    with pytest.raises(cal.PairFileError) as info:
        cal.load_pairs(p)
    msg = str(info.value)
    assert "related/tune: 9 row(s), need at least 10 (1 short)" in msg
    assert "unrelated/holdout: 4 row(s), need at least 5 (1 short)" in msg
    assert "same/" not in msg


def test_exactly_the_minimum_counts_pass(tmp_path):
    p, _ = _full_file(tmp_path)
    assert len(cal.load_pairs(p)) == 45


# ── confusion, sweep, plateau ───────────────────────────────────────────────
def test_confusion_and_f1_math():
    c = cal.confusion([(0.9, True), (0.8, True), (0.7, False), (0.6, True), (0.95, False)], 0.75)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 1)
    assert c.precision == pytest.approx(2 / 3) and c.recall == pytest.approx(2 / 3)
    assert c.f1 == pytest.approx(2 / 3)
    assert cal.Confusion(0, 0, 0, 3).f1 == 0.0
    lopsided = cal.Confusion(tp=1, fp=0, fn=3, tn=0)  # precision 1.0, recall 0.25
    assert lopsided.f1 == pytest.approx(0.4)  # the harmonic mean, not the average (0.625)


def test_a_cosine_exactly_on_the_threshold_counts_as_positive():
    assert cal.confusion([(np.float32(0.8), True)], 0.80).tp == 1


def test_plateau_centre_of_a_clean_separation():
    points = [(0.90, True), (0.92, True), (0.95, True), (0.50, False), (0.60, False), (0.70, False)]
    rows = cal.sweep(points)
    assert [t for t, _ in rows][:2] == [0.30, 0.31] and rows[-1][0] == 0.99 and len(rows) == 70
    # F1 = 1.0 for every threshold in 0.71..0.90 -> centre 0.81 (half-up on 0.805)
    assert cal.plateau_centre(rows) == (0.81, 0.71, 0.90)


def test_plateau_takes_the_contiguous_run_around_the_best_not_a_far_tie():
    rows = [(i / 100, cal.Confusion(0, 0, 0, 0)) for i in cal.SWEEP]

    def f1(value):  # a Confusion with this exact F1 (precision = recall = value)
        tp = round(value * 1000)
        return cal.Confusion(tp, 1000 - tp, 1000 - tp, 0)

    rows = [(t, f1(0.5)) for t, _ in rows]
    rows[9] = (rows[9][0], f1(0.895))  # 0.39: inside 0.01 of the best, below it
    rows[10] = (rows[10][0], f1(0.90))  # 0.40: the best
    rows[11] = (rows[11][0], f1(0.895))  # 0.41: inside 0.01, above it -> plateau 0.39-0.41
    rows[50] = (rows[50][0], f1(0.90))  # 0.80: a tie far away, not contiguous
    assert cal.plateau_centre(rows) == (0.40, 0.39, 0.41)


def test_the_sweep_is_deterministic(tmp_path):
    p, cos = _full_file(tmp_path)
    pairs = cal.load_pairs(p)
    first = cal.calibrate(pairs, cos)
    assert cal.calibrate(list(reversed(pairs)), dict(reversed(list(cos.items())))).__dict__ == first.__dict__


def test_calibrate_finds_both_boundaries_and_scores_the_holdout(tmp_path):
    p, cos = _full_file(tmp_path)
    r = cal.calibrate(cal.load_pairs(p), cos)
    # TUNE rows only (i < 10): same 0.920-0.946, related 0.750-0.814, unrelated 0.400-0.496
    # -> HAVE plateau 0.82..0.92 (centre 0.87); ADJ plateau 0.50..0.75 (centre 0.63)
    assert r.plateaus["have"] == (0.87, 0.82, 0.92)
    assert r.plateaus["adj"] == (0.63, 0.50, 0.75)
    assert (r.have, r.adj) == (0.87, 0.63)
    assert r.holdout["have"]["all"].f1 == 1.0 and r.holdout["adj"]["all"].f1 == 1.0
    assert sum(c.tp + c.fp + c.fn + c.tn for c in [r.holdout["have"]["trap"], r.holdout["have"]["real"]]) == 15
    assert r.flags == [] and not r.broken


def test_tune_only_drives_the_pick(tmp_path):
    # wreck the HOLDOUT cosines: the thresholds must not move, only the holdout score
    p, cos = _full_file(tmp_path)
    pairs = cal.load_pairs(p)
    bad = {k: (0.1 if any(x.key == k and x.split == "holdout" for x in pairs) else v) for k, v in cos.items()}
    clean, wrecked = cal.calibrate(pairs, cos), cal.calibrate(pairs, bad)
    assert (wrecked.have, wrecked.adj) == (clean.have, clean.adj)
    assert wrecked.holdout["have"]["all"].f1 < 1.0


def test_gate_have_not_above_adj_is_broken(tmp_path):
    # related looks like same -> both boundaries land in the same place
    p, cos = _full_file(tmp_path, cosine_of=lambda label, i: {"same": 0.95, "related": 0.95, "unrelated": 0.30}[label])
    r = cal.calibrate(cal.load_pairs(p), cos)
    assert r.broken and any(f.startswith("STOP: HAVE") for f in r.flags)


def test_gate_small_gap_asks_for_a_check_in(tmp_path):
    p, cos = _full_file(
        tmp_path, cosine_of=lambda label, i: {"same": 0.80 + i * 0.001, "related": 0.77, "unrelated": 0.70}[label]
    )
    r = cal.calibrate(cal.load_pairs(p), cos)
    assert not r.broken and any(f.startswith("CHECK-IN") for f in r.flags)


def test_gate_low_holdout_f1_triggers_s5_3(tmp_path):
    p, cos = _full_file(tmp_path)
    pairs = cal.load_pairs(p)
    # 2 of the 5 'same' holdout pairs fall to 0.45: HAVE holdout precision 1.0, recall 0.6 -> F1 0.75
    dropped = [x.key for x in pairs if x.split == "holdout" and x.label == "same"][:2]
    r = cal.calibrate(pairs, {k: (0.45 if k in dropped else v) for k, v in cos.items()})
    assert r.holdout["have"]["all"].f1 == pytest.approx(0.75)
    assert any(f.startswith("S5.3: holdout F1 for HAVE is 0.75") for f in r.flags)


# ── cosines ─────────────────────────────────────────────────────────────────
def test_pair_cosines_from_nodes_aliases_and_fresh_embeddings(conn, tmp_path):
    cfg = _cfg()
    _node(conn, cfg, "kubernetes", _basis(0))
    _node(conn, cfg, "docker", _tilted(0, 0.755))
    conn.execute("INSERT INTO skill_alias (alias, canonical, created_at) VALUES ('k8s', 'kubernetes', now())")
    conn.commit()
    p = _write(tmp_path, "k8s,docker,related,tune,real,\nkubernetes,helm chart,related,holdout,trap,\n")
    pairs = cal.load_pairs(p, enforce_counts=False)
    asked: list[list[str]] = []

    def factory():
        def embed(names):
            asked.append(list(names))
            return np.asarray([_tilted(0, 0.6)], dtype=np.float32)

        return embed

    cache = tmp_path / "cos.csv"
    cos = cal.pair_cosines(conn, cfg, pairs, factory, cache_path=cache)
    assert cos[("docker", "k8s")] == pytest.approx(0.755, abs=1e-6)
    assert cos[("helm chart", "kubernetes")] == pytest.approx(0.6, abs=1e-6)
    assert asked == [["helm chart"]]  # only the name that is not a node

    # a re-run reuses the snapshot: no node and no embedder needed
    conn.execute("DELETE FROM skill_vocab")
    conn.commit()
    assert cal.pair_cosines(conn, cfg, pairs, None, cache_path=cache) == cos
    assert cache.read_text(encoding="utf-8").splitlines()[0] == "a,b,cosine,embed_tag"


def test_node_cosines_are_saved_before_a_budget_stop(conn, tmp_path):
    cfg = _cfg()
    _node(conn, cfg, "kafka", _basis(2))
    _node(conn, cfg, "fastapi", _tilted(2, 0.616))
    p = _write(tmp_path, "kafka,fastapi,unrelated,tune,real,\nkafka,flink sql,related,tune,trap,\n")
    pairs = cal.load_pairs(p, enforce_counts=False)

    def factory():
        def embed(names):
            raise DailyQuotaExceeded("PerDay")

        return embed

    cache = tmp_path / "cos.csv"
    with pytest.raises(DailyQuotaExceeded):
        cal.pair_cosines(conn, cfg, pairs, factory, cache_path=cache)
    assert "fastapi,kafka,0.616000" in cache.read_text(encoding="utf-8")


def test_no_embedder_is_built_when_every_name_is_a_node(conn, tmp_path):
    cfg = _cfg()
    _node(conn, cfg, "kafka", _basis(2))
    _node(conn, cfg, "fastapi", _tilted(2, 0.616))
    pairs = cal.load_pairs(_write(tmp_path, "kafka,fastapi,unrelated,tune,real,\n"), enforce_counts=False)
    cal.pair_cosines(conn, cfg, pairs, lambda: pytest.fail("embedder built"), cache_path=tmp_path / "c.csv")


# ── the blind sampler ───────────────────────────────────────────────────────
def _extracted_job(conn, job_id, skills):
    upsert_job(
        conn, {"job_id": job_id, "url": f"u://{job_id}", "title": "Engineer", "company": "Acme", "source": "greenhouse"}
    )
    set_extraction(conn, job_id, skills=[{"name": s, "requirement": "required"} for s in skills],
                   domain_bucket=None, legitimacy=None, extraction_confidence="high")  # fmt: skip
    set_status(conn, job_id, "filtered")
    set_status(conn, job_id, "extracted")


def _sampler_graph(conn, cfg):
    _node(conn, cfg, "python", _basis(0), mine=True)
    _node(conn, cfg, "postgres", _basis(10), mine=True)
    names = {
        "django": (0, 0.97),
        "flask": (0, 0.85),
        "pandas": (0, 0.72),
        "mysql": (10, 0.88),
        "redis": (10, 0.55),
        "cobol": (10, 0.45),
    }
    for name, (axis, c) in names.items():
        _node(conn, cfg, name, _tilted(axis, c))
    _node(conn, cfg, "fortran", _tilted(20, 0.9))  # nearest mine is far below 0.40 -> never suggested
    _extracted_job(conn, "greenhouse:j1", ["Django", "flask", "pandas", "python"])
    _extracted_job(conn, "greenhouse:j2", ["mysql", "redis", "cobol", "fortran", "not a node"])


def test_suggest_draws_across_bands_and_keeps_the_cosines_private(conn):
    cfg = _cfg()
    _sampler_graph(conn, cfg)
    pairs, sidecar = cal.suggest_pairs(conn, cfg, n=12, seed=1)
    assert pairs == sorted(pairs)
    assert set(pairs) == {
        ("cobol", "postgres"),
        ("django", "python"),
        ("flask", "python"),
        ("mysql", "postgres"),
        ("pandas", "python"),
        ("redis", "postgres"),
    }  # "python" is mine (skipped as a JD skill), fortran is below the bands, "not a node" has no vector
    by_pair = {(p["a"], p["b"]): p for p in sidecar["pairs"]}
    assert by_pair[("django", "python")]["cosine"] == pytest.approx(0.97, abs=1e-5)
    assert by_pair[("django", "python")]["band"] == 5 and by_pair[("cobol", "postgres")]["band"] == 0


def test_suggest_caps_each_band(conn):
    cfg = _cfg()
    _sampler_graph(conn, cfg)
    pairs, sidecar = cal.suggest_pairs(conn, cfg, n=6, seed=1, bands=2, low=0.40)  # bands 0.40-0.70, 0.70-1.00
    # 3 per band asked; the low band has only cobol and redis
    assert len(pairs) == 5 and sorted(p["band"] for p in sidecar["pairs"]) == [0, 0, 1, 1, 1]
    pairs, sidecar = cal.suggest_pairs(conn, cfg, n=2, seed=1, bands=2, low=0.40)
    assert len(pairs) == 2 and sorted(p["band"] for p in sidecar["pairs"]) == [0, 1]


def test_suggest_needs_mine_skills(conn):
    cfg = _cfg()
    _node(conn, cfg, "django", _basis(0))
    with pytest.raises(ValueError, match="no embedded 'mine' skills"):
        cal.suggest_pairs(conn, cfg)


# ── config write-back ───────────────────────────────────────────────────────
def test_write_fit_config_keeps_comments_and_loads_back(tmp_path):
    path = tmp_path / "graph.yml"
    shutil.copy(REPO / "config" / "graph.yml", path)
    before = path.read_text(encoding="utf-8")
    cal.write_fit_config(path, 0.87, 0.66, "step5-2026-11-01")
    after = path.read_text(encoding="utf-8")
    cfg = load_graph_config(path)
    assert (cfg.have, cfg.adj, cfg.calibration_id) == (0.87, 0.66, "step5-2026-11-01")
    assert after.count("#") == before.count("#")  # every comment survived
    assert [ln for ln in after.splitlines() if ln.startswith("embedding:")] == ["embedding:"]

    cal.write_fit_config(path, 0.88, 0.65, "step5-2026-11-02")  # a second calibration replaces, never duplicates
    again = path.read_text(encoding="utf-8")
    assert again.count("calibration_id:") == 1 and "step5-2026-11-02" in again
    assert load_graph_config(path).have == 0.88


def test_the_shipped_config_is_calibrated_and_has_its_report():
    cfg = load_graph_config()
    assert cfg.calibration_id and cfg.calibration_id.startswith("step5-")
    assert (REPO / "calibration" / "reports" / f"{cfg.calibration_id}.md").exists()


# ── CLI ─────────────────────────────────────────────────────────────────────
def _cli_env(tmp_path, pg_url, cfg_cosines=None):
    """A real DB with every pair's two names as nodes at the pair's exact cosine."""
    config = tmp_path / "graph.yml"
    shutil.copy(REPO / "config" / "graph.yml", config)
    cfg = load_graph_config(config)
    pairs_path, cosines = _full_file(tmp_path, cfg_cosines)
    db = pg_url
    c = connect(db)
    init_db(c)
    for i, ((a, b), cosine) in enumerate(cosines.items()):
        v_a = np.zeros(cfg.dimensions, dtype=np.float32)
        v_a[2 * i] = 1.0
        v_b = np.zeros(cfg.dimensions, dtype=np.float32)
        v_b[2 * i], v_b[2 * i + 1] = cosine, math.sqrt(1 - cosine**2)
        for name, v in ((a, v_a), (b, v_b)):
            add_market_skill(c, name)
            save_vectors(c, [name], np.asarray([v]), cfg.embed_tag)
    c.close()
    args = [
        "--db", str(db), "--config", str(config),
        "calibrate", "--pairs", str(pairs_path),
        "--cosines-path", str(tmp_path / "cos.csv"), "--reports-dir", str(tmp_path / "reports"),
    ]  # fmt: skip
    return args, config


def test_cli_dry_run_writes_nothing(tmp_path, capsys, pg_url):
    args, config = _cli_env(tmp_path, pg_url)
    before = config.read_text(encoding="utf-8")
    assert graph_main(args) == 0
    out = capsys.readouterr().out
    assert "HAVE=0.87 ADJ=0.63" in out and "DRY RUN" in out
    assert config.read_text(encoding="utf-8") == before and not (tmp_path / "reports").exists()
    assert (tmp_path / "cos.csv").exists()  # the snapshot is always kept


def test_cli_write_updates_config_and_saves_the_report(tmp_path, pg_url):
    args, config = _cli_env(tmp_path, pg_url)
    assert graph_main([*args, "--write"]) == 0
    cfg = load_graph_config(config)
    assert (cfg.have, cfg.adj) == (0.87, 0.63) and cfg.calibration_id.startswith("step5-")
    report = next((tmp_path / "reports").glob("step5-*.md")).read_text(encoding="utf-8")
    assert "**HAVE = 0.87**" in report and "## Misclassified holdout pairs" in report


def test_cli_broken_result_exits_3_and_writes_nothing(tmp_path, pg_url):
    args, config = _cli_env(
        tmp_path, pg_url, lambda label, i: {"same": 0.95, "related": 0.95, "unrelated": 0.30}[label]
    )
    before = config.read_text(encoding="utf-8")
    assert graph_main([*args, "--write"]) == 3
    assert config.read_text(encoding="utf-8") == before


def test_cli_bad_pair_file_exits_1(tmp_path, capsys, pg_url):
    args, _ = _cli_env(tmp_path, pg_url)
    bad = _write(tmp_path, "a,b,same,,trap,\n", name="bad.csv")
    args[args.index("--pairs") + 1] = str(bad)
    assert graph_main(args) == 1
    assert "line 5: no split" in capsys.readouterr().out


def test_cli_suggest_prints_only_names(tmp_path, capsys, pg_url):
    config = tmp_path / "graph.yml"
    shutil.copy(REPO / "config" / "graph.yml", config)
    cfg = load_graph_config(config)
    db = pg_url
    c = connect(db)
    init_db(c)

    def vec(axis, cosine):
        v = np.zeros(cfg.dimensions, dtype=np.float32)
        v[axis], v[axis + 1] = cosine, math.sqrt(1 - cosine**2)
        return v

    for name, v, mine in (
        ("python", vec(0, 1.0), True),
        ("django", vec(0, 0.913), False),
        ("pandas", vec(0, 0.71), False),
    ):
        add_market_skill(c, name)
        save_vectors(c, [name], np.asarray([v]), cfg.embed_tag)
        if mine:
            c.execute(
                "UPDATE skill_vocab SET mine = true, source = 'profile', level = 'production' WHERE name = %s", (name,)
            )
    c.commit()
    _extracted_job(c, "greenhouse:j1", ["django", "pandas"])
    c.close()
    sidecar = tmp_path / "private" / "bands.json"

    code = graph_main(
        ["--db", str(db), "--config", str(config), "calibrate", "--suggest", "--sidecar-path", str(sidecar)]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert out.splitlines() == ["a,b", "django,python", "pandas,python"]
    assert not re.search(r"\d", out)  # no cosine, no band, no count reaches the labelers
    assert json.loads(sidecar.read_text(encoding="utf-8"))["pairs"][0]["cosine"] == pytest.approx(0.913, abs=1e-5)
