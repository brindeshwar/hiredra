"""Tests for the scoring run (plan unit S6.2): migration v6, the preconditions, the freeze lock,
rescore triggers, F2 alias-before-cache, gating and release, budget stops, and the CLI views.
A fake embedder only; vectors sit at exact cosines so every fit is known."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import psycopg
import pytest
import yaml

from hiredra.graph.config import GraphConfig
from hiredra.quota import DailyQuotaExceeded
from hiredra.graph.index import save_vectors
from hiredra.graph.lock import RunLock
from hiredra.profile import parse_skills
from hiredra.scoring import run as sr
from hiredra.scoring.__main__ import main as scoring_main
from hiredra.scoring.score import load_weights
from hiredra.storage import (
    add_market_skill,
    apply_profile_overlay,
    connect,
    db,
    get_events,
    get_job,
    init_db,
    merge_skill,
    set_extraction,
    set_status,
    upsert_job,
)

DIM = 128
REPO = Path(__file__).resolve().parents[1]


def _graph_cfg(prefix="") -> GraphConfig:
    return GraphConfig(
        model="gemini-embedding-2",
        dimensions=DIM,
        prefix=prefix,
        inputs_per_request=50,
        dedup_threshold=0.9,
        have=0.90,
        adj=0.70,
    )


def _vec(axis: int, cosine: float = 1.0) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[axis], v[axis + 1] = cosine, math.sqrt(max(0.0, 1 - cosine**2))
    return v


def _weights(tmp_path, cfg, **over):
    data = {
        "calibration_id": "placeholder",
        "domain_fit": {"agent_llm": 1.0, "backend": 0.5, "data": 0.5, "ml": 0.5, "generic_swe": 0.0, "other": 0.0},
        "w_domain": 1.0,
        "w_skills": 1.0,
        "w_gap": 0.5,
        "cover_bar": 0.85,
        "good_to_have_factor": 0.4,
        "apply_line": None,
        "min_skills": 1,  # the few_skills rule has its own test below
        **over,
    }
    p = tmp_path / f"weights-{abs(hash(json.dumps(data, sort_keys=True)))}.yml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_weights(cfg, p), p


class World:
    """A small graph: mine python (axis 0) and postgresql (axis 10); JD skills at known cosines."""

    def __init__(self, conn, tmp_path):
        self.conn, self.tmp, self.cfg = conn, tmp_path, _graph_cfg()
        for name, vec in {
            "python": _vec(0),
            "postgresql": _vec(10),
            "django": _vec(0, 0.95),  # -> have
            "flask": _vec(0, 0.80),  # -> adjacent (fit 0.80, under the 0.85 bar)
            "mysql": _vec(10, 0.88),  # -> adjacent (fit 0.88, covered)
            "cobol": _vec(20),  # -> missing
        }.items():
            add_market_skill(conn, name)
            save_vectors(conn, [name], np.asarray([vec]), self.cfg.embed_tag)
        apply_profile_overlay(conn, mine={"python": "production", "postgresql": "learned"}, new_nodes={}, aliases=[])
        self.profile_path = tmp_path / "profile.md"
        self.profile_path.write_text(
            "Last updated: 2026-10-18\n## Skills\n**Strong:**\npython\n**Working:**\ndocker\n**Familiar:**\npostgresql\n",
            encoding="utf-8",
        )
        self.profile = parse_skills(self.profile_path)
        self.overlay_path = tmp_path / "overlay.json"
        self.overlay_path.write_text(json.dumps({"skills_hash": self.profile.skills_hash}), encoding="utf-8")

    def job(self, job_id, skills, *, bucket="backend", legitimacy="legitimate", confidence="high", described=True):
        upsert_job(
            self.conn,
            {
                "job_id": job_id,
                "url": f"u://{job_id}",
                "title": f"Engineer {job_id}",
                "company": "Acme",
                "source": "greenhouse",
            },
        )
        set_status(self.conn, job_id, "filtered")
        set_extraction(
            self.conn,
            job_id,
            skills=[{"name": n, "requirement": r} for n, r in skills],
            domain_bucket=bucket,
            legitimacy=legitimacy,
            extraction_confidence=confidence,
        )
        self.conn.execute("UPDATE jobs SET extraction_version = 'x:1' WHERE job_key = %s", (job_id,))
        if described:  # the describe step has settled every skill (decision D3)
            self.conn.execute("UPDATE skill_vocab SET description_source = 'declined' WHERE description_source IS NULL")
        self.conn.commit()
        set_status(self.conn, job_id, "extracted")
        return job_id

    def run(self, weights, embed_factory=None, force=False):
        return sr.run_scoring(self.conn, self.cfg, weights, self.profile, embed_factory, force=force)


@pytest.fixture()
def world(tmp_path, pg_url):
    c = connect(pg_url)
    init_db(c)
    yield World(c, tmp_path)
    c.close()


def _no_embed():
    pytest.fail("the embedder was built")


# ── scoring ─────────────────────────────────────────────────────────────────
def test_a_job_is_scored_with_its_features_and_stamps(world):
    w, _ = _weights(world.tmp, world.cfg)
    jid = world.job(
        "greenhouse:j1",
        [("django", "required"), ("flask", "required"), ("mysql", "good_to_have"), ("cobol", "required")],
    )
    stats = world.run(w, _no_embed)
    assert (stats.selected, stats.scored, stats.embedded) == (1, 1, 0)

    row = get_job(world.conn, jid)
    assert row["status"] == "scored" and row["review_reason"] is None
    # weights 1, 1, 0.4, 1 ; covered: django (1.0) + mysql (0.88) -> 1.4 / 3.4 ; gaps: flask (1-0.8), cobol (1-0)
    coverage, gap = 1.4 / 3.4, math.sqrt(0.2**2 + 1.0)
    assert row["score"] == pytest.approx(0.5 + coverage - 0.5 * gap, abs=1e-5)
    f = row["score_features"]
    assert f["v"] == 1 and f["domain_bucket"] == "backend" and f["coverage"] == pytest.approx(coverage, abs=1e-6)
    assert [(s["skill"], s["band"], s["nearest"]) for s in f["skills"]] == [
        ("django", "have", "python"),
        ("flask", "adjacent", "python"),
        ("mysql", "adjacent", "postgresql"),
        ("cobol", "missing", "postgresql"),  # a tie at cosine 0: the first mine node by name
    ]
    assert f["counts"] == {"required": 3, "good_to_have": 1, "have": 1, "adjacent": 2, "missing": 1}
    assert f["nearest_tier"] == {"strong": 2, "working": 0, "familiar": 1}  # matched (have/adjacent) only
    assert f["params"]["cover_bar"] == 0.85 and f["score_terms"]["gap"] == pytest.approx(-0.5 * gap, abs=1e-6)
    assert (row["score_version"], row["score_embed_tag"]) == (w.score_version, world.cfg.embed_tag)
    assert (row["score_skills_hash"], row["score_profile_hash"]) == (world.profile.skills_hash, world.profile.file_hash)
    assert row["scored_at"]


def test_review_reasons_are_stored_but_the_job_is_still_scored(world):
    w, _ = _weights(world.tmp, world.cfg)
    world.job("greenhouse:j1", [("django", "required")], legitimacy="likely_ghost", confidence="low")
    world.run(w)
    row = get_job(world.conn, "greenhouse:j1")
    assert row["status"] == "scored" and row["review_reason"] == ["likely_ghost", "low_confidence"]
    assert row["score"] is not None


def test_a_job_with_too_few_skills_goes_to_review_and_is_never_promoted(world):
    w, _ = _weights(world.tmp, world.cfg, min_skills=3, apply_line=-10.0)
    world.job("greenhouse:few", [("django", "required"), ("flask", "required")])
    world.job("greenhouse:enough", [("django", "required"), ("flask", "required"), ("mysql", "good_to_have")])
    world.run(w)
    few, enough = get_job(world.conn, "greenhouse:few"), get_job(world.conn, "greenhouse:enough")
    assert few["status"] == "scored" and few["review_reason"] == ["few_skills"]
    assert enough["review_reason"] is None
    assert [j["job_id"] for j in sr.promoted(world.conn, w)] == [
        "greenhouse:enough"
    ]  # below-floor apply line: only skill count gates


def test_force_rescores_fresh_jobs_and_picks_up_a_merge(world):
    w, _ = _weights(world.tmp, world.cfg)
    world.job("greenhouse:j1", [("django", "required")])
    assert world.run(w).scored == 1
    assert world.run(w).selected == 0  # nothing stale: a plain run selects nothing
    stats = world.run(w, force=True)
    assert (stats.selected, stats.rescored, stats.scored) == (1, 1, 0)


def test_f2_a_merge_after_extraction_scores_against_the_survivor_with_zero_embeddings(world):
    w, _ = _weights(world.tmp, world.cfg)
    add_market_skill(world.conn, "django framework")  # B, extracted before the merge
    save_vectors(world.conn, ["django framework"], np.asarray([_vec(30)]), world.cfg.embed_tag)
    world.job("greenhouse:j1", [("django framework", "required")])
    merge_skill(world.conn, "django framework", "django")  # B merged into A after extraction
    stats = world.run(w, _no_embed)
    f = get_job(world.conn, "greenhouse:j1")["score_features"]
    assert stats.embedded == 0 and f["skills"][0]["skill"] == "django" and f["skills"][0]["band"] == "have"


def test_a_cache_miss_is_embedded_once(world):
    w, _ = _weights(world.tmp, world.cfg)
    world.job("greenhouse:j1", [("brand new skill", "required")])
    world.job("greenhouse:j2", [("brand new skill", "required"), ("django", "required")])
    built: list[int] = []

    def factory():
        built.append(1)
        return lambda names: np.asarray([_vec(0, 0.92) for _ in names], dtype=np.float32)

    stats = world.run(w, factory)
    assert stats.embedded == 1 and built == [1] and stats.scored == 2
    assert get_job(world.conn, "greenhouse:j2")["score_features"]["skills"][0]["band"] == "have"


def test_the_embedder_is_built_once_for_many_misses(world):
    w, _ = _weights(world.tmp, world.cfg)
    world.job("greenhouse:j1", [("new skill a", "required")])
    world.job("greenhouse:j2", [("new skill b", "required")])
    built: list[int] = []

    def factory():
        built.append(1)
        return lambda names: np.asarray([_vec(40) for _ in names], dtype=np.float32)

    stats = world.run(w, factory)
    assert stats.embedded == 2 and built == [1]


def test_an_unknown_error_is_raised_not_turned_into_a_stop(world):
    w, _ = _weights(world.tmp, world.cfg)
    world.job("greenhouse:j1", [("django", "required")])
    world.job("greenhouse:j2", [("new skill", "required")])

    def factory():
        def embed(names):
            raise RuntimeError("a bug")

        return embed

    with pytest.raises(RuntimeError, match="a bug"):
        world.run(w, factory)
    assert get_job(world.conn, "greenhouse:j1")["status"] == "scored"  # finished rows stay


def test_a_miss_without_an_embedder_names_the_sync_command(world):
    w, _ = _weights(world.tmp, world.cfg)
    world.job("greenhouse:j1", [("new skill", "required")])
    with pytest.raises(sr.PreconditionError, match="graph sync"):
        world.run(w, None)


def test_a_budget_stop_keeps_finished_rows(world):
    w, _ = _weights(world.tmp, world.cfg)
    world.job("greenhouse:j1", [("django", "required")])
    world.job("greenhouse:j2", [("brand new skill", "required")])
    world.job("greenhouse:j3", [("flask", "required")])

    def factory():
        def embed(names):
            raise DailyQuotaExceeded("PerDay")

        return embed

    stats = world.run(w, factory)
    assert stats.stopped == "daily_quota" and stats.scored == 1
    assert [get_job(world.conn, j)["status"] for j in ("greenhouse:j1", "greenhouse:j2", "greenhouse:j3")] == [
        "scored",
        "extracted",
        "extracted",
    ]


def test_the_store_refuses_malformed_skills_so_scoring_never_sees_them(world):
    world.job("greenhouse:bad", [])
    with pytest.raises(psycopg.errors.CheckViolation):
        set_extraction(world.conn, "greenhouse:bad", skills=[{"name": "x", "requirement": "maybe"}],
                       domain_bucket=None, legitimacy=None, extraction_confidence="high")  # fmt: skip
    with pytest.raises(KeyError):
        set_extraction(world.conn, "greenhouse:bad", skills=[{"requirement": "required"}],
                       domain_bucket=None, legitimacy=None, extraction_confidence="high")  # fmt: skip


def test_a_job_with_no_skills_scores_on_domain_alone(world):
    w, _ = _weights(world.tmp, world.cfg)
    world.job("greenhouse:j1", [], bucket="agent_llm")
    world.run(w)
    assert get_job(world.conn, "greenhouse:j1")["score"] == pytest.approx(1.0)


# ── rescore triggers ────────────────────────────────────────────────────────
@pytest.mark.parametrize("column", ["score_version", "score_embed_tag", "score_skills_hash"])
def test_each_stale_stamp_triggers_a_rescore(world, column):
    w, _ = _weights(world.tmp, world.cfg)
    world.job("greenhouse:j1", [("django", "required")])
    world.job("greenhouse:j2", [("flask", "required")])
    world.run(w)
    assert world.run(w).selected == 0  # nothing stale
    world.conn.execute(f"UPDATE jobs SET {column} = 'old' WHERE job_key = 'greenhouse:j1'")
    world.conn.commit()
    stats = world.run(w)
    assert (stats.selected, stats.rescored, stats.scored) == (1, 1, 0)
    assert (
        get_job(world.conn, "greenhouse:j1")[column] != "old"
        and get_job(world.conn, "greenhouse:j1")["status"] == "scored"
    )
    assert any("re-scored" in e["detail"] for e in get_events(world.conn, "greenhouse:j1") if e["kind"] == "note")


def test_new_weights_rescore_everything_and_the_profile_file_hash_does_not(world):
    w, _ = _weights(world.tmp, world.cfg)
    world.job("greenhouse:j1", [("django", "required")])
    world.run(w)
    world.conn.execute("UPDATE jobs SET score_profile_hash = 'old'")  # tracing only
    world.conn.commit()
    assert world.run(w).selected == 0
    w2, _ = _weights(world.tmp, world.cfg, w_gap=0.7)
    assert world.run(w2).rescored == 1


# ── preconditions ───────────────────────────────────────────────────────────
def test_preconditions(world):
    sr.check_preconditions(world.conn, world.cfg, world.profile, world.overlay_path)  # all good
    with pytest.raises(sr.PreconditionError, match="no profile overlay yet"):
        sr.check_preconditions(world.conn, world.cfg, world.profile, world.tmp / "missing.json")
    stale = world.tmp / "stale.json"
    stale.write_text(json.dumps({"skills_hash": "greenhouse:old"}), encoding="utf-8")
    with pytest.raises(sr.PreconditionError, match="re-run `python -m hiredra.graph overlay`"):
        sr.check_preconditions(world.conn, world.cfg, world.profile, stale)
    add_market_skill(world.conn, "docker")
    apply_profile_overlay(world.conn, mine={"python": "production", "docker": "built-with"}, new_nodes={}, aliases=[])
    with pytest.raises(sr.PreconditionError, match=r"1 mine skill\(s\) have no current vector \(docker\).*graph sync"):
        sr.check_preconditions(world.conn, world.cfg, world.profile, world.overlay_path)
    apply_profile_overlay(world.conn, mine={}, new_nodes={}, aliases=[])
    with pytest.raises(sr.PreconditionError, match="no 'mine' skills"):
        sr.check_preconditions(world.conn, world.cfg, world.profile, world.overlay_path)


# ── gating and release ──────────────────────────────────────────────────────
def test_gating_release_and_reject(world):
    w, _ = _weights(world.tmp, world.cfg)
    world.job("greenhouse:high", [("django", "required")], bucket="agent_llm")  # 2.0
    world.job("greenhouse:ghost", [("django", "required")], bucket="agent_llm", legitimacy="suspect")  # 2.0 but review
    world.job("greenhouse:low", [("cobol", "required")], bucket="other")  # -0.5
    world.run(w)
    assert sr.promoted(world.conn, w) == []  # no apply line yet: nothing promoted

    lined, _ = _weights(world.tmp, world.cfg, apply_line=1.0)
    assert [r["job_id"] for r in sr.promoted(world.conn, lined)] == ["greenhouse:high"]
    assert [r["job_id"] for r in sr.review_list(world.conn)] == ["greenhouse:ghost"]
    assert [r["job_id"] for r in sr.all_scored(world.conn)] == [
        "greenhouse:ghost",
        "greenhouse:high",
        "greenhouse:low",
    ]  # score desc, ties by id

    db.release_job(world.conn, "greenhouse:low")  # D13: a below-line job can be released
    db.release_job(world.conn, "greenhouse:ghost")  # and a review-list job
    assert {r["job_id"] for r in sr.promoted(world.conn, lined)} == {
        "greenhouse:high",
        "greenhouse:low",
        "greenhouse:ghost",
    }
    assert sr.review_list(world.conn) == []
    assert [r["job_id"] for r in sr.promoted(world.conn, w)] == [
        "greenhouse:ghost",
        "greenhouse:low",
    ]  # released: promoted with no line

    set_status(world.conn, "greenhouse:high", "dropped", detail="review: rejected")
    with pytest.raises(ValueError, match="only a scored job"):
        db.release_job(world.conn, "greenhouse:high")


# ── CLI ─────────────────────────────────────────────────────────────────────
def _cli_world(tmp_path, pg_url):
    """A file DB plus the shipped graph config's dimensions/tag, for the CLI."""
    from hiredra.graph.config import load_graph_config

    cfg = load_graph_config()
    path = pg_url
    c = connect(path)
    init_db(c)
    for name, axis in (("python", 0), ("django", 0)):
        v = np.zeros(cfg.dimensions, dtype=np.float32)
        v[axis] = 1.0
        add_market_skill(c, name)
        save_vectors(c, [name], np.asarray([v]), cfg.embed_tag)
    apply_profile_overlay(c, mine={"python": "production"}, new_nodes={}, aliases=[])
    w = World.__new__(World)
    w.conn, w.tmp, w.cfg = c, tmp_path, cfg
    w.job("greenhouse:j1", [("django", "required")], bucket="agent_llm")
    w.job("greenhouse:j2", [("django", "required")], legitimacy="suspect")
    c.close()
    profile = tmp_path / "profile.md"
    profile.write_text(
        "Last updated: 2026-10-18\n## Skills\n**Strong:**\npython\n**Working:**\ndocker\n**Familiar:**\nrust\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay.json"
    overlay.write_text(json.dumps({"skills_hash": parse_skills(profile).skills_hash}), encoding="utf-8")
    _, weights_path = _weights(tmp_path, cfg, cover_bar=None, apply_line=1.0)
    return [
        "--db", str(path), "--weights", str(weights_path), "--profile", str(profile),
        "--overlay-path", str(overlay),
    ]  # fmt: skip


def test_cli_run_and_views(tmp_path, capsys, pg_url):
    base = _cli_world(tmp_path, pg_url)
    assert scoring_main([*base, "run"]) == 0
    assert "selected=2 scored=2" in capsys.readouterr().out

    assert scoring_main([*base, "list"]) == 0
    out = capsys.readouterr().out
    assert "promoted jobs: 1" in out and "greenhouse:j1" in out and "greenhouse:j2" not in out
    assert scoring_main([*base, "list", "--all"]) == 0
    assert "every scored job (D13: below the line too): 2" in capsys.readouterr().out
    assert scoring_main([*base, "review"]) == 0
    out = capsys.readouterr().out
    assert "review list: 1" in out and "[suspect]" in out

    assert scoring_main([*base, "show", "greenhouse:j1"]) == 0
    out = capsys.readouterr().out
    assert "have (1):" in out and "django" in out and "nearest python" in out

    assert scoring_main([*base, "release", "greenhouse:j2"]) == 0
    capsys.readouterr()
    assert scoring_main([*base, "list"]) == 0
    assert "promoted jobs: 2" in capsys.readouterr().out
    assert scoring_main([*base, "reject", "greenhouse:j1"]) == 0
    capsys.readouterr()
    assert scoring_main([*base, "release", "greenhouse:j1"]) == 1  # dropped now

    assert scoring_main([*base, "status"]) == 0
    out = capsys.readouterr().out
    assert "Last updated 2026-10-18" in out and "STALE" not in out and "x:1" in out


def test_cli_run_refuses_while_the_graph_lock_is_held(tmp_path, capsys, pg_url):
    base = _cli_world(tmp_path, pg_url)
    with RunLock(what="sync", url=pg_url):
        assert scoring_main([*base, "run"]) == 4
    assert "another graph maintenance run is in progress" in capsys.readouterr().out


def test_cli_run_exits_2_on_a_stale_overlay(tmp_path, capsys, pg_url):
    base = _cli_world(tmp_path, pg_url)
    (tmp_path / "overlay.json").write_text(json.dumps({"skills_hash": "greenhouse:old"}), encoding="utf-8")
    assert scoring_main([*base, "run"]) == 2
    assert "re-run `python -m hiredra.graph overlay`" in capsys.readouterr().out


def test_cli_bad_weights_exit_1(tmp_path, capsys, pg_url):
    base = _cli_world(tmp_path, pg_url)
    bad = tmp_path / "bad.yml"
    bad.write_text("w_gap: 1\n", encoding="utf-8")
    base[base.index("--weights") + 1] = str(bad)
    assert scoring_main([*base, "status"]) == 1
    assert "config error" in capsys.readouterr().out


def test_d3_a_job_waits_until_its_skills_are_described(world):
    w, _ = _weights(world.tmp, world.cfg)
    world.job("greenhouse:ready", [("django", "required")])
    add_market_skill(world.conn, "airflow")
    save_vectors(world.conn, ["airflow"], np.asarray([_vec(40)]), world.cfg.embed_tag)
    world.job("greenhouse:waits", [("django", "required"), ("airflow", "required")], described=False)
    world.job("greenhouse:miss", [("not a node yet", "required")], described=False)  # nothing to wait for
    assert sr.waiting_for_descriptions(world.conn) == 1
    stats = world.run(w, lambda: lambda names: np.stack([_vec(50) for _ in names]))
    assert stats.scored == 2 and get_job(world.conn, "greenhouse:waits")["status"] == "extracted"
    world.conn.execute("UPDATE skill_vocab SET description_source = 'generated' WHERE name = 'airflow'")
    assert sr.waiting_for_descriptions(world.conn) == 0
    assert world.run(w).scored == 1 and get_job(world.conn, "greenhouse:waits")["status"] == "scored"
