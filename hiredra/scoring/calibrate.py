"""Step 7 weight calibration (plan unit S7.3): fit the weights to the owner's 30 job ratings.

    python -m hiredra.scoring calibrate                       # dry run: report only, nothing written
    python -m hiredra.scoring calibrate --have 1.0 --adj 0.615  # the same, under other match thresholds
    python -m hiredra.scoring calibrate --write               # freeze config/weights.yml and write the reports

* Metric (D3): pairwise rank agreement ``A = (concordant + 0.5 * tied) / pairs with different ratings``, rating
  order apply > maybe > skip. A tie is a pair the score cannot tell apart.
* Grid (D15): w_domain x w_gap x cover_bar x good-to-have factor x the backend, data/ml and generic_swe steps,
  6,300 combinations with ``w_skills`` fixed at 1.0. ``agent_llm`` = 1 and ``other`` = 0 stay fixed.
* Choice rule: among the combinations within 0.01 of the best TUNE agreement, the one that changes the fewest
  knobs from the current config. Twenty jobs cannot tell near-equal combinations apart, so the smallest move
  wins.
* Apply line (D13): the score that best separates tune ``apply`` from tune ``skip`` (balanced accuracy; ``maybe``
  left out); ties go to the higher line. A quality floor, never a cap on count.
* Holdout: agreement on the held-back jobs, computed once for the chosen combination, with a bootstrap 90%
  interval over jobs.

Every score is computed by ``hiredra.scoring.score.score_job`` itself, from the per-skill similarities stored in
each job's ``score_features``: no embedding, no model call. Rated jobs the filters have since dropped are left
out: they never reach the scorer.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
import re
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from hiredra.scoring.rate import CALIBRATION_DIR, RATINGS_PATH
from hiredra.scoring.score import DEFAULT_WEIGHTS_PATH, Weights, score_job

RANK = {"skip": 0, "maybe": 1, "apply": 2}
NEAR_BEST = 0.01
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20261126
_EPS = 1e-6

GRID_W_DOMAIN = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
GRID_W_GAP = (0.0, 0.25, 0.5, 0.75, 1.0)
GRID_X = (0.2, 0.4, 0.6)
GRID_BACKEND = (0.5, 0.75, 1.0)
GRID_DATA_ML = (0.25, 0.5)
GRID_GENERIC = (0.0, 0.25)

REPORT_PATH = CALIBRATION_DIR / "step7-report.md"
SNAPSHOT_PATH = CALIBRATION_DIR / "step7-snapshot.json"
SUMMARY_PATH = Path(__file__).resolve().parents[2] / "calibration" / "reports" / "step7-summary.md"


class CalibrationError(ValueError):
    """The ratings cannot support a calibration as they stand."""


@dataclass(frozen=True)
class RatedJob:
    job_id: str
    split: str
    rating: str
    reason: str
    bucket: str | None
    skills: tuple[tuple[float, str], ...]  # (similarity, requirement) per extracted skill


@dataclass(frozen=True)
class Combo:
    w_domain: float
    w_gap: float
    cover_bar: float
    x: float
    backend: float
    data_ml: float
    generic: float

    def weights(self, base: Weights) -> Weights:
        domain = {**base.domain_fit, "backend": self.backend, "data": self.data_ml, "ml": self.data_ml,
                  "generic_swe": self.generic}  # fmt: skip
        return replace(base, w_domain=self.w_domain, w_gap=self.w_gap, cover_bar=self.cover_bar,
                       good_to_have_factor=self.x, domain_fit=domain)  # fmt: skip


# ─────────────────────────────────────────────────────────────────────────────
# the metric, the fit, the grid, the choice
# ─────────────────────────────────────────────────────────────────────────────
def agreement(scores: list[float], ratings: list[str]) -> tuple[float | None, int]:
    """(A, number of pairs with different ratings). A is None when no pair differs."""
    good = pairs = 0.0
    for i, j in itertools.combinations(range(len(scores)), 2):
        dr = RANK[ratings[i]] - RANK[ratings[j]]
        if dr == 0:
            continue
        pairs += 1
        ds = scores[i] - scores[j]
        if abs(ds) <= _EPS:
            good += 0.5
        elif (ds > 0) == (dr > 0):
            good += 1
    return (good / pairs if pairs else None), int(pairs)


def fit(similarity: float, have: float, adj: float) -> float:
    """The per-skill fit (hiredra.graph.fit): 1 from ``have``, the similarity itself from ``adj``, else 0."""
    if similarity >= have - _EPS:
        return 1.0
    if similarity >= adj - _EPS:
        return similarity
    return 0.0


def cover_bars(adj: float, have: float) -> tuple[float, ...]:
    """Five evenly spaced values strictly inside (adj, have); the middle one is the default midpoint."""
    return tuple(round(adj + (have - adj) * k / 6, 6) for k in range(1, 6))


def grid(adj: float, have: float) -> list[Combo]:
    return [Combo(*c) for c in itertools.product(GRID_W_DOMAIN, GRID_W_GAP, cover_bars(adj, have), GRID_X,
                                                  GRID_BACKEND, GRID_DATA_ML, GRID_GENERIC)]  # fmt: skip


def default_combo(base: Weights) -> Combo:
    d = base.domain_fit
    return Combo(base.w_domain, base.w_gap, base.cover_bar, base.good_to_have_factor, d["backend"], d["data"],
                 d["generic_swe"])  # fmt: skip


def changed_knobs(combo: Combo, default: Combo) -> int:
    return sum(abs(a - b) > _EPS for a, b in zip(vars(combo).values(), vars(default).values()))


def score_all(jobs: list[RatedJob], combo: Combo, base: Weights, have: float, adj: float) -> list[float]:
    w = combo.weights(base)
    return [score_job([(fit(s, have, adj), req) for s, req in j.skills], j.bucket, w).score for j in jobs]


def choose(results: list[tuple[Combo, float]], default: Combo) -> tuple[Combo, float]:
    """Within NEAR_BEST of the best agreement: fewest changed knobs, then higher agreement, then grid order."""
    best = max(a for _, a in results)
    near = [(i, c, a) for i, (c, a) in enumerate(results) if a >= best - NEAR_BEST - _EPS]
    _, combo, a = min(near, key=lambda t: (changed_knobs(t[1], default), -t[2], t[0]))
    return combo, a


def apply_line(scores: list[float], ratings: list[str]) -> tuple[float | None, float | None]:
    """(line, balanced accuracy): the apply-job score that best splits apply from skip. None without both."""
    applies = [s for s, r in zip(scores, ratings) if r == "apply"]
    skips = [s for s, r in zip(scores, ratings) if r == "skip"]
    if not applies or not skips:
        return None, None
    best: tuple[float, float] | None = None
    for line in sorted(set(applies)):
        tpr = sum(s >= line - _EPS for s in applies) / len(applies)
        tnr = sum(s < line - _EPS for s in skips) / len(skips)
        bacc = (tpr + tnr) / 2
        if best is None or bacc >= best[1] - _EPS:  # ascending lines: a tie goes to the higher one
            best = (line, bacc)
    return best


def bootstrap(scores: list[float], ratings: list[str], *, draws: int = BOOTSTRAP_DRAWS,
              seed: int = BOOTSTRAP_SEED) -> tuple[float, float] | None:  # fmt: skip
    """5th and 95th percentile of A over job resamples (resamples with no differing pair are skipped)."""
    rng = random.Random(seed)
    n, values = len(scores), []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        a, _ = agreement([scores[i] for i in idx], [ratings[i] for i in idx])
        if a is not None:
            values.append(a)
    if not values:
        return None
    values.sort()
    return values[int(0.05 * (len(values) - 1))], values[int(0.95 * (len(values) - 1))]


# ─────────────────────────────────────────────────────────────────────────────
# data
# ─────────────────────────────────────────────────────────────────────────────
def load_rated(conn, ratings_path: Path = RATINGS_PATH) -> tuple[list[RatedJob], list[str]]:
    """The latest rating per job, joined to its stored features. Returns (usable jobs, left-out notes)."""
    latest: dict[str, dict] = {}
    with ratings_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            latest[row["job_id"]] = row  # later rows win
    jobs, left_out = [], []
    for jid, row in latest.items():
        rec = conn.execute(
            "SELECT status, status_detail, score_features FROM jobs WHERE job_key = %s", (jid,)
        ).fetchone()
        if rec is None or rec["score_features"] is None:
            left_out.append(f"{jid}: no stored score")
            continue
        if rec["status"] == "dropped":
            left_out.append(f"{jid} ({row['split']}, {row['rating']}): dropped, {rec['status_detail']}")
            continue
        f = rec["score_features"]
        skills = tuple((float(s["s"]), s["req"]) for s in f["skills"])
        jobs.append(RatedJob(jid, row["split"], row["rating"], row["reason"], f["domain_bucket"], skills))
    return jobs, left_out


# ─────────────────────────────────────────────────────────────────────────────
# the run
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Calibration:
    have: float
    adj: float
    tune: list[RatedJob]
    holdout: list[RatedJob]
    left_out: list[str]
    default: Combo
    default_tune_a: float
    default_holdout_a: float | None
    chosen: Combo
    tune_a: float
    best_tune_a: float
    near_best: int
    tune_pairs: int
    holdout_a: float | None
    holdout_pairs: int
    holdout_interval: tuple[float, float] | None
    apply_line: float | None
    apply_bacc: float | None
    tune_scores: list[float]
    holdout_scores: list[float]


def calibrate(jobs: list[RatedJob], base: Weights, have: float, adj: float, left_out: list[str] | None = None
              ) -> Calibration:  # fmt: skip
    tune = [j for j in jobs if j.split == "tune"]
    holdout = [j for j in jobs if j.split == "holdout"]
    t_ratings, h_ratings = [j.rating for j in tune], [j.rating for j in holdout]
    if agreement([0.0] * len(holdout), h_ratings)[1] == 0 or agreement([0.0] * len(tune), t_ratings)[1] == 0:
        raise CalibrationError("tune or holdout has no pair of jobs with different ratings; add ratings first")

    default = default_combo(base)
    if all(abs(default.cover_bar - b) > _EPS for b in cover_bars(adj, have)):  # off this grid: use the midpoint
        default = replace(default, cover_bar=cover_bars(adj, have)[2])
    results = [(c, agreement(score_all(tune, c, base, have, adj), t_ratings)[0]) for c in grid(adj, have)]
    chosen, tune_a = choose(results, default)
    best = max(a for _, a in results)

    t_scores = score_all(tune, chosen, base, have, adj)
    h_scores = score_all(holdout, chosen, base, have, adj)
    line, bacc = apply_line(t_scores, t_ratings)
    h_a, h_pairs = agreement(h_scores, h_ratings)
    return Calibration(
        have=have, adj=adj, tune=tune, holdout=holdout, left_out=left_out or [], default=default,
        default_tune_a=agreement(score_all(tune, default, base, have, adj), t_ratings)[0],
        default_holdout_a=agreement(score_all(holdout, default, base, have, adj), h_ratings)[0],
        chosen=chosen, tune_a=tune_a, best_tune_a=best,
        near_best=sum(a >= best - NEAR_BEST - _EPS for _, a in results),
        tune_pairs=agreement(t_scores, t_ratings)[1], holdout_a=h_a, holdout_pairs=h_pairs,
        holdout_interval=bootstrap(h_scores, h_ratings), apply_line=line, apply_bacc=bacc,
        tune_scores=t_scores, holdout_scores=h_scores,
    )  # fmt: skip


def _combo_text(c: Combo) -> str:
    return (f"w_domain {c.w_domain}, w_gap {c.w_gap}, cover_bar {c.cover_bar:.3f}, good-to-have {c.x}, "
            f"backend {c.backend}, data/ml {c.data_ml}, generic_swe {c.generic}")  # fmt: skip


def summary(cal: Calibration) -> str:
    """Counts and numbers only: safe to print, and the body of the public summary."""
    iv = cal.holdout_interval
    lines = [
        f"match thresholds: full credit from {cal.have}, partial from {cal.adj}",
        f"rated jobs used: {len(cal.tune)} tune + {len(cal.holdout)} holdout ({len(cal.left_out)} left out: dropped by the filters since)",
        f"grid: {len(GRID_W_DOMAIN) * len(GRID_W_GAP) * 5 * len(GRID_X) * len(GRID_BACKEND) * len(GRID_DATA_ML) * len(GRID_GENERIC)} combinations",
        f"current weights:  tune A {cal.default_tune_a:.3f}, holdout A {cal.default_holdout_a:.3f}",
        f"best tune A {cal.best_tune_a:.3f}; {cal.near_best} combinations within {NEAR_BEST}",
        f"chosen ({changed_knobs(cal.chosen, cal.default)} knobs changed): {_combo_text(cal.chosen)}",
        f"chosen:           tune A {cal.tune_a:.3f} ({cal.tune_pairs} pairs), "
        f"holdout A {cal.holdout_a:.3f} ({cal.holdout_pairs} pairs), 90% interval "
        + (f"{iv[0]:.2f}-{iv[1]:.2f}" if iv else "n/a"),
        f"apply line: {cal.apply_line:.3f} (balanced accuracy on tune apply vs skip {cal.apply_bacc:.2f})"
        if cal.apply_line is not None else "apply line: none (tune needs both apply and skip ratings)",
    ]  # fmt: skip
    return "\n".join(lines)


def private_report(cal: Calibration) -> str:
    """Per-job detail for the owner: ratings, reasons, scores and where the score disagrees. Never committed."""
    out = ["# Step 7 calibration report (private)", "", "```", summary(cal), "```", ""]
    for name, jobs, scores in (("Tune", cal.tune, cal.tune_scores), ("Holdout", cal.holdout, cal.holdout_scores)):
        out += [f"## {name}", "", "| score | rating | above line | job | your reason |", "|---|---|---|---|---|"]
        for j, s in sorted(zip(jobs, scores), key=lambda t: -t[1]):
            above = "" if cal.apply_line is None else ("yes" if s >= cal.apply_line - _EPS else "no")
            out.append(f"| {s:.3f} | {j.rating} | {above} | `{j.job_id}` | {j.reason.replace('|', '/')} |")
        out.append("")
    if cal.left_out:
        out += ["## Left out", "", *[f"- {x}" for x in cal.left_out], ""]
    return "\n".join(out)


def write_weights(cal: Calibration, path: Path = DEFAULT_WEIGHTS_PATH, *, today: date | None = None) -> str:
    """Freeze the chosen values into ``config/weights.yml``, keeping its comments. Returns the calibration id."""
    cid = f"step7-{(today or date.today()).isoformat()}"
    c = cal.chosen
    text = path.read_text(encoding="utf-8")
    subs = {
        "calibration_id": cid,
        "domain_fit": (f"{{agent_llm: 1.0, backend: {c.backend}, data: {c.data_ml}, ml: {c.data_ml}, "
                       f"generic_swe: {c.generic}, other: 0.0}}"),
        "w_domain": c.w_domain,
        "w_gap": c.w_gap,
        "cover_bar": round(c.cover_bar, 6),
        "good_to_have_factor": c.x,
        "apply_line": round(cal.apply_line, 6) if cal.apply_line is not None else "null",
    }  # fmt: skip
    for key, value in subs.items():
        text, n = re.subn(rf"^{key}:[^\n#]*?(\s*#.*)?$", lambda m, k=key, v=value: f"{k}: {v}{m.group(1) or ''}",
                          text, count=1, flags=re.M)  # fmt: skip
        if n != 1:
            raise CalibrationError(f"{path}: no '{key}:' line to update")
    path.write_text(text, encoding="utf-8")
    return cid


def write_reports(cal: Calibration, cid: str) -> None:
    REPORT_PATH.write_text(private_report(cal), encoding="utf-8")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        f"# Step 7 weight calibration: {cid}\n\nOwner-rated jobs only; no job text or employer names.\n\n```\n"
        f"{summary(cal)}\n```\n",
        encoding="utf-8",
    )
    SNAPSHOT_PATH.write_text(
        json.dumps(
            {"calibration_id": cid, "have": cal.have, "adj": cal.adj, "chosen": vars(cal.chosen),
             "holdout_a": cal.holdout_a, "apply_line": cal.apply_line,
             "holdout": [{"rating": j.rating, "bucket": j.bucket, "skills": j.skills} for j in cal.holdout]},
            indent=1,
        ),
        encoding="utf-8",
    )  # fmt: skip
