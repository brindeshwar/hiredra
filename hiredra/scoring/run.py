"""Step 6: the scoring run -- score every extracted job against the frozen graph and log its
features (plan unit S6.2, docs/scoring.md secs 2 and 7).

The caller holds the graph ``RunLock`` for the whole run: that is the freeze rule (no sync,
merges or overlay can change the graph mid-run). ``check_preconditions`` refuses a stale profile
overlay or unembedded ``mine`` nodes; ``run_scoring`` then:

* resolves every JD skill name through the alias table BEFORE the vector cache (F2), so a merge
  applied after extraction scores against the survivor's vector and costs 0 embeddings;
* scores ``extracted`` jobs, and re-scores ``scored`` ones whose stamps went stale (weights
  version, embedding tag or profile skills hash);
* writes the score, the feature JSON (the ledger row for a future learner), the review reason
  and the four stamps. Only cache misses (a JD skill newer than the last sync) are embedded,
  through the daily counter; a budget stop ends the run cleanly with finished rows kept.

The model never sees a score: it only supplied skill names and the domain bucket (rule 3).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hiredra.graph.config import GraphConfig
from hiredra.graph.embedder import Embedder, classify_failure
from hiredra.graph.fit import fit_skills
from hiredra.graph.index import load_skills
from hiredra.profile import ProfileSkills
from hiredra.scoring.score import Weights, review_reasons, score_job, skill_weight
from hiredra.storage import add_note, resolve_alias, set_score, set_status

#: level -> profile tier, for the ``nearest_tier`` count (Familiar-inflation check, OD8).
_TIER_OF_LEVEL = {"production": "strong", "built-with": "working", "learned": "familiar"}


class PreconditionError(RuntimeError):
    """Scoring would not be reproducible right now. The message says the exact command to run."""


def check_preconditions(conn, graph_cfg: GraphConfig, profile: ProfileSkills, overlay_path: Path | str) -> None:
    path = Path(overlay_path)
    if not path.exists():
        raise PreconditionError(f"no profile overlay yet ({path}) -- run `python -m hiredra.graph overlay --apply`")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("skills_hash") != profile.skills_hash:
        raise PreconditionError(
            "the profile's Skills section changed since the overlay was applied -- "
            "re-run `python -m hiredra.graph overlay` and `overlay --apply`"
        )
    mine = [s for s in load_skills(conn, graph_cfg) if s.mine]
    if not mine:
        raise PreconditionError("no 'mine' skills in the graph -- run `python -m hiredra.graph overlay --apply`")
    unembedded = sorted(s.name for s in mine if s.vector is None)
    if unembedded:
        raise PreconditionError(
            f"{len(unembedded)} mine skill(s) have no current vector ({', '.join(unembedded[:5])}"
            f"{', ...' if len(unembedded) > 5 else ''}) -- run `python -m hiredra.graph sync`"
        )


def select_jobs(conn, weights: Weights, graph_cfg: GraphConfig, profile: ProfileSkills) -> list[dict]:
    """``extracted`` jobs, plus ``scored`` ones with any stale stamp (oldest first)."""
    rows = conn.execute(
        "SELECT * FROM jobs WHERE status = 'extracted' OR (status = 'scored' AND ("
        "score_version IS NOT ? OR score_embed_tag IS NOT ? OR score_skills_hash IS NOT ?)) "
        "ORDER BY first_seen_at, job_id",
        (weights.score_version, graph_cfg.embed_tag, profile.skills_hash),
    )
    return [dict(r) for r in rows]


def _band(fit: float) -> str:
    return "have" if fit >= 1.0 else "adjacent" if fit > 0.0 else "missing"


def build_features(job: dict, skills: list[dict], fits, result, reasons, weights: Weights, graph_cfg, levels) -> dict:
    """The ``score_features`` ledger row (plan contract sec 3.4)."""
    rows, counts, tiers = [], Counter(), Counter()
    for item, f in zip(skills, fits):
        band = _band(f.fit)
        counts[item["requirement"]] += 1
        counts[band] += 1
        if band != "missing" and f.nearest in levels:
            tiers[_TIER_OF_LEVEL.get(levels[f.nearest], "unknown")] += 1
        rows.append(
            {
                "skill": f.skill,
                "req": item["requirement"],
                "w": skill_weight(item["requirement"], weights),
                "nearest": f.nearest,
                "s": round(f.similarity, 4),
                "fit": round(f.fit, 4),
                "band": band,
            }
        )
    return {
        "v": 1,
        "domain_bucket": job["domain_bucket"],
        "domain_fit": result.domain_fit,
        "coverage": round(result.coverage, 6),
        "gap_distance": round(result.gap_distance, 6),
        "score_terms": {
            "domain": round(weights.w_domain * result.domain_fit, 6),
            "skills": round(weights.w_skills * result.coverage, 6),
            "gap": round(-weights.w_gap * result.gap_distance, 6),
        },
        "skills": rows,
        "counts": {k: counts[k] for k in ("required", "good_to_have", "have", "adjacent", "missing")},
        "nearest_tier": {k: tiers[k] for k in ("strong", "working", "familiar")},
        "params": {
            "have": graph_cfg.have,
            "adj": graph_cfg.adj,
            "cover_bar": weights.cover_bar,
            "x": weights.good_to_have_factor,
            "w_domain": weights.w_domain,
            "w_skills": weights.w_skills,
            "w_gap": weights.w_gap,
        },
        "review_reason": reasons,
    }


@dataclass
class ScoringStats:
    selected: int = 0
    scored: int = 0
    rescored: int = 0
    malformed: int = 0
    embedded: int = 0  # cache misses sent to the embedder
    stopped: str | None = None
    stop_detail: str = ""


def run_scoring(
    conn,
    graph_cfg: GraphConfig,
    weights: Weights,
    profile: ProfileSkills,
    embed_factory: Callable[[], Embedder] | None = None,
) -> ScoringStats:
    """Score the selected jobs. The caller holds the graph lock and has checked preconditions."""
    stats = ScoringStats()
    skills = [s for s in load_skills(conn, graph_cfg) if s.vector is not None]
    reference = [s for s in skills if s.mine]
    levels = {s.name: s.level for s in reference}
    cache: dict[str, np.ndarray] = {s.name: s.vector for s in skills}
    embedder: list[Embedder] = []

    def embed(names):
        stats.embedded += len(names)
        if embed_factory is None:
            raise PreconditionError(f"{len(names)} JD skill(s) have no vector -- run `python -m hiredra.graph sync`")
        if not embedder:
            embedder.append(embed_factory())
        return embedder[0](names)

    jobs = select_jobs(conn, weights, graph_cfg, profile)
    stats.selected = len(jobs)
    for job in jobs:
        try:
            raw = json.loads(job["skills"] or "[]")
            if not isinstance(raw, list):
                raise ValueError("skills is not a list")
            items = [
                {"name": str(i["name"]), "requirement": str(i["requirement"])}
                for i in raw
                if isinstance(i, dict) and str(i.get("name", "")).strip()
            ]
            if len(items) != len(raw):
                raise ValueError("an entry without a name")
            for i in items:
                skill_weight(i["requirement"], weights)  # an unknown requirement is malformed, not a crash
        except (ValueError, TypeError, KeyError) as err:
            add_note(conn, job["job_id"], f"scoring: malformed skills, not scored ({err})")
            stats.malformed += 1
            continue

        names = [resolve_alias(conn, i["name"]) or i["name"] for i in items]  # F2: alias BEFORE the cache
        try:
            fits = fit_skills(names, reference, graph_cfg, embed, cache=cache)
        except Exception as err:
            kind = classify_failure(err)
            if kind is None:
                raise
            stats.stopped, stats.stop_detail = kind, str(err).splitlines()[0][:300] if str(err) else kind
            break

        result = score_job([(f.fit, i["requirement"]) for f, i in zip(fits, items)], job["domain_bucket"], weights)
        reasons = review_reasons(job)
        features = build_features(job, items, fits, result, reasons, weights, graph_cfg, levels)
        set_score(
            conn,
            job["job_id"],
            score=round(result.score, 6),
            features=json.dumps(features, sort_keys=True),
            review_reason=json.dumps(reasons) if reasons else None,
            score_version=weights.score_version,
            embed_tag=graph_cfg.embed_tag,
            skills_hash=profile.skills_hash,
            profile_hash=profile.file_hash,
        )
        if job["status"] == "extracted":
            set_status(conn, job["job_id"], "scored")
            stats.scored += 1
        else:
            add_note(conn, job["job_id"], f"scoring: re-scored under {weights.score_version}")
            stats.rescored += 1
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# views (D13: nothing below the line is hidden, only not promoted)
# ─────────────────────────────────────────────────────────────────────────────
_LIST_COLS = "job_id, title, company, score, review_reason, released_at, score_features"


def promoted(conn, weights: Weights) -> list[dict]:
    """Released jobs, plus clean (no review reason) jobs at or above the apply line, best first.
    With no apply line yet, only released jobs are promoted (Step 8 refuses to start)."""
    if weights.apply_line is None:
        where, params = "released_at IS NOT NULL", ()
    else:
        where, params = "released_at IS NOT NULL OR (review_reason IS NULL AND score >= ?)", (weights.apply_line,)
    rows = conn.execute(
        f"SELECT {_LIST_COLS} FROM jobs WHERE status = 'scored' AND ({where}) ORDER BY score DESC, job_id", params
    )
    return [dict(r) for r in rows]


def all_scored(conn) -> list[dict]:
    rows = conn.execute(f"SELECT {_LIST_COLS} FROM jobs WHERE status = 'scored' ORDER BY score DESC, job_id")
    return [dict(r) for r in rows]


def review_list(conn) -> list[dict]:
    rows = conn.execute(
        f"SELECT {_LIST_COLS} FROM jobs WHERE status = 'scored' AND review_reason IS NOT NULL AND released_at IS NULL "
        "ORDER BY score DESC, job_id"
    )
    return [dict(r) for r in rows]
