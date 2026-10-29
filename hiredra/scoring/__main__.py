"""CLI for the scoring run and its views (plan unit S6.2).

    python -m hiredra.scoring run                  # score extracted jobs + re-score stale ones (holds the graph lock)
    python -m hiredra.scoring list [--top N]       # the promoted jobs (above the apply line, or released)
    python -m hiredra.scoring list --all           # every scored job, below the line too (D13)
    python -m hiredra.scoring review               # the review list (suspect / likely_ghost / low_confidence / few_skills)
    python -m hiredra.scoring show <job_id>        # the full feature breakdown
    python -m hiredra.scoring release <job_id>     # the owner's override: promote it (review list or below the line)
    python -m hiredra.scoring reject <job_id>      # drop it with detail "review: rejected"
    python -m hiredra.scoring status               # stamps in use, stale counts, the profile export
    python -m hiredra.scoring rate --sample        # Step 7: write the blind rating sheet (30 jobs)
    python -m hiredra.scoring rate --ingest        # Step 7: read the ticked sheet into job_ratings.csv

Exit codes: 0 ok, 1 config or input error, 2 stopped (a precondition, or the model failed), 4 lock held.
"""

from __future__ import annotations

import argparse
import sys

from hiredra.graph.config import DEFAULT_CONFIG_PATH, load_graph_config
from hiredra.graph.lock import RunLock
from hiredra.graph.overlay import DEFAULT_OVERLAY_PATH
from hiredra.lock import LockHeld
from hiredra.profile import DEFAULT_PROFILE_PATH, ProfileError, parse_skills
from hiredra.scoring import rate as srate
from hiredra.scoring import run as sr
from hiredra.scoring.score import DEFAULT_WEIGHTS_PATH, load_weights
from hiredra.storage import connect, get_job, init_db, release_job, set_status

for _stream in (sys.stdout, sys.stderr):  # titles carry any Unicode; a cp1252 console must not crash
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _row(r: dict) -> str:
    f = r["score_features"] or {}
    flags = " ".join(r["review_reason"] or ())
    released = " RELEASED" if r["released_at"] else ""
    return (
        f"{r['score']:7.3f}  cov {f.get('coverage', 0):.2f}  gap {f.get('gap_distance', 0):.2f}  "
        f"{r['job_id']:<28} {r['company'][:20]:<20} {r['title'][:52]}{released}{'  [' + flags + ']' if flags else ''}"
    )


def _cmd_run(conn, args, graph_cfg, weights) -> int:
    try:
        profile = parse_skills(args.profile)
    except (OSError, ProfileError) as exc:
        _log(f"NOT RUN: {exc}")
        return 1
    try:
        with RunLock(what="scoring run", url=args.db):
            sr.check_preconditions(conn, graph_cfg, profile, args.overlay_path)

            def embedder():
                from hiredra.graph.embedder import build_embedder

                return build_embedder(graph_cfg)

            stats = sr.run_scoring(conn, graph_cfg, weights, profile, embedder, force=args.force)
    except LockHeld as exc:
        _log(f"NOT RUN: {exc}")
        return 4
    except sr.PreconditionError as exc:
        _log(f"NOT RUN: {exc}")
        return 2
    _log(
        f"scoring {weights.score_version}: selected={stats.selected} scored={stats.scored} rescored={stats.rescored} "
        f"embedded_misses={stats.embedded}"
    )
    if stats.embedded:
        _log("  hint: run `python -m hiredra.graph sync` after each extraction day, so scoring needs no embeddings")
    if stats.stopped:
        _log(f"STOPPED ({stats.stopped}): {stats.stop_detail} -- finished rows are kept")
        return 2
    if weights.apply_line is None:
        _log("  note: apply_line is not set yet (Step 7) -- nothing is promoted until it is, except released jobs")
    return 0


def _cmd_list(conn, args, graph_cfg, weights) -> int:
    rows = sr.all_scored(conn) if args.all else sr.promoted(conn, weights)
    if args.top:
        rows = rows[: args.top]
    title = "every scored job (D13: below the line too)" if args.all else "promoted jobs"
    _log(f"{title}: {len(rows)}")
    for r in rows:
        _log("  " + _row(r))
    return 0


def _cmd_review(conn, args, graph_cfg, weights) -> int:
    rows = sr.review_list(conn)
    _log(f"review list: {len(rows)} (release or reject each one)")
    for r in rows:
        _log("  " + _row(r))
    return 0


def _cmd_show(conn, args, graph_cfg, weights) -> int:
    job = get_job(conn, args.job_id)
    if job is None or not job["score_features"]:
        _log(f"no scored job {args.job_id!r}")
        return 1
    f = job["score_features"]
    _log(f"{job['title']} -- {job['company']}  [{job['status']}]")
    _log(f"  {job['url']}")
    _log(
        f"  score {job['score']:.3f} = domain {f['score_terms']['domain']:.3f} + skills {f['score_terms']['skills']:.3f}"
        f" + gap {f['score_terms']['gap']:.3f}   (bucket {f['domain_bucket']}, coverage {f['coverage']:.2f},"
        f" gap {f['gap_distance']:.3f})"
    )
    if f["review_reason"]:
        _log(f"  review: {', '.join(f['review_reason'])}")
    for band in ("have", "adjacent", "missing"):
        rows = [s for s in f["skills"] if s["band"] == band]
        _log(f"  {band} ({len(rows)}):")
        for s in rows:
            _log(
                f"    {s['skill']:<32} {s['req']:<13} nearest {s['nearest'] or '-':<28} s {s['s']:.3f}  fit {s['fit']:.3f}"
            )
    _log(f"  stamps: {job['score_version']} | {job['score_embed_tag']} | skills {job['score_skills_hash'][:12]}")
    return 0


def _cmd_release(conn, args, graph_cfg, weights) -> int:
    try:
        release_job(conn, args.job_id)
    except (KeyError, ValueError) as exc:
        _log(f"NOT RELEASED: {exc}")
        return 1
    _log(f"released {args.job_id}: it is promoted now, whatever its score or review reason")
    return 0


def _cmd_reject(conn, args, graph_cfg, weights) -> int:
    job = get_job(conn, args.job_id)
    if job is None or job["status"] != "scored":
        _log(f"NOT REJECTED: {args.job_id!r} is not a scored job")
        return 1
    set_status(conn, args.job_id, "dropped", detail="review: rejected")
    _log(f"rejected {args.job_id} (dropped, detail 'review: rejected')")
    return 0


def _cmd_status(conn, args, graph_cfg, weights) -> int:
    _log(f"current: score_version {weights.score_version} | embed_tag {graph_cfg.embed_tag}")
    try:
        profile = parse_skills(args.profile)
        _log(f"profile: Last updated {profile.last_updated or '(missing!)'} | file sha256 {profile.file_hash[:16]}")
        for w in profile.warnings:
            _log(f"  warning: {w}")
        skills_hash = profile.skills_hash
    except (OSError, ProfileError) as exc:
        _log(f"profile: cannot read it ({str(exc).splitlines()[0]})")
        skills_hash = None
    rows = conn.execute(
        "SELECT score_version, score_embed_tag, score_skills_hash, COUNT(*) AS n FROM jobs WHERE status = 'scored' "
        "GROUP BY 1, 2, 3 ORDER BY n DESC"
    ).fetchall()
    _log("scored rows by stamp:")
    if not rows:
        _log("  (none yet)")
    for r in rows:
        stale = (
            r["score_version"] != weights.score_version
            or r["score_embed_tag"] != graph_cfg.embed_tag
            or r["score_skills_hash"] != skills_hash
        )
        _log(
            f"  {r['n']:>6}  {r['score_version']} | {r['score_embed_tag']} | {str(r['score_skills_hash'])[:12]}{'  STALE' if stale else ''}"
        )
    _log("extraction stamps in use:")
    for r in conn.execute(
        "SELECT extraction_version, COUNT(*) AS n FROM jobs WHERE status IN ('extracted', 'scored') GROUP BY 1 ORDER BY n DESC"
    ):
        _log(f"  {r['n']:>6}  {r['extraction_version'] or '(no stamp)'}")
    return 0


def _cmd_rate(conn, args, graph_cfg, weights) -> int:
    try:
        if args.sample:
            n = srate.write_sample(conn, force=args.force)
            _log(f"wrote {srate.SHEET_PATH} ({n} jobs, two sittings); tick one box and write a reason per job")
        else:
            new, same = srate.ingest()
            _log(f"saved {new} rating(s) to {srate.RATINGS_PATH} ({same} unchanged)")
    except (srate.SheetError, OSError) as exc:
        _log(f"rate: {exc}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hiredra.scoring", description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    ap.add_argument("--graph-config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS_PATH))
    ap.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH))
    ap.add_argument("--overlay-path", default=str(DEFAULT_OVERLAY_PATH))
    sub = ap.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--force", action="store_true", help="re-score every scored job (after graph merges; 0 API)")
    lst = sub.add_parser("list")
    lst.add_argument("--top", type=int, default=None)
    lst.add_argument("--all", action="store_true")
    sub.add_parser("review")
    for name in ("show", "release", "reject"):
        sub.add_parser(name).add_argument("job_id")
    sub.add_parser("status")
    rate_p = sub.add_parser("rate", help="Step 7: the blind rating sheet")
    mode = rate_p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", action="store_true", help="draw 30 jobs and write the sheet")
    mode.add_argument("--ingest", action="store_true", help="save the ticked ratings")
    rate_p.add_argument("--force", action="store_true", help="with --sample: replace an existing sheet")
    args = ap.parse_args(argv)

    try:
        graph_cfg = load_graph_config(args.graph_config)
        weights = load_weights(graph_cfg, args.weights)
    except (OSError, ValueError) as exc:
        _log(f"config error: {exc}")
        return 1
    handlers = {
        "run": _cmd_run,
        "list": _cmd_list,
        "review": _cmd_review,
        "show": _cmd_show,
        "release": _cmd_release,
        "reject": _cmd_reject,
        "status": _cmd_status,
        "rate": _cmd_rate,
    }
    conn = connect(args.db)
    try:
        init_db(conn)
        return handlers[args.command](conn, args, graph_cfg, weights)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
