"""CLI for the between-runs graph maintenance steps (build 4.3).

    python -m hiredra.graph status                    # index counts + today's request budget (offline)
    python -m hiredra.graph harvest [--dry-run]       # open-jobs market vocabulary -> skill_vocab
    python -m hiredra.graph consistency [--sample N]  # batch vs one-at-a-time vs reversed, against stored vectors
    python -m hiredra.graph sync [--limit N]          # version gate (canary) + embed pending nodes
    python -m hiredra.graph sync --reembed-all        # FALLBACK: clear every vector, re-embed under one method
    python -m hiredra.graph merges [--show-low]       # DRY RUN: propose merges in three bands, write the review file
    python -m hiredra.graph merges --apply            # apply exactly the entries marked "approve": true
    python -m hiredra.graph overlay                   # DRY RUN: match the master profile's skills, write the links file
    python -m hiredra.graph overlay --apply           # rebuild the `mine` nodes from the profile (approved links only)
    python -m hiredra.graph calibrate --suggest       # blind real-pair suggestions (a,b only) for labeling
    python -m hiredra.graph calibrate [--write]       # Step 5: sweep HAVE/ADJ on the pair file (dry run without --write)

`merges` never changes the database unless `--apply` is given. The dry-run sorts
pairs into AUTO (pre-approved), ALIAS (your alias file, pre-approved) and REVIEW
(not approved -- you set "approve": true), and `--apply` applies only entries
marked true. `sync` and `consistency` check today's request budget first and
refuse WITHOUT calling the API if Google's daily quota is already known to be gone.
Every command that writes or spends budget (sync, consistency, harvest, merges
--apply) takes a run lock first, so two of them can never overlap (exit code 4).
Every command opens the DB through `init_db`, which migrates an older schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from hiredra.graph.config import DEFAULT_CONFIG_PATH, load_graph_config
from hiredra.graph.curation import (
    DEFAULT_ALIAS_PATH,
    DEFAULT_NEVER_MERGE_PATH,
    load_aliases,
    load_never_merge,
)
from hiredra.graph.harvest import DEFAULT_DF_PATH, harvest, load_df
from hiredra.graph.lock import DEFAULT_LOCK_PATH, LockHeld, RunLock
from hiredra.graph.merge import DEFAULT_REVIEW_PATH, apply_review, propose_merges, write_review
from hiredra.graph.usage import DEFAULT_USAGE_PATH, DailyUsage, reset_text
from hiredra.storage import (
    connect,
    init_db,
    invalidate_skill_vectors,
    skill_index_status,
    skills_needing_embedding,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _usage(cfg, args) -> DailyUsage:
    return DailyUsage(cfg.rpd_budget, path=args.usage_path)


def _warn_if_no_df(conn, weights, args) -> None:
    """Without the harvest's frequency file, embed order and merge-survivor choice
    silently fall back to shorter-name / alphabetical -- say so."""
    if not weights and conn.execute("SELECT 1 FROM skill_vocab WHERE source = 'market' LIMIT 1").fetchone():
        _log(
            f"WARNING: no market-frequency file at {args.df_path}; embed order and merge survivors "
            f"fall back to shorter name / alphabetical. Re-run `harvest` to recreate it."
        )


def _after_reset() -> str:
    return f"Run `sync` again after {reset_text(datetime.now(timezone.utc))}."


#: What to do next, per stop reason (see hiredra.graph.sync.EmbedReport.stopped).
_RESUME_HINT = {
    "daily_budget": _after_reset,
    "daily_quota": _after_reset,
    "transient": lambda: (
        "Google or the network kept failing after retries; the data is fine. Run `sync` again in a few minutes."
    ),
    "api_error": lambda: (
        "Google rejected the request (check GEMINI_API_KEY, the model name and access). Fix that first, then run `sync` again."
    ),
    "bad_response": lambda: (
        "Google returned vectors we can't use (wrong count or size). Read the error above before re-running: it may repeat."
    ),
}


def _refuse_if_exhausted(usage: DailyUsage) -> bool:
    if usage.exhausted():
        _log(f"NOT RUN (no API call made): {usage.summary()}")
        return True
    return False


def _cmd_status(conn, cfg, args) -> int:
    st = skill_index_status(conn, cfg.embed_tag)
    _log(f"embed tag : {cfg.embed_tag}")
    _log(f"nodes     : {st['total']}  (market {st['market']}, jd {st['jd']}, profile {st['profile']})")
    _log(f"embedded  : {st['current']} current, {st['stale']} stale, {st['unembedded']} not yet")
    _log(f"aliases   : {st['aliases']}")
    _log(f"queue     : {len(skills_needing_embedding(conn, cfg.embed_tag))} node(s) need embedding")
    _log(f"today     : {_usage(cfg, args).summary()}")
    _log(
        f"limits    : Google {cfg.requests_per_minute} req/min, {cfg.tokens_per_minute} tokens/min, "
        f"{cfg.requests_per_day} req/day (each input in a batch = 1 request); we stay at "
        f"{cfg.rpm_budget}/{cfg.tpm_budget}/{cfg.rpd_budget}"
    )
    return 0


def _cmd_harvest(conn, cfg, args) -> int:
    if cfg.harvest is None:
        raise SystemExit("graph config has no harvest section")
    report = harvest(
        conn,
        cfg.harvest,
        export_date=args.export_date,
        dry_run=args.dry_run,
        log=_log,
        df_path=Path(args.df_path),
        allow_partial=args.allow_partial,
    )
    _log("")
    _log(f"HARVEST {'(dry run -- nothing written) ' if report.dry_run else ''}from export {report.export_date}")
    _log("-" * 72)
    _log(f"  enriched jobs read            : {report.enriched_jobs}")
    _log(f"  in target role families       : {report.in_family_jobs}   ({', '.join(cfg.harvest.role_families)})")
    _log(f"  distinct folded skill tokens  : {report.distinct_tokens}")
    _log(f"  seen in >= {cfg.harvest.min_df} jobs (df filter)  : {report.after_min_df}")
    for reason, n in sorted(report.rejected_by_sanity.items()):
        _log(f"    - rejected by sanity ({reason}): {n}   e.g. {report.rejected_samples[reason][:5]}")
    _log(f"  CANDIDATE MARKET NODES        : {report.candidates}")
    if report.written:
        _log(f"    new nodes inserted          : {report.inserted}")
        _log(f"    already a node / alias      : {report.already_known}")
    elif not report.dry_run:
        _log(f"  NOTHING WRITTEN: {report.not_written_reason}")
    if report.failed:
        _log(f"  FAILED FILES (yield is understated): {report.failed}  {[report.per_ats[a] for a in report.failed]}")
    names = list(report.df)
    _log(f"\n  top 25 by document frequency : {[(n, report.df[n]) for n in names[:25]]}")
    _log(f"  bottom 25 (df={cfg.harvest.min_df} tail)     : {[(n, report.df[n]) for n in names[-25:]]}")
    return 1 if report.failed else 0


def _cmd_consistency(conn, cfg, args) -> int:
    from hiredra.graph.consistency import check_batch_consistency
    from hiredra.graph.embedder import build_embedder

    usage = _usage(cfg, args)
    if _refuse_if_exhausted(usage):
        return 2
    need = 3 * args.sample
    if usage.remaining() < need:
        _log(f"NOT RUN (no API call made): the check needs ~{need} requests but {usage.summary()}")
        return 2
    _log(f"consistency check: {args.sample} skills x 3 ways = ~{need} counted requests. {usage.summary()}")
    r = check_batch_consistency(conn, cfg, build_embedder(cfg, usage=usage), sample=args.sample)
    if r.skipped_reason:
        _log(f"SKIPPED: {r.skipped_reason}")
        return 2
    _log(f"sample: {list(r.sample)}")
    for c in r.conditions:
        _log(f"  {c.name:<11} min cosine vs stored {c.min_cosine:.6f}   (worst: {c.worst!r})")
    if r.error:
        _log(f"COULD NOT FINISH: {r.error}")
        return 2
    _log(f"threshold {r.threshold}; {r.requests_used} counted requests used")
    if r.consistent:
        _log("CONSISTENT: batching does not change the vectors. Safe to embed the rest the same way.")
        return 0
    _log("INCONSISTENT: the same skill gives a different vector depending on how it is sent.")
    _log("Do NOT embed the pending skills yet -- that would mix two kinds of vector in one index.")
    _log(
        "Fallback: `python -m hiredra.graph sync --reembed-all` clears every vector and re-embeds all "
        "nodes under ONE method (about two days at the daily budget), most common skills first."
    )
    return 3


def _cmd_sync(conn, cfg, args) -> int:
    from hiredra.graph.embedder import build_embedder
    from hiredra.graph.sync import ensure_index_current

    usage = _usage(cfg, args)
    if _refuse_if_exhausted(usage):
        return 2
    weights = load_df(args.df_path)
    _warn_if_no_df(conn, weights, args)
    if args.reembed_all:
        cleared = invalidate_skill_vectors(conn)
        _log(f"--reembed-all: cleared {cleared} stored vectors; the whole index will be re-embedded")
    embed = build_embedder(cfg, usage=usage)
    _log(
        f"embed tag {cfg.embed_tag}; {cfg.inputs_per_request} skills per request, pacing "
        f"{cfg.rpm_budget} req/min and {cfg.tpm_budget} tokens/min (each skill = 1 request)"
    )
    _log(f"today: {usage.summary()}")
    gate = ensure_index_current(
        conn, cfg, embed, limit=args.limit, weights=weights, daily_remaining=usage.remaining, log=_log
    )
    e = gate.embed
    _log("")
    _log(
        f"queue at start {e.queue_size}; attempted {e.attempted}; embedded {e.embedded} "
        f"in {e.chunks} chunk(s), {e.elapsed_s:.0f}s; still needing embedding: {e.remaining}"
    )
    _log(f"index now: {gate.after}")
    _log(f"today: {usage.summary()}")
    if e.stopped:
        _log(f"STOPPED ({e.stopped}): {e.error}")
        _log("Every finished chunk is saved. " + _RESUME_HINT[e.stopped]())
        return 2
    _log(
        "READY -- the whole index is embedded under the live version."
        if gate.ready
        else "not ready yet" + (f" (run `sync` again{'' if args.limit is None else ', without --limit'})")
    )
    return 0


def _print_band(title: str, proposals, note: str = "") -> None:
    """One band of the dry-run, grouped by survivor."""
    _log(f"\n{title}  --  {len(proposals)} pair(s){('   ' + note) if note else ''}")
    _log("-" * 96)
    by_survivor: dict[str, list] = {}
    for m in proposals:
        by_survivor.setdefault(m.survivor, []).append(m)
    for survivor, ms in by_survivor.items():
        first = ms[0]
        _log(f"{survivor}   [seen {first.survivor_seen}, df {first.survivor_df}]")
        for m in ms:
            cos = "  n/a " if m.cosine is None else f"{m.cosine:.3f}"
            _log(f"    <- {m.dup:<34} cos {cos}   [seen {m.dup_seen}, df {m.dup_df}]   ({m.reason})")


def _cmd_merges(conn, cfg, args) -> int:
    review = Path(args.review_path)
    try:
        never = load_never_merge(args.never_merge_file)
        aliases = {} if args.apply else load_aliases(args.alias_file)  # apply reads its aliases from the review file
    except ValueError as exc:
        _log(f"NOT RUN: a curation file is invalid -- {exc}")
        return 1
    if args.apply:
        try:
            report = apply_review(conn, cfg, review, never=never, log=_log)
        except (FileNotFoundError, ValueError) as exc:
            _log(f"NOT APPLIED: {exc}")
            return 1
        _log(
            f"\napplied {len(report.applied)} merge(s), skipped {len(report.skipped)}, "
            f"left {report.not_approved} not approved; review file archived to {report.archived_to}"
        )
        return 0

    weights = load_df(args.df_path)
    _warn_if_no_df(conn, weights, args)
    report = propose_merges(conn, cfg, weights=weights, aliases=aliases, never=never, show_low=args.show_low)
    write_review(report, review)

    auto, alias, rev = report.by_band("auto"), report.by_band("alias"), report.by_band("review")
    _log(
        f"MERGE PROPOSALS -- DRY RUN, nothing was changed  ({report.nodes_considered} embedded nodes considered, "
        f"{report.nodes_skipped} not eligible yet for the cosine bands)"
    )
    _log('Only entries marked "approve": true in the review file are ever applied.')
    _print_band(
        "AUTO -- pre-approved (approve: true)",
        auto,
        f"same spelling, or cosine >= {cfg.auto_cosine} and not a specialization / version variant",
    )
    _print_band("ALIAS FILE -- your entries, pre-approved (approve: true)", alias, "config/skill_aliases.yml")
    _print_band(
        'REVIEW -- NOT approved (approve: false); set "approve": true to merge one',
        rev,
        f"cosine >= {cfg.review_cosine}"
        + (f" (incl. the {cfg.dedup_threshold}-{cfg.review_cosine} low band)" if args.show_low else ""),
    )
    if report.hidden_low:
        _log(
            f"\n  hidden: {report.hidden_low} more review-type pair(s) between {cfg.dedup_threshold} and "
            f"{cfg.review_cosine} -- about half were wrong on the first real data. `merges --show-low` lists them."
        )
    for a, c, why in report.alias_notes:
        _log(f"  alias file: {a!r} -> {c!r} not proposed: {why}")
    if report.never_blocked:
        _log(f"  never-merge list: {report.never_blocked} pair(s) suppressed")

    _log("\n" + "=" * 96)
    _log(
        f"{len(auto)} AUTO + {len(alias)} ALIAS (pre-approved)   |   {len(rev)} REVIEW (unapproved)   |   "
        f"{report.hidden_low} hidden low-band"
    )
    _log(f"\nNEAR MISSES just under {report.threshold} (NOT proposed; for judging the threshold):")
    for a, b, c in report.near_misses:
        _log(f"    {a:<34} ~ {b:<34} cos {c:.3f}")
    _log(f"\nreview file written: {review}")
    _log(
        'Edit it (set "approve" on the entries you want, delete the ones you do not), then run '
        "`python -m hiredra.graph merges --apply`. Nothing applies until you do."
    )
    return 0


#: commands that write to the DB or spend Google budget take the run lock
def _cmd_overlay(conn, cfg, args) -> int:
    from hiredra.graph import overlay as ov
    from hiredra.graph.embedder import classify_failure
    from hiredra.profile import ProfileError, parse_skills

    try:
        profile = parse_skills(args.profile)
    except FileNotFoundError:
        _log(f"NOT RUN: no master profile at {args.profile} -- export it from the source doc first (owner task O1a)")
        return 1
    except (OSError, ProfileError) as exc:
        _log(f"NOT RUN: {exc}")
        return 1
    for warning in profile.warnings:
        _log(f"warning: {warning}")

    if args.apply:
        try:
            r = ov.apply_overlay(conn, profile, args.links_path, args.overlay_path)
        except (ov.OverlayError, ValueError) as exc:
            _log(f"NOT APPLIED: {exc}")
            return 1
        c = r.counts
        _log(
            f"overlay applied: {len(r.mine) + len(r.new_nodes)} mine node(s) -- exact {c['exact']}, hyphen "
            f"{c['hyphen']}, alias {c['alias']}, linked {c['linked']}, new {c['new']}; {len(r.aliases)} alias(es) added"
        )
        for node, tokens in r.collisions.items():
            _log(f"  collision: {', '.join(tokens)} all resolve to {node!r} -- tidy the profile in the source doc")
        _log(f"state written: {args.overlay_path}. Next: `python -m hiredra.graph sync` embeds the new profile nodes.")
        return 0

    usage = _usage(cfg, args)

    def embedder():
        from hiredra.graph.embedder import build_embedder

        return build_embedder(cfg, usage=usage)

    try:
        plan = ov.plan_overlay(conn, cfg, profile, embedder)
    except Exception as exc:
        kind = classify_failure(exc)
        if kind is None:
            raise
        _log(f"STOPPED ({kind}): {exc} -- nothing was written")
        return 2
    ov.write_links(plan, args.links_path)
    _log("PROFILE OVERLAY -- DRY RUN, nothing in the graph was changed")
    for step, title in (
        ("exact", "will apply: exact node name"),
        ("hyphen", "will apply: hyphens as spaces"),
        ("alias", "will apply: recorded alias"),
        ("vector", f'PROPOSED link (cosine >= {ov.LINK_COSINE}; set "approve": true to accept)'),
        ("new", "new profile node (no match)"),
    ):
        matches = plan.by_step(step)
        _log(f"\n{title}: {len(matches)}")
        for m in matches:
            arrow = f" -> {m.node}" if m.node and m.node != m.token else ""
            cos = f"  (cos {m.cosine:.3f})" if m.cosine is not None else ""
            _log(f"    [{m.tier}] {m.token}{arrow}{cos}")
    for node, tokens in plan.collisions().items():
        _log(f"  collision: {', '.join(tokens)} all resolve to {node!r}")
    _log(f"\nlinks file written: {args.links_path}. Approve the links you want, then `overlay --apply`.")
    return 0


def _cmd_calibrate(conn, cfg, args) -> int:
    from hiredra.graph import calibrate as cal
    from hiredra.graph.embedder import classify_failure

    if args.suggest:
        try:
            pairs, sidecar = cal.suggest_pairs(conn, cfg, n=args.n, seed=args.seed)
        except ValueError as exc:
            _log(f"NOT RUN: {exc}")
            return 1
        sidecar_path = Path(args.sidecar_path)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
        # stdout carries ONLY the pairs: no cosine, no band, no count (F5 blind labeling)
        print("a,b")
        for a, b in pairs:
            print(f"{a},{b}")
        print("(the cosine bands are in the private sidecar -- do not open it before labeling)", file=sys.stderr)
        return 0

    try:
        pairs = cal.load_pairs(args.pairs)
    except (OSError, cal.PairFileError) as exc:
        _log(f"NOT RUN: {exc}")
        return 1

    usage = _usage(cfg, args)

    def embedder():
        from hiredra.graph.embedder import build_embedder

        return build_embedder(cfg, usage=usage)

    try:
        cosines = cal.pair_cosines(conn, cfg, pairs, embedder, cache_path=args.cosines_path)
    except Exception as exc:
        kind = classify_failure(exc)
        if kind is None:
            raise
        _log(f"STOPPED ({kind}): {exc} -- cosines computed so far are cached; re-run later")
        return 2

    result = cal.calibrate(pairs, cosines)
    cid = cal.calibration_id_for(date.today())
    report = cal.render_report(result, pairs, cosines, cfg, cid)
    _log(
        f"HAVE={result.have:.2f} ADJ={result.adj:.2f} holdout F1: HAVE {result.holdout['have']['all'].f1:.3f}, "
        f"ADJ {result.holdout['adj']['all'].f1:.3f}"
    )
    for flag in result.flags:
        _log(f"  {flag}")
    if result.broken:
        return 3
    if not args.write:
        _log("DRY RUN: nothing written. Re-run with --write to update config/graph.yml and save the report.")
        return 0
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{cid}.md").write_text(report, encoding="utf-8")
    cal.write_fit_config(args.config, result.have, result.adj, cid)
    _log(f"wrote fit.have / fit.adj / fit.calibration_id={cid} to {args.config}, report {reports / f'{cid}.md'}")
    return 0


def _lock_label(args) -> str | None:
    if args.command in ("sync", "consistency"):
        return args.command
    if args.command == "harvest" and not args.dry_run:
        return "harvest"
    if args.command == "merges" and args.apply:
        return "merges --apply"
    if args.command == "calibrate" and not args.suggest:
        return "calibrate"
    if args.command == "overlay":
        return "overlay --apply" if args.apply else "overlay"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hiredra.graph", description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--df-path", default=str(DEFAULT_DF_PATH), help="market document-frequency sidecar")
    ap.add_argument("--usage-path", default=str(DEFAULT_USAGE_PATH), help="local daily request counter")
    ap.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH), help="run lock shared by every writing command")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    h = sub.add_parser("harvest")
    h.add_argument("--dry-run", action="store_true")
    h.add_argument("--export-date", default=None)
    h.add_argument(
        "--allow-partial",
        action="store_true",
        help="write nodes even if a file failed (document frequencies then cover a subset)",
    )
    c = sub.add_parser("consistency")
    c.add_argument("--sample", type=int, default=8, help="skills to compare (uses 3 x N counted requests)")
    s = sub.add_parser("sync")
    s.add_argument("--limit", type=int, default=None, help="embed at most N skills this run")
    s.add_argument(
        "--reembed-all",
        action="store_true",
        help="FALLBACK: clear every stored vector first, then re-embed everything under one method",
    )
    m = sub.add_parser("merges")
    m.add_argument(
        "--apply", action="store_true", help='apply the entries marked "approve": true (never without a dry-run file)'
    )
    m.add_argument(
        "--show-low",
        action="store_true",
        help="also list review-type pairs in the hidden low band (threshold .. review_cosine)",
    )
    m.add_argument("--review-path", default=str(DEFAULT_REVIEW_PATH))
    m.add_argument("--alias-file", default=str(DEFAULT_ALIAS_PATH), help="owner-editable canonical: [aliases] list")
    m.add_argument("--never-merge-file", default=str(DEFAULT_NEVER_MERGE_PATH), help="owner-editable never-merge pairs")
    o = sub.add_parser("overlay")
    o.add_argument("--apply", action="store_true", help="rebuild the mine nodes (only approved links are used)")
    o.add_argument("--profile", default=None, help="the master profile export (default data/master_profile.md)")
    o.add_argument("--links-path", default=None)
    o.add_argument("--overlay-path", default=None)
    k = sub.add_parser("calibrate")
    k.add_argument("--suggest", action="store_true", help="print blind real-pair suggestions (a,b only)")
    k.add_argument("--n", type=int, default=60, help="suggestions to draw (spread evenly across cosine bands)")
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--pairs", default=None, help="the labeled pair file (default calibration/skill_pairs.csv)")
    k.add_argument("--write", action="store_true", help="update config/graph.yml and save the report")
    k.add_argument("--cosines-path", default=None, help="cosine cache + regression snapshot")
    k.add_argument("--sidecar-path", default=None, help="private suggest sidecar (cosine bands)")
    k.add_argument("--reports-dir", default=None)
    args = ap.parse_args(argv)
    if args.command == "overlay":
        from hiredra.graph import overlay as ov
        from hiredra.profile import DEFAULT_PROFILE_PATH

        args.profile = args.profile or str(DEFAULT_PROFILE_PATH)
        args.links_path = args.links_path or str(ov.DEFAULT_LINKS_PATH)
        args.overlay_path = args.overlay_path or str(ov.DEFAULT_OVERLAY_PATH)
    if args.command == "calibrate":
        from hiredra.graph import calibrate as cal

        args.pairs = args.pairs or str(cal.DEFAULT_PAIRS_PATH)
        args.cosines_path = args.cosines_path or str(cal.DEFAULT_COSINES_PATH)
        args.sidecar_path = args.sidecar_path or str(cal.DEFAULT_SIDECAR_PATH)
        args.reports_dir = args.reports_dir or str(cal.DEFAULT_REPORTS_DIR)

    args.never_merge_file = getattr(args, "never_merge_file", str(DEFAULT_NEVER_MERGE_PATH))
    cfg = load_graph_config(args.config)
    handlers = {
        "status": _cmd_status,
        "harvest": _cmd_harvest,
        "consistency": _cmd_consistency,
        "sync": _cmd_sync,
        "merges": _cmd_merges,
        "calibrate": _cmd_calibrate,
        "overlay": _cmd_overlay,
    }

    def run() -> int:
        conn = connect(args.db)
        init_db(conn)
        try:
            return handlers[args.command](conn, cfg, args)
        finally:
            conn.close()

    label = _lock_label(args)
    if label is None:
        return run()
    try:
        with RunLock(args.lock_path, what=label):
            return run()
    except LockHeld as exc:
        _log(f"NOT RUN: {exc}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
