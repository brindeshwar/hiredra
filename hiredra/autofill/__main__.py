"""The application filler's command line. It prepares; it never submits.

    python -m hiredra.autofill scan --job KEY | --tailored [--no-drafts]   # read forms, build answers
    python -m hiredra.autofill batch [--min N]                             # the next batch of up to 10
    python -m hiredra.autofill batches                                     # list batches
    python -m hiredra.autofill fill --fixture three_page [--headless]      # fill a LOCAL test form, stop before Submit

``fill`` runs only against the local test forms in this build (tests/fixtures/forms/): the first fill on a real job
site happens later, with the owner watching. Draft answers use the model in config/resume.yml (NVIDIA while building).
"""

from __future__ import annotations

import argparse
import sys


def _scan(args) -> int:
    from hiredra.autofill.facts import load_facts
    from hiredra.autofill.prepare import prepare_job
    from hiredra.lock import LockHeld, RunLock
    from hiredra.quota import DailyUsage
    from hiredra.resume.llm import Caller
    from hiredra.resume.packet import build_gate
    from hiredra.resume.settings import load_resume_config
    from hiredra.resume.source import read_profile
    from hiredra.sourcing.http import HttpClient, HttpConfig
    from hiredra.storage import connect, init_db

    facts, profile, cfg = load_facts(), read_profile(), load_resume_config()
    conn = connect(args.db)
    try:
        init_db(conn)
        if args.job:
            jobs = [args.job]
        else:
            jobs = [r[0] for r in conn.execute(
                "SELECT j.job_key FROM jobs j WHERE j.status = 'tailored' AND NOT EXISTS "
                "(SELECT 1 FROM artifacts a WHERE a.job_id = j.id AND a.kind = 'form_scan') ORDER BY j.score DESC NULLS LAST"
            )]  # fmt: skip
        caller = None if args.no_drafts else Caller(conn, cfg, usage=DailyUsage(cfg.requests_per_day, conn, "nvidia",
                                                                                 cfg.model, count_errors=False))  # fmt: skip
        gate, http = build_gate(conn, profile), HttpClient(HttpConfig())
        print(f"{len(jobs)} tailored job(s) to read", flush=True)
        with RunLock("llm", what="form answers", kind=cfg.provider, url=args.db):
            for job_id in jobs:
                try:
                    r = prepare_job(conn, job_id, facts=facts, profile=profile, gate=gate, caller=caller, http=http,
                                    model=cfg.model)  # fmt: skip
                except Exception as exc:  # one bad form never stops the rest
                    print(f"  {job_id}: FAILED {type(exc).__name__}: {str(exc)[:200]}", flush=True)
                    continue
                state = "closed (dropped)" if r.dead else ", ".join(f"{k} {v}" for k, v in sorted(r.counts.items()))
                print(f"  {job_id}: {r.via}; {state or 'no fields'}", flush=True)
        return 0
    except LockHeld as exc:
        print(f"NOT RUN: {exc}")
        return 4
    finally:
        conn.close()


def _batch(args) -> int:
    from hiredra.autofill.batch import list_batches, make_batch
    from hiredra.storage import connect, init_db

    conn = connect(args.db)
    try:
        init_db(conn)
        if args.command == "batches":
            for b in list_batches(conn):
                decided = sum(1 for i in b["items"] if i["decision"])
                print(f"{b['batch_id']} {b['status']}: {len(b['items'])} job(s), {decided} decided")
            return 0
        b = make_batch(conn, min_size=args.min)
        if b is None:
            print(f"no batch: fewer than {args.min} tailored job(s) with a read form are waiting")
            return 0
        print(f"{b['batch_id']}: {len(b['items'])} job(s) queued for you")
        for i in b["items"]:
            print(f"  {i['position']:>2}. {i['title']} at {i['company']}  ({i['job_id']})")
        return 0
    finally:
        conn.close()


def _fill(args) -> int:
    from hiredra.autofill.driver import run_fixture

    keep = (
        None
        if args.headless
        else (lambda: input("\nThe form is filled up to its last page. Look at it, then press Enter to close... "))
    )
    report, counters = run_fixture(args.fixture, headless=args.headless, keep_open=keep)
    print(f"pages {report.pages}, last page reached: {report.last_page_reached}")
    print(f"filled: {', '.join(report.filled) or '-'}")
    print(f"left for you: {', '.join(report.paused) or '-'}")
    if report.skipped:
        print(f"skipped: {'; '.join(report.skipped)}")
    for r in report.refused:
        print(f"refused: {r}")
    print(f"submit events: {counters['submits']}, final-button clicks: {counters['finals']}")
    return 0 if counters["submits"] == 0 and counters["finals"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hiredra.autofill", description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="read the forms of tailored jobs and build their answers")
    pick = s.add_mutually_exclusive_group(required=True)
    pick.add_argument("--job")
    pick.add_argument("--tailored", action="store_true", help="every tailored job whose form is not read yet")
    s.add_argument("--no-drafts", action="store_true", help="no model call: free-text questions are left for you")
    b = sub.add_parser("batch", help="queue the next batch of up to 10")
    b.add_argument("--min", type=int, default=1, help="make a batch only when at least this many are ready")
    sub.add_parser("batches", help="list batches")
    f = sub.add_parser("fill", help="fill a local test form up to its last page (never submits)")
    f.add_argument("--fixture", required=True, choices=["three_page", "single_page"])
    f.add_argument("--headless", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "scan":
        return _scan(args)
    if args.command in ("batch", "batches"):
        return _batch(args)
    return _fill(args)


if __name__ == "__main__":
    sys.exit(main())
