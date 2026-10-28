"""``python -m hiredra.sourcing run --config PATH [--db URL] [--resume]``
``python -m hiredra.sourcing runs [--db URL] [--limit N]``

The CLI owns the one shared connection to the operational DB: opens it once,
closes it once in a ``finally``. ``ScoringSink`` and ``PgRunStore`` share it.
"""

from __future__ import annotations

import argparse
import sys

# Job titles/companies/detail strings can carry arbitrary Unicode; a Windows
# console's default codepage (cp1252 etc.) can't encode all of it. Replace
# rather than crash the whole run's output after the DB write already succeeded.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

from hiredra.sourcing.config import load_run_config
from hiredra.sourcing.orchestrator import RunReport, run as run_orchestrator
from hiredra.sourcing.runstore import PgRunStore
from hiredra.sourcing.sink import ScoringSink
from hiredra.storage import db


def _print_report(report: RunReport) -> None:
    tag = "RESUMED " if report.resumed else ""
    print(f"run {report.run_id} [{report.kind}/{report.config_name}] {tag}-> {report.status.upper()}")
    print(f"  {report.started_at} .. {report.finished_at}")

    print("  companies:")
    if not report.companies:
        print("    (none configured)")
    for c in report.companies:
        line = f"    {c.name:<24} {c.status:<10}"
        if c.provider:
            line += f" via {c.provider:<10}"
        line += f" jobs={c.jobs}"
        if c.detail:
            line += f"  ({c.detail[:100]})"
        print(line)
    if report.unroutable:
        print(f"  [!] {len(report.unroutable)} unroutable -- fix the company list: {list(report.unroutable)}")

    print("  jobspy:")
    for o in report.jobspy:
        line = f"    {o.site:<10} {o.status:<9} jobs={o.jobs}"
        if o.detail:
            line += f"  ({o.detail[:100]})"
        print(line)

    s = report.stats
    print(
        f"  stats: fetched={s.fetched} collapsed_in_run={s.collapsed_in_run} "
        f"matched_in_store={s.matched_in_store} persisted_new={s.persisted_new} "
        f"touched={s.touched}"
    )


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_run_config(args.config)
    conn = db.connect(args.db)
    try:
        db.init_db(conn)
        run_store = PgRunStore(conn)
        report = run_orchestrator(
            config,
            sink_factory=lambda run_id: ScoringSink(conn, run_id),
            run_store=run_store,
            resume=args.resume,
        )
        _print_report(report)
        return 0 if report.status != "failed" else 1
    finally:
        conn.close()


def _cmd_runs(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    try:
        rows = db.list_runs(conn, limit=args.limit)
        if not rows:
            print("(no runs yet)")
            return 0
        for r in rows:
            print(
                f"{r['run_id']:<24} {r['kind']:<9} {r['status']:<8} "
                f"started={r['started_at']} finished={r['finished_at'] or '-'} "
                f"fetched={r['fetched']} new={r['persisted_new']} touched={r['touched']}"
            )
        return 0
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hiredra.sourcing")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a sourcing config")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    p_run.add_argument("--resume", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    p_runs = sub.add_parser("runs", help="list recent runs")
    p_runs.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    p_runs.add_argument("--limit", type=int, default=20)
    p_runs.set_defaults(func=_cmd_runs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
