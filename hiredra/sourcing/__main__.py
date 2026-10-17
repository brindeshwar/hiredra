"""``python -m hiredra.sourcing run --config PATH [--db PATH] [--resume]``
``python -m hiredra.sourcing runs [--db PATH] [--limit N]``
``python -m hiredra.sourcing backfill-slugs [--db PATH] [--lists YML ...] [--roster JSON]``

The CLI owns the one shared connection to the operational DB: opens it once,
closes it once in a ``finally``. ``ScoringSink`` and ``SqliteRunStore`` share it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Job titles/companies/detail strings can carry arbitrary Unicode; a Windows
# console's default codepage (cp1252 etc.) can't encode all of it. Replace
# rather than crash the whole run's output after the DB write already succeeded.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

from hiredra.sourcing.config import load_companies, load_run_config
from hiredra.sourcing.orchestrator import RunReport, run as run_orchestrator
from hiredra.sourcing.persist import backfill_slugs
from hiredra.sourcing.runstore import SqliteRunStore
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
        run_store = SqliteRunStore(conn)
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


def _cmd_backfill_slugs(args: argparse.Namespace) -> int:
    entries = []
    for path in args.lists:
        if Path(path).exists():
            entries.extend(load_companies(path))
        else:
            print(f"  (list not found, skipped: {path})")
    roster: set[tuple[str, str]] = set()
    if Path(args.roster).exists():
        data = json.loads(Path(args.roster).read_text(encoding="utf-8"))["ats"]
        roster = {(ats, slug.lower()) for ats, slugs in data.items() for slug in slugs}
    else:
        print(f"  (roster not found, skipped: {args.roster})")

    conn = db.connect(args.db)
    try:
        db.init_db(conn)  # migrates to v5 first if needed
        s = backfill_slugs(conn, entries, roster)
        total = conn.execute("SELECT COUNT(*) FROM jobs WHERE source IN ('greenhouse', 'lever', 'ashby')").fetchone()[0]
        with_slug = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE source IN ('greenhouse', 'lever', 'ashby') AND company_slug IS NOT NULL"
        ).fetchone()[0]
    finally:
        conn.close()
    print(
        f"backfill-slugs: {s.rows} row(s) looked at -- slug from url={s.from_url} list={s.from_list} "
        f"roster={s.from_roster} unresolved={s.unresolved}; tagged={s.tagged}"
    )
    print(f"  ATS rows with a slug: {with_slug}/{total} ({100 * with_slug / total if total else 100:.1f}%)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hiredra.sourcing")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run a sourcing config")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--db", default=str(db.DEFAULT_DB_PATH))
    p_run.add_argument("--resume", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    p_runs = sub.add_parser("runs", help="list recent runs")
    p_runs.add_argument("--db", default=str(db.DEFAULT_DB_PATH))
    p_runs.add_argument("--limit", type=int, default=20)
    p_runs.set_defaults(func=_cmd_runs)

    p_bf = sub.add_parser("backfill-slugs", help="fill company_slug / list_tag on rows stored before v5")
    p_bf.add_argument("--db", default=str(db.DEFAULT_DB_PATH))
    p_bf.add_argument("--lists", nargs="+", default=["config/companies.yml", "config/companies.wide.yml"])
    p_bf.add_argument("--roster", default="reference/open-jobs/slugs.json")
    p_bf.set_defaults(func=_cmd_backfill_slugs)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
