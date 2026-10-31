"""``python -m hiredra.sourcing run --config PATH [--db URL] [--resume]``
``python -m hiredra.sourcing runs [--db URL] [--limit N]``
``python -m hiredra.sourcing boards import PATH [--provenance NAME] [--db URL]``

A run crawls the ``boards`` table, picked by the run config's ``boards`` key (decision D2.1). A company list such
as ``config/companies.yml`` reaches the table through ``boards import``: new entries are added as ``pending``,
known ones only get blank fields filled, and ``enabled: false`` retires a board.

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

from pathlib import Path

from hiredra.sourcing.config import board_entry, load_companies, load_run_config
from hiredra.sourcing.orchestrator import RunReport, run as run_orchestrator
from hiredra.sourcing.provider import company_slug
from hiredra.sourcing.registry import REGISTRY, route
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
        companies = [board_entry(row) for row in db.load_boards(conn, config.boards)]
        report = run_orchestrator(
            config,
            companies=companies,
            sink_factory=lambda run_id: ScoringSink(conn, run_id),
            run_store=run_store,
            resume=args.resume,
        )
        _print_report(report)
        return 0 if report.status != "failed" else 1
    finally:
        conn.close()


def _cmd_boards_import(args: argparse.Namespace) -> int:
    entries = load_companies(args.path, args.provenance, include_disabled=True)
    counts: dict[str, int] = {}
    unroutable: list[str] = []
    conn = db.connect(args.db)
    try:
        db.init_db(conn)
        with conn.transaction():
            for entry in entries:
                provider = route(entry, REGISTRY)
                slug = company_slug(provider, entry) if provider is not None else None
                if provider is None or slug is None:
                    unroutable.append(entry.name)
                    continue
                outcome = db.import_board(
                    conn,
                    provider=provider.name,
                    board=slug,
                    name=entry.name,
                    provenance=str(entry.extra["provenance"]),
                    source_name=Path(args.path).name,
                    list_tag=entry.extra.get("tag"),
                    careers_url=entry.careers_url,
                    enabled=entry.extra.get("enabled", True),
                )
                counts[outcome] = counts.get(outcome, 0) + 1
    finally:
        conn.close()
    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "nothing"
    print(f"boards import {args.path}: {len(entries)} entries -> {summary}")
    if unroutable:
        print(f"  [!] no provider or board slug for {len(unroutable)}: {', '.join(unroutable)}")
    return 1 if unroutable else 0


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

    p_boards = sub.add_parser("boards", help="manage the boards table (what a run crawls)")
    boards_sub = p_boards.add_subparsers(dest="boards_command", required=True)
    p_import = boards_sub.add_parser("import", help="write a company list (e.g. config/companies.yml) into boards")
    p_import.add_argument("path")
    p_import.add_argument("--provenance", default=None, help="default: curated for companies.yml, else the file stem")
    p_import.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    p_import.set_defaults(func=_cmd_boards_import)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
