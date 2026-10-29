"""``python -m hiredra.storage <init|status> [--db URL]``

init    apply every pending migration (the same as ``python -m hiredra.pg migrate``) — idempotent
status  print the funnel snapshot (job count per status)
"""

from __future__ import annotations

import argparse
import sys

from hiredra.storage import db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hiredra.storage")
    parser.add_argument("command", choices=["init", "status"])
    parser.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL from .env)")
    args = parser.parse_args(argv)

    conn = db.connect(args.db)
    try:
        if args.command == "init":
            db.init_db(conn)
            print("database is up to date (every migration applied)")
        elif args.command == "status":
            counts = db.status_counts(conn)
            width = max(len(s) for s in db.STATUSES)
            for status in db.STATUSES:
                print(f"  {status:<{width}}  {counts[status]:>6}")
            print(f"  {'total':<{width}}  {sum(counts.values()):>6}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
