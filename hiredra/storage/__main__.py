"""``python -m hiredra.storage <init|status> [--db PATH]``

  init    create (or migrate) the SQLite DB file — idempotent
  status  print the funnel snapshot (job count per status)
"""

from __future__ import annotations

import argparse
import sys

from hiredra.storage import db


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hiredra.storage")
    parser.add_argument("command", choices=["init", "status"])
    parser.add_argument(
        "--db",
        default=str(db.DEFAULT_DB_PATH),
        help=f"SQLite file path (default: {db.DEFAULT_DB_PATH})",
    )
    args = parser.parse_args(argv)

    conn = db.connect(args.db)
    try:
        if args.command == "init":
            db.init_db(conn)
            print(f"initialized {args.db} (schema v{db.SCHEMA_VERSION})")
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
