"""Postgres connection and the migration runner (plan unit DB1; decisions D1.2, D1.3).

    python -m hiredra.pg migrate     # apply every pending migrations/NNNN_*.sql
    python -m hiredra.pg status      # applied and pending migrations

Migrations are plain numbered ``.sql`` files in ``migrations/``, applied in name order. Each one runs in its own
transaction and is recorded in ``schema_migrations`` in that same transaction, so a failed file leaves nothing
half-applied. A session advisory lock keeps two processes from migrating at once.

An applied migration is never edited (decision D15.2): its sha256 is recorded with it, and ``migrate`` and
``status`` refuse to run when an applied file's text has changed or the file is gone, because the database would no
longer match the repo and nothing else would say so. A change goes in a new numbered file. Rows applied before the
checksum existed take today's file's hash the first time they are checked.

The connection string comes from ``HIREDRA_DATABASE_URL`` (.env); it is never written in code (CLAUDE.md rule 5).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"
_MIGRATE_LOCK = "hiredra:migrate"


def database_url() -> str:
    load_dotenv()
    url = os.environ.get("HIREDRA_DATABASE_URL")
    if not url:
        raise RuntimeError("HIREDRA_DATABASE_URL is not set (see .env.example)")
    return url


def connect(url: str | None = None) -> psycopg.Connection:
    return psycopg.connect(url or database_url())


def _files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.sql") if p.stem[:4].isdigit())


class MigrationChanged(RuntimeError):
    """An applied migration's file was edited or removed. The message names every one."""


def checksum(path: Path) -> str:
    """sha256 of the file's text with CRLF read as LF, so a Windows checkout hashes like the commit."""
    return hashlib.sha256(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _applied(conn: psycopg.Connection, directory: Path) -> set[str]:
    """The applied versions, after checking every applied file is unchanged (``MigrationChanged`` if not)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    conn.execute("ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS checksum text")
    files = {p.stem: p for p in _files(directory)}
    problems = []
    for version, recorded in conn.execute(
        "SELECT version, checksum FROM schema_migrations ORDER BY version"
    ).fetchall():
        path = files.get(version)
        if path is None:
            problems.append(f"{version}: applied, but its file is gone from {directory.name}/")
        elif recorded is None:
            conn.execute("UPDATE schema_migrations SET checksum = %s WHERE version = %s", (checksum(path), version))
        elif recorded != checksum(path):
            problems.append(f"{version}: applied, but {path.name} has been edited since")
    if problems:
        conn.rollback()
        raise MigrationChanged(
            "applied migrations no longer match the repo (put a change in a new numbered file):\n  "
            + "\n  ".join(problems)
        )
    return set(files) & {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}


def pending(conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[Path]:
    done = _applied(conn, directory)
    conn.commit()
    return [p for p in _files(directory) if p.stem not in done]


def migrate(conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every pending migration, in order. Returns the versions applied."""
    conn.execute("SELECT pg_advisory_lock(hashtext(%s))", (_MIGRATE_LOCK,))
    conn.commit()
    applied = []
    try:
        for path in pending(conn, directory):
            with conn.transaction():
                conn.execute(path.read_text(encoding="utf-8"))
                conn.execute(
                    "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)", (path.stem, checksum(path))
                )
            applied.append(path.stem)
    finally:
        conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (_MIGRATE_LOCK,))
        conn.commit()
    return applied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hiredra.pg", description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("migrate", "status"))
    args = ap.parse_args(argv)
    try:
        with connect() as conn:
            if args.command == "migrate":
                done = migrate(conn)
                print(f"applied {len(done)}: {', '.join(done) or 'nothing pending'}")
            else:
                todo = pending(conn)
                print(f"pending {len(todo)}: {', '.join(p.stem for p in todo) or 'none'}")
    except (RuntimeError, psycopg.Error) as exc:
        print(f"pg: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
