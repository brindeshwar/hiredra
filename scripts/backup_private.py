"""Off-machine backup of the files git can't give back (plan unit BK1, decision A6).

Copies the owner's hand-made, git-ignored files into a private folder outside the repo
(for example a cloud-synced personal drive), set as HIREDRA_BACKUP_DIR in .env:

    <dir>/hiredra-backup-<YYYY-MM-DD>/   the copied files, same relative paths as the repo
                                         + MANIFEST.sha256 (one "<sha256>  <path>" line per file)

A re-run on the same day replaces that day's folder only. The newest 8 dated folders are kept.
The DB is NOT copied by default (big, and rebuildable by re-running stages). --with-db adds a
`pg_dump` of the Postgres database (decision D1.13) as data/hiredra.dump, then checks it: the dump
is restored into a scratch database and every table's row count must match the live one. The
counts are saved beside it as data/hiredra.counts.json. .env is never copied: keys can be
re-issued, and a secret has no place in a synced folder.

pg_dump and pg_restore run inside the docker compose service `db` (D1.1), so nothing is installed
on the host and the tools always match the server version. Restore a dump with:

    docker compose exec -T db pg_restore -U hiredra -d <empty database> --no-owner < hiredra.dump

    uv run python scripts/backup_private.py
    uv run python scripts/backup_private.py --with-db
    uv run python scripts/backup_private.py --verify <dir>/hiredra-backup-2026-10-17
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

REPO = Path(__file__).resolve().parents[1]

# Repo-relative files, folders and globs. A missing one is reported and skipped, not an error:
# most appear only as later units land (the profile, the calibration ratings, the facts file).
ITEMS = (
    "CLAUDE.md",  # git-ignored: lives only on this disk
    "docs",  # git-ignored: the design docs and the project plan
    "config",  # tracked, plus the git-ignored companies.wide.yml
    "calibration",
    "data/calibration",
    "data/master_profile.md",
    "data/autofill_facts.yml",
    "data/profile_links.json",
    "data/profile_overlay.json",
    "data/merge_proposals.applied-*.json",  # the owner's reviewed skill-merge decisions
)
DUMP = "data/hiredra.dump"  # pg_dump custom format
COUNTS = "data/hiredra.counts.json"
RESTORE_CHECK_DB = "hiredra_restore_check"
PG_TOOLS = ("docker", "compose", "exec", "-T", "db")  # the compose service HIREDRA_DATABASE_URL points at
MANIFEST = "MANIFEST.sha256"
KEEP = 8
FOLDER_RE = re.compile(r"^hiredra-backup-\d{4}-\d{2}-\d{2}$")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _expand(repo: Path) -> tuple[list[Path], list[str]]:
    """Every file under ITEMS (relative to repo), and the items that matched nothing."""
    files: list[Path] = []
    missing: list[str] = []
    for item in ITEMS:
        matches = sorted(repo.glob(item)) if "*" in item else [repo / item]
        found = [m for m in matches if m.exists()]
        if not found:
            missing.append(item)
        for m in found:
            if m.is_dir():
                files.extend(sorted(p for p in m.rglob("*") if p.is_file() and "__pycache__" not in p.parts))
            else:
                files.append(m)
    return [f.relative_to(repo) for f in files], missing


def _check_target(repo: Path, backup_dir: Path) -> None:
    repo, backup_dir = repo.resolve(), backup_dir.resolve()
    if backup_dir == repo or backup_dir.is_relative_to(repo):
        raise SystemExit(f"refusing: HIREDRA_BACKUP_DIR ({backup_dir}) is inside the repo. Pick a folder outside it.")


def _pg_tool(args: list[str], *, stdin=None, stdout=None) -> None:
    subprocess.run([*PG_TOOLS, *args], stdin=stdin, stdout=stdout, cwd=REPO, check=True)


def table_counts(url: str) -> dict[str, int]:
    """Rows per table in the public schema."""
    with psycopg.connect(url) as conn:
        tables = [r[0] for r in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1")]
        count = sql.SQL("SELECT count(*) FROM {}")
        return {t: conn.execute(count.format(sql.Identifier(t))).fetchone()[0] for t in tables}


def dump_db(url: str, dest: Path) -> dict[str, int]:
    """pg_dump the database to ``dest`` and prove the dump restores. Returns the row counts it holds."""
    info = conninfo_to_dict(url)
    before = table_counts(url)
    with dest.open("wb") as f:
        _pg_tool(["pg_dump", "-U", info["user"], "-d", info["dbname"], "-Fc"], stdout=f)
    if table_counts(url) != before:  # the dump's snapshot is then of unknown counts: nothing to check it against
        raise SystemExit("refusing: the database changed during the dump. Run the backup when no stage is running.")
    restore_check(url, dest, before)
    return before


def restore_check(url: str, dump: Path, expected: dict[str, int]) -> None:
    """Restore ``dump`` into a scratch database on the same server and compare row counts; the scratch database
    is dropped either way."""
    admin = make_conninfo(url, dbname="postgres")
    drop = sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(RESTORE_CHECK_DB))
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(drop)
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(RESTORE_CHECK_DB)))
    try:
        with dump.open("rb") as f:
            user = conninfo_to_dict(url)["user"]
            _pg_tool(["pg_restore", "-U", user, "-d", RESTORE_CHECK_DB, "--no-owner", "--exit-on-error"], stdin=f)
        got = table_counts(make_conninfo(url, dbname=RESTORE_CHECK_DB))
    finally:
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute(drop)
    if got != expected:
        diff = {
            t: (expected.get(t), got.get(t)) for t in sorted(set(expected) | set(got)) if expected.get(t) != got.get(t)
        }
        raise SystemExit(f"restore check failed (table: live rows, restored rows): {diff}")


def backup(repo: Path, backup_dir: Path, today: dt.date, db_url: str | None = None) -> Path:
    """Write today's backup folder, prune to the newest KEEP, return the folder."""
    """``db_url`` given: the database is dumped too (``--with-db``)."""
    _check_target(repo, backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    final = backup_dir / f"hiredra-backup-{today.isoformat()}"
    work = backup_dir / (final.name + ".partial")  # built aside, so a failed run never eats a good copy
    if work.exists():
        shutil.rmtree(work)
    work.mkdir()

    files, missing = _expand(repo)
    for rel in files:
        dest = work / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / rel, dest)
    counts = None
    if db_url:
        (work / DUMP).parent.mkdir(parents=True, exist_ok=True)
        counts = dump_db(db_url, work / DUMP)
        (work / COUNTS).write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
        files += [Path(DUMP), Path(COUNTS)]

    lines = [f"{_sha256(work / rel)}  {rel.as_posix()}" for rel in sorted(files)]
    (work / MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")

    if final.exists():
        shutil.rmtree(final)
    work.rename(final)
    pruned = _prune(backup_dir)

    print(f"backup: {len(files)} file(s) -> {final}")
    if counts is not None:
        print(f"  database: {len(counts)} tables, {sum(counts.values()):,} rows; restore check OK")
    for item in missing:
        print(f"  skipped (not present yet): {item}")
    for p in pruned:
        print(f"  pruned old copy: {p.name}")
    return final


def _prune(backup_dir: Path) -> list[Path]:
    """Delete dated backup folders beyond the newest KEEP. Touches only folders named by this script."""
    dated = sorted(p for p in backup_dir.iterdir() if p.is_dir() and FOLDER_RE.match(p.name))
    old = dated[:-KEEP]
    for p in old:
        shutil.rmtree(p)
    return old


def verify(folder: Path) -> list[str]:
    """Re-hash a backup folder against its manifest. Returns the problems (empty = good)."""
    problems: list[str] = []
    listed: set[str] = set()
    for line in (folder / MANIFEST).read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        listed.add(rel)
        path = folder / rel
        if not path.is_file():
            problems.append(f"missing: {rel}")
        elif _sha256(path) != digest:
            problems.append(f"changed: {rel}")
    for path in folder.rglob("*"):
        rel = path.relative_to(folder).as_posix()
        if path.is_file() and rel != MANIFEST and rel not in listed:
            problems.append(f"not in manifest: {rel}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-db", action="store_true", help="also pg_dump the database, with a restore check")
    parser.add_argument("--verify", type=Path, metavar="FOLDER", help="check a backup folder against its manifest")
    args = parser.parse_args(argv)

    if args.verify:
        problems = verify(args.verify)
        for p in problems:
            print(p)
        print(f"verify: {'OK' if not problems else f'{len(problems)} problem(s)'} ({args.verify})")
        return 1 if problems else 0

    load_dotenv(REPO / ".env")
    target = os.environ.get("HIREDRA_BACKUP_DIR", "").strip()
    if not target:
        print("HIREDRA_BACKUP_DIR is not set. Add it to .env (a private folder outside the repo).", file=sys.stderr)
        return 2
    db_url = os.environ.get("HIREDRA_DATABASE_URL") if args.with_db else None
    if args.with_db and not db_url:
        print("HIREDRA_DATABASE_URL is not set (see .env.example).", file=sys.stderr)
        return 2
    backup(REPO, Path(target).expanduser(), dt.date.today(), db_url=db_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
