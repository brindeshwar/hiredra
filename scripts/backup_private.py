"""Off-machine backup of the files git can't give back (plan unit BK1, decision A6).

Copies the owner's hand-made, git-ignored files into a private folder outside the repo
(for example a cloud-synced personal drive), set as HIREDRA_BACKUP_DIR in .env:

    <dir>/hiredra-backup-<YYYY-MM-DD>/   the copied files, same relative paths as the repo
                                         + MANIFEST.sha256 (one "<sha256>  <path>" line per file)

A re-run on the same day replaces that day's folder only. The newest 8 dated folders are kept.
The DB is NOT copied by default (big, and rebuildable by re-running stages); --with-db adds a
consistent copy through SQLite's online backup. .env is never copied: keys can be re-issued, and
a secret has no place in a synced folder.

    uv run python scripts/backup_private.py
    uv run python scripts/backup_private.py --with-db
    uv run python scripts/backup_private.py --verify <dir>/hiredra-backup-2026-10-17
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

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
DB = "data/hiredra.db"
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


def backup(repo: Path, backup_dir: Path, today: dt.date, with_db: bool = False) -> Path:
    """Write today's backup folder, prune to the newest KEEP, return the folder."""
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
    if with_db:
        if (repo / DB).exists():
            (work / DB).parent.mkdir(parents=True, exist_ok=True)
            src, dst = sqlite3.connect(repo / DB), sqlite3.connect(work / DB)
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
            files.append(Path(DB))
        else:
            missing.append(DB)

    lines = [f"{_sha256(work / rel)}  {rel.as_posix()}" for rel in sorted(files)]
    (work / MANIFEST).write_text("\n".join(lines) + "\n", encoding="utf-8")

    if final.exists():
        shutil.rmtree(final)
    work.rename(final)
    pruned = _prune(backup_dir)

    print(f"backup: {len(files)} file(s) -> {final}")
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
    parser.add_argument("--with-db", action="store_true", help="also copy data/hiredra.db (SQLite online backup)")
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
    backup(REPO, Path(target).expanduser(), dt.date.today(), with_db=args.with_db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
