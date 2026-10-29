"""Tests for scripts/backup_private.py: what gets copied, the manifest, same-day overwrite, pruning, the guards."""

import datetime as dt
import json
import subprocess
from pathlib import Path

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from scripts import backup_private as bp

DAY = dt.date(2026, 10, 17)


def _write(root: Path, rel: str, text: str = "x") -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    _write(r, "CLAUDE.md", "constitution")
    _write(r, "docs/project-plan/README.md", "plan")
    _write(r, "config/companies.yml", "companies: []")
    _write(r, "config/companies.wide.yml", "companies: []")
    _write(r, "config/__pycache__/junk.pyc", "cache")
    _write(r, "data/master_profile.md", "profile")
    _write(r, "data/calibration/ratings.csv", "job,rating")
    _write(r, "data/merge_proposals.applied-20260922T205312Z.json", "{}")
    _write(r, "data/market_df.json", "rebuildable")
    _write(r, ".env", "GROQ_API_KEY=secret")
    return r


def _copied(folder: Path) -> set[str]:
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def test_copies_the_listed_files_and_nothing_else(repo: Path, tmp_path: Path):
    out = bp.backup(repo, tmp_path / "bk", DAY)
    assert out.name == "hiredra-backup-2026-10-17"
    assert _copied(out) == {
        "CLAUDE.md",
        "docs/project-plan/README.md",
        "config/companies.yml",
        "config/companies.wide.yml",
        "data/master_profile.md",
        "data/calibration/ratings.csv",
        "data/merge_proposals.applied-20260922T205312Z.json",
        bp.MANIFEST,
    }  # no .env, no rebuildable data, no caches, no DB


def test_manifest_hashes_match_and_verify_passes(repo: Path, tmp_path: Path):
    out = bp.backup(repo, tmp_path / "bk", DAY)
    lines = (out / bp.MANIFEST).read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(_copied(out)) - 1
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert digest == bp._sha256(repo / rel)
    assert bp.verify(out) == []


def test_verify_catches_changed_missing_and_extra_files(repo: Path, tmp_path: Path):
    out = bp.backup(repo, tmp_path / "bk", DAY)
    (out / "CLAUDE.md").write_text("tampered", encoding="utf-8")
    (out / "data/master_profile.md").unlink()
    _write(out, "stray.txt")
    assert sorted(bp.verify(out)) == [
        "changed: CLAUDE.md",
        "missing: data/master_profile.md",
        "not in manifest: stray.txt",
    ]
    assert bp.main(["--verify", str(out)]) == 1


def test_same_day_rerun_replaces_only_that_day(repo: Path, tmp_path: Path):
    bk = tmp_path / "bk"
    yesterday = bp.backup(repo, bk, DAY - dt.timedelta(days=1))
    _write(repo, "data/master_profile.md", "profile v2")
    first = bp.backup(repo, bk, DAY)
    (repo / "data/calibration/ratings.csv").unlink()  # backed up by the first run, gone from the repo since
    _write(repo, "data/master_profile.md", "profile v3")
    second = bp.backup(repo, bk, DAY)

    assert second == first
    assert (second / "data/master_profile.md").read_text(encoding="utf-8") == "profile v3"
    assert not (second / "data/calibration/ratings.csv").exists()  # the first run's own files do not survive
    assert (yesterday / "data/master_profile.md").read_text(encoding="utf-8") == "profile"
    assert sorted(p.name for p in bk.iterdir()) == ["hiredra-backup-2026-10-16", "hiredra-backup-2026-10-17"]


def test_keeps_newest_eight_and_leaves_other_folders_alone(repo: Path, tmp_path: Path):
    bk = tmp_path / "bk"
    (bk / "my-photos").mkdir(parents=True)
    for i in range(10):
        bp.backup(repo, bk, DAY + dt.timedelta(days=i))
    dated = sorted(p.name for p in bk.iterdir() if bp.FOLDER_RE.match(p.name))
    assert len(dated) == bp.KEEP
    assert dated[0] == "hiredra-backup-2026-10-19"  # the two oldest went
    assert (bk / "my-photos").is_dir()


def test_a_same_day_rerun_refuses_to_delete_files_it_did_not_write(repo: Path, tmp_path: Path):
    """2026-10-29: the owner moved the archived databases into today's folder; a re-run must not eat them."""
    bk = tmp_path / "bk"
    first = bp.backup(repo, bk, DAY)
    _write(first, "archive/old.db", "precious")
    with pytest.raises(SystemExit, match="did not write"):
        bp.backup(repo, bk, DAY)
    assert (first / "archive/old.db").read_text(encoding="utf-8") == "precious"
    assert not (bk / (first.name + ".partial")).exists()  # refused before any work


def test_pruning_skips_a_dated_folder_holding_someone_elses_files(repo: Path, tmp_path: Path, capsys):
    bk = tmp_path / "bk"
    oldest = bp.backup(repo, bk, DAY)
    _write(oldest, "notes.txt", "mine")
    for i in range(1, bp.KEEP + 1):
        bp.backup(repo, bk, DAY + dt.timedelta(days=i))
    assert (oldest / "notes.txt").exists()
    assert "NOT pruned (holds files this script did not write): hiredra-backup-2026-10-17" in capsys.readouterr().out


@pytest.mark.parametrize("inside", [".", "data/backups"])
def test_refuses_a_target_inside_the_repo(repo: Path, inside: str):
    with pytest.raises(SystemExit, match="inside the repo"):
        bp.backup(repo, repo / inside, DAY)
    assert not (repo / "hiredra-backup-2026-10-17").exists()


def _compose_db_up() -> bool:
    try:
        return subprocess.run([*bp.PG_TOOLS, "true"], cwd=bp.REPO, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _seed(url: str, rows: int) -> None:
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("CREATE TABLE jobs (id text)")
        conn.execute("INSERT INTO jobs SELECT 'j' || n FROM generate_series(1, %s) n", (rows,))
        conn.execute("CREATE TABLE runs (id text)")


def test_with_db_dumps_and_proves_the_dump_restores(repo: Path, tmp_path: Path, db_url: str):
    """Real pg_dump/pg_restore in the compose service: the dump restores to the same row counts."""
    if not _compose_db_up():
        pytest.skip("the docker compose service `db` is not running")
    _seed(db_url, 3)

    assert bp.DUMP not in _copied(bp.backup(repo, tmp_path / "plain", DAY))
    out = bp.backup(repo, tmp_path / "bk", DAY, db_url=db_url)
    assert {bp.DUMP, bp.COUNTS} <= _copied(out)
    assert json.loads((out / bp.COUNTS).read_text(encoding="utf-8")) == {"jobs": 3, "runs": 0}
    assert bp.verify(out) == []
    with psycopg.connect(make_conninfo(db_url, dbname="postgres")) as conn:  # the scratch database is gone
        assert conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (bp.RESTORE_CHECK_DB,)).fetchone() is None


def test_a_dump_that_restores_short_fails_and_leaves_no_folder(repo: Path, tmp_path: Path, db_url: str, monkeypatch):
    """pg_restore is faked to restore nothing: the counts differ, the run stops, today's folder is not written."""
    _seed(db_url, 2)
    monkeypatch.setattr(bp, "_pg_tool", lambda args, **kw: None)
    with pytest.raises(SystemExit, match=r"restore check failed .*'jobs': \(2, None\)"):
        bp.backup(repo, tmp_path / "bk", DAY, db_url=db_url)
    assert not (tmp_path / "bk" / "hiredra-backup-2026-10-17").exists()


def test_a_database_written_during_the_dump_is_refused(tmp_path: Path, monkeypatch):
    counts = iter([{"jobs": 5}, {"jobs": 6}])
    monkeypatch.setattr(bp, "table_counts", lambda url: next(counts))
    monkeypatch.setattr(bp, "_pg_tool", lambda args, **kw: None)
    restored = []
    monkeypatch.setattr(bp, "restore_check", lambda *a: restored.append(a))
    with pytest.raises(SystemExit, match="changed during the dump"):
        bp.dump_db("postgresql://hiredra@localhost/hiredra", tmp_path / "x.dump")
    assert restored == []


def test_missing_items_are_skipped_not_fatal(tmp_path: Path, capsys):
    empty = tmp_path / "repo"
    empty.mkdir()
    out = bp.backup(empty, tmp_path / "bk", DAY)
    assert _copied(out) == {bp.MANIFEST}
    assert "skipped (not present yet): data/master_profile.md" in capsys.readouterr().out


def test_main_needs_the_env_key(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(bp, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("HIREDRA_BACKUP_DIR", raising=False)
    assert bp.main([]) == 2
