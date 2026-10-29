"""Company discovery + one-time harvest -- a separate, off-to-the-side program.

NOT part of the live apply pipeline. See discovery/README.md. Writes to its own
database, never Hiredra's: a separate Postgres database on the same server
(``HIREDRA_DISCOVERY_DATABASE_URL``, default ``hiredra_discovery``, created on first
use). The 2026-09 corpus is frozen: data/archive/discovery-2026-09.db (the SQLite file of
the original harvest) and data/discovery/exports/.

Reuses the existing sourcing/filter machinery unchanged: the ATS providers,
the http layer (pacing + cooldowns), the orchestrator (runs table + --resume)
for the self-fetch pass, dedup + ScoringSink for persistence, and the funnel
(location -> seniority -> role-type) for scoring the corpus. This file is
orchestration glue over those pieces plus the open-jobs ingest and the CSV/
parquet export -- no new provider, dedup, or funnel logic.

    python -m discovery.harvest pool         # build the covered/not-covered split
    python -m discovery.harvest ingest       # pull + ingest open-jobs' 3 ATS parquet
    python -m discovery.harvest self-fetch   # fetch the not-covered companies live
    python -m discovery.harvest funnel       # run the existing 3-stage funnel
    python -m discovery.harvest export       # companies.csv + raw.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import duckdb
import pandas
import psycopg

from hiredra.filter import load_filter_config, run_funnel
from hiredra.sourcing._html import html_to_text
from hiredra.sourcing.config import Pacing, RunConfig, load_companies
from hiredra.sourcing.job import Job
from hiredra.sourcing.jobspy_source import JobSpyConfig, JobSpyResult
from hiredra.sourcing.http import HttpConfig
from hiredra.sourcing.orchestrator import run as run_orchestrator
from hiredra.sourcing.providers._common import iso_date
from hiredra.sourcing.providers.ashby import _CAREERS_RE as _ASHBY_RE
from hiredra.sourcing.providers.greenhouse import _SLUG_RE as _GH_RE
from hiredra.sourcing.providers.lever import _CAREERS_RE as _LEVER_HOST_RE
from hiredra.sourcing.runstore import PgRunStore
from hiredra.sourcing.sink import ScoringSink
from hiredra import pg
from hiredra.storage import connect, init_db

DATA_DIR = Path("data/discovery")
NOT_COVERED_PATH = DATA_DIR / "not_covered.yml"
POOL_SUMMARY_PATH = DATA_DIR / "pool_summary.json"
EXPORT_DIR = DATA_DIR / "exports"

ATS = ("greenhouse", "lever", "ashby")
OPENJOBS_BASE = "https://backend.dehnbostele.workers.dev"
OPENJOBS_SLUGS = Path("reference/open-jobs/slugs.json")
OPENJOBS_V2 = Path("reference/OpenJobs/data/companies_v2.json")
UA = {"User-Agent": "hiredra-discovery/0.1 (one-time company harvest, github.com/hiredra)"}


def _discovery_url() -> str:
    """Discovery's own database: ``HIREDRA_DISCOVERY_DATABASE_URL``, else ``hiredra_discovery`` on the server
    ``HIREDRA_DATABASE_URL`` names. Created (empty) on first use; ``init_db`` then migrates it."""
    url = os.environ.get("HIREDRA_DISCOVERY_DATABASE_URL")
    if url:
        return url
    url = psycopg.conninfo.make_conninfo(pg.database_url(), dbname="hiredra_discovery")
    admin = psycopg.conninfo.make_conninfo(url, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = 'hiredra_discovery'").fetchone():
            conn.execute('CREATE DATABASE "hiredra_discovery"')
    return url


def _q(value: str) -> str:
    """A YAML-safe double-quoted scalar -- some slugs are purely numeric or
    YAML-boolean-like and would otherwise parse as int/bool, not str."""
    return json.dumps(str(value))


# ── slug resolution (the one small piece of new glue) ──────────────────────
def resolve_ats_slug(careers_url: str) -> tuple[str, str] | None:
    """``careers_url`` -> ``(ats, slug)``, reusing each provider's own compiled
    regex (imported directly, not re-implemented) so this never drifts from
    what the providers themselves recognize."""
    m = _GH_RE.search(careers_url)
    if m:
        return ("greenhouse", m.group(1))
    m = _ASHBY_RE.search(careers_url)
    if m:
        return ("ashby", m.group(1))
    parts = urlsplit(careers_url)
    m = _LEVER_HOST_RE.match(parts.hostname or "")
    if m:
        segments = [s for s in parts.path.split("/") if s]
        if segments:
            return ("lever", segments[0])
    return None


# ── phase: pool ──────────────────────────────────────────────────────────────
def _our_lists() -> dict[tuple[str, str], str]:
    pool: dict[tuple[str, str], str] = {}
    for path in ("config/companies.yml", "config/companies.wide.yml"):
        if not Path(path).exists():
            continue
        for entry in load_companies(path):
            slug = entry.extra.get("slug") if entry.extra else None
            if entry.provider in ATS and slug:
                pool[(entry.provider, str(slug).lower())] = entry.name
            elif entry.careers_url:
                resolved = resolve_ats_slug(entry.careers_url)
                if resolved:
                    pool[(resolved[0], resolved[1].lower())] = entry.name
    return pool


def _openjobs_v2_links() -> dict[tuple[str, str], str]:
    data = json.loads(OPENJOBS_V2.read_text(encoding="utf-8"))
    pool: dict[tuple[str, str], str] = {}
    for company in data:
        for link in company.get("ats_links") or []:
            resolved = resolve_ats_slug(link)
            if resolved:
                key = (resolved[0], resolved[1].lower())
                pool.setdefault(key, company.get("name") or resolved[1])
                break
    return pool


def _openjobs_roster() -> set[tuple[str, str]]:
    data = json.loads(OPENJOBS_SLUGS.read_text(encoding="utf-8"))
    return {(ats, slug.lower()) for ats in ATS for slug in data["ats"][ats]}


def cmd_pool(_args: argparse.Namespace) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pool: dict[tuple[str, str], str] = {}
    pool.update(_openjobs_v2_links())
    pool.update(_our_lists())  # our own names win if a key appears in both

    roster = _openjobs_roster()
    covered = {k: v for k, v in pool.items() if k in roster}
    not_covered = {k: v for k, v in pool.items() if k not in roster}

    lines = ["companies:"]
    for (ats, slug), name in sorted(not_covered.items()):
        lines.append(f"  - name: {_q(name)}")
        lines.append(f"    slug: {_q(slug)}")
        lines.append(f"    provider: {ats}")
    NOT_COVERED_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "pooled": len(pool),
        "covered": len(covered),
        "not_covered": len(not_covered),
        "by_source": {
            "openjobs_v2_links": len(_openjobs_v2_links()),
            "our_lists": len(_our_lists()),
        },
    }
    POOL_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {NOT_COVERED_PATH}")
    return 0


# ── phase: ingest (open-jobs' crawled corpus for our 3 ATS) ────────────────
def _latest_export_date() -> str:
    req = urllib.request.Request(f"{OPENJOBS_BASE}/data/exports/", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    return sorted(x.rstrip("/") for x in d["dirs"])[-1]


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _ingest_parquet(conn, path: Path, ats: str) -> None:
    sink = ScoringSink(conn, run_id=f"open-jobs-ingest-{ats}")
    db = duckdb.connect()
    total = db.execute(f"select count(*) from read_parquet('{path.as_posix()}')").fetchone()[0]
    print(f"{ats}: {total:,} rows in the export")
    cur = db.execute(
        "select ats, slug, id, title, location, url, content, published_at, is_open "
        f"from read_parquet('{path.as_posix()}')"
    )
    done = 0
    while True:
        rows = cur.fetchmany(5000)
        if not rows:
            break
        jobs: list[Job] = []
        for row_ats, slug, jid, title, location, url, content, published_at, is_open in rows:
            if is_open is False or not url or not title:
                continue
            company = str(slug).replace("-", " ").replace("_", " ").title() if slug else row_ats
            try:
                jobs.append(
                    Job(
                        job_id=f"open-jobs-{row_ats}:{jid}",
                        url=url,
                        title=title,
                        company=company,
                        source=f"open-jobs-{row_ats}",
                        description=html_to_text(content) if content else None,
                        location=location,
                        date_posted=iso_date(published_at),
                    )
                )
            except (ValueError, TypeError):
                continue
        if jobs:
            sink.consume(jobs)
        done += len(rows)
        print(f"\r  {ats}: {done:,}/{total:,}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    print(f"{ats} ingested: {sink.stats()}")


def cmd_ingest(args: argparse.Namespace) -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect(_discovery_url())
    init_db(conn)
    date = args.date or _latest_export_date()
    print(f"open-jobs export date: {date}")
    try:
        for ats in ATS:
            local = DATA_DIR / f"_{ats}.parquet"
            if not local.exists():
                url = f"{OPENJOBS_BASE}/data/exports/{date}/jobs/{ats}.parquet"
                print(f"downloading {url}")
                _download(url, local)
            _ingest_parquet(conn, local, ats)
            local.unlink()
    finally:
        conn.close()
    return 0


# ── phase: self-fetch (companies open-jobs does not cover) ─────────────────
class _NoJobSpy:
    """Discovery is ATS-only -- no aggregator search. Matches the FakeJobSpySource
    injection seam orchestrator.run() already exposes for tests; reused here for
    the same reason (skip Path-3 without touching orchestrator.py)."""

    def fetch(self, ctx):
        return JobSpyResult(jobs=[], outcomes=())


def cmd_self_fetch(_args: argparse.Namespace) -> int:
    if not NOT_COVERED_PATH.exists():
        sys.exit("run `pool` first")
    conn = connect(_discovery_url())
    init_db(conn)
    try:
        config = RunConfig(
            kind="backfill",
            config_name="discovery-self-fetch",
            companies_path=str(NOT_COVERED_PATH),
            jobspy=JobSpyConfig(sites=()),
            http=HttpConfig(),
            pacing=Pacing(between_companies=1.0, jitter=0.5),  # same pacing proven at 1,000 companies, zero blocks
        )
        report = run_orchestrator(
            config,
            sink_factory=lambda run_id: ScoringSink(conn, run_id),
            run_store=PgRunStore(conn),
            jobspy_source=_NoJobSpy(),
            resume=True,  # safe when nothing is unfinished; resumes cleanly if interrupted
        )
        print(f"self-fetch: {report.status}")
        print(f"  stats: {report.stats}")
        non_ok = [c for c in report.companies if c.status != "ok"]
        if non_ok:
            print(f"  non-ok companies ({len(non_ok)}):")
            for c in non_ok:
                print(f"    {c.name:<30} {c.status:<10} {c.detail or ''}")
    finally:
        conn.close()
    return 0


# ── phase: funnel ────────────────────────────────────────────────────────────
def cmd_funnel(_args: argparse.Namespace) -> int:
    conn = connect(_discovery_url())
    init_db(conn)
    try:
        config = load_filter_config("config/filters.yml")
        stats = run_funnel(conn, config)
        print(stats)
    finally:
        conn.close()
    return 0


# ── phase: export ────────────────────────────────────────────────────────────
def cmd_export(_args: argparse.Namespace) -> int:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect(_discovery_url())
    try:
        jobs = conn.execute(
            "select job_key, company, source, url, status, status_detail from jobs order by id"
        ).fetchall()
        unknown_ids = {
            row["job_key"]
            for row in conn.execute(
                "select distinct j.job_key from job_events e join jobs j on j.id = e.job_id "
                "where e.kind = 'note' and e.detail like 'location:%'"
            )
        }
        columns = [r[0] for r in conn.execute(
            "select column_name from information_schema.columns where table_name = 'jobs' "
            "and column_name not in ('id', 'search') order by ordinal_position"
        )]  # fmt: skip
        raw = pandas.DataFrame(
            conn.execute(f"select {', '.join(columns)} from jobs order by id").fetchall(), columns=columns
        )
    finally:
        conn.close()

    # Keyed by (ats, slug) -- the real board identity -- not (ats, company):
    # `company` is a display label (open-jobs-ingested rows get it from a
    # lossy kebab-case -> Title Case transform of the slug; self-fetched rows
    # get it from whatever display name the source list used), never
    # guaranteed to be the exact fetchable URL token. The slug is instead
    # recovered from the job's own posting `url` via resolve_ats_slug() --
    # confirmed by a live spot-check: a naive company-as-slug 404s on real
    # boards, the url-derived slug 200s.
    rows: dict[tuple[str, str], tuple[str, str]] = {}  # (ats, slug) -> (company, tag)
    for job_id, company, source, url, status, status_detail in jobs:
        ats = source[len("open-jobs-") :] if source.startswith("open-jobs-") else source
        if ats not in ATS:
            continue
        dropped_on_location = status == "dropped" and (status_detail or "").startswith("location:")
        if dropped_on_location:
            continue  # never passed location -- not an India/remote-eligible company via this job
        resolved = resolve_ats_slug(url) if url else None
        if not resolved or resolved[0] != ats:
            continue  # shouldn't happen for real ATS-sourced job urls; skip rather than guess
        key = (ats, resolved[1])
        tag = "unknown" if job_id in unknown_ids else "confirmed"
        prev = rows.get(key)
        if prev is None or (prev[1] == "unknown" and tag == "confirmed"):
            rows[key] = (company, tag)  # confirmed always wins, regardless of row order

    companies_csv = EXPORT_DIR / "companies.csv"
    if companies_csv.exists():
        companies_csv.unlink()  # regenerable deliverable, not the frozen corpus
    with open(companies_csv, "w", encoding="utf-8", newline="") as f:
        f.write("ats,slug,company,tag\n")
        for (ats, slug), (company, t) in sorted(rows.items()):
            f.write(f'{ats},{slug},"{company.replace(chr(34), chr(39))}",{t}\n')
    confirmed = sum(1 for _, t in rows.values() if t == "confirmed")
    unknown_only = sum(1 for _, t in rows.values() if t == "unknown")
    print(f"wrote {companies_csv}: {len(rows)} companies ({confirmed} confirmed, {unknown_only} unknown-only)")

    raw_parquet = EXPORT_DIR / "raw.parquet"
    if raw_parquet.exists():
        sys.exit(f"{raw_parquet} already exists -- refusing to overwrite the frozen export")
    db = duckdb.connect()
    db.register("raw", raw)
    db.execute(f"COPY (SELECT * FROM raw) TO '{raw_parquet.as_posix()}' (FORMAT PARQUET)")
    print(f"wrote {raw_parquet}")
    print("back the database up with pg_dump (the frozen 2026-09 corpus stays in data/archive/discovery-2026-09.db)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m discovery.harvest")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_pool = sub.add_parser("pool", help="pool + split covered vs not-covered")
    p_pool.set_defaults(func=cmd_pool)

    p_ingest = sub.add_parser("ingest", help="pull + ingest open-jobs' 3 ATS parquet")
    p_ingest.add_argument("--date", default=None, help="override the export date (default: latest)")
    p_ingest.set_defaults(func=cmd_ingest)

    p_fetch = sub.add_parser("self-fetch", help="fetch not-covered companies live")
    p_fetch.set_defaults(func=cmd_self_fetch)

    p_funnel = sub.add_parser("funnel", help="run the existing 3-stage funnel")
    p_funnel.set_defaults(func=cmd_funnel)

    p_export = sub.add_parser("export", help="companies.csv + raw.parquet")
    p_export.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
