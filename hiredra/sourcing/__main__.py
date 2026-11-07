"""``python -m hiredra.sourcing run --config PATH [--db URL] [--resume]``
``python -m hiredra.sourcing crawl [--config PATH] [--db URL] [--resume]``
``python -m hiredra.sourcing runs [--db URL] [--limit N]``
``python -m hiredra.sourcing boards import PATH [--provenance NAME] [--provider NAME] [--db URL]``
``python -m hiredra.sourcing companies merge [--apply] [--curated PATH] [--db URL]``
``python -m hiredra.sourcing boards discover-yc PATH [--limit N] [--dry-run] [--db URL]``
``python -m hiredra.sourcing boards discover-careers PATH [--limit N] [--dry-run] [--db URL]``
``python -m hiredra.sourcing boards discover-orphans [--limit N] [--dry-run] [--db URL]``
``python -m hiredra.sourcing boards discover-getro HOST... [--limit N] [--per-company N] [--dry-run] [--db URL]``

A run crawls the ``boards`` table, picked by the run config's ``boards`` key (decision D2.1). A company list such
as ``config/companies.yml``, or a public ``name,slug,url`` CSV of one ATS's companies (D2.2, ``--provider``
defaults to the file stem), reaches the table through ``boards import``: new entries are added as ``pending``,
known ones only get blank fields filled, and ``enabled: false`` retires a board.

The CLI owns the one shared connection to the operational DB: opens it once,
closes it once in a ``finally``. ``ScoringSink`` and ``PgRunStore`` share it.
"""

from __future__ import annotations

import argparse
import logging
import sys

# Job titles/companies/detail strings can carry arbitrary Unicode; a Windows
# console's default codepage (cp1252 etc.) can't encode all of it. Replace
# rather than crash the whole run's output after the DB write already succeeded.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from hiredra.filter import load_filter_config, passes_location, passes_role_type, passes_seniority_gate
from hiredra.sourcing import companies as company_merge
from hiredra.sourcing import discover
from hiredra.sourcing.config import INVENTORY_PROVENANCE, board_entry, load_companies, load_inventory, load_run_config
from hiredra.sourcing.http import FetchError, HttpClient, HttpConfig, HTTPError
from hiredra.sourcing.identity import company_slug as company_key
from hiredra.sourcing.orchestrator import RunReport, run as run_orchestrator
from hiredra.sourcing.provider import FetchContext, company_slug
from hiredra.sourcing.registry import REGISTRY, route
from hiredra.sourcing.providers._common import Robots, sitemap_locs
from hiredra.sourcing.runstore import PgRunStore
from hiredra.sourcing.sink import ScoringSink
from hiredra.lock import LockHeld, RunLock
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


def _known_eligible(job, funnel) -> bool:
    """For a board's cadence (D2.3): a posting whose READABLE place is India or remote. An unknown location is kept
    by the funnel but is no evidence: a re-sighted posting from a detail-page adapter carries none at all, so a
    board counted on it would never go weekly (measured 2026-11-03: iCIMS/SuccessFactors listings, all unknown)."""
    verdict = passes_location(job.title, job.location, job.description, funnel)
    return verdict.keep and verdict.reason != "unknown"


def _cmd_crawl(args: argparse.Namespace) -> int:
    """D5.1: crawl the boards that are due (``next_due_at`` passed, not resting after failures), under the
    ``sourcing`` lock so two crawls never overlap; then exit. A scheduler (Windows Task Scheduler, later the desktop
    app) triggers it; each crawl sets every board's next due time from its cadence (daily, or weekly for a board
    with no India/remote-eligible posting in its last crawls, D2.3)."""
    try:
        with RunLock("sourcing", what="crawl", kind="sourcing", url=args.db):
            return _cmd_run(args, due_only=True)
    except LockHeld as err:
        print(f"crawl: {err}")
        return 1


def _show_status() -> None:
    """The belt's status line in the crawl's output (stderr, unbuffered), and nothing else of the INFO chatter."""
    status = logging.getLogger("hiredra.sourcing.status")
    if not status.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
        status.addHandler(handler)
        status.setLevel(logging.INFO)
        status.propagate = False


def _cmd_run(args: argparse.Namespace, *, due_only: bool = False) -> int:
    config = load_run_config(args.config)
    _show_status()
    funnel = load_filter_config()
    conn = db.connect(args.db)
    # ``seen`` is asked from the fetch workers (D4.1): with more than one, it gets a connection of its own, one
    # question at a time; the run's connection stays the main thread's (sink, checkpoints, cooldowns).
    seen_conn = db.connect(args.db) if config.pacing.workers > 1 else conn
    seen_lock = threading.Lock()

    def seen(job_id: str) -> bool:
        with seen_lock:
            return db.job_stored(seen_conn, job_id)

    def stored_many(keys: list[str]) -> set[str]:
        with seen_lock:
            return db.jobs_stored(seen_conn, keys)

    try:
        db.init_db(conn)
        run_store = PgRunStore(conn)
        companies = [board_entry(row) for row in db.load_boards(conn, config.boards, due_only=due_only)]
        if due_only:
            print(f"crawl: {len(companies)} boards due")
        try:
            report = run_orchestrator(
                config,
                companies=companies,
                sink_factory=lambda run_id: ScoringSink(conn, run_id),
                run_store=run_store,
                resume=args.resume,
                seen=seen,
                seen_many=lambda keys: stored_many(keys),  # the dispatcher's question, never on the writer's connection
                eligible=lambda jobs: sum(_known_eligible(j, funnel) for j in jobs),
            )
        except KeyboardInterrupt:
            print("stopped: the boards finished so far are saved; go on with the same command and --resume")
            return 130
        _print_report(report)
        return 0 if report.status != "failed" else 1
    finally:
        if seen_conn is not conn:
            seen_conn.close()
        conn.close()


def _cmd_boards_import(args: argparse.Namespace) -> int:
    if Path(args.path).suffix.lower() == ".csv":
        provider = args.provider or Path(args.path).stem
        entries = load_inventory(
            args.path, provider, args.provenance or INVENTORY_PROVENANCE, board_from_url=args.board_from_url
        )
    else:
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
                    name=None if getattr(provider, "many_companies", False) else entry.name,
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
        more = f" and {len(unroutable) - 20} more" if len(unroutable) > 20 else ""
        print(f"  [!] no provider or board slug for {len(unroutable)}: {', '.join(unroutable[:20])}{more}")
    return 1 if unroutable else 0


def _cmd_companies_merge(args: argparse.Namespace) -> int:
    curated = company_merge.load_curated(args.curated)
    conn = db.connect(args.db)
    try:
        db.init_db(conn)
        rows = company_merge.company_rows(conn)
        renames, groups = company_merge.plan_merges(rows, curated)
        print(
            f"companies: {len(rows)}; renamed by the key rule: {len(renames)}; merge groups: {len(groups)} "
            f"({sum(len(g.retired) for g in groups)} companies folded in, {sum(g.jobs for g in groups)} jobs)"
        )
        for g in groups:
            members = " + ".join([g.old_slug, *(s for _, s in g.retired)])
            print(f"  [{g.reason}] {members} -> {g.slug} ({g.name}, {g.jobs} jobs)")
        for r in renames[: args.show]:
            print(f"  [rule] {r.old_slug} -> {r.new_slug}")
        if len(renames) > args.show:
            print(f"  ... and {len(renames) - args.show} more renames")
        hints = company_merge.suggestions(rows)
        if hints:
            print(f"possibly one employer, NOT merged (add to {company_merge.CURATED_PATH.name} to merge):")
            for a, b, n in hints:
                print(f"  {a} / {b} ({n} jobs)")
        if not args.apply:
            print("DRY RUN: nothing written. Re-run with --apply to merge.")
            return 0
        company_merge.apply_merges(conn, renames, groups)
        print(f"applied: {len(groups)} merges, {len(renames)} renames; old spellings kept as aliases")
        return 0
    finally:
        conn.close()


def _boarded_folds(conn) -> set[str]:
    """Squashed keys (and aliases) of every company that already has a board."""
    return discover.known_folds(
        conn.execute(
            "SELECT c.slug FROM companies c JOIN boards b ON b.company_id = c.id UNION "
            "SELECT a.alias FROM company_slug_aliases a JOIN boards b ON b.company_id = a.company_id"
        ).fetchall()
    )


def _funnel_counts(jobs, funnel) -> tuple[int, int]:
    """(India/remote-eligible, passing all three text stages): the funnel run in memory, nothing stored."""
    eligible = kept = 0
    for job in jobs:
        if not passes_location(job.title, job.location, job.description, funnel).keep:
            continue
        eligible += 1
        if passes_seniority_gate(job.title, job.description, funnel).keep and passes_role_type(job.title, funnel).keep:
            kept += 1
    return eligible, kept


def _cmd_boards_discover_careers(args: argparse.Namespace) -> int:
    """Slice 2.5: YC companies hiring in India or remote, still without a board -> their own careers site is read
    for a Greenhouse, Lever or Ashby board link (robots.txt respected, private addresses refused by the client).
    A board found is imported as pending with provenance ``atsdetect`` once the ATS answers for it and its slug
    carries the company's name, or its jobs name the company's web domain. Anything else is printed with ``?`` for
    the owner to check: measured 2026-11-01, a site's board link is not always the company's own (a careers page
    listing customers' jobs, a site redirected to its acquirer, an embed placeholder)."""
    funnel = load_filter_config()
    conn = db.connect(args.db)
    http = HttpClient(HttpConfig())
    ctx = FetchContext(http=http)

    def text(url: str) -> str | None:
        try:
            return http.get_text(url, follow_redirects=True, timeout=args.timeout)
        except FetchError:
            return None

    robots = Robots(text, http.config.user_agent)
    pages = refused = 0

    def page(url: str) -> str | None:
        nonlocal pages, refused
        if not robots.allowed(url):
            refused += 1
            return None
        pages += 1
        time.sleep(args.pause)
        return text(url)

    try:
        db.init_db(conn)
        candidates = [c for c in discover.yc_candidates(args.path, _boarded_folds(conn)) if c.website]
        candidates = candidates[: args.limit or None]
        print(f"discover-careers {args.path}: {len(candidates)} candidates with a website and no board yet")
        found = linked = doubted = eligible = kept = 0
        started = time.monotonic()
        by_provider: dict[str, int] = {}
        outcomes: dict[str, int] = {}  # import_board's answer: new / known / ...
        for i, cand in enumerate(candidates, 1):
            hit = discover.careers_walk(cand.website, page)
            if hit is not None:
                linked += 1
                provider, slug, where = hit
                result = discover.probe(
                    discover.Candidate(cand.name, (slug,), cand.website, cand.team_size),
                    [p for p in REGISTRY if p.name == provider],
                    ctx,
                )
                if result.provider is None:
                    print(f"  - {cand.name}: {provider}/{slug} linked from {where}, but the ATS does not answer")
                elif not (discover.slug_names(cand.name, slug) or discover.confirm(result) == "domain"):
                    doubted += 1
                    print(f"  ? {cand.name}: {provider}/{slug} ({len(result.jobs)} jobs) linked from {where}")
                else:
                    found += 1
                    by_provider[provider] = by_provider.get(provider, 0) + 1
                    e, k = _funnel_counts(result.jobs, funnel)
                    eligible, kept = eligible + e, kept + k
                    print(f"  + {cand.name}: {provider}/{slug} ({len(result.jobs)} jobs) linked from {where}")
                    if not args.dry_run:
                        with conn.transaction():
                            outcome = db.import_board(
                                conn, provider=provider, board=slug, name=cand.name, provenance="atsdetect",
                                source_name=Path(args.path).name, careers_url=where, alias=True,
                            )  # fmt: skip
                        outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if i % 25 == 0:
                print(
                    f"  .. {i}/{len(candidates)} sites, {found} found, {pages} pages, {time.monotonic() - started:.0f}s",
                    flush=True,
                )
        print(
            f"sites {len(candidates)}: board link found on {linked}, ATS answering and the company's {found} "
            f"({', '.join(f'{k} {v}' for k, v in sorted(by_provider.items()))}), not clearly theirs {doubted} (?); "
            f"pages read {pages}, "
            f"refused by robots.txt {refused}; jobs India/remote-eligible {eligible}, passing all three stages {kept}"
        )
        print(
            "DRY RUN: no board written."
            if args.dry_run
            else "boards: "
            + (", ".join(f"{v} {k}" for k, v in sorted(outcomes.items())) or "none found")
            + " (new ones added as pending, provenance atsdetect)"
        )
        return 0
    finally:
        http.close()
        conn.close()


def _cmd_boards_discover_yc(args: argparse.Namespace) -> int:
    """Slice 2.4: YC companies hiring in India or remote, without a board yet -> guessed slugs, checked live.
    A board found is imported as pending with provenance ``yc`` only when its jobs name the company's own web
    domain. A name-only match is printed with ``?`` for the owner to check: measured 2026-10-31, 3 of 7 checked were
    namesakes (Haven, Morph, Runway). The imported boards' jobs are run through the funnel's three text stages in
    memory, so the report says what they are worth (nothing about jobs is stored here)."""
    funnel = load_filter_config()
    conn = db.connect(args.db)
    ctx = FetchContext(http=HttpClient(HttpConfig()))
    try:
        db.init_db(conn)
        candidates = discover.yc_candidates(args.path, _boarded_folds(conn))[: args.limit or None]
        print(f"discover-yc {args.path}: {len(candidates)} candidates (hiring, India/remote region, no board yet)")
        found = eligible = kept = errors = 0
        by_provider: dict[str, int] = {}
        by_confirm: dict[str, int] = {}
        started = time.monotonic()
        for i, cand in enumerate(candidates, 1):
            result = discover.probe(cand, REGISTRY, ctx, sleep=time.sleep, pause=args.pause)
            errors += bool(result.errors)
            if result.provider is not None:
                confirmed = discover.confirm(result)
                by_confirm[confirmed] = by_confirm.get(confirmed, 0) + 1
                print(
                    f"  {dict(domain='+', name='?').get(confirmed, '-')} {cand.name}: {result.provider}/{result.slug} "
                    f"({len(result.jobs)} jobs, confirmed: {confirmed})"
                )
                if confirmed != "domain":
                    continue  # name-only or nothing: too likely a namesake to import unchecked
                found += 1
                by_provider[result.provider] = by_provider.get(result.provider, 0) + 1
                e, k = _funnel_counts(result.jobs, funnel)
                eligible, kept = eligible + e, kept + k
                if not args.dry_run:
                    with conn.transaction():
                        db.import_board(
                            conn, provider=result.provider, board=result.slug, name=cand.name, provenance="yc",
                            source_name=Path(args.path).name, alias=True,
                        )  # fmt: skip
            if i % 25 == 0:
                print(
                    f"  .. {i}/{len(candidates)} checked, {found} found, {time.monotonic() - started:.0f}s", flush=True
                )
        print(f"boards answering: {sum(by_confirm.values())}; confirmed by: {by_confirm}")
        print(
            f"imported-worthy {found} of {len(candidates)} ({', '.join(f'{k} {v}' for k, v in sorted(by_provider.items()))}); "
            f"probe errors on {errors}; jobs India/remote-eligible {eligible}, passing all three stages {kept}"
        )
        print("DRY RUN: no board written." if args.dry_run else f"imported {found} board(s) as pending, provenance yc")
        return 0
    finally:
        ctx.http.close()
        conn.close()


_ORPHAN_JOBS = """
SELECT j.company_id, c.name, j.title, j.apply_url, j.url
FROM jobs j JOIN companies c ON c.id = j.company_id
WHERE j.source LIKE 'jobspy-%%'
  AND NOT EXISTS (SELECT 1 FROM boards b WHERE b.company_id = j.company_id)
  AND NOT EXISTS (SELECT 1 FROM jobs k WHERE k.company_id = j.company_id AND k.source NOT LIKE 'jobspy-%%')
"""


def _cmd_boards_discover_orphans(args: argparse.Namespace) -> int:
    """Slice 2.6: companies we hold only through JobSpy (no board, no ATS job) -> their board. An apply link that
    names a board on an ATS we have an adapter for is taken as is (a ``grnh.se`` or Workable ``/j/`` short link is
    followed once, the board read from where it lands); otherwise the
    name's guessed slugs are probed, and a board found is kept only if it lists a job title we saw on the
    aggregator for that company. A company whose apply links show another ATS but name no board is counted, not
    guessed. A board found that we already hold under another company is that company under a second name: a
    merge suggestion is printed, nothing is written for it. New boards are imported as pending, provenance
    ``aggregator``."""
    funnel = load_filter_config()
    conn = db.connect(args.db)
    http = HttpClient(HttpConfig())
    ctx = FetchContext(http=http, seen=lambda job_id: True)  # a probe lists a board: never a detail request

    def text(url: str) -> str | None:
        try:
            return http.get_text(url, follow_redirects=True, timeout=10)
        except FetchError:
            return None

    def landing(url: str) -> str | None:
        """Where a short link points, else the page it lands on: the board is in one or the other. A redirect that
        names the board is read without requesting its target (Workable's job page answers with a challenge); one
        that stays on the short link's host and names none (Workable's ``/oops`` for a closed job) is the end."""
        try:
            return http.get_text(url, follow_redirects=False, timeout=10)
        except HTTPError as err:
            if err.location and discover.detect_board(err.location):
                return err.location
            if err.location is None or urlsplit(err.location).hostname == urlsplit(url).hostname:
                return None
        except FetchError:
            return None
        return text(url)

    robots = Robots(text, http.config.user_agent)
    try:
        db.init_db(conn)
        worklist = discover.orphans(conn.execute(_ORPHAN_JOBS).fetchall())
        if args.links_only:
            worklist = [o for o in worklist if o.board or o.short_links]
        worklist = worklist[: args.limit or None]
        print(f"discover-orphans: {len(worklist)} companies seen only on JobSpy")
        tally: dict[str, int] = {}
        other_ats: dict[str, int] = {}
        merges: list[tuple[str, str]] = []
        found = eligible = kept = 0
        started = time.monotonic()
        for i, o in enumerate(worklist):
            if i and i % 50 == 0:
                print(f"  .. {i}/{len(worklist)} companies, {found} new, {time.monotonic() - started:.0f}s", flush=True)
            board, how = o.board, "apply link"
            if board is None:
                for link in o.short_links:
                    if robots.allowed(link) and (page := landing(link)) and (board := discover.detect_board(page)):
                        how = "short link"
                        break
            if board is None and o.other_ats:
                other_ats[o.other_ats] = other_ats.get(o.other_ats, 0) + 1
                tally["another ATS, no board in its links"] = tally.get("another ATS, no board in its links", 0) + 1
                continue
            if board is None and args.links_only:
                tally["no board in its links"] = tally.get("no board in its links", 0) + 1
                continue
            providers = REGISTRY if board is None else [p for p in REGISTRY if p.name == board[0]]
            slugs = o.slugs if board is None else (board[1],)
            result = discover.probe(
                discover.Candidate(o.name, slugs), providers, ctx, sleep=time.sleep, pause=args.pause
            )
            if result.provider is None:
                tally["no board"] = tally.get("no board", 0) + 1
                if board is not None:
                    print(f"  - {o.name}: {board[0]}/{board[1]} from an {how}, but the ATS does not answer")
                continue
            if board is None:
                how = "guess"
                if not discover.same_opening(result.jobs, o.titles):
                    tally["guess, no shared job title"] = tally.get("guess, no shared job title", 0) + 1
                    print(f"  ? {o.name}: {result.provider}/{result.slug} ({len(result.jobs)} jobs), no shared title")
                    continue
            owner = conn.execute(
                "SELECT c.name FROM boards b JOIN companies c ON c.id = b.company_id"
                " WHERE b.provider = %s AND b.board = %s",
                (result.provider, result.slug),
            ).fetchone()
            if owner is not None:
                tally["board already ours"] = tally.get("board already ours", 0) + 1
                merges.append((o.name, owner[0]))
                print(f"  = {o.name}: {result.provider}/{result.slug} is already {owner[0]}'s board ({how})")
                continue
            found += 1
            tally[f"new, by {how}"] = tally.get(f"new, by {how}", 0) + 1
            e, k = _funnel_counts(result.jobs, funnel)
            eligible, kept = eligible + e, kept + k
            print(f"  + {o.name}: {result.provider}/{result.slug} ({len(result.jobs)} jobs, by {how})")
            if not args.dry_run:
                with conn.transaction():
                    db.import_board(
                        conn, provider=result.provider, board=result.slug, name=o.name, provenance="aggregator",
                        source_name="jobspy", alias=True,
                    )  # fmt: skip
        print(f"companies {len(worklist)}: " + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())))
        print(
            "on another ATS, board not in the link: "
            + (", ".join(f"{k} {v}" for k, v in sorted(other_ats.items())) or "none")
        )
        print(f"new boards {found}: jobs India/remote-eligible {eligible}, passing all three stages {kept}")
        if merges:
            print(f"one employer under two names ({len(merges)}); to merge, add to config/company_merges.yml:")
            for ours, theirs in merges:
                print(f"  - [{ours!r}, {theirs!r}]")
        print(
            "DRY RUN: no board written."
            if args.dry_run
            else f"imported {found} board(s) as pending, provenance aggregator"
        )
        return 0
    finally:
        http.close()
        conn.close()


def _cmd_boards_discover_getro(args: argparse.Namespace) -> int:
    """Slice 3.13: the employers' own boards behind VC networks' Getro job boards (``discover.getro_job``). Per
    portfolio company, its job pages are read in sitemap order until one's source link names a board on an ATS we
    have an adapter for (at most ``--per-company``). The board must carry the company's name (``slug_names``: an
    acquired company's jobs link to its parent's board, Frame.io's to Adobe's), must answer, must not be one we hold,
    and must list an India- or remote-eligible job: a network's portfolio is global, and a board with none would only cost crawl
    time. New boards are imported as pending, provenance ``getro``. robots.txt is obeyed; ``--pause`` keeps to the
    boards' ``Crawl-delay: 1``."""
    funnel = load_filter_config()
    conn = db.connect(args.db)
    http = HttpClient(HttpConfig())
    ctx = FetchContext(http=http, seen=lambda job_id: True)  # a probe lists a board: never a detail request

    def text(url: str) -> str | None:
        time.sleep(args.pause)
        try:
            return http.get_text(url, follow_redirects=True, timeout=30)
        except FetchError:
            return None

    robots = Robots(text, http.config.user_agent)
    tally: dict[str, int] = {}
    merges: list[tuple[str, str]] = []
    found = eligible = kept = 0
    try:
        db.init_db(conn)
        for host in args.hosts:
            sitemap = f"https://{host}/sitemap.xml"
            body = text(sitemap) if robots.allowed(sitemap) else None
            if body is None:
                print(f"{host}: no sitemap (or robots.txt refuses it)")
                continue
            pages, subs = sitemap_locs(body)
            for sub in subs:
                if robots.allowed(sub) and (more := text(sub)):
                    pages += sitemap_locs(more)[0]
            companies = discover.getro_job_pages(loc for loc, _ in pages)
            companies = dict(list(companies.items())[: args.limit or None])
            print(f"{host}: {len(companies)} companies with job pages", flush=True)
            for i, pages_of in enumerate(companies.values()):
                if i and i % 50 == 0:
                    print(f"  .. {i}/{len(companies)} companies, {found} new", flush=True)
                name = board = None
                for page_url in pages_of[: args.per_company]:
                    page = text(page_url) if robots.allowed(page_url) else None
                    job = discover.getro_job(page) if page else None
                    if job and (board := discover.detect_board(job[1])):
                        name = job[0]
                        break
                if board is None:
                    tally["no ATS link in its first jobs"] = tally.get("no ATS link in its first jobs", 0) + 1
                    continue
                if not discover.slug_names(name, board[1]):  # an acquired company's jobs on its parent's board
                    tally["board not named for it: for the owner"] = (
                        tally.get("board not named for it: for the owner", 0) + 1
                    )
                    print(f"  ? {name}: {board[0]}/{board[1]} does not carry its name (a parent's board?)")
                    continue
                owner = conn.execute(
                    "SELECT c.name FROM boards b JOIN companies c ON c.id = b.company_id"
                    " WHERE b.provider = %s AND lower(b.board) = lower(%s)",
                    board,
                ).fetchone()
                if owner is not None:
                    tally["board already ours"] = tally.get("board already ours", 0) + 1
                    if company_key(owner[0]) != company_key(name):
                        merges.append((name, owner[0]))
                    continue
                providers = [p for p in REGISTRY if p.name == board[0]]
                result = discover.probe(discover.Candidate(name, (board[1],)), providers, ctx)
                if result.provider is None:
                    tally["the ATS does not answer"] = tally.get("the ATS does not answer", 0) + 1
                    print(f"  - {name}: {board[0]}/{board[1]} does not answer")
                    continue
                e, k = _funnel_counts(result.jobs, funnel)
                if not e:
                    tally["no India/remote-eligible job"] = tally.get("no India/remote-eligible job", 0) + 1
                    continue
                found += 1
                eligible, kept = eligible + e, kept + k
                tally["new"] = tally.get("new", 0) + 1
                print(f"  + {name}: {result.provider}/{result.slug} ({len(result.jobs)} jobs, {e} eligible, {k} pass)")
                if not args.dry_run:
                    with conn.transaction():
                        db.import_board(
                            conn, provider=result.provider, board=result.slug, name=name, provenance="getro",
                            source_name=host, alias=True,
                        )  # fmt: skip
        print("companies: " + (", ".join(f"{v} {k}" for k, v in sorted(tally.items())) or "none"))
        print(f"new boards {found}: jobs India/remote-eligible {eligible}, passing all three stages {kept}")
        if merges:  # Getro's spelling is not stored: a better display name, never a second company to merge
            print(f"a board we hold under another spelling ({len(merges)}): Getro's name, ours")
            for getro_name, ours in merges:
                print(f"  - [{getro_name!r}, {ours!r}]")
        print(
            "DRY RUN: no board written." if args.dry_run else f"imported {found} board(s) as pending, provenance getro"
        )
        return 0
    finally:
        http.close()
        conn.close()


def _cmd_boards_discover_phenom(args: argparse.Namespace) -> int:
    """Phenom career sites (parity step, 2026-11-04): most only front another ATS, so each site's first listing page
    is read for its apply links (``discover.phenom_boards``). A board behind it on an ATS we read is imported unless
    we hold it; the site itself is imported as a ``phenom`` board only when most of its jobs apply where we cannot
    read them. Each needs a job of its own on that page whose READABLE place is India or remote. Unlike Getro's, a
    board is not checked for the company's name: the site is the employer's own, and its boards are too (GE
    HealthCare's is ``gehc``, TJX's ``tjx``: the check refused 11 of them). Provenance ``phenom``; robots.txt is
    obeyed; one request a ``--pause``."""
    from hiredra.sourcing.providers import phenom

    funnel = load_filter_config()
    conn = db.connect(args.db)
    http = HttpClient(HttpConfig())
    ctx = FetchContext(http=http, seen=lambda job_id: True)
    pace = phenom.Pace(args.pause)

    def text(url: str) -> str | None:
        try:
            return http.get_text(url, timeout=30)
        except FetchError:
            return None

    robots = Robots(text, http.config.user_agent)
    sites = load_inventory(args.path, "phenom", "phenom", board_from_url=False) if args.path.endswith(".csv") else []
    if not sites:
        print(f"{args.path}: no sites (a CSV with name, slug, url columns: the slug is the site host)")
        return 1
    tally: dict[str, int] = {}
    imported: list[tuple[str, str, str]] = []
    try:
        db.init_db(conn)
        for entry in sites[: args.limit or None]:
            host, name = entry.extra["slug"], entry.name
            if not robots.allowed(f"https://{host}/widgets"):
                tally["robots.txt refuses"] = tally.get("robots.txt refuses", 0) + 1
                continue
            try:
                rows, total = phenom.listing_page(ctx, host, 0, pace)
            except FetchError as err:
                tally["site does not answer"] = tally.get("site does not answer", 0) + 1
                print(f"  - {name} ({host}): {err}")
                continue

            def eligible(group: list[dict]) -> int:
                jobs = (phenom.to_job(name, host, r, None) for r in group)
                return sum(_known_eligible(j, funnel) for j in jobs if j is not None)

            groups = discover.phenom_boards(rows, (p.name for p in REGISTRY if p.name != "phenom"))
            wanted = {key: eligible(group) for key, group in groups.items() if key is not None}
            if len(groups.get(None, ())) * 2 > len(rows):  # most of its jobs reach us only through the site
                wanted[("phenom", host)] = eligible(groups[None])
            if not any(wanted.values()):
                tally["no India/remote job on its first page"] = (
                    tally.get("no India/remote job on its first page", 0) + 1
                )
                continue
            for (p, b), count in wanted.items():
                if not count:
                    tally["board with no India/remote job"] = tally.get("board with no India/remote job", 0) + 1
                    continue
                held = conn.execute(
                    "SELECT 1 FROM boards WHERE provider = %s AND lower(board) = lower(%s)", (p, b)
                ).fetchone()
                if held:
                    tally["board already ours"] = tally.get("board already ours", 0) + 1
                    continue
                tally["new"] = tally.get("new", 0) + 1
                imported.append((name, p, b))
                print(
                    f"  + {name}: {p}/{b} ({total} jobs on the site; {count} of its India/remote on page 1)", flush=True
                )
                if not args.dry_run:
                    with conn.transaction():
                        db.import_board(conn, provider=p, board=b, name=name, provenance="phenom",
                                        source_name=host, alias=True)  # fmt: skip
        print("sites and boards: " + (", ".join(f"{v} {k}" for k, v in sorted(tally.items())) or "none"))
        print(
            "DRY RUN: no board written."
            if args.dry_run
            else f"imported {len(imported)} board(s) as pending, provenance phenom"
        )
        return 0
    finally:
        http.close()
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
    p_crawl = sub.add_parser("crawl", help="crawl the boards that are due, then exit (D5.1; for a scheduler)")
    p_crawl.add_argument("--config", default="config/scoring.wide.yml", help="run settings (default: the wide one)")
    p_crawl.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    p_crawl.add_argument("--resume", action="store_true")
    p_crawl.set_defaults(func=_cmd_crawl)

    p_runs = sub.add_parser("runs", help="list recent runs")
    p_runs.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    p_runs.add_argument("--limit", type=int, default=20)
    p_runs.set_defaults(func=_cmd_runs)

    p_boards = sub.add_parser("boards", help="manage the boards table (what a run crawls)")
    boards_sub = p_boards.add_subparsers(dest="boards_command", required=True)
    p_import = boards_sub.add_parser("import", help="write a company list (e.g. config/companies.yml) into boards")
    p_import.add_argument("path")
    p_import.add_argument(
        "--provenance",
        default=None,
        help="default: curated for companies.yml, ats-inventory for a CSV, else the file stem",
    )
    p_import.add_argument("--provider", default=None, help="a CSV's ATS (default: the file stem, e.g. lever.csv)")
    p_import.add_argument(
        "--board-from-url",
        action="store_true",
        help="a CSV: read each board from its URL (the step-3 ATS lists, whose slug is not our board form)",
    )
    p_import.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    p_import.set_defaults(func=_cmd_boards_import)
    p_yc = boards_sub.add_parser("discover-yc", help="find boards for YC companies hiring in India/remote (D2.2)")
    p_yc.add_argument("path", help="yc-oss companies/all.json, downloaded into data/ (no licence stated: local only)")
    p_yc.add_argument("--limit", type=int, default=0, help="check only the first N candidates")
    p_yc.add_argument("--pause", type=float, default=0.5, help="seconds between probe requests")
    p_yc.add_argument("--dry-run", action="store_true", help="report only; write no board")
    p_yc.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    p_yc.set_defaults(func=_cmd_boards_discover_yc)
    p_cs = boards_sub.add_parser("discover-careers", help="find boards linked from YC companies' careers sites (D2.2)")
    p_cs.add_argument("path", help="yc-oss companies/all.json, downloaded into data/ (no licence stated: local only)")
    p_cs.add_argument("--limit", type=int, default=0, help="read only the first N sites")
    p_cs.add_argument("--pause", type=float, default=0.5, help="seconds between page requests")
    p_cs.add_argument("--timeout", type=float, default=10.0, help="seconds per page")
    p_cs.add_argument("--dry-run", action="store_true", help="report only; write no board")
    p_cs.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    p_cs.set_defaults(func=_cmd_boards_discover_careers)

    p_or = boards_sub.add_parser("discover-orphans", help="find boards for companies seen only on JobSpy (D2.2)")
    p_or.add_argument("--limit", type=int, default=0, help="check only the first N companies")
    p_or.add_argument("--pause", type=float, default=0.5, help="seconds between probe requests")
    p_or.add_argument("--links-only", action="store_true", help="only companies whose links name a board: no guessing")
    p_or.add_argument("--dry-run", action="store_true", help="report only; write no board")
    p_or.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    p_or.set_defaults(func=_cmd_boards_discover_orphans)
    p_gt = boards_sub.add_parser("discover-getro", help="find the boards behind VC networks' Getro job boards (3.13)")
    p_gt.add_argument("hosts", nargs="+", help="a network's job board host, e.g. jobs.3one4capital.com")
    p_gt.add_argument("--limit", type=int, default=0, help="read only the first N companies of each network")
    p_gt.add_argument("--per-company", type=int, default=3, help="job pages read per company to find its board")
    p_gt.add_argument("--pause", type=float, default=1.0, help="seconds between Getro page requests (Crawl-delay: 1)")
    p_gt.add_argument("--dry-run", action="store_true", help="report only; write no board")
    p_gt.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    p_gt.set_defaults(func=_cmd_boards_discover_getro)
    p_ph = boards_sub.add_parser("discover-phenom", help="find the boards behind Phenom career sites (parity step)")
    p_ph.add_argument("path", help="a CSV of Phenom sites: name, slug (the site host), url")
    p_ph.add_argument("--limit", type=int, default=0, help="read only the first N sites")
    p_ph.add_argument("--pause", type=float, default=1.0, help="seconds between Phenom requests (one pace for all)")
    p_ph.add_argument("--dry-run", action="store_true", help="report only; write no board")
    p_ph.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    p_ph.set_defaults(func=_cmd_boards_discover_phenom)

    p_companies = sub.add_parser("companies", help="one company per employer (D2.4)")
    companies_sub = p_companies.add_subparsers(dest="companies_command", required=True)
    p_merge = companies_sub.add_parser("merge", help="merge companies that are one employer (dry run by default)")
    p_merge.add_argument("--apply", action="store_true", help="write the merges")
    p_merge.add_argument("--curated", default=str(company_merge.CURATED_PATH), help="owner-listed merges")
    p_merge.add_argument("--show", type=int, default=20, help="renames to print")
    p_merge.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    p_merge.set_defaults(func=_cmd_companies_merge)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
