"""Copy today's SQLite database into Postgres and prove nothing was lost (plan unit DB2;
docs/freehire-merge/05-schema.md §4 mapping, §5 steps 3-4).

    python -m hiredra.pg_import              # import into the empty, migrated database, then verify
    python -m hiredra.pg_import --reset      # empty every Postgres table first, then import and verify
    python -m hiredra.pg_import --verify     # verify only

Until the switch (DB5) Postgres holds a derived copy: SQLite stays the source of truth and is opened
read-only. Each table is imported in its own transaction, so a failure leaves the earlier tables done and
the failed one empty; ``--reset`` then starts over.

The report is printed and written to ``data/logs/pg-import-<UTC time>.md``. Every check is exact. The
differences the mapping makes on purpose are listed apart, with counts, and do not fail the run; anything
else does (exit code 1).
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

import psycopg
import yaml

from hiredra import pg
from hiredra.location import parse
from hiredra.sourcing.config import load_companies
from hiredra.sourcing.identity import company_slug, content_hash, role_fingerprint
from hiredra.sourcing.provider import company_slug as list_board
from hiredra.sourcing.registry import REGISTRY, route
from hiredra.storage import DEFAULT_DB_PATH

REPO = Path(__file__).resolve().parents[1]
DATA = Path(DEFAULT_DB_PATH).parent
COMPANY_LISTS = (("openjobs", REPO / "config" / "companies.wide.yml"), ("curated", REPO / "config" / "companies.yml"))
USAGE_DIR = DATA / "usage"
EMBED_USAGE = DATA / "embed_usage.json"
GRAPH_CONFIG = REPO / "config" / "graph.yml"
DESCRIPTIONS = REPO / "hiredra" / "graph" / "skill_descriptions.tsv"
SAMPLE_SEED = 20261028
# Child tables first; schema_migrations is never touched.
TABLES = ("api_usage", "skill_alias", "skill_vocab", "job_events", "job_skills", "jobs", "board_crawls", "runs",
          "board_health", "boards", "company_slug_aliases", "companies")  # fmt: skip

JOB_COLUMNS = (
    "source", "external_id", "board_id", "company_id", "url", "apply_url", "title", "company", "location",
    "countries", "is_remote", "date_posted", "job_type", "salary", "description", "content_hash",
    "role_fingerprint", "years_required", "status", "status_detail", "status_changed_at", "domain_bucket",
    "legitimacy", "extraction_confidence", "extraction_version", "score", "score_features", "review_reason",
    "released_at", "scored_at", "score_version", "score_embed_tag", "score_skills_hash", "score_profile_hash",
    "outcome", "outcome_at", "outcome_detail", "first_seen_run", "first_seen_at", "last_seen_at", "updated_at",
)  # fmt: skip
SAME_TEXT = (
    "url", "apply_url", "title", "company", "location", "job_type", "salary", "description", "years_required",
    "source", "status", "status_detail", "domain_bucket", "legitimacy", "extraction_confidence",
    "extraction_version", "score_version", "score_embed_tag", "score_skills_hash", "score_profile_hash", "outcome",
    "outcome_detail", "first_seen_run", "score",
)  # fmt: skip
TIMESTAMPS = (
    "status_changed_at",
    "released_at",
    "scored_at",
    "outcome_at",
    "first_seen_at",
    "last_seen_at",
    "updated_at",
)


def open_sqlite(path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    lite = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    lite.row_factory = sqlite3.Row
    return lite


def _ts(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text else None


def _date(text: str | None) -> date | None:
    try:
        return date.fromisoformat(text) if text else None
    except ValueError:
        return None


@lru_cache(maxsize=None)
def _countries(location: str | None, title: str) -> list[str]:
    return sorted(parse(location or "", "", title)["countries"])  # the same call dedup makes


@lru_cache(maxsize=None)
def _company_key(name: str) -> str:
    return company_slug(name)


def _top(counts: Counter) -> str:
    """The most frequent value; ties go to the alphabetically first, so the choice is reproducible."""
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


def skill_descriptions(path: Path = DESCRIPTIONS) -> dict[str, str]:
    """Lowercased skill name -> description, keyed by freehire's slug, the slug with spaces, and its label.

    Only exact names: freehire's alias table would hand "entity framework" the .NET description, which would
    pull two different skills together once embedded.
    """
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        slug, label, text = line.split("\t")
        for form in (slug, slug.replace("-", " "), label.lower()):
            if form:
                out.setdefault(form, text)
    return out


_TAG_RANK = {"confirmed": 2, "unknown": 1}  # confirmed = India/remote eligibility was seen; unknown = not yet


def listed_boards() -> dict[tuple[str, str], dict]:
    """``(provider, lower(board))`` -> the company list's entry. The curated list wins over the wide one."""
    out = {}
    for provenance, path in COMPANY_LISTS:
        if not path.exists():
            continue
        for entry in load_companies(path):
            provider = route(entry, REGISTRY)
            board = list_board(provider, entry) if provider else None
            if not board:
                continue
            key, tag = (provider.name, board.lower()), (entry.extra or {}).get("tag")
            old = out.get(key)
            if old and old["provenance"] == provenance and _TAG_RANK.get(old["tag"], 0) >= _TAG_RANK.get(tag, 0):
                continue  # one list naming a board twice (case spellings): the stronger tag stands
            out[key] = {
                "board": board,
                "name": entry.name,
                "tag": tag,
                "careers_url": entry.careers_url,
                "provenance": provenance,
            }
    return out


def usage_rows() -> list[tuple[str, str, str, int]]:
    """``(provider, model, day, calls)`` from today's quota files (Q5). The day-only ``exhausted`` flag is not kept."""
    rows = []
    for path in sorted(USAGE_DIR.glob("*.json")):
        provider, _, model = path.stem.partition("_") if "_" in path.stem else ("gemini", "", path.stem)
        rec = json.loads(path.read_text(encoding="utf-8"))
        rows.append((provider, model, rec["day"], int(rec.get("requests") or 0)))
    if EMBED_USAGE.exists():
        rec = json.loads(EMBED_USAGE.read_text(encoding="utf-8"))
        model = yaml.safe_load(GRAPH_CONFIG.read_text(encoding="utf-8"))["embedding"]["model"]
        rows.append(("gemini", model, rec["day"], int(rec.get("requests") or 0)))
    return rows


# ── import ──────────────────────────────────────────────────────────────────────────────────


def reset(conn: psycopg.Connection) -> None:
    conn.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")
    conn.commit()


def _step(name: str, started: float) -> None:
    print(f"  {name:<12} {time.monotonic() - started:6.1f}s", flush=True)


def import_all(lite: sqlite3.Connection, conn: psycopg.Connection, listed: dict, usage: list) -> dict:
    """Every table, in dependency order (§5 step 3). ``listed`` is ``listed_boards()``, ``usage`` is
    ``usage_rows()``. Returns the facts the report states."""
    facts: dict = {}
    t = time.monotonic()

    with conn.transaction():
        conn.cursor().executemany(
            "INSERT INTO runs (run_id, kind, config_name, started_at, finished_at, status, fetched, collapsed_in_run, "
            "matched_in_store, persisted_new, touched, detail) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
            [(r["run_id"], r["kind"], r["config_name"], _ts(r["started_at"]), _ts(r["finished_at"]), r["status"],
              r["fetched"], r["collapsed_in_run"], r["matched_in_store"], r["persisted_new"], r["touched"], r["detail"])
             for r in lite.execute("SELECT * FROM runs")],
        )  # fmt: skip
    _step("runs", t)

    # Companies and boards, from the distinct values on jobs plus the company lists (§4).
    spellings: defaultdict[str, Counter] = defaultdict(Counter)  # company key -> raw names
    for company, n in lite.execute("SELECT company, COUNT(*) FROM jobs GROUP BY company"):
        if _company_key(company):
            spellings[_company_key(company)][company] += n
    for meta in listed.values():
        if _company_key(meta["name"]):
            spellings[_company_key(meta["name"])][meta["name"]] += 0
    with conn.transaction():
        conn.cursor().executemany(
            "INSERT INTO companies (slug, name) VALUES (%s, %s)", [(k, _top(c)) for k, c in sorted(spellings.items())]
        )
    company_ids = dict(conn.execute("SELECT slug, id FROM companies").fetchall())

    seen: defaultdict[tuple[str, str], Counter] = defaultdict(Counter)  # board key -> spellings on jobs
    owners: defaultdict[tuple[str, str], Counter] = defaultdict(Counter)  # board key -> company keys
    job_tags: defaultdict[tuple[str, str], Counter] = defaultdict(Counter)
    for source, board, company, tag, n in lite.execute(
        "SELECT source, company_slug, company, list_tag, COUNT(*) FROM jobs "
        "WHERE company_slug IS NOT NULL AND source NOT LIKE 'jobspy-%' GROUP BY 1, 2, 3, 4"
    ):
        key = (source, board.lower())
        seen[key][board] += n
        owners[key][_company_key(company)] += n
        if tag:
            job_tags[key][tag] += n
    boards = []
    for key in sorted(set(seen) | set(listed)):
        meta = listed.get(key)
        owner = _top(owners[key]) if owners[key] else _company_key(meta["name"])
        boards.append((
            key[0],
            meta["board"] if meta else _top(seen[key]),
            company_ids.get(owner),
            meta["careers_url"] if meta else None,
            "active" if key in seen else "pending",
            meta["provenance"] if meta else "orphan",
            (meta["tag"] if meta else None) or (_top(job_tags[key]) if job_tags[key] else None),
        ))  # fmt: skip
    with conn.transaction():
        conn.cursor().executemany(
            "INSERT INTO boards (provider, board, company_id, careers_url, status, provenance, list_tag) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            boards,
        )
    board_ids = {(p, b.lower()): i for p, b, i in conn.execute("SELECT provider, board, id FROM boards")}
    facts["case_merged_boards"] = sorted(f"{k[0]}:{' / '.join(sorted(c))}" for k, c in seen.items() if len(c) > 1)
    tag_of = {(p, b.lower()): tag for p, b, _, _, _, _, tag in boards}
    _step("companies", t)

    # Jobs, streamed with COPY.
    tag_moved = jobspy_slugs = bad_dates = 0
    with conn.transaction(), conn.cursor().copy(f"COPY jobs ({', '.join(JOB_COLUMNS)}) FROM STDIN") as copy:
        for r in lite.execute("SELECT * FROM jobs ORDER BY rowid"):
            source, _, external_id = r["job_id"].partition(":")
            board_key = (r["source"], (r["company_slug"] or "").lower())
            board_id = None
            if r["source"].startswith("jobspy-"):
                jobspy_slugs += r["company_slug"] is not None
            elif r["company_slug"]:
                board_id = board_ids[board_key]
                tag_moved += r["list_tag"] != tag_of[board_key]
            if r["date_posted"] and _date(r["date_posted"]) is None:
                bad_dates += 1
            key = _company_key(r["company"])
            countries = _countries(r["location"], r["title"] or "")
            copy.write_row((
                source, external_id, board_id, company_ids.get(key), r["url"], r["apply_url"], r["title"],
                r["company"], r["location"], countries, None if r["is_remote"] is None else bool(r["is_remote"]),
                _date(r["date_posted"]), r["job_type"], r["salary"], r["description"],
                content_hash(r["title"], r["company"], r["location"], r["description"]),
                role_fingerprint(key, r["title"], r["description"], countries), r["years_required"], r["status"],
                r["status_detail"], r["status_changed_at"], r["domain_bucket"], r["legitimacy"],
                r["extraction_confidence"], r["extraction_version"], r["score"], r["score_features"],
                json.loads(r["review_reason"]) if r["review_reason"] else None, r["released_at"], r["scored_at"],
                r["score_version"], r["score_embed_tag"], r["score_skills_hash"], r["score_profile_hash"],
                r["outcome"], r["outcome_at"], r["outcome_detail"], r["first_seen_run"], r["first_seen_at"],
                r["last_seen_at"], r["updated_at"],
            ))  # fmt: skip
    facts.update(list_tag_moved=tag_moved, jobspy_slugs_dropped=jobspy_slugs, unparseable_dates=bad_dates)
    _step("jobs", t)

    ids = dict(conn.execute("SELECT job_key, id FROM jobs").fetchall())
    repeats = 0
    with (
        conn.transaction(),
        conn.cursor().copy("COPY job_skills (job_id, skill, requirement, position) FROM STDIN") as copy,
    ):
        for job_id, skills in lite.execute("SELECT job_id, skills FROM jobs WHERE skills IS NOT NULL ORDER BY rowid"):
            names: set[str] = set()
            for item in json.loads(skills):
                if item["name"] in names:
                    repeats += 1  # the same skill twice in one extraction: kept once, first occurrence
                    continue
                copy.write_row((ids[job_id], item["name"], item["requirement"], len(names)))
                names.add(item["name"])
    facts["repeated_skills_merged"] = repeats
    _step("job_skills", t)

    with (
        conn.transaction(),
        conn.cursor().copy("COPY job_events (job_id, kind, from_status, to_status, detail, at) FROM STDIN") as copy,
    ):
        for job_id, kind, a, b, detail, at in lite.execute(
            "SELECT job_id, kind, from_status, to_status, detail, at FROM job_events ORDER BY id"
        ):
            copy.write_row((ids[job_id], kind, a, b, detail, at))
    _step("job_events", t)

    described = skill_descriptions()
    with conn.transaction():
        conn.cursor().executemany(
            "INSERT INTO skill_vocab (name, description, first_seen_at, seen_count, mine, level, source, edges) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
            [(r["name"], described.get(r["name"].lower()), _ts(r["first_seen_at"]), r["seen_count"], bool(r["mine"]),
              r["level"], r["source"], r["edges"]) for r in lite.execute("SELECT * FROM skill_vocab")],
        )  # fmt: skip
        conn.cursor().executemany(
            "INSERT INTO skill_alias (alias, canonical, created_at) VALUES (%s, %s, %s)",
            [(a, c, _ts(at)) for a, c, at in lite.execute("SELECT alias, canonical, created_at FROM skill_alias")],
        )
    facts["described_skills"] = sum(
        1 for (n,) in lite.execute("SELECT name FROM skill_vocab") if n.lower() in described
    )
    _step("skills", t)

    with conn.transaction():
        conn.cursor().executemany("INSERT INTO api_usage (provider, model, day, calls) VALUES (%s, %s, %s, %s)", usage)
    _step("api_usage", t)
    return facts


# ── verify (§5 step 4) ──────────────────────────────────────────────────────────────────────


def _expected(r: sqlite3.Row, field: str):
    """The value Postgres must hold for SQLite row ``r``'s ``field``, after the §4 transform."""
    if field in SAME_TEXT:
        return r[field]
    if field in TIMESTAMPS:
        return _ts(r[field])
    if field == "is_remote":
        return None if r["is_remote"] is None else bool(r["is_remote"])
    if field == "date_posted":
        return _date(r["date_posted"])
    if field == "score_features":
        return json.loads(r["score_features"]) if r["score_features"] else None
    if field == "review_reason":
        return json.loads(r["review_reason"]) if r["review_reason"] else None
    if field == "skills":  # NULL = never extracted, [] = extracted with none (both are zero job_skills rows)
        if r["extraction_version"] is None:
            return None
        names, out = set(), []
        for item in json.loads(r["skills"] or "[]"):
            if item["name"] not in names:
                names.add(item["name"])
                out.append({"name": item["name"], "requirement": item["requirement"]})
        return out
    raise KeyError(field)


def verify(
    lite: sqlite3.Connection, conn: psycopg.Connection, usage: list, *, sample: int = 1000
) -> list[tuple[str, bool, str]]:
    """``(check, passed, detail)`` for every §5 step-4 check."""
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    def one(sql_lite: str, sql_pg: str):
        return lite.execute(sql_lite).fetchone()[0], conn.execute(sql_pg).fetchone()[0]

    for table in ("runs", "jobs", "job_events", "skill_vocab", "skill_alias"):
        a, b = one(f"SELECT COUNT(*) FROM {table}", f"SELECT COUNT(*) FROM {table}")
        check(f"rows: {table}", a == b, f"SQLite {a:,} / Postgres {b:,}")

    entries = repeats = 0
    for (skills,) in lite.execute("SELECT skills FROM jobs WHERE skills IS NOT NULL"):
        names = [i["name"] for i in json.loads(skills)]
        entries += len(names)
        repeats += len(names) - len(set(names))
    a, b = entries - repeats, conn.execute("SELECT COUNT(*) FROM job_skills").fetchone()[0]
    check(
        "rows: job_skills",
        a == b,
        f"{entries:,} skill entries - {repeats} repeated in one job = {a:,} / Postgres {b:,}",
    )

    for label, col in (("status", "status"), ("source", "source"), ("status x source", "status || '/' || source")):
        a = Counter(dict(lite.execute(f"SELECT {col}, COUNT(*) FROM jobs GROUP BY 1").fetchall()))
        b = Counter(dict(conn.execute(f"SELECT {col}, COUNT(*) FROM jobs GROUP BY 1").fetchall()))
        check(
            f"jobs by {label}",
            a == b,
            ", ".join(f"{k} {v:,}" for k, v in sorted(a.items())) if label == "status" else "",
        )
    a, b = one(
        "SELECT COUNT(*) FROM jobs WHERE review_reason IS NOT NULL",
        "SELECT COUNT(*) FROM jobs WHERE review_reason IS NOT NULL",
    )
    check("jobs on the review list", a == b, f"{a:,} / {b:,}")
    a = Counter(dict(lite.execute("SELECT kind, COUNT(*) FROM job_events GROUP BY 1").fetchall()))
    b = Counter(dict(conn.execute("SELECT kind, COUNT(*) FROM job_events GROUP BY 1").fetchall()))
    check("events by kind", a == b, ", ".join(f"{k} {v:,}" for k, v in sorted(a.items())))

    old = {r[0] for r in lite.execute("SELECT job_id FROM jobs")}
    new = {r[0] for r in conn.execute("SELECT job_key FROM jobs")}
    check("every old job_id resolves through job_key", old == new, f"missing {len(old - new)}, extra {len(new - old)}")

    # Boards and companies, every job.
    pg_board = dict(conn.execute(
        "SELECT j.job_key, b.provider || ':' || lower(b.board) FROM jobs j JOIN boards b ON b.id = j.board_id"
    ).fetchall())  # fmt: skip
    want_board = {jid: f"{src}:{slug.lower()}" for jid, src, slug in lite.execute(
        "SELECT job_id, source, company_slug FROM jobs WHERE company_slug IS NOT NULL AND source NOT LIKE 'jobspy-%'")}  # fmt: skip
    check(
        "every ATS job is on its board",
        pg_board == want_board,
        f"{len(want_board):,} jobs; {sum(pg_board.get(k) != v for k, v in want_board.items())} differ",
    )
    pg_company = dict(
        conn.execute("SELECT j.job_key, c.slug FROM jobs j JOIN companies c ON c.id = j.company_id").fetchall()
    )
    want_company = {
        jid: _company_key(c) for jid, c in lite.execute("SELECT job_id, company FROM jobs") if _company_key(c)
    }
    check("every job is linked to its company", pg_company == want_company, f"{len(want_company):,} jobs linked")

    # A random sample, field by field.
    keys = sorted(old)
    picked = random.Random(SAMPLE_SEED).sample(keys, min(sample, len(keys)))
    fields = ("is_remote", "date_posted", "score_features", "review_reason", "skills", *SAME_TEXT, *TIMESTAMPS)
    diffs: defaultdict[str, list[str]] = defaultdict(list)
    for jid in picked:
        r = lite.execute("SELECT * FROM jobs WHERE job_id = ?", (jid,)).fetchone()
        row = conn.execute(
            f"SELECT id, {', '.join(f for f in fields if f != 'skills')} FROM jobs WHERE job_key = %s", (jid,)
        ).fetchone()
        got = dict(zip([f for f in fields if f != "skills"], row[1:]))
        skills = conn.execute(
            "SELECT skill, requirement FROM job_skills WHERE job_id = %s ORDER BY position", (row[0],)
        ).fetchall()
        got["skills"] = None if r["extraction_version"] is None else [{"name": s, "requirement": q} for s, q in skills]
        for f in fields:
            if got[f] != _expected(r, f):
                diffs[f].append(jid)
        old_events = [(k, a, b, d, _ts(at)) for k, a, b, d, at in lite.execute(
            "SELECT kind, from_status, to_status, detail, at FROM job_events WHERE job_id = ? ORDER BY id", (jid,))]  # fmt: skip
        new_events = conn.execute(
            "SELECT kind, from_status, to_status, detail, at FROM job_events WHERE job_id = %s ORDER BY id", (row[0],)
        ).fetchall()
        if old_events != [tuple(e) for e in new_events]:
            diffs["events"].append(jid)
    check(
        f"{len(picked):,}-job random sample, field by field ({len(fields)} fields + events)",
        not diffs,
        "; ".join(f"{f}: {len(j)} (e.g. {', '.join(j[:3])})" for f, j in diffs.items()) or "all equal",
    )

    old_vocab = {r["name"]: (_ts(r["first_seen_at"]), r["seen_count"], bool(r["mine"]), r["level"], r["source"],
                             json.loads(r["edges"])) for r in lite.execute("SELECT * FROM skill_vocab")}  # fmt: skip
    new_vocab = {r[0]: tuple(r[1:]) for r in conn.execute(
        "SELECT name, first_seen_at, seen_count, mine, level, source, edges FROM skill_vocab")}  # fmt: skip
    check("skill_vocab, every row", old_vocab == new_vocab, f"{len(old_vocab):,} skills")
    old_alias = {a: (c, _ts(at)) for a, c, at in lite.execute("SELECT alias, canonical, created_at FROM skill_alias")}
    new_alias = {a: (c, at) for a, c, at in conn.execute("SELECT alias, canonical, created_at FROM skill_alias")}
    check("skill_alias, every row", old_alias == new_alias, f"{len(old_alias)} aliases")
    old_runs = {r["run_id"]: (r["kind"], r["status"], _ts(r["started_at"]), _ts(r["finished_at"]), r["touched"],
                              json.loads(r["detail"]) if r["detail"] else None) for r in lite.execute("SELECT * FROM runs")}  # fmt: skip
    new_runs = {r[0]: tuple(r[1:]) for r in conn.execute(
        "SELECT run_id, kind, status, started_at, finished_at, touched, detail FROM runs")}  # fmt: skip
    check("runs, every row", old_runs == new_runs, f"{len(old_runs)} runs")
    want_usage = sorted((p, m, date.fromisoformat(d), c) for p, m, d, c in usage)
    got_usage = sorted(tuple(r) for r in conn.execute("SELECT provider, model, day, calls FROM api_usage"))
    check(
        "api_usage = today's quota files",
        want_usage == got_usage,
        ", ".join(f"{p}/{m} {d}: {c}" for p, m, d, c in want_usage),
    )
    return checks


_BUILT = {
    "companies": "SELECT COUNT(*) FROM companies",
    "no_company": "SELECT COUNT(*) FROM jobs WHERE company_id IS NULL",
    "boards": "SELECT COUNT(*) FROM boards",
    "active": "SELECT COUNT(*) FROM boards WHERE status = 'active'",
    "with_country": "SELECT COUNT(*) FROM jobs WHERE countries <> '{}'",
    "india": "SELECT COUNT(*) FROM jobs WHERE 'IN' = ANY (countries)",
    "clusters": "SELECT COUNT(*) FROM (SELECT 1 FROM jobs GROUP BY role_fingerprint HAVING COUNT(*) > 1) x",
    "vocab": "SELECT COUNT(*) FROM skill_vocab",
    "vectors": "SELECT COUNT(vector) FROM skill_vocab",
}


def render(checks: list[tuple[str, bool, str]], facts: dict, conn: psycopg.Connection) -> str:
    n = {k: conn.execute(sql).fetchone()[0] for k, sql in _BUILT.items()}
    provenance = conn.execute("SELECT provenance, COUNT(*) FROM boards GROUP BY 1 ORDER BY 2 DESC").fetchall()
    failed = [c for c in checks if not c[1]]
    out = [
        "# SQLite -> Postgres import: verification report",
        "",
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC. **{'PASS' if not failed else f'FAIL ({len(failed)} checks)'}**: "
        f"{len(checks) - len(failed)} of {len(checks)} checks pass.",
        "",
        "## Checks (docs/freehire-merge/05-schema.md §5 step 4)",
        "",
        *[f"- {'PASS' if ok else '**FAIL**'} {name}{f': {detail}' if detail else ''}" for name, ok, detail in checks],
        "",
        "## Built at import (new in Postgres)",
        "",
        f"- companies: {n['companies']:,}; jobs without one (the name has no letter or digit): {n['no_company']:,}",
        f"- boards: {n['boards']:,} ({', '.join(f'{p} {c:,}' for p, c in provenance)}); active (has jobs) {n['active']:,}",
        f"- jobs with a parsed country: {n['with_country']:,}; India among them: {n['india']:,}",
        f"- role clusters (fingerprints shared by more than one job): {n['clusters']:,}",
        f"- skills with a freehire description: {facts.get('described_skills', 'n/a')} of {n['vocab']:,}",
        "",
        "## Differences made on purpose (§4)",
        "",
        f"- Same skill twice in one job's extraction, kept once: {facts.get('repeated_skills_merged', 'n/a')}.",
        f"- Board slugs differing only in case, now one board each (boards are unique on lower(board)): "
        f"{', '.join(facts.get('case_merged_boards', [])) or 'none'}.",
        f"- Jobs whose own list_tag differs from their board's (list_tag moved to boards; the company list wins): "
        f"{facts.get('list_tag_moved', 'n/a')}.",
        f"- JobSpy jobs that carried a board slug (JobSpy jobs get no board; their company link stays): "
        f"{facts.get('jobspy_slugs_dropped', 'n/a')}.",
        f"- Unparseable date_posted, stored NULL: {facts.get('unparseable_dates', 'n/a')}.",
        f"- Skill vectors not copied (Gemini, 768 dims): every skill is re-embedded with Qwen in DB4. Vectors now: {n['vectors']}.",
        "- `cooldowns` (0 rows) is dropped: board_health replaces it. Quota files' day-only `exhausted` flag is not kept.",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hiredra.pg_import", description=__doc__.split("\n\n")[0])
    ap.add_argument("--sqlite", default=str(DEFAULT_DB_PATH), help="the SQLite file to copy (opened read-only)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--reset", action="store_true", help="empty every Postgres table first")
    mode.add_argument("--verify", action="store_true", help="verify only; import nothing")
    ap.add_argument("--sample", type=int, default=1000, help="jobs compared field by field (default 1000)")
    args = ap.parse_args(argv)

    lite = open_sqlite(args.sqlite)
    usage = usage_rows()
    facts: dict = {}
    try:
        with pg.connect() as conn:
            if pg.pending(conn):
                print("pg_import: migrations are pending; run `python -m hiredra.pg migrate` first")
                return 1
            if not args.verify:
                if args.reset:
                    reset(conn)
                elif conn.execute("SELECT EXISTS (SELECT 1 FROM jobs)").fetchone()[0]:
                    print("pg_import: Postgres already holds jobs; pass --reset to start over, or --verify")
                    return 1
                print("importing:", flush=True)
                facts = import_all(lite, conn, listed_boards(), usage)
            report = render(verify(lite, conn, usage, sample=args.sample), facts, conn)
    except (RuntimeError, psycopg.Error) as exc:
        print(f"pg_import: {exc}")
        return 1
    finally:
        lite.close()
    path = DATA / "logs" / f"pg-import-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"report: {path}")
    return 0 if "**FAIL" not in report else 1


if __name__ == "__main__":
    sys.exit(main())
