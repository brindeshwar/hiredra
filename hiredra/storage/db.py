"""Postgres storage layer for Hiredra (build sub-slice 2a.1; moved from SQLite in plan unit DB3).

Single source of truth for jobs, their pipeline status, the scoring ledger, and
outcomes. Every later stage reads and updates rows here; cross-run dedup and
status tracking work against it. The schema is ``migrations/*.sql`` (applied by
``hiredra.pg.migrate``; ``init_db`` is that call).

This module is persistence only -- no sourcing, filtering, or scoring logic. The
one exception is that ``upsert_job`` stamps a new job's derived identity columns
(company, countries, content hash, role fingerprint), so every write path keeps
them consistent.

Conventions
-----------
* ``connect()`` returns an AUTOCOMMIT connection: each statement commits on its
  own, and every multi-statement write here runs in ``conn.transaction()``.
  Never ``with conn:`` -- in psycopg that CLOSES the connection.
* Rows read by position or by column name (``Row``), as ``sqlite3.Row`` did.
* A job is addressed by its readable key ``job_key`` (``source:external_id``,
  e.g. ``greenhouse:105736``); every job dict carries it as ``job_id``, the name
  every caller already uses. The bigint ``id`` stays inside this module.
* Timestamps are ``timestamptz``. ``_now()`` returns an ISO-8601 UTC string
  (tests monkeypatch it); reads return aware ``datetime`` values.
* ``years_required`` is ``None`` when unknown and ``0`` only for an explicitly
  zero-experience role. The two are never conflated.
* A re-sight (``upsert_job`` on an existing ``job_id``) bumps ``last_seen_at``
  and appends a ``seen`` event. It never regresses ``status`` and never clobbers
  ``score`` / ``score_features`` / ``outcome*``; Job content columns are written
  once, on first insert.
"""

from __future__ import annotations

import secrets
import struct
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from hiredra import pg
from hiredra.location import parse as parse_location
from hiredra.sourcing.identity import company_slug as company_key
from hiredra.sourcing.identity import content_hash, role_fingerprint

# data/ at the repo root (this file lives at hiredra/storage/db.py): calibration files, logs, the archived SQLite files.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR: Path = _REPO_ROOT / "data"

#: run statuses (see schema `runs.status`).
RUN_STATUSES: tuple[str, ...] = ("running", "ok", "partial", "failed")

#: Ordered pipeline statuses plus the off-path ``dropped`` (funnel exit — kept in
#: the DB so cross-run dedup never re-processes it).
STATUSES: tuple[str, ...] = (
    "fetched",
    "filtered",
    "extracted",
    "scored",
    "tailored",
    "queued",
    "applied",
    "outcome",
    "dropped",
)

#: Documented outcome vocabulary (docs/scoring.md §7). NOT enforced by the DB —
#: the learning loop (v2) may extend it.
OUTCOMES: tuple[str, ...] = (
    "silence",
    "rejected",
    "reply",
    "interview",
    "offer",
    "hired",
)

#: skill_vocab vocabularies (schema CHECKs).
SKILL_LEVELS: tuple[str, ...] = ("production", "built-with", "learned", "interview-ready")
SKILL_SOURCES: tuple[str, ...] = ("market", "jd", "profile")

#: Job-object columns written by ``upsert_job`` (docs/sourcing.md §2). Written
#: once, on first insert; a re-sight does not overwrite them.
_JOB_COLUMNS: tuple[str, ...] = (
    "url",
    "title",
    "company",
    "description",
    "years_required",
    "source",
    "location",
    "date_posted",
    "job_type",
    "salary",
    "apply_url",
    "is_remote",
)

_REQUIRED_JOB_FIELDS = ("url", "title", "company", "source")

# Every job column a caller reads, plus the board's slug and tag and the
# skills list, in the shapes the SQLite rows had them: ``job_id`` is the key, ``skills`` is None until extraction
# ran and then a list of {name, requirement, source} in order: the LLM's skills, then the dictionary's (0005).
_JOB_SELECT = """
SELECT j.job_key AS job_id, j.id, j.source, j.external_id, j.url, j.apply_url, j.title, j.company, j.location,
       j.countries, j.is_remote, j.work_mode, j.date_posted, j.job_type, j.employment_type, j.salary,
       j.description, j.content_hash, j.role_fingerprint, j.duplicate_of, j.years_required,
       j.status, j.status_detail, j.status_changed_at,
       j.domain_bucket, j.legitimacy, j.extraction_confidence, j.extraction_version, j.dictionary_version,
       CASE WHEN j.extraction_confidence IS NULL THEN NULL ELSE COALESCE(
           (SELECT json_agg(json_build_object('name', s.skill, 'requirement', s.requirement, 'source', s.source)
                            ORDER BY s.position)
              FROM job_skills s WHERE s.job_id = j.id), '[]'::json) END AS skills,
       j.score, j.score_features, j.review_reason, j.released_at, j.scored_at, j.score_version, j.score_embed_tag,
       j.score_skills_hash, j.score_profile_hash, j.outcome, j.outcome_at, j.outcome_detail,
       j.first_seen_run, j.first_seen_at, j.last_seen_at, j.closed_at, j.updated_at,
       j.board_id, j.company_id, b.board AS company_slug, b.list_tag
  FROM jobs j LEFT JOIN boards b ON b.id = j.board_id
"""


def _now() -> str:
    """Current UTC time as an ISO-8601 string, second precision.

    Tests monkeypatch this module attribute to get a deterministic clock.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    """``<YYYYMMDDThhmmssZ>-<4hex>`` derived from ``_now()``.

    Compact basic-format ISO (no colons → filename/URL safe), explicit ``Z``,
    lexicographically sortable by start time, plus a random suffix so same-second
    starts don't collide. Deterministic under a monkeypatched ``_now`` except the
    hex.
    """
    compact = _now().replace("-", "").replace(":", "").replace("+0000", "Z").replace("+00:00", "Z")
    return f"{compact}-{secrets.token_hex(2)}"


# ─────────────────────────────────────────────────────────────────────────────
# connection / setup
# ─────────────────────────────────────────────────────────────────────────────
class Row(tuple):
    """A result row readable by position (``row[0]``) and by column name (``row["status"]``), and convertible
    with ``dict(row)`` -- what ``sqlite3.Row`` gave every caller."""

    def __new__(cls, values: Sequence[Any], index: dict[str, int]) -> Row:
        row = super().__new__(cls, values)
        row._index = index
        return row

    def __getitem__(self, key):  # type: ignore[override]
        return super().__getitem__(self._index[key] if isinstance(key, str) else key)

    def keys(self) -> list[str]:
        return list(self._index)


def _row_factory(cursor: psycopg.Cursor) -> Any:
    index = {c.name: i for i, c in enumerate(cursor.description or ())}
    return lambda values: Row(values, index)


def connect(url: str | None = None) -> psycopg.Connection:
    """Open an autocommit connection to ``url`` (default: ``HIREDRA_DATABASE_URL``)."""
    return psycopg.connect(url or pg.database_url(), autocommit=True, row_factory=_row_factory)


def init_db(conn: psycopg.Connection) -> None:
    """Apply every pending migration (``hiredra.pg.migrate``) -- idempotent."""
    pg.migrate(conn)


def _id(conn: psycopg.Connection, job_id: str) -> int:
    """The bigint id behind ``job_id`` (a ``job_key``). ``KeyError`` if unknown."""
    row = conn.execute("SELECT id FROM jobs WHERE job_key = %s", (job_id,)).fetchone()
    if row is None:
        raise KeyError(f"no job with job_id {job_id!r}")
    return row[0]


# ─────────────────────────────────────────────────────────────────────────────
# writes
# ─────────────────────────────────────────────────────────────────────────────
def _company_id(conn: psycopg.Connection, name: str) -> int | None:
    """The company ``name`` belongs to (D2.4): by its key, else a recorded alias, else a company whose key differs
    only by hyphens ("Together AI" -> ``togetherai``; the new spelling is then recorded as an alias). Only a name
    none of these reach makes a new company."""
    key = company_key(name)
    if not key:
        return None
    for by_fold, sql, arg in (
        (False, "SELECT id FROM companies WHERE slug = %s", key),
        (False, "SELECT company_id FROM company_slug_aliases WHERE alias = %s", key),
        (True, "SELECT id FROM companies WHERE replace(slug, '-', '') = %s ORDER BY id LIMIT 1", key.replace("-", "")),
    ):
        row = conn.execute(sql, (arg,)).fetchone()
        if row is not None:
            if by_fold:  # found by the squashed spelling: remember this one
                conn.execute(
                    "INSERT INTO company_slug_aliases (alias, company_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (key, row[0]),
                )
            return row[0]
    conn.execute("INSERT INTO companies (slug, name) VALUES (%s, %s) ON CONFLICT (slug) DO NOTHING", (key, name))
    return conn.execute("SELECT id FROM companies WHERE slug = %s", (key,)).fetchone()[0]


def upsert_job(
    conn: psycopg.Connection,
    job: Mapping[str, Any],
    *,
    first_seen_run: str | None = None,
) -> str:
    """Insert a new job, or re-sight an existing one. Returns the ``job_id``.

    ``job_id`` is ``source:raw_id`` (it becomes ``job_key``); it falls back to
    ``source:url`` when absent. ``url``, ``title``, ``company`` and ``source`` are
    required. ``years_required``, if given, must be an ``int`` (``0`` is valid;
    unknown is ``None``).

    New row: every Job column written, ``first_seen_run`` set (if given),
    ``status`` defaults to ``fetched``, ``first_seen_at``/``last_seen_at``/
    ``updated_at`` set, the company linked (created if new), the countries,
    content hash and role fingerprint computed, a ``seen`` event appended.

    Existing row (matched by ``job_id``): only ``last_seen_at``/``updated_at``
    are bumped and a ``seen`` event appended. Status, the scoring ledger, the
    outcome fields, and the Job content columns are all left untouched — a flaky
    re-fetch must never corrupt a good row or move a job backwards through the
    funnel. One exception: a job liveness dropped as ``liveness: gone`` that is
    listed again is reopened to ``filtered`` (the only status liveness drops
    from) -- a take-down and repost under the same id (2026-10-27).
    """
    for field in _REQUIRED_JOB_FIELDS:
        if not job.get(field):
            raise ValueError(f"job {job.get('job_id') or job.get('url')!r} is missing required field {field!r}")
    source = str(job["source"])
    job_id = str(job.get("job_id") or f"{source}:{job['url']}")
    if not job_id.startswith(source + ":"):
        raise ValueError(f"job_id {job_id!r} must start with its source {source!r} and ':' (D1.5)")

    yr = job.get("years_required")
    if yr is not None and not isinstance(yr, int):
        raise TypeError("years_required must be an int or None (0 = explicitly zero-experience role; unknown = None)")

    now = _now()
    reopen = False
    with conn.transaction():
        row = conn.execute("SELECT id, status, status_detail FROM jobs WHERE job_key = %s", (job_id,)).fetchone()
        if row is None:
            title, company = job["title"], job["company"]
            countries = sorted(parse_location(job.get("location") or "", "", title)["countries"])
            values = [job.get(c) for c in _JOB_COLUMNS]
            if values[_JOB_COLUMNS.index("is_remote")] is not None:  # a 0/1 from an older caller is a bool
                values[_JOB_COLUMNS.index("is_remote")] = bool(values[_JOB_COLUMNS.index("is_remote")])
            values += [
                job_id[len(source) + 1 :], _company_id(conn, company), countries,
                content_hash(title, company, job.get("location"), job.get("description")),
                role_fingerprint(company_key(company), title, job.get("description"), countries),
                now, now, now, now, first_seen_run,
            ]  # fmt: skip
            cols = (*_JOB_COLUMNS, "external_id", "company_id", "countries", "content_hash", "role_fingerprint",
                    "status_changed_at", "first_seen_at", "last_seen_at", "updated_at", "first_seen_run")  # fmt: skip
            try:
                with conn.transaction():
                    new_id = conn.execute(
                        f"INSERT INTO jobs ({', '.join(cols)}) VALUES ({', '.join('%s' for _ in cols)}) RETURNING id",
                        values,
                    ).fetchone()[0]
            except psycopg.errors.UniqueViolation as err:
                raise ValueError(
                    f"url {job.get('url')!r} is already stored under a different "
                    f"job_id (dedup should have merged these upstream)"
                ) from err
            _add_event(conn, new_id, "seen", detail="first sighting", at=now)
        else:
            _touch(conn, row[0], detail="re-sighted", now=now)
            reopen = row[1] == "dropped" and (row[2] or "").startswith("liveness: gone")

    if reopen:
        set_status(conn, job_id, "filtered", detail="reopened: listed again after liveness: gone")
    return job_id


def fill_provenance(
    conn: psycopg.Connection,
    job_ids: Iterable[str],
    *,
    company_slug: str | None,
    list_tag: str | None,
    provenance: str = "orphan",
) -> None:
    """Put each ATS job on its board (``provider`` = the job's source, ``board`` = ``company_slug``), where the
    job has no board yet. A board new to the store is created with ``provenance`` (the company list it came
    from) and ``list_tag``; an existing board's tag is filled only if it has none. JobSpy jobs get no board.

    Never overwrites a stored value: the first sighting's provenance wins, so a later fetch
    from a differently tagged list can't rewrite history. Changes no other job column.
    """
    job_ids = list(job_ids)
    if company_slug is None or not job_ids:
        return
    with conn.transaction():
        for source in {r[0] for r in _sources(conn, job_ids)}:
            if source.startswith("jobspy-"):
                continue
            conn.execute(
                "INSERT INTO boards (provider, board, status, provenance, list_tag) VALUES (%s, %s, 'active', %s, %s) "
                "ON CONFLICT (provider, lower(board), region) DO UPDATE SET list_tag = COALESCE(boards.list_tag, "
                "excluded.list_tag)",
                (source, company_slug, provenance, list_tag),
            )
            conn.execute(
                "UPDATE jobs SET board_id = (SELECT id FROM boards WHERE provider = %s AND lower(board) = lower(%s) "
                "AND region = '') WHERE job_key = ANY(%s) AND source = %s AND board_id IS NULL",
                (source, company_slug, job_ids, source),
            )


def load_boards(conn: psycopg.Connection, provenances: Sequence[str]) -> list[Row]:
    """The boards a run crawls (decision D2.1): status pending or active, of the given provenances (``("all",)``
    = every provenance), each with its company's display name. Ordered by provider, then board, so a run and its
    ``--resume`` walk the same list."""
    if not provenances:
        return []
    sql = (
        "SELECT b.id, b.provider, b.board, b.careers_url, b.provenance, b.list_tag, c.name "
        "FROM boards b LEFT JOIN companies c ON c.id = b.company_id "
        "WHERE b.status IN ('pending', 'active')"
    )
    params: tuple = ()
    if tuple(provenances) != ("all",):
        sql += " AND b.provenance = ANY(%s)"
        params = (list(provenances),)
    return conn.execute(sql + " ORDER BY b.provider, lower(b.board), b.region", params).fetchall()


def import_board(
    conn: psycopg.Connection,
    *,
    provider: str,
    board: str,
    name: str,
    provenance: str,
    source_name: str,
    list_tag: str | None = None,
    careers_url: str | None = None,
    enabled: bool = True,
) -> str:
    """Write one company-list entry into ``boards`` (``boards import``). Returns what happened:

    ``new``        not stored yet: added as ``pending`` (D5.5), with its company.
    ``known``      already stored: only blank fields are filled; provenance and status never change (the first
                   sighting wins, as in ``fill_provenance``).
    ``re-enabled`` stored, retired earlier because the list disabled it, and enabled again: back to ``pending``.
    ``retired``    ``enabled: false`` in the list: the board is retired with that reason and is no longer crawled.
    ``skipped``    disabled and never stored: nothing to do.
    """
    row = conn.execute(
        "SELECT id, status, status_reason, company_id FROM boards "
        "WHERE provider = %s AND lower(board) = lower(%s) AND region = ''",
        (provider, board),
    ).fetchone()
    reason = f"disabled in {source_name}"
    if not enabled:
        if row is None:
            return "skipped"
        conn.execute("UPDATE boards SET status = 'retired', status_reason = %s WHERE id = %s", (reason, row["id"]))
        return "retired"
    if row is None:
        conn.execute(
            "INSERT INTO boards (provider, board, company_id, careers_url, status, provenance, list_tag) "
            "VALUES (%s, %s, %s, %s, 'pending', %s, %s)",
            (provider, board, _company_id(conn, name), careers_url, provenance, list_tag),
        )
        return "new"
    # A known board keeps its company. Only a board without one looks the list's name up: looking it up anyway
    # would mint a company row nothing points at for every spelling the list differs on ("1910" vs "1910 Genetics").
    company_id = row["company_id"] if row["company_id"] is not None else _company_id(conn, name)
    conn.execute(
        "UPDATE boards SET company_id = %s, careers_url = COALESCE(careers_url, %s), "
        "list_tag = COALESCE(list_tag, %s) WHERE id = %s",
        (company_id, careers_url, list_tag, row["id"]),
    )
    if row["status"] == "retired" and row["status_reason"] == reason:
        conn.execute("UPDATE boards SET status = 'pending', status_reason = NULL WHERE id = %s", (row["id"],))
        return "re-enabled"
    return "known"


def _sources(conn: psycopg.Connection, job_ids: list[str]) -> list[Row]:
    return conn.execute("SELECT DISTINCT source FROM jobs WHERE job_key = ANY(%s)", (job_ids,)).fetchall()


def set_status(
    conn: psycopg.Connection,
    job_id: str,
    to_status: str,
    detail: str | None = None,
) -> None:
    """Move a job to ``to_status`` and append a ``status`` event.

    This is the deliberate stage-transition API; callers own the state machine.
    (The automatic no-regression guard is on ``upsert_job``, not here.)
    """
    if to_status not in STATUSES:
        raise ValueError(f"unknown status {to_status!r}; must be one of {STATUSES}")

    now = _now()
    with conn.transaction():
        row = conn.execute("SELECT id, status FROM jobs WHERE job_key = %s FOR UPDATE", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"no job with job_id {job_id!r}")
        conn.execute(
            "UPDATE jobs SET status = %s, status_detail = %s, status_changed_at = %s, updated_at = %s WHERE id = %s",
            (to_status, detail, now, now, row["id"]),
        )
        _add_event(conn, row["id"], "status", from_status=row["status"], to_status=to_status, detail=detail, at=now)


def set_score(
    conn: psycopg.Connection,
    job_id: str,
    *,
    score: float,
    features: Mapping[str, Any],
    review_reason: Sequence[str] | None,
    score_version: str,
    embed_tag: str,
    skills_hash: str,
    profile_hash: str,
) -> None:
    """Store one scoring result and its four reproducibility stamps (v6). Not a status change:
    the caller moves ``extracted`` -> ``scored`` with ``set_status``."""
    now = _now()
    cur = conn.execute(
        "UPDATE jobs SET score = %s, score_features = %s, review_reason = %s, scored_at = %s, score_version = %s, "
        "score_embed_tag = %s, score_skills_hash = %s, score_profile_hash = %s, updated_at = %s WHERE job_key = %s",
        (score, Jsonb(dict(features)), list(review_reason) if review_reason is not None else None, now,
         score_version, embed_tag, skills_hash, profile_hash, now, job_id),
    )  # fmt: skip
    if cur.rowcount != 1:
        raise KeyError(f"no job with job_id {job_id!r}")


def release_job(conn: psycopg.Connection, job_id: str) -> None:
    """The owner's override (D13): promote a scored job out of the review list and/or from below
    the apply line. Only a ``scored`` job can be released; a note event records it."""
    now = _now()
    with conn.transaction():
        row = conn.execute("SELECT id, status FROM jobs WHERE job_key = %s FOR UPDATE", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"no job with job_id {job_id!r}")
        if row["status"] != "scored":
            raise ValueError(f"job {job_id!r} is {row['status']!r}; only a scored job can be released")
        conn.execute("UPDATE jobs SET released_at = %s, updated_at = %s WHERE id = %s", (now, now, row["id"]))
        _add_event(conn, row["id"], "note", detail="scoring: released by the owner", at=now)


def set_years_required(conn: psycopg.Connection, job_id: str, years: int | None) -> None:
    """Store the filter funnel's extracted years-required feature.

    Not a status change and not an event — ``years_required`` is a stored
    feature (docs/scoring.md §1), written once by the seniority gate regardless
    of whether the gate then keeps or drops the job.
    """
    cur = conn.execute(
        "UPDATE jobs SET years_required = %s, updated_at = %s WHERE job_key = %s", (years, _now(), job_id)
    )
    if cur.rowcount != 1:
        raise KeyError(f"no job with job_id {job_id!r}")


def set_extraction(
    conn: psycopg.Connection,
    job_id: str,
    *,
    skills: Sequence[Mapping[str, str]],
    domain_bucket: str | None,
    legitimacy: str | None,
    extraction_confidence: str,
    extraction_version: str | None = None,
) -> None:
    """Store Step 3's skill-extraction output.

    Not a status change and not an event — same convention as
    ``set_years_required``: a stored feature, written once per job regardless
    of what the caller does with ``status`` next. ``skills`` is a list of
    ``{name, requirement}``, already vocab-canonicalized by the caller (this
    function does not touch the vocabulary); it replaces the job's
    ``job_skills`` rows in order, a name repeated in one list kept once (its
    first occurrence). ``domain_bucket``/``legitimacy`` are ``None`` on a
    too-thin-to-call skip; ``skills``/``extraction_confidence`` are always given
    (``[]`` / ``"low"`` for a skip). ``extraction_version`` (v5) stamps which
    model + prompt + schema produced it; the extraction stage always passes it.
    """
    with conn.transaction():
        jid = _id(conn, job_id)
        conn.execute(
            "UPDATE jobs SET domain_bucket = %s, legitimacy = %s, extraction_confidence = %s, "
            "extraction_version = %s, updated_at = %s WHERE id = %s",
            (domain_bucket, legitimacy, extraction_confidence, extraction_version, _now(), jid),
        )
        _replace_skills(conn, jid, skills)


def _replace_skills(conn: psycopg.Connection, jid: int, skills: Sequence[Mapping[str, str]]) -> None:
    conn.execute("DELETE FROM job_skills WHERE job_id = %s", (jid,))
    _insert_skills(conn, jid, skills, 0)


def _insert_skills(conn: psycopg.Connection, jid: int, skills: Sequence[Mapping[str, str]], start: int) -> None:
    """Append ``skills`` from position ``start``; a name repeated (or already stored) is kept once."""
    rows, names = [], set()
    for item in skills:
        if item["name"] not in names:
            rows.append((jid, item["name"], item["requirement"], item.get("source", "llm"), start + len(names)))
            names.add(item["name"])
    if rows:
        conn.cursor().executemany(
            "INSERT INTO job_skills (job_id, skill, requirement, source, position) VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (job_id, skill) DO NOTHING",
            rows,
        )


def set_dictionary_skills(conn: psycopg.Connection, job_id: str, names: Sequence[str], version: str) -> None:
    """Replace a job's dictionary-found skills (D10.3) with ``names``, as good_to_have after the LLM's own, and
    stamp ``dictionary_version``. The LLM's rows are not touched; a name the LLM already listed is skipped."""
    with conn.transaction():
        jid = _id(conn, job_id)
        conn.execute("DELETE FROM job_skills WHERE job_id = %s AND source = 'dictionary'", (jid,))
        start = conn.execute(
            "SELECT COALESCE(MAX(position) + 1, 0) AS n FROM job_skills WHERE job_id = %s", (jid,)
        ).fetchone()["n"]
        _insert_skills(
            conn, jid, [{"name": n, "requirement": "good_to_have", "source": "dictionary"} for n in names], start
        )
        conn.execute("UPDATE jobs SET dictionary_version = %s WHERE id = %s", (version, jid))


def set_skills(conn: psycopg.Connection, job_id: str, skills: Sequence[Mapping[str, str]], **columns: Any) -> None:
    """Rewrite a job's skills list (and any extraction ``columns`` given) without touching the rest -- for the
    re-canonicalization pass that rewrites spellings after a vocabulary merge."""
    allowed = {"extraction_confidence", "extraction_version"}
    if set(columns) - allowed:
        raise ValueError(f"set_skills only updates {sorted(allowed)}")
    with conn.transaction():
        jid = _id(conn, job_id)
        for col, value in columns.items():
            conn.execute(f"UPDATE jobs SET {col} = %s WHERE id = %s", (value, jid))
        _replace_skills(conn, jid, skills)


def record_outcome(
    conn: psycopg.Connection,
    job_id: str,
    outcome: str,
    detail: str | None = None,
    at: str | None = None,
) -> None:
    """Record a post-application outcome and append an ``outcome`` event.

    Does not change ``status`` — when a job moves to the ``outcome`` stage is
    the tracking slice's decision (build step 11). ``outcome`` is free text; the
    ``OUTCOMES`` tuple is the documented vocabulary, not a DB constraint.
    """
    now = _now()
    with conn.transaction():
        jid = _id(conn, job_id)
        conn.execute(
            "UPDATE jobs SET outcome = %s, outcome_at = %s, outcome_detail = %s, updated_at = %s WHERE id = %s",
            (outcome, at or now, detail, now, jid),
        )
        _add_event(conn, jid, "outcome", detail=f"{outcome}: {detail}" if detail else outcome, at=now)


def add_note(conn: psycopg.Connection, job_id: str, text: str) -> None:
    """Append a free-text ``note`` event. Changes no column."""
    _add_event(conn, _id(conn, job_id), "note", detail=text, at=_now())


def touch_job(conn: psycopg.Connection, job_id: str, *, run_id: str | None = None) -> None:
    """Record a re-sighting (dedup's cross-run match).

    Bumps ``last_seen_at``/``updated_at`` and appends a ``seen`` event. Does not
    change ``status``, ``first_seen_run``, or any Job column. Raises ``KeyError``
    if unknown. ``run_id`` (when given) is noted in the event detail.
    """
    now = _now()
    detail = "re-sighted (dedup match)" + (f" [{run_id}]" if run_id else "")
    with conn.transaction():
        _touch(conn, _id(conn, job_id), detail=detail, now=now)


def _touch(conn: psycopg.Connection, jid: int, *, detail: str, now: str) -> None:
    conn.execute("UPDATE jobs SET last_seen_at = %s, updated_at = %s WHERE id = %s", (now, now, jid))
    _add_event(conn, jid, "seen", detail=detail, at=now)


def _add_event(
    conn: psycopg.Connection,
    jid: int,
    kind: str,
    *,
    from_status: str | None = None,
    to_status: str | None = None,
    detail: str | None = None,
    at: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO job_events (job_id, kind, from_status, to_status, detail, at) VALUES (%s, %s, %s, %s, %s, %s)",
        (jid, kind, from_status, to_status, detail, at or _now()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# reads
# ─────────────────────────────────────────────────────────────────────────────
def get_job(conn: psycopg.Connection, job_id: str) -> dict | None:
    row = conn.execute(_JOB_SELECT + " WHERE j.job_key = %s", (job_id,)).fetchone()
    return dict(row) if row is not None else None


def job_exists(conn: psycopg.Connection, job_id: str) -> bool:
    """True if ``job_id`` is already stored — the cross-run dedup check (§8)."""
    return conn.execute("SELECT 1 FROM jobs WHERE job_key = %s", (job_id,)).fetchone() is not None


def select_jobs(conn: psycopg.Connection, where: str = "", params: Sequence[Any] = (), order: str = "") -> list[dict]:
    """Job dicts (``get_job``'s shape) for a ``where`` clause over the alias ``j`` (jobs) / ``b`` (boards)."""
    sql = _JOB_SELECT + (f" WHERE {where}" if where else "") + (f" ORDER BY {order}" if order else "")
    return [dict(r) for r in conn.execute(sql, params)]


def iter_jobs(
    conn: psycopg.Connection,
    *,
    status: str | None = None,
    source: str | None = None,
) -> Iterator[dict]:
    """Yield job rows (oldest first), optionally filtered by status and/or source."""
    clauses: list[str] = []
    params: list[Any] = []
    if status is not None:
        clauses.append("j.status = %s")
        params.append(status)
    if source is not None:
        clauses.append("j.source = %s")
        params.append(source)
    yield from select_jobs(conn, " AND ".join(clauses), params, "j.first_seen_at, j.job_key")


def get_events(conn: psycopg.Connection, job_id: str) -> list[dict]:
    """All events for a job, oldest first."""
    cur = conn.execute(
        "SELECT e.id, j.job_key AS job_id, e.kind, e.from_status, e.to_status, e.detail, e.at "
        "FROM job_events e JOIN jobs j ON j.id = e.job_id WHERE j.job_key = %s ORDER BY e.id",
        (job_id,),
    )
    return [dict(r) for r in cur]


def status_counts(conn: psycopg.Connection) -> dict[str, int]:
    """``{status: count}`` over every known status — the dashboard funnel snapshot."""
    counts = dict.fromkeys(STATUSES, 0)
    for row in conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"):
        counts[row["status"]] = row["n"]
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# runs (v2) — the orchestrator's run bookkeeping. See docs/sourcing.md §9-10.
# ─────────────────────────────────────────────────────────────────────────────
def open_run(conn: psycopg.Connection, run_id: str, *, kind: str, config_name: str | None) -> str:
    """Insert the ``running`` row. Returns ``started_at`` (the caller often wants it)."""
    if kind != "scoring":  # the schema also allows 'backfill' (old rows) and 'crawl' (the sourcing unit)
        raise ValueError(f"unknown run kind {kind!r}")
    now = _now()
    conn.execute(
        "INSERT INTO runs (run_id, kind, config_name, started_at, status) VALUES (%s, %s, %s, %s, 'running')",
        (run_id, kind, config_name, now),
    )
    return now


def checkpoint_run(conn: psycopg.Connection, run_id: str, *, detail: str) -> None:
    """Update only ``detail`` (the JSON progress blob) — a mid-run checkpoint."""
    conn.execute("UPDATE runs SET detail = %s::jsonb WHERE run_id = %s", (detail, run_id))


def finish_run(
    conn: psycopg.Connection,
    run_id: str,
    *,
    status: str,
    fetched: int = 0,
    collapsed_in_run: int = 0,
    matched_in_store: int = 0,
    persisted_new: int = 0,
    touched: int = 0,
    detail: str | None = None,
) -> None:
    if status not in RUN_STATUSES or status == "running":
        raise ValueError(f"cannot finish a run with status {status!r}")
    conn.execute(
        "UPDATE runs SET finished_at = %s, status = %s, fetched = %s, "
        "collapsed_in_run = %s, matched_in_store = %s, persisted_new = %s, "
        "touched = %s, detail = COALESCE(%s::jsonb, detail) WHERE run_id = %s",
        (_now(), status, fetched, collapsed_in_run, matched_in_store, persisted_new, touched, detail, run_id),
    )


def unfinished_run(conn: psycopg.Connection, kind: str, config_name: str | None) -> dict | None:
    """Most recent ``running`` row for this (kind, config_name) — for ``--resume``."""
    row = conn.execute(
        "SELECT * FROM runs WHERE kind = %s AND config_name IS NOT DISTINCT FROM %s "
        "AND finished_at IS NULL ORDER BY started_at DESC LIMIT 1",
        (kind, config_name),
    ).fetchone()
    return dict(row) if row is not None else None


def sweep_stale_runs(conn: psycopg.Connection, kind: str, config_name: str | None) -> int:
    """Mark still-``running`` rows for this (kind, config_name) as ``failed``.

    Called ONLY on a fresh (non-``--resume``) start, before ``open_run``. Scoped
    to the same operation so one config's start never fails another config's unfinished run.
    Assumes at most one live run per (kind, config_name) — true for the manual
    single-user CLI; add a lock/heartbeat if concurrent same-config runs appear.
    Returns the count swept.
    """
    cur = conn.execute(
        "UPDATE runs SET status = 'failed', finished_at = %s "
        "WHERE kind = %s AND config_name IS NOT DISTINCT FROM %s AND finished_at IS NULL",
        (_now(), kind, config_name),
    )
    return cur.rowcount


def list_runs(conn: psycopg.Connection, *, limit: int = 20) -> list[dict]:
    cur = conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT %s", (limit,))
    return [dict(r) for r in cur]


# ─────────────────────────────────────────────────────────────────────────────
# cooldowns (v2, migration 0002) — cross-process durability for CooldownRegistry
# (docs §5). Kept until the sourcing unit's board_health replaces it.
# ─────────────────────────────────────────────────────────────────────────────
def load_cooldowns(conn: psycopg.Connection) -> dict[str, float]:
    """``{key: until}`` (epoch seconds) for cooldowns that haven't expired."""
    now = time.time()
    return {r["key"]: r["until"] for r in conn.execute("SELECT key, until FROM cooldowns") if r["until"] > now}


def save_cooldown(conn: psycopg.Connection, key: str, until: float, reason: str | None = None) -> None:
    """Write-through from ``CooldownRegistry.on_trip``. Extends, never shortens.

    NOTE: called on the same thread as the ``trip()`` that produced ``until``.
    When the fetch loop is parallelized (deferred), a worker thread must NOT call
    this directly — marshal the save to the main thread. The single-connection /
    main-thread rule is load-bearing.
    """
    conn.execute(
        "INSERT INTO cooldowns (key, until, tripped_at, reason) VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (key) DO UPDATE SET until = GREATEST(cooldowns.until, excluded.until), "
        "tripped_at = excluded.tripped_at, reason = excluded.reason",
        (key, until, _now(), reason),
    )


def clear_cooldown(conn: psycopg.Connection, key: str | None = None) -> None:
    """Drop one cooldown, or all of them (manual override)."""
    if key is None:
        conn.execute("DELETE FROM cooldowns")
    else:
        conn.execute("DELETE FROM cooldowns WHERE key = %s", (key,))


# ─────────────────────────────────────────────────────────────────────────────
# skill_vocab (v3) — the growing canonical skill vocabulary (build step 3,
# see hiredra/vocab.py for the resolve logic that calls these).
# ─────────────────────────────────────────────────────────────────────────────
def touch_skill_vocab(conn: psycopg.Connection, name: str) -> None:
    """Record a sighting of ``name``: insert it (``seen_count=1``) if new,
    else bump ``seen_count``. ``name`` is stored exactly as given — callers
    normalize case/whitespace before calling, this function does no folding.
    """
    conn.execute(
        "INSERT INTO skill_vocab (name, first_seen_at, seen_count) VALUES (%s, %s, 1) "
        "ON CONFLICT (name) DO UPDATE SET seen_count = skill_vocab.seen_count + 1",
        (name, _now()),
    )


# Vectors cross this boundary as opaque little-endian float32 ``bytes`` (hiredra.graph.index owns the codec), so
# the graph code is unchanged by the move; Postgres stores them as pgvector ``vector`` (float4 too: exact).
_VOCAB_SELECT = (
    "SELECT name, description, first_seen_at, seen_count, vector::real[] AS vector, embed_tag, mine, level, source, "
    "edges FROM skill_vocab"
)


def _vocab_row(row: Row) -> dict:
    d = dict(row)
    if d["vector"] is not None:
        d["vector"] = struct.pack(f"<{len(d['vector'])}f", *d["vector"])
    return d


def _vector_text(blob: bytes) -> str:
    return "[" + ",".join(repr(x) for x in struct.unpack(f"<{len(blob) // 4}f", blob)) + "]"


def get_skill_vocab(conn: psycopg.Connection, name: str) -> dict | None:
    """Exact lookup by canonical name (case-sensitive — fold before calling)."""
    row = conn.execute(_VOCAB_SELECT + " WHERE name = %s", (name,)).fetchone()
    return _vocab_row(row) if row is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# skill graph nodes + aliases (v4) — build step 4. Persistence only: the Skill
# object, the vector codec and the embed/merge policy live in hiredra.graph.
# ─────────────────────────────────────────────────────────────────────────────
def add_market_skill(conn: psycopg.Connection, name: str) -> bool:
    """Insert ``name`` as a ``source='market'`` node (``seen_count=0``: it has
    never been seen in one of OUR jobs). Insert-or-ignore: an existing node is
    left untouched (its ``seen_count``/``source`` are not bumped or rewritten),
    and a name that is already a merged alias is skipped -- a string is a node
    or an alias, never both. Returns True only if a new row was inserted.
    ``name`` is stored exactly as given (fold before calling).
    """
    with conn.transaction():
        if conn.execute("SELECT 1 FROM skill_alias WHERE alias = %s", (name,)).fetchone():
            return False
        cur = conn.execute(
            "INSERT INTO skill_vocab (name, first_seen_at, seen_count, source) "
            "VALUES (%s, %s, 0, 'market') ON CONFLICT (name) DO NOTHING",
            (name, _now()),
        )
        return cur.rowcount == 1


def resolve_alias(conn: psycopg.Connection, name: str) -> str | None:
    """The canonical node ``name`` was merged into, or None if it isn't an alias."""
    row = conn.execute("SELECT canonical FROM skill_alias WHERE alias = %s", (name,)).fetchone()
    return row["canonical"] if row is not None else None


def add_alias(conn: psycopg.Connection, alias: str, canonical: str) -> None:
    """Record that ``alias`` is another spelling of the existing node ``canonical``.

    Raises ``ValueError`` if ``canonical`` is not a node, if ``alias`` is itself a
    node (merge it with ``merge_skill`` instead -- that is what keeps seen_count
    and existing aliases from being lost), or if ``alias`` is already an alias of
    a DIFFERENT node (silently retargeting an alias would rewrite history).
    Re-adding an identical alias is a no-op.
    """
    if alias == canonical:
        raise ValueError(f"alias {alias!r} cannot point at itself")
    with conn.transaction():
        if conn.execute("SELECT 1 FROM skill_vocab WHERE name = %s", (canonical,)).fetchone() is None:
            raise ValueError(f"canonical skill {canonical!r} does not exist")
        if conn.execute("SELECT 1 FROM skill_vocab WHERE name = %s", (alias,)).fetchone() is not None:
            raise ValueError(f"{alias!r} is a canonical skill, not an alias -- use merge_skill")
        existing = resolve_alias(conn, alias)
        if existing is not None:
            if existing != canonical:
                raise ValueError(f"alias {alias!r} already points at {existing!r}, not {canonical!r}")
            return
        conn.execute(
            "INSERT INTO skill_alias (alias, canonical, created_at) VALUES (%s, %s, %s)", (alias, canonical, _now())
        )


def merge_skill(conn: psycopg.Connection, dup: str, canonical: str) -> None:
    """Fold node ``dup`` into node ``canonical`` (one transaction).

    ``canonical`` gains ``dup``'s ``seen_count`` and the earlier ``first_seen_at``;
    any aliases that pointed at ``dup`` are re-pointed at ``canonical`` (aliases
    stay one hop, never a chain); ``dup`` is deleted and recorded as an alias of
    ``canonical``. ``dup``'s vector is dropped -- the surviving node's is kept.
    Which node survives is the caller's decision (this is mechanism, not policy).

    A ``mine`` node is never absorbed (``ValueError``): the owner's capability is
    edited in the master profile, not silently folded away. ``job_skills`` still
    stores whatever spelling extraction wrote; ``resolve_alias`` maps it back.
    """
    if dup == canonical:
        raise ValueError("cannot merge a skill into itself")
    with conn.transaction():
        d = conn.execute("SELECT * FROM skill_vocab WHERE name = %s FOR UPDATE", (dup,)).fetchone()
        c = conn.execute("SELECT * FROM skill_vocab WHERE name = %s FOR UPDATE", (canonical,)).fetchone()
        if d is None or c is None:
            missing = dup if d is None else canonical
            raise ValueError(f"skill {missing!r} does not exist")
        if d["mine"]:
            raise ValueError(f"{dup!r} is a 'mine' skill and is never merged away")
        conn.execute(
            "UPDATE skill_vocab SET seen_count = seen_count + %s, first_seen_at = LEAST(first_seen_at, %s) "
            "WHERE name = %s",
            (d["seen_count"], d["first_seen_at"], canonical),
        )
        # The alias rows are moved before the node goes: skill_alias.canonical cascades on delete.
        conn.execute("UPDATE skill_alias SET canonical = %s WHERE canonical = %s", (canonical, dup))
        conn.execute("DELETE FROM skill_vocab WHERE name = %s", (dup,))
        conn.execute(
            "INSERT INTO skill_alias (alias, canonical, created_at) VALUES (%s, %s, %s)", (dup, canonical, _now())
        )


def iter_skill_vocab(conn: psycopg.Connection) -> list[dict]:
    """Every node as a plain dict (all columns incl. ``vector`` bytes), by name."""
    return [_vocab_row(r) for r in conn.execute(_VOCAB_SELECT + " ORDER BY name")]


def apply_profile_overlay(
    conn: psycopg.Connection,
    *,
    mine: Mapping[str, str],
    new_nodes: Mapping[str, str],
    aliases: Iterable[tuple[str, str]],
) -> None:
    """Rebuild the ``mine`` flags from the master profile, in ONE transaction (PF1).

    Every node first loses ``mine``/``level`` (the graph is rebuilt from the profile file, the
    source-of-truth rule). Then each existing node in ``mine`` ({name: level}) gets them back,
    each name in ``new_nodes`` ({name: level}) is created as a ``source='profile'`` node, and
    each ``(alias, canonical)`` is recorded. Before committing, the alias invariant is
    re-checked (no string is both a node and an alias; no alias points at another alias); any
    violation rolls the whole overlay back. ``ValueError`` if a ``mine`` name is not a node or
    a new name already exists.
    """
    now = _now()
    with conn.transaction():
        conn.execute("UPDATE skill_vocab SET mine = false, level = NULL WHERE mine OR level IS NOT NULL")
        for name, level in mine.items():
            cur = conn.execute("UPDATE skill_vocab SET mine = true, level = %s WHERE name = %s", (level, name))
            if cur.rowcount != 1:
                raise ValueError(f"profile overlay: {name!r} is not a node")
        for name, level in new_nodes.items():
            if conn.execute("SELECT 1 FROM skill_vocab WHERE name = %s", (name,)).fetchone():
                raise ValueError(f"profile overlay: new node {name!r} already exists")
            conn.execute(
                "INSERT INTO skill_vocab (name, first_seen_at, seen_count, source, mine, level) "
                "VALUES (%s, %s, 0, 'profile', true, %s)",
                (name, now, level),
            )
        for alias, canonical in aliases:
            existing = conn.execute("SELECT canonical FROM skill_alias WHERE alias = %s", (alias,)).fetchone()
            if existing is not None:
                if existing["canonical"] != canonical:
                    raise ValueError(f"profile overlay: alias {alias!r} already points at {existing['canonical']!r}")
                continue
            if conn.execute("SELECT 1 FROM skill_vocab WHERE name = %s", (canonical,)).fetchone() is None:
                raise ValueError(
                    f"profile overlay broke the alias invariant: node+alias [], chains [], dangling [{alias!r}]"
                )
            conn.execute(
                "INSERT INTO skill_alias (alias, canonical, created_at) VALUES (%s, %s, %s)", (alias, canonical, now)
            )
        both = conn.execute("SELECT alias FROM skill_alias WHERE alias IN (SELECT name FROM skill_vocab)").fetchall()
        chains = conn.execute(
            "SELECT alias FROM skill_alias WHERE canonical IN (SELECT alias FROM skill_alias)"
        ).fetchall()
        if both or chains:
            raise ValueError(
                "profile overlay broke the alias invariant: "
                f"node+alias {[r[0] for r in both]}, chains {[r[0] for r in chains]}, dangling []"
            )


def list_skill_aliases(conn: psycopg.Connection) -> dict[str, list[str]]:
    """``{canonical: [aliases…]}`` (aliases sorted), only for nodes that have any."""
    out: dict[str, list[str]] = {}
    for r in conn.execute("SELECT canonical, alias FROM skill_alias ORDER BY canonical, alias"):
        out.setdefault(r["canonical"], []).append(r["alias"])
    return out


def skills_needing_embedding(
    conn: psycopg.Connection,
    embed_tag: str,
    limit: int | None = None,
    weights: Mapping[str, int] | None = None,
) -> list[str]:
    """Nodes with no vector yet, or one made under a different ``embed_tag``
    (stale). This IS the resumable work queue for an index build: a chunk's
    vectors commit before the next chunk starts, so re-running just continues.
    Highest-value first -- ``mine`` nodes, then the most-sighted in OUR jobs,
    then (if ``weights`` is given, e.g. market document frequency) the most
    common, then by name -- so an interrupted build has embedded what scoring
    is most likely to need.
    """
    rows = conn.execute(
        "SELECT name, mine, seen_count FROM skill_vocab WHERE vector IS NULL OR embed_tag IS DISTINCT FROM %s",
        (embed_tag,),
    ).fetchall()
    w = weights or {}
    rows.sort(key=lambda r: (-r["mine"], -r["seen_count"], -w.get(r["name"], 0), r["name"]))
    names = [r["name"] for r in rows]
    return names if limit is None else names[:limit]


def skill_index_status(conn: psycopg.Connection, embed_tag: str) -> dict[str, int]:
    """Counts describing the embedded index against the live ``embed_tag``:
    ``total`` nodes, ``current`` (vector under this tag), ``stale`` (a vector
    under another tag), ``unembedded`` (no vector), plus ``market``/``jd``/
    ``profile`` node counts and ``aliases``."""
    row = conn.execute(
        "SELECT COUNT(*) AS total, "
        "COUNT(*) FILTER (WHERE vector IS NOT NULL AND embed_tag IS NOT DISTINCT FROM %s) AS current, "
        "COUNT(*) FILTER (WHERE vector IS NOT NULL AND embed_tag IS DISTINCT FROM %s) AS stale, "
        "COUNT(*) FILTER (WHERE vector IS NULL) AS unembedded, "
        "COUNT(*) FILTER (WHERE source = 'market') AS market, COUNT(*) FILTER (WHERE source = 'jd') AS jd, "
        "COUNT(*) FILTER (WHERE source = 'profile') AS profile FROM skill_vocab",
        (embed_tag, embed_tag),
    ).fetchone()
    out = {k: int(row[k] or 0) for k in row.keys()}
    out["aliases"] = conn.execute("SELECT COUNT(*) FROM skill_alias").fetchone()[0]
    return out


def invalidate_skill_vectors(conn: psycopg.Connection) -> int:
    """Drop EVERY stored vector and its tag, so the whole index re-enters the
    embedding queue. Used when the canary shows the provider changed weights
    under an unchanged model id: those vectors are no longer comparable to
    anything embedded from now on. Returns the number of rows cleared."""
    cur = conn.execute("UPDATE skill_vocab SET vector = NULL, embed_tag = NULL WHERE vector IS NOT NULL")
    return cur.rowcount


def set_skill_vectors(conn: psycopg.Connection, items: list[tuple[str, bytes]], embed_tag: str) -> None:
    """Write ``(name, vector_blob)`` pairs and stamp them with ``embed_tag``, all
    in ONE transaction: a chunk lands whole or not at all. An unknown ``name``
    raises ``KeyError`` and rolls the whole batch back -- vectors are never
    silently dropped."""
    with conn.transaction():
        for name, blob in items:
            cur = conn.execute(
                "UPDATE skill_vocab SET vector = %s::vector, embed_tag = %s WHERE name = %s",
                (_vector_text(blob), embed_tag, name),
            )
            if cur.rowcount != 1:
                raise KeyError(f"no skill_vocab row named {name!r}")
