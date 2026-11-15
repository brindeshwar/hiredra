"""Batches of prepared applications, and the human's decision on each one.

A batch takes up to 10 ``tailored`` jobs whose form has been read (a ``form_scan`` artifact), best score first, and
moves them to ``queued``. A job is never in two open batches. A batch closes when every job in it has a decision.

Decisions are the human's own (the status machine, plan contract 3.2). Code never decides that an application was
sent:

- ``applied``: only from ``queued``, and only with the confirmation text the site showed, typed or pasted by the human;
- ``not_yet``: stays ``queued`` (a note records it);
- ``deferred``: back to ``tailored``, out of the batch;
- ``skipped`` / ``closed``: ``dropped`` (``gate: skipped`` / ``closed``).
"""

from __future__ import annotations

from hiredra.storage import add_note, set_status

BATCH_SIZE = 10
DECISIONS = ("applied", "not_yet", "deferred", "skipped", "closed")


def _next_id(conn) -> str:
    row = conn.execute("SELECT max(batch_id) FROM batches").fetchone()
    n = int(row[0][1:]) + 1 if row and row[0] else 1
    return f"B{n:04d}"


def ready_jobs(conn, limit: int = BATCH_SIZE) -> list[str]:
    """Tailored jobs with a form scan that are in no open batch, best score first."""
    rows = conn.execute(
        "SELECT j.job_key FROM jobs j WHERE j.status = 'tailored' "
        "AND EXISTS (SELECT 1 FROM artifacts a WHERE a.job_id = j.id AND a.kind = 'form_scan') "
        "AND NOT EXISTS (SELECT 1 FROM batch_items i JOIN batches b ON b.batch_id = i.batch_id "
        "                WHERE i.job_id = j.id AND b.status <> 'closed' AND i.decision IS NULL) "
        "ORDER BY j.score DESC NULLS LAST, j.job_key LIMIT %s",
        (limit,),
    )
    return [r[0] for r in rows]


def make_batch(conn, *, size: int = BATCH_SIZE, min_size: int = 1) -> dict | None:
    """The next batch of up to ``size`` ready jobs, or None when fewer than ``min_size`` are ready."""
    jobs = ready_jobs(conn, size)
    if len(jobs) < max(1, min_size):
        return None
    with conn.transaction():
        batch_id = _next_id(conn)
        conn.execute("INSERT INTO batches (batch_id) VALUES (%s)", (batch_id,))
        for pos, job_key in enumerate(jobs, 1):
            conn.execute(
                "INSERT INTO batch_items (batch_id, job_id, position) SELECT %s, id, %s FROM jobs WHERE job_key = %s",
                (batch_id, pos, job_key),
            )
            set_status(conn, job_key, "queued", detail=f"batch: {batch_id}")
    return get_batch(conn, batch_id)


def get_batch(conn, batch_id: str) -> dict | None:
    row = conn.execute("SELECT batch_id, created_at, status FROM batches WHERE batch_id = %s", (batch_id,)).fetchone()
    if row is None:
        return None
    items = conn.execute(
        "SELECT j.job_key AS job_id, j.title, j.company, j.status, i.position, i.decision, i.decided_at "
        "FROM batch_items i JOIN jobs j ON j.id = i.job_id WHERE i.batch_id = %s ORDER BY i.position",
        (batch_id,),
    ).fetchall()
    return {**dict(row), "items": [dict(i) for i in items]}


def list_batches(conn) -> list[dict]:
    ids = [r[0] for r in conn.execute("SELECT batch_id FROM batches ORDER BY batch_id DESC")]
    return [get_batch(conn, b) for b in ids]


def decide(conn, job_key: str, decision: str, confirmation: str = "") -> dict:
    """Record the human's decision. ValueError when it is not allowed from the job's current state."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    row = conn.execute(
        "SELECT j.status, i.batch_id FROM jobs j LEFT JOIN batch_items i ON i.job_id = j.id AND i.decision IS NULL "
        "WHERE j.job_key = %s",
        (job_key,),
    ).fetchone()
    if row is None:
        raise KeyError(job_key)
    status, batch_id = row["status"], row["batch_id"]
    if status != "queued":
        raise ValueError(f"only a queued job takes a decision (this one is {status}); put it in a batch first")
    with conn.transaction():
        if decision == "applied":
            text = " ".join(confirmation.split())
            if len(text) < 5:
                raise ValueError("'applied' needs the confirmation text the site showed after you submitted")
            set_status(conn, job_key, "applied", detail=f"confirmed: {text[:160]}")
        elif decision == "not_yet":
            add_note(conn, job_key, "gate: not yet")
            return {"job_id": job_key, "status": "queued", "decision": decision}
        elif decision == "deferred":
            set_status(conn, job_key, "tailored", detail="gate: deferred")
        else:
            set_status(conn, job_key, "dropped", detail="gate: skipped" if decision == "skipped" else "closed")
        if batch_id:
            conn.execute(
                "UPDATE batch_items SET decision = %s, decided_at = now() WHERE batch_id = %s "
                "AND job_id = (SELECT id FROM jobs WHERE job_key = %s)",
                (decision, batch_id, job_key),
            )
            conn.execute(
                "UPDATE batches SET status = CASE WHEN NOT EXISTS (SELECT 1 FROM batch_items "
                "WHERE batch_id = %s AND decision IS NULL) THEN 'closed' ELSE 'in_review' END WHERE batch_id = %s",
                (batch_id, batch_id),
            )
    return {"job_id": job_key, "status": conn.execute("SELECT status FROM jobs WHERE job_key = %s", (job_key,)).fetchone()[0],
            "decision": decision}  # fmt: skip


def record_attention(conn, job_key: str, fields: list[dict]) -> int:
    """Replace the job's open attention items with the PAUSE fields (labels and categories only)."""
    pauses = [f for f in fields if f.get("class") == "PAUSE"]
    with conn.transaction():
        conn.execute(
            "DELETE FROM attention WHERE job_id = (SELECT id FROM jobs WHERE job_key = %s) AND resolved_at IS NULL",
            (job_key,),
        )
        for f in pauses:
            conn.execute(
                "INSERT INTO attention (job_id, category, field_label) SELECT id, %s, %s FROM jobs WHERE job_key = %s",
                (f.get("category") or "unknown", f["label"][:300], job_key),
            )
    return len(pauses)
