"""The sidecar API: the only way the desktop app reaches the pipeline (plan contract 3.10).

Local only: the server binds 127.0.0.1, and every request but ``/health`` needs ``Authorization: Bearer <token>``
with the token the app generated for this launch. Read-mostly: it lists and shows jobs, packets and runs, starts
pipeline stages (``runs.py``), and records the owner's own triage. It never submits or sends anything, and it never
returns a secret or the database address.
"""

from __future__ import annotations

import secrets
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pydantic import BaseModel

from hiredra.api.runs import Runs
from hiredra.resume.packet import PACKETS_DIR, folder_name
from hiredra.storage import artifacts_for, connect, get_job, release_job, set_status, status_counts

VIEWS = {
    "ranked": "j.status = 'scored' AND j.review_reason IS NULL",
    "review": "j.status = 'scored' AND j.review_reason IS NOT NULL AND j.released_at IS NULL",
    "tailored": "j.status = 'tailored'",
    "queued": "j.status = 'queued'",
    "applied": "j.status IN ('applied', 'outcome')",
    "all": "j.status IN ('scored', 'tailored', 'queued', 'applied', 'outcome')",
}
PACKET_FILES = {"cv.docx", "cv.pdf", "cover.docx", "cover.pdf", "payload.json", "claims.json", "form_scan.json",
                "answers.json"}  # fmt: skip


class StartRun(BaseModel):
    top: int = 10


class Reject(BaseModel):
    reason: str = "rejected"


def create_app(*, token: str, db_url: str | None = None, runs: Runs | None = None,
               packets_dir: Path = PACKETS_DIR) -> FastAPI:  # fmt: skip
    if not token:
        raise ValueError("the API needs a token")
    run_table = runs or Runs()

    @asynccontextmanager
    async def lifespan(_app):
        yield
        run_table.stop_all()  # the app quit: no stage keeps running without it

    app = FastAPI(title="Hiredra sidecar", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.runs = run_table

    def auth(authorization: str = Header(default="")) -> None:
        if not secrets.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="missing or wrong token")

    def db() -> Iterator:
        conn = connect(db_url)
        try:
            yield conn
        finally:
            conn.close()

    def job_or_404(conn, job_key: str) -> dict:
        job = get_job(conn, job_key)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/status", dependencies=[Depends(auth)])
    def status(conn=Depends(db)):
        usage = conn.execute(
            "SELECT provider, model, calls, exhausted_at IS NOT NULL AS exhausted FROM api_usage "
            "WHERE day = (SELECT max(day) FROM api_usage) ORDER BY provider, model"
        ).fetchall()
        audits = sorted((Path(__file__).resolve().parents[2] / "data" / "graph-audit").glob("*.md"))
        return jsonable_encoder({
            "jobs": status_counts(conn),
            "usage": [dict(r) for r in usage],
            "latest_audit": audits[-1].name if audits else None,
            "runs": [r.view() | {"log": r.view()["log"][-3:]} for r in app.state.runs.all()[:10]],
        })  # fmt: skip

    @app.get("/jobs", dependencies=[Depends(auth)])
    def jobs(view: str = "ranked", limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
             conn=Depends(db)):  # fmt: skip
        if view not in VIEWS:
            raise HTTPException(status_code=400, detail=f"view must be one of {sorted(VIEWS)}")
        where = VIEWS[view]
        total = conn.execute(f"SELECT count(*) FROM jobs j WHERE {where}").fetchone()[0]
        rows = conn.execute(
            "SELECT j.job_key AS job_id, j.title, j.company, j.location, j.score, j.status, j.review_reason, "
            f"j.released_at, j.apply_url, j.url FROM jobs j WHERE {where} "
            "ORDER BY j.score DESC NULLS LAST, j.job_key LIMIT %s OFFSET %s",
            (limit, offset),
        ).fetchall()
        return jsonable_encoder({"view": view, "total": total, "items": [dict(r) for r in rows]})

    @app.get("/jobs/{job_key:path}/events", dependencies=[Depends(auth)])
    def job_events(job_key: str, conn=Depends(db)):
        job_or_404(conn, job_key)
        rows = conn.execute(
            "SELECT e.kind, e.at, e.detail FROM job_events e JOIN jobs j ON j.id = e.job_id WHERE j.job_key = %s "
            "ORDER BY e.at DESC, e.id DESC LIMIT 100",
            (job_key,),
        ).fetchall()
        return jsonable_encoder([dict(r) for r in rows])

    @app.post("/jobs/{job_key:path}/release", dependencies=[Depends(auth)])
    def release(job_key: str, conn=Depends(db)):
        job_or_404(conn, job_key)
        try:
            release_job(conn, job_key)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/jobs/{job_key:path}/reject", dependencies=[Depends(auth)])
    def reject(job_key: str, body: Reject | None = None, conn=Depends(db)):
        job = job_or_404(conn, job_key)
        if job["status"] != "scored":
            raise HTTPException(status_code=409, detail=f"only a scored job can be rejected (it is {job['status']})")
        set_status(conn, job_key, "dropped", detail=f"review: {(body or Reject()).reason}")
        return {"ok": True}

    @app.get("/jobs/{job_key:path}", dependencies=[Depends(auth)])
    def job(job_key: str, conn=Depends(db)):
        return jsonable_encoder(job_or_404(conn, job_key))

    @app.post("/runs/{stage}", dependencies=[Depends(auth)])
    def start_run(stage: str, body: StartRun | None = None):
        if stage not in app.state.runs.stages:
            raise HTTPException(status_code=404, detail=f"stage must be one of {sorted(app.state.runs.stages)}")
        try:
            run = app.state.runs.start(stage, top=(body or StartRun()).top)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return run.view()

    @app.get("/runs", dependencies=[Depends(auth)])
    def list_runs():
        return [r.view() | {"log": r.view()["log"][-3:]} for r in app.state.runs.all()]

    @app.get("/runs/{run_id}", dependencies=[Depends(auth)])
    def get_run(run_id: str):
        run = app.state.runs.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="no such run")
        return run.view()

    @app.get("/packets/{job_key:path}/files/{name}", dependencies=[Depends(auth)])
    def packet_file(job_key: str, name: str, conn=Depends(db)):
        job_or_404(conn, job_key)
        if name not in PACKET_FILES:
            raise HTTPException(status_code=400, detail="not a packet file")
        path = (packets_dir / folder_name(job_key) / name).resolve()
        if not path.is_relative_to(packets_dir.resolve()) or not path.exists():
            raise HTTPException(status_code=404, detail="no such file in this packet")
        return FileResponse(path, filename=name)

    @app.get("/packets/{job_key:path}", dependencies=[Depends(auth)])
    def packet(job_key: str, conn=Depends(db)):
        job_or_404(conn, job_key)
        folder = packets_dir / folder_name(job_key)
        files = sorted(p.name for p in folder.iterdir() if p.name in PACKET_FILES) if folder.exists() else []
        notes = conn.execute(
            "SELECT e.detail FROM job_events e JOIN jobs j ON j.id = e.job_id WHERE j.job_key = %s AND e.kind = 'note' "
            "AND e.detail LIKE 'packet:%%' ORDER BY e.at DESC LIMIT 1",
            (job_key,),
        ).fetchone()
        return jsonable_encoder({
            "job_id": job_key, "folder": folder.name, "files": files, "artifacts": artifacts_for(conn, job_key),
            "flags": notes[0].removeprefix("packet: ").split(", ") if notes else [],
        })  # fmt: skip

    return app
