"""Prepare one tailored job for the human: read its form, build the answers, store both, list what needs the human.

Writes ``form_scan.json`` and ``answers.json`` into the job's packet folder, records them as artifacts, and replaces the
job's open attention items with the form's PAUSE fields. A posting that is gone (404) is dropped as ``closed``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from hiredra.autofill.answers import PROMPT_VERSION, build_answers
from hiredra.autofill.batch import record_attention
from hiredra.autofill.scan import scan_job
from hiredra.resume.packet import PACKETS_DIR, folder_name
from hiredra.storage import add_artifact, get_job, set_status

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Prepared:
    job_id: str
    via: str
    dead: bool
    counts: dict[str, int]


def _record(conn, job_id: str, kind: str, path: Path, model: str | None) -> None:
    rel = path.resolve().relative_to(_REPO_ROOT).as_posix() if path.resolve().is_relative_to(_REPO_ROOT) else str(path)
    add_artifact(conn, job_id, kind, rel, hashlib.sha256(path.read_bytes()).hexdigest(), model=model,
                 prompt_version=PROMPT_VERSION if model else None)  # fmt: skip


def prepare_job(conn, job_id: str, *, facts: dict, profile, gate, caller=None, http=None, model: str | None = None,
                packets_dir: Path = PACKETS_DIR) -> Prepared:  # fmt: skip
    job = get_job(conn, job_id)
    if job is None or job["status"] not in ("tailored", "queued"):
        raise ValueError(f"{job_id}: only a tailored job is prepared (it is {job and job['status']})")
    folder = packets_dir / folder_name(job_id)
    folder.mkdir(parents=True, exist_ok=True)
    scan = scan_job(job, http)
    (folder / "form_scan.json").write_text(json.dumps(scan, indent=1), encoding="utf-8")
    if scan["dead"]:
        with conn.transaction():
            _record(conn, job_id, "form_scan", folder / "form_scan.json", None)
            set_status(conn, job_id, "dropped", detail="closed")
        return Prepared(job_id, scan["via"], True, {})
    summary = ""
    payload = folder / "payload.json"
    if payload.exists():
        summary = json.loads(payload.read_text(encoding="utf-8")).get("summary", "")
    answers = build_answers(scan, facts, job=job, profile=profile, gate=gate, caller=caller, cv_summary=summary,
                            packet_dir=folder)  # fmt: skip
    (folder / "answers.json").write_text(json.dumps(answers, indent=1), encoding="utf-8")
    drafted = any(f["class"] == "DRAFT" for f in answers["fields"])
    with conn.transaction():
        _record(conn, job_id, "form_scan", folder / "form_scan.json", None)
        _record(conn, job_id, "answers", folder / "answers.json", model if drafted else None)
        record_attention(conn, job_id, answers["fields"])
    counts: dict[str, int] = {}
    for f in answers["fields"]:
        counts[f["class"]] = counts.get(f["class"], 0) + 1
    return Prepared(job_id, scan["via"], False, counts)
