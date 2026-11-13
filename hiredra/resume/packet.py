"""A packet: everything prepared for one job, in one folder, and the job moved to ``tailored``.

``data/packets/<folder>/`` (git-ignored) holds ``cv.docx``, ``cv.pdf``, ``cover.docx``, ``cover.pdf``,
``payload.json`` (the résumé's data) and ``claims.json`` (the claim gate's findings). Each file is recorded in the
``artifacts`` table with its sha256, the model and the prompt version. A packet whose text needed repair is flagged
in a note on the job; it is still built, because a repaired piece uses only the profile's own words.

Which jobs get packets (``select_jobs``): ``scored`` jobs with no review reason and no packet yet, best score first.
The weights are placeholders until Step 7, so this order is provisional.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from hiredra.resume import cover, tailor
from hiredra.resume.build import write_cover, write_cv
from hiredra.resume.claims import Gate
from hiredra.resume.settings import ResumeConfig, SkillDisplay
from hiredra.resume.skills import fold
from hiredra.resume.source import ProfileText
from hiredra.storage import add_artifact, add_note, get_job, set_status

_REPO_ROOT = Path(__file__).resolve().parents[2]
PACKETS_DIR = _REPO_ROOT / "data" / "packets"


@dataclass
class PacketResult:
    job_id: str
    folder: Path
    pages: int | None
    flags: list[str] = field(default_factory=list)
    calls: int = 0


def folder_name(job_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", job_id)[:80]


def build_gate(conn, profile: ProfileText) -> Gate:
    """The claim gate for this run: every skill name and alias in the graph (what it looks for in text), the alias
    table (so "mcp servers" is checked as "mcp"), and extraction's generic words (never a claim on their own)."""
    from hiredra.extraction import load_extraction_config

    names = {fold(r[0]) for r in conn.execute("SELECT name FROM skill_vocab")}
    aliases = {fold(r[0]): fold(r[1]) for r in conn.execute("SELECT alias, canonical FROM skill_alias")}
    generic = load_extraction_config().generic_skills
    from hiredra.resume.settings import load_resume_config

    banned = tuple(load_resume_config().banned_phrases)
    return Gate(profile, names | set(aliases), aliases=aliases, ignore=frozenset(generic), banned=banned)


def select_jobs(conn, top: int) -> list[str]:
    rows = conn.execute(
        "SELECT j.job_key FROM jobs j WHERE j.status = 'scored' AND j.review_reason IS NULL "
        "AND NOT EXISTS (SELECT 1 FROM artifacts a WHERE a.job_id = j.id AND a.kind = 'cv_docx') "
        "ORDER BY j.score DESC NULLS LAST, j.job_key LIMIT %s",
        (top,),
    )
    return [r[0] for r in rows]


def _record(conn, job_id: str, kind: str, path: Path, model: str | None, prompt: str | None) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rel = path.resolve().relative_to(_REPO_ROOT).as_posix() if path.resolve().is_relative_to(_REPO_ROOT) else str(path)
    add_artifact(conn, job_id, kind, rel, digest, model=model, prompt_version=prompt)


def build_packet(
    conn,
    job_id: str,
    *,
    profile: ProfileText,
    candidate,
    gate: Gate,
    caller,
    display: SkillDisplay,
    cfg: ResumeConfig,
    out_root: Path = PACKETS_DIR,
    rebuild: bool = False,
) -> PacketResult:
    """``rebuild`` remakes the packet of a job already ``tailored`` (new files, new artifact rows)."""
    job = get_job(conn, job_id)
    allowed = ("scored", "tailored") if rebuild else ("scored",)
    if job is None or job["status"] not in allowed:
        raise ValueError(
            f"{job_id}: a packet needs a scored job (it is {job and job['status']}; --rebuild for tailored)"
        )
    before = caller.calls
    folder = out_root / folder_name(job_id)
    folder.mkdir(parents=True, exist_ok=True)

    cv = tailor.tailor(job, profile, candidate, gate, caller, display, cfg)
    letter, cover_claims, cover_flags = cover.write_letter(job, profile, cv.payload, gate, caller, cfg)
    cv_files = write_cv(cv.payload, folder)
    cover_files = write_cover(letter, folder)

    (folder / "payload.json").write_text(cv.payload.model_dump_json(indent=1), encoding="utf-8")
    claims = {"cv": cv.claims, "cover": cover_claims, "rounds": cv.rounds}
    (folder / "claims.json").write_text(json.dumps(claims, indent=1), encoding="utf-8")

    model = cfg.model
    with conn.transaction():
        _record(conn, job_id, "cv_docx", cv_files.docx, model, tailor.PROMPT_VERSION)
        if cv_files.pdf:
            _record(conn, job_id, "cv_pdf", cv_files.pdf, model, tailor.PROMPT_VERSION)
        _record(conn, job_id, "cover_docx", cover_files.docx, model, cover.PROMPT_VERSION)
        if cover_files.pdf:
            _record(conn, job_id, "cover_pdf", cover_files.pdf, model, cover.PROMPT_VERSION)
        _record(conn, job_id, "payload", folder / "payload.json", None, None)
        _record(conn, job_id, "claims", folder / "claims.json", None, None)
        flags = sorted(set(cv.flags + [f"cover:{f}" for f in cover_flags]))
        if cv_files.pages is not None and cv_files.pages > 1:
            flags.append("page_overflow")
        if cv_files.pdf is None:
            flags.append("no_pdf")
        if flags:
            add_note(conn, job_id, "packet: " + ", ".join(flags))
        if job["status"] == "scored":
            set_status(conn, job_id, "tailored")
    return PacketResult(job_id, folder, cv_files.pages, flags, caller.calls - before)
