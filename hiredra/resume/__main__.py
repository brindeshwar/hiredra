"""The résumé step's command line.

    python -m hiredra.resume sample [--out DIR]          # a résumé straight from the master profile, no model call
    python -m hiredra.resume render PAYLOAD.json         # render a payload file (the contract in payload.py)
    python -m hiredra.resume packets --top N [--dry-run] # tailored packets for the N best scored jobs
    python -m hiredra.resume packets --job JOB_KEY       # one job

Every résumé is an editable ``cv.docx`` plus a ``cv.pdf`` fitted to one page. Output goes under ``data/packets/``
(git-ignored): a résumé is personal data. ``packets`` calls the model in ``config/resume.yml`` (NVIDIA while the
system is being built) and takes the ``llm`` run lock; it never submits anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hiredra.profile import DEFAULT_PROFILE_PATH
from hiredra.resume.build import write_cv
from hiredra.resume.payload import CvPayload
from hiredra.resume.source import DEFAULT_FACTS_PATH, baseline_payload, load_candidate, read_profile

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = _REPO_ROOT / "data" / "packets"


def _packets(args) -> int:
    from hiredra.lock import LockHeld, RunLock
    from hiredra.quota import DailyQuotaExceeded, DailyUsage
    from hiredra.resume import packet
    from hiredra.resume.llm import Caller
    from hiredra.resume.settings import load_resume_config, load_skill_display
    from hiredra.storage import connect, init_db

    cfg, display = load_resume_config(), load_skill_display()
    profile = read_profile(args.profile)
    candidate = load_candidate(profile, args.facts)
    conn = connect(args.db)
    try:
        init_db(conn)
        jobs = [args.job] if args.job else packet.select_jobs(conn, args.top)
        usage = DailyUsage(cfg.requests_per_day, conn, "nvidia", cfg.model, count_errors=False)
        print(f"{len(jobs)} job(s) for packets; {cfg.provider} {cfg.model}; today: {usage.summary()}", flush=True)
        if args.dry_run:
            for job_id in jobs:
                print(f"  would build: {job_id}")
            return 0
        gate = packet.build_gate(conn, profile)
        caller = Caller(conn, cfg, usage=usage)
        built = flagged = 0
        with RunLock("llm", what="resume packets", kind=cfg.provider, url=args.db):
            for job_id in jobs:
                try:
                    r = packet.build_packet(conn, job_id, profile=profile, candidate=candidate, gate=gate,
                                            caller=caller, display=display, cfg=cfg, out_root=Path(args.out),
                                            rebuild=args.rebuild and bool(args.job))  # fmt: skip
                except DailyQuotaExceeded as exc:
                    print(f"STOPPED (daily budget): {str(exc).splitlines()[0]}")
                    break
                built += 1
                flagged += bool(r.flags)
                print(f"  {job_id}: {r.pages or '?'} page(s), {r.calls} call(s)"
                      + (f", flags: {', '.join(r.flags)}" if r.flags else ""), flush=True)  # fmt: skip
        print(f"built {built} packet(s), {flagged} flagged; files under {args.out}")
        return 0
    except LockHeld as exc:
        print(f"NOT RUN: {exc}")
        return 4
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hiredra.resume", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="a résumé straight from the master profile (no model call)")
    r = sub.add_parser("render", help="render a payload JSON file")
    r.add_argument("payload")
    k = sub.add_parser("packets", help="tailored packets (résumé + cover letter) for scored jobs")
    pick = k.add_mutually_exclusive_group(required=True)
    pick.add_argument("--top", type=int, help="the N best scored jobs that have no packet yet")
    pick.add_argument("--job", help="one job key")
    k.add_argument("--dry-run", action="store_true", help="list the jobs; no model call")
    k.add_argument("--rebuild", action="store_true", help="with --job: remake the packet of a tailored job")
    k.add_argument("--db", default=None, help="database URL (default: HIREDRA_DATABASE_URL)")
    for p in (s, r, k):
        p.add_argument("--out", default=None)
        p.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH))
        p.add_argument("--facts", default=str(DEFAULT_FACTS_PATH))
    args = ap.parse_args(argv)

    if args.command == "packets":
        args.out = args.out or str(DEFAULT_OUT)
        return _packets(args)
    if args.command == "sample":
        profile = read_profile(args.profile)
        payload = baseline_payload(profile, load_candidate(profile, args.facts))
        out = Path(args.out or DEFAULT_OUT / "_sample")
    else:
        payload = CvPayload.model_validate_json(Path(args.payload).read_text(encoding="utf-8"))
        out = Path(args.out or DEFAULT_OUT / Path(args.payload).stem)
    result = write_cv(payload, out)
    pages = "unknown (no PDF converter)" if result.pages is None else result.pages
    print(f"wrote {result.docx}")
    print(f"  pdf: {result.pdf or 'not made'}; pages: {pages}; body {result.layout.body_pt} pt; "
          f"bullets left out to fit: {result.dropped}")  # fmt: skip
    if result.note:
        print(f"  note: {result.note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
