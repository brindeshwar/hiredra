"""The whole pipeline in one command, in order, stopping at the first stage that fails. It prepares; it never submits.

    python -m hiredra.e2e [--top N] [--skip STAGE ...] [--no-drafts]

Stages, each the same command you could run by hand:

    filter      python -m hiredra.filter                     location, seniority, role, liveness (no model)
    extraction  python -m hiredra.extraction                 skills per job (Gemini, then the NVIDIA fallback)
    sync        python -m hiredra.graph sync                 embed new skills (local model)
    describe    python -m hiredra.graph.describe             describe new skills, fold in other names
    scoring     python -m hiredra.scoring run                score against the profile (code)
    audit       python -m hiredra.graph audit                graph health checks
    packets     python -m hiredra.resume packets --top N     tailored resume + cover letter (NVIDIA)
    forms       python -m hiredra.autofill scan --tailored   read the application forms, prepare answers
    batch       python -m hiredra.autofill batch             queue the next batch of up to 10 for you

Each runs as its own process. A stage that exits with an error stops the run, and the summary says which one and
why; a stage refused because another process holds its lock (exit code 4) stops it too. The last step leaves a batch
for the human: open the app's Batches and Apply screens. Nothing in any stage submits or sends anything.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

STAGES: list[tuple[str, list[str]]] = [
    ("filter", ["-m", "hiredra.filter"]),
    ("extraction", ["-m", "hiredra.extraction"]),
    ("sync", ["-m", "hiredra.graph", "sync"]),
    ("describe", ["-m", "hiredra.graph.describe"]),
    ("scoring", ["-m", "hiredra.scoring", "run"]),
    ("audit", ["-m", "hiredra.graph", "audit"]),
    ("packets", ["-m", "hiredra.resume", "packets", "--top", "{top}"]),
    ("forms", ["-m", "hiredra.autofill", "scan", "--tailored"]),
    ("batch", ["-m", "hiredra.autofill", "batch"]),
]
NAMES = [name for name, _ in STAGES]


@dataclass
class StageResult:
    name: str
    exit_code: int
    seconds: float
    last_line: str


def run(stages=STAGES, *, top: int = 10, skip=(), no_drafts: bool = False, python: str = sys.executable,
        out=print) -> list[StageResult]:  # fmt: skip
    """Run the stages in order; stop at the first non-zero exit. Each stage's output is shown as it runs."""
    results: list[StageResult] = []
    for name, args in stages:
        if name in skip:
            continue
        cmd = [python, *(a.format(top=top) for a in args)]
        if name == "forms" and no_drafts:
            cmd.append("--no-drafts")
        out(f"\n=== {name}: {' '.join(cmd[1:])}")
        started = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=_REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")  # fmt: skip
        last = ""
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                last = line
                out(f"  {line}")
        code = proc.wait()
        results.append(StageResult(name, code, time.monotonic() - started, last))
        if code != 0:
            break
    return results


def summary(results: list[StageResult]) -> str:
    lines = ["", "stage        exit  seconds  last line", "-----------  ----  -------  ---------"]
    lines += [f"{r.name:<11}  {r.exit_code:>4}  {r.seconds:>7.0f}  {r.last_line[:90]}" for r in results]
    failed = next((r for r in results if r.exit_code != 0), None)
    if failed:
        why = "another run holds its lock" if failed.exit_code == 4 else "see its output above"
        lines.append(f"\nSTOPPED at {failed.name} (exit {failed.exit_code}): {why}. Later stages did not run.")
    else:
        lines.append("\nDone. Open the app's Batches and Apply screens: you review each job and submit it yourself.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hiredra.e2e", description=__doc__.split("\n\n")[0])
    ap.add_argument("--top", type=int, default=10, help="packets for the N best scored jobs (default 10)")
    ap.add_argument("--skip", action="append", default=[], choices=NAMES, help="leave a stage out (repeatable)")
    ap.add_argument(
        "--no-drafts", action="store_true", help="no model call for form answers: free text is left for you"
    )
    args = ap.parse_args(argv)
    results = run(top=args.top, skip=set(args.skip), no_drafts=args.no_drafts)
    print(summary(results), flush=True)
    return 0 if results and all(r.exit_code == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
