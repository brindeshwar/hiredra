"""The résumé step's command line.

    python -m hiredra.resume sample [--out DIR]     # a résumé straight from the master profile, no model call
    python -m hiredra.resume render PAYLOAD.json [--out DIR]   # render a payload file (the contract in payload.py)

Both write an editable ``cv.docx`` and a ``cv.pdf`` fitted to one page. Output goes under ``data/packets/``
(git-ignored): a résumé is personal data.
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hiredra.resume", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sample", help="a résumé straight from the master profile (no model call)")
    r = sub.add_parser("render", help="render a payload JSON file")
    r.add_argument("payload")
    for p in (s, r):
        p.add_argument("--out", default=None)
        p.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH))
        p.add_argument("--facts", default=str(DEFAULT_FACTS_PATH))
    args = ap.parse_args(argv)

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
