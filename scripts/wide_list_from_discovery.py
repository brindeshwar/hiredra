"""Generate config/companies.wide.yml from the discovery harvest's output.

Source: data/discovery/exports/companies.csv (ats, slug, company, tag), the
real deliverable of discovery/harvest.py -- a separate, one-time program (see
discovery/README.md). This script does no fetching and no discovery of its
own; it is pure config generation from an already-produced CSV. It replaced an
earlier generator that sampled open-jobs' raw slug roster (removed 2026-10-29).

Includes BOTH tags (confirmed and unknown-only): the runtime location stage
does the final per-job India check, so an unknown-only company can't leak a
non-eligible job through -- excluding it here would only risk losing a real
lead the location stage would have caught correctly. `tag` is carried into
each entry's `extra` so a wide run can report confirmed-vs-unknown yield
separately later.

    python scripts/wide_list_from_discovery.py
    python scripts/wide_list_from_discovery.py --csv path/to/companies.csv --out config/companies.wide.yml

The file is an IMPORT list (decision D2.1): runs crawl the boards table, so load the result with
`python -m hiredra.sourcing boards import config/companies.wide.yml` (provenance openjobs).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

_ATS_ORDER = ("greenhouse", "lever", "ashby")

_HEADER = """# {out} -- generated from the discovery harvest's companies.csv.
# DO NOT HAND-EDIT: regenerate with `python scripts/wide_list_from_discovery.py`.
# Source: {csv} ({total} companies: {confirmed} confirmed India/remote-eligible,
# {unknown} unknown-only -- kept per the owner's call, since the funnel's
# location stage does the final per-job check anyway). See discovery/README.md
# for how this was produced and config/companies.yml for the curated small list.

companies:
"""


def _q(value: str) -> str:
    """A YAML-safe double-quoted scalar (JSON string syntax is valid YAML) --
    some slugs/names are purely numeric or YAML-boolean-like and would
    otherwise parse as int/bool, not str."""
    return json.dumps(str(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/discovery/exports/companies.csv")
    parser.add_argument("--out", default="config/companies.wide.yml")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))

    by_ats: dict[str, list[dict]] = {ats: [] for ats in _ATS_ORDER}
    for row in rows:
        ats = row["ats"]
        by_ats.setdefault(ats, []).append(row)

    confirmed = sum(1 for r in rows if r["tag"] == "confirmed")
    unknown = sum(1 for r in rows if r["tag"] == "unknown")
    lines = [
        _HEADER.format(out=args.out, csv=csv_path, total=len(rows), confirmed=confirmed, unknown=unknown).rstrip("\n")
    ]

    ats_order = list(_ATS_ORDER) + [a for a in by_ats if a not in _ATS_ORDER]
    for ats in ats_order:
        entries = by_ats.get(ats) or []
        if not entries:
            continue
        lines.append(f"  # --- {ats} ({len(entries)}) ---")
        for row in sorted(entries, key=lambda r: r["slug"]):
            lines.append(f"  - name: {_q(row['company'])}")
            lines.append(f"    slug: {_q(row['slug'])}")
            lines.append(f"    provider: {ats}")
            lines.append(f"    extra: {{tag: {row['tag']}}}")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print(f"wrote {out_path}: {len(rows)} companies ({confirmed} confirmed, {unknown} unknown-only)")
    print("  by ats: " + ", ".join(f"{a}={len(by_ats.get(a) or [])}" for a in ats_order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
