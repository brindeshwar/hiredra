"""Generate a wide Path-1 company list for a max-scale sourcing run.

Samples from the open-jobs slug roster already on disk
(reference/open-jobs/slugs.json), restricted to the three ATS Hiredra has
providers for (greenhouse, lever, ashby). Pure config generation -- no new
provider code, no ingestion pipeline. The curated list is preserved verbatim
by reading it from disk (never hardcoded here), so this never goes stale.

    python scripts/generate_company_list.py
    python scripts/generate_company_list.py --total 5000 --seed 7
    python scripts/generate_company_list.py --keep config/companies.yml \\
        --out config/companies.wide.yml --slugs reference/open-jobs/slugs.json

The output file (default config/companies.wide.yml) is git-ignored -- it's
regenerable data, not hand-authored config. Only this generator is committed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

_ATS = ("greenhouse", "lever", "ashby")

_HEADER = """# {out} -- generated Path-1 company list for a max-scale sourcing run.
# DO NOT HAND-EDIT: regenerate with `python scripts/generate_company_list.py`.
# Sampled (seed {seed}) from {slugs}'s live-board roster, restricted to
# greenhouse/lever/ashby, on top of the curated list from {keep}. No industry
# or location metadata -- source wide, the 2b funnel filters it down.
# See config/companies.yml for the curated small list and docs/sourcing.md
# sec 9.5 for how to point a run config at whichever list you want.

companies:
"""


def _existing_keys(keep_path: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    """The curated list's raw entries (kept verbatim) + their (ats, slug) keys
    (so the sample never duplicates one of them)."""
    doc = yaml.safe_load(keep_path.read_text(encoding="utf-8")) or {}
    entries = list(doc.get("companies") or [])
    keys = set()
    for e in entries:
        ats = e.get("provider")
        slug = e.get("slug") or e.get("extra", {}).get("slug") if isinstance(e.get("extra"), dict) else e.get("slug")
        if ats in _ATS and slug:
            keys.add((ats, str(slug)))
    return entries, keys


def _q(value: str) -> str:
    """A YAML-safe double-quoted scalar (JSON string syntax is valid YAML) --
    some slugs are purely numeric ("550") or YAML-boolean-like ("off", "no",
    ...) and would otherwise parse as int/bool, not str."""
    return json.dumps(str(value))


def _entry_lines(entry: dict) -> list[str]:
    lines = [f"  - name: {_q(entry['name'])}"]
    for key in ("careers_url", "slug", "api", "provider"):
        if entry.get(key):
            lines.append(f"    {key}: {_q(entry[key])}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slugs", default="reference/open-jobs/slugs.json")
    parser.add_argument("--keep", default="config/companies.yml", help="curated list to preserve verbatim")
    parser.add_argument("--out", default="config/companies.wide.yml")
    parser.add_argument("--total", type=int, default=1000, help="total companies in the output, curated + sampled")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    keep_path = Path(args.keep)
    kept_entries, kept_keys = _existing_keys(keep_path)

    data = json.loads(Path(args.slugs).read_text(encoding="utf-8"))
    pool = [(ats, slug) for ats in _ATS for slug in data["ats"][ats] if (ats, slug) not in kept_keys]

    n = max(0, args.total - len(kept_entries))
    if n > len(pool):
        raise SystemExit(f"--total {args.total} needs {n} sampled companies but the pool only has {len(pool)}")
    rng = random.Random(args.seed)
    sample = rng.sample(pool, n)

    lines = [_HEADER.format(out=args.out, seed=args.seed, slugs=args.slugs, keep=args.keep).rstrip("\n")]
    for entry in kept_entries:
        lines.extend(_entry_lines(entry))
    by_ats = {ats: 0 for ats in _ATS}
    for ats, slug in sample:
        by_ats[ats] += 1
        lines.append(f"  - name: {_q(slug)}")
        lines.append(f"    slug: {_q(slug)}")
        lines.append(f"    provider: {ats}")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    total = len(kept_entries) + len(sample)
    print(f"wrote {out_path}: {total} companies ({len(kept_entries)} kept from {keep_path} + {len(sample)} sampled)")
    print(f"  by ATS (sampled only): {by_ats}")
    print(f"  pool sizes: " + ", ".join(f"{a}={len(data['ats'][a])}" for a in _ATS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
