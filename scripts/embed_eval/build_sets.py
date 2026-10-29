"""Build the evaluation sets for the local-embedding comparison (docs/model-selection-embedding.md).

Gold: the owner-accepted 174 pairs (calibration/skill_pairs.csv), used exactly as Step 5 uses them.
Silver: a larger set built from labels that already exist, at no labelling cost:
  same      -- freehire's alias dictionary (alias -> canonical label) + our config/skill_aliases.yml;
  related   -- freehire's curated skill-adjacency map;
  unrelated -- random pairs of distinct canonical skills, minus anything aliased, adjacent or sharing a word.
Our own graph merges are left out on purpose: they were proposed by Gemini cosines and would favour Gemini.

Writes data/calibration/embed-eval/{sets.json, strings.json}. Run with the project venv:
    uv run python scripts/embed_eval/build_sets.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import yaml

from hiredra.graph.calibrate import load_pairs

ROOT = Path(__file__).resolve().parents[2]
FH = ROOT / "reference" / "freehire" / "internal" / "dict"
OUT = ROOT / "data" / "calibration" / "embed-eval"
SEED = 20261025
N_UNRELATED = 2000


def fold(s: str) -> str:
    return " ".join(s.lower().split())


def squash(s: str) -> str:
    return re.sub(r"[^a-z0-9+#]", "", s.lower())


def go_string_maps(path: Path) -> dict[str, str]:
    """Every `"key": "value",` entry in a Go source file (the skilltag alias tables)."""
    out = {}
    for k, v in re.findall(r'^\s*"([^"\n]+)"\s*:\s*"([^"\n]+)"\s*,', path.read_text(encoding="utf-8"), re.M):
        out.setdefault(k, v)
    return out


def labels() -> dict[str, str]:
    """canonical slug -> display label (freehire labels.go)."""
    return go_string_maps(FH / "skilltag" / "labels.go")


def descriptions() -> dict[str, str]:
    out = {}
    for line in (FH / "skilltag" / "descriptions.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "\t" in line:
            slug, desc = line.split("\t", 1)
            out[slug.strip()] = desc.strip()
    return out


def adjacency() -> list[tuple[str, str]]:
    text = (FH / "skilladjacency" / "skilladjacency.go").read_text(encoding="utf-8")
    pairs = set()
    for key, vals in re.findall(r'^\s*"([^"]+)"\s*:\s*\{([^}]*)\}', text, re.M):
        for v in re.findall(r'"([^"]+)"', vals):
            pairs.add(tuple(sorted((key, v))))
    return sorted(pairs)


def main() -> None:
    lab = labels()
    desc = descriptions()
    alias_map = {}  # alias text -> canonical slug
    dicts = (FH / "skilltag" / "dictionaries.go").read_text(encoding="utf-8")
    entries = list(go_string_maps(FH / "skilltag" / "dictionaries.go").items())  # wordAliases, acronym maps
    entries += re.findall(r'\{"([^"\n]+)",\s*"([^"\n]+)"\}', dicts)  # phraseAliases: {"phrase", "slug"}
    entries += re.findall(r'"([^"\n]+)":\s*\{canonical:\s*"([^"\n]+)"', dicts)  # category-scoped acronyms
    for k, v in entries:
        if v in lab or v in desc:
            alias_map.setdefault(fold(k), v)
    surfaces = (FH / "skilltag" / "interchangeable.go").read_text(encoding="utf-8")
    for slug, vals in re.findall(r'^\s*"([^"]+)"\s*:\s*\{([^}]*)\}', surfaces, re.M):
        for v in re.findall(r'"([^"]+)"', vals):
            alias_map.setdefault(fold(v), slug)
    canon = sorted(set(lab) | set(desc))
    name_of = {s: fold(lab.get(s, s.replace("-", " "))) for s in canon}

    # ---- silver: same ----------------------------------------------------------------------
    same = set()
    for alias, slug in alias_map.items():
        a, b = alias, name_of.get(slug, slug)
        if squash(a) != squash(b):  # skip pure spelling variants (our merge step handles those)
            same.add(tuple(sorted((a, b))))
    ours = yaml.safe_load((ROOT / "config" / "skill_aliases.yml").read_text(encoding="utf-8"))["aliases"]
    for canonical, alist in ours.items():
        for a in alist:
            if squash(a) != squash(canonical):
                same.add(tuple(sorted((fold(a), fold(canonical)))))

    # ---- silver: related -------------------------------------------------------------------
    related = {tuple(sorted((name_of.get(a, a), name_of.get(b, b)))) for a, b in adjacency()}
    # freehire's curated skill bundles (skills used together, e.g. docker/kubernetes/ci-cd): pairs inside a bundle
    btext = (FH / "skillbundle" / "skillbundle.go").read_text(encoding="utf-8")
    for members in re.findall(r"\[\]string\{([^}]*)\}\}", btext):
        ms = [name_of.get(m, m) for m in re.findall(r'"([^"]+)"', members)]
        related |= {tuple(sorted((a, b))) for i, a in enumerate(ms) for b in ms[i + 1 :]}
    related -= same

    # ---- silver: unrelated -----------------------------------------------------------------
    blocked = same | related
    rng = random.Random(SEED)
    names = sorted(set(name_of.values()))
    unrelated = set()
    tries = 0
    while len(unrelated) < N_UNRELATED and tries < 200000:
        tries += 1
        a, b = rng.sample(names, 2)
        key = tuple(sorted((a, b)))
        if key in blocked or key in unrelated or set(a.split()) & set(b.split()):
            continue
        unrelated.add(key)

    # ---- gold -------------------------------------------------------------------------------
    gold = [
        {"a": fold(p.a), "b": fold(p.b), "label": p.label, "split": p.split, "source": p.source}
        for p in load_pairs(ROOT / "calibration" / "skill_pairs.csv")
    ]

    silver = (
        [{"a": a, "b": b, "label": "same"} for a, b in sorted(same)]
        + [{"a": a, "b": b, "label": "related"} for a, b in sorted(related)]
        + [{"a": a, "b": b, "label": "unrelated"} for a, b in sorted(unrelated)]
    )

    # ---- strings + their "name: description" variant ------------------------------------------
    slug_of = {v: k for k, v in name_of.items()}
    for alias, slug in alias_map.items():
        slug_of.setdefault(alias, slug)
    strings = sorted({p["a"] for p in gold + silver} | {p["b"] for p in gold + silver})
    described = {}
    hit = 0
    for s in strings:
        d = desc.get(slug_of.get(s, ""), "") or desc.get(s.replace(" ", "-"), "")
        if d:
            hit += 1
        described[s] = f"{s}: {d}" if d else s

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sets.json").write_text(json.dumps({"gold": gold, "silver": silver}, indent=1), encoding="utf-8")
    (OUT / "strings.json").write_text(json.dumps({"bare": strings, "described": described}, indent=1), encoding="utf-8")
    print(
        f"gold {len(gold)} | silver same {len(same)} related {len(related)} unrelated {len(unrelated)} "
        f"| strings {len(strings)}, with a description {hit}"
    )


if __name__ == "__main__":
    main()
