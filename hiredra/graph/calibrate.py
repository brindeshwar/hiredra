"""Step 5: threshold calibration -- set HAVE and ADJ from labeled skill pairs (plan unit S5.1,
docs/scoring.md sec 5a).

The pieces, in the order a calibration uses them:

1. ``load_pairs``: read ``calibration/skill_pairs.csv`` (format: plan contracts sec 3.8). Every
   row carries its ``split`` BEFORE any cosine is computed; the loader refuses a row without
   one, and refuses the whole file if any label is short of tune or holdout rows.
2. ``suggest_pairs``: the blind real-pair sampler. Each JD skill (from extracted jobs) is paired
   with its nearest ``mine`` node and pairs are drawn evenly across cosine bands. The labelers
   see only ``a,b`` (sorted, so band order doesn't leak); the cosines and bands go to a private
   sidecar they never open.
3. ``pair_cosines``: one cosine per pair. A name resolves fold -> alias -> node vector; only
   names that are not nodes are embedded (real pairs are always nodes, so they cost 0). Every
   cosine is stored in ``calibration/step5_cosines.csv``, which is both the resume cache and
   the regression snapshot.
4. ``calibrate``: sweep thresholds 0.30-0.99 on the TUNE split for each boundary, pick the
   centre of the max-F1 plateau, then score both thresholds once on the HOLDOUT split.

The code only computes; the model never sees a label or a threshold (CLAUDE.md rule 3).
"""

from __future__ import annotations

import csv
import io
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from hiredra.graph.config import GraphConfig
from hiredra.graph.embedder import Embedder
from hiredra.graph.index import load_skills
from hiredra.storage import resolve_alias
from hiredra.vocab import normalize

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAIRS_PATH = _REPO_ROOT / "calibration" / "skill_pairs.csv"
DEFAULT_COSINES_PATH = _REPO_ROOT / "calibration" / "step5_cosines.csv"
DEFAULT_REPORTS_DIR = _REPO_ROOT / "calibration" / "reports"
DEFAULT_SIDECAR_PATH = _REPO_ROOT / "data" / "calibration" / "suggest_bands.json"

LABELS = ("same", "related", "unrelated")
SPLITS = ("tune", "holdout")
SOURCES = ("trap", "real")
HEADER = ["a", "b", "label", "split", "source", "note"]

#: The loader refuses a file with fewer rows than this for any label (F5).
MIN_TUNE, MIN_HOLDOUT = 10, 5
#: Thresholds swept, as hundredths: 0.30, 0.31, ... 0.99. Integers, so no float drift.
SWEEP = range(30, 100)
#: Every threshold within this much F1 of the best one is on the plateau.
PLATEAU_F1 = 0.01
#: Gates (docs/scoring.md sec 5a).
MIN_GAP = 0.10
MIN_HOLDOUT_F1 = 0.80

#: float32 dot products of unit vectors can land a hair under an exact threshold.
_EPS = 1e-6


class PairFileError(ValueError):
    """The pair file is malformed or too small. The message lists every problem found."""


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    label: str
    split: str
    source: str
    note: str = ""
    line: int = 0

    @property
    def key(self) -> tuple[str, str]:
        return (self.a, self.b) if self.a <= self.b else (self.b, self.a)


# ─────────────────────────────────────────────────────────────────────────────
# 1. the pair file
# ─────────────────────────────────────────────────────────────────────────────
def count_shortfalls(pairs: Sequence[Pair]) -> list[str]:
    """Each label short of MIN_TUNE tune rows or MIN_HOLDOUT holdout rows, as a sentence."""
    counts = Counter((p.label, p.split) for p in pairs)
    short: list[str] = []
    for label in LABELS:
        for split, need in (("tune", MIN_TUNE), ("holdout", MIN_HOLDOUT)):
            have = counts[(label, split)]
            if have < need:
                short.append(f"{label}/{split}: {have} row(s), need at least {need} ({need - have} short)")
    return short


def load_pairs(path: Path | str = DEFAULT_PAIRS_PATH, *, enforce_counts: bool = True) -> list[Pair]:
    """Read and validate the pair file. Raises ``PairFileError`` listing every problem with its
    line number; nothing partial is ever returned."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    pairs: list[Pair] = []
    seen: dict[tuple[str, str], int] = {}
    header_seen = False
    for lineno, raw in enumerate(lines, start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        row = next(csv.reader(io.StringIO(raw)))
        if not header_seen:
            header_seen = True
            if [c.strip() for c in row] != HEADER:
                errors.append(f"line {lineno}: header must be {','.join(HEADER)}")
                break
            continue
        if len(row) < 5 or len(row) > 6:
            errors.append(f"line {lineno}: expected 5 or 6 columns ({','.join(HEADER)}), got {len(row)}")
            continue
        a, b, label, split, source = (normalize(c) for c in row[:5])
        note = row[5].strip() if len(row) == 6 else ""
        problems = []
        if not a or not b:
            problems.append("both names are required")
        elif a == b:
            problems.append(f"a and b are the same skill after folding ({a!r})")
        if label not in LABELS:
            problems.append(f"label {label!r} is not one of {', '.join(LABELS)}")
        if not split:
            problems.append("no split (assign tune/holdout before any cosine is seen)")
        elif split not in SPLITS:
            problems.append(f"split {split!r} is not one of {', '.join(SPLITS)}")
        if source not in SOURCES:
            problems.append(f"source {source!r} is not one of {', '.join(SOURCES)}")
        if problems:
            errors.extend(f"line {lineno}: {p}" for p in problems)
            continue
        pair = Pair(a=a, b=b, label=label, split=split, source=source, note=note, line=lineno)
        if pair.key in seen:
            errors.append(f"line {lineno}: duplicate of line {seen[pair.key]} ({a} / {b})")
            continue
        seen[pair.key] = lineno
        pairs.append(pair)
    if not header_seen:
        errors.append("no header line found")
    if errors:
        raise PairFileError(f"{path}: {len(errors)} problem(s)\n  " + "\n  ".join(errors))
    if enforce_counts:
        short = count_shortfalls(pairs)
        if short:
            raise PairFileError(f"{path}: not enough labeled pairs\n  " + "\n  ".join(short))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 2. the blind real-pair sampler
# ─────────────────────────────────────────────────────────────────────────────
def _jd_skill_names(conn) -> set[str]:
    """Every skill name on an extracted (or later) job, resolved through the alias table."""
    names: set[str] = set()
    rows = conn.execute("SELECT skills FROM jobs WHERE skills IS NOT NULL AND status NOT IN ('fetched', 'filtered')")
    for (blob,) in rows:
        for item in json.loads(blob or "[]"):
            name = normalize(str(item.get("name", "")))
            if name:
                names.add(resolve_alias(conn, name) or name)
    return names


def suggest_pairs(
    conn,
    cfg: GraphConfig,
    *,
    n: int = 60,
    seed: int = 0,
    bands: int = 6,
    low: float = 0.40,
) -> tuple[list[tuple[str, str]], dict]:
    """Draw up to ``n`` real pairs (JD skill, its nearest ``mine`` node), evenly across
    ``bands`` equal cosine bands from ``low`` to 1.0. Returns the pairs sorted by name (for the
    labelers) and the private sidecar (cosines and bands, for the report only)."""
    skills = [s for s in load_skills(conn, cfg) if s.vector is not None]
    mine = [s for s in skills if s.mine]
    if not mine:
        raise ValueError("no embedded 'mine' skills yet -- apply the master-profile overlay (PF1) first")
    by_name = {s.name: s for s in skills}
    jd = sorted(name for name in _jd_skill_names(conn) if name in by_name and not by_name[name].mine)
    if not jd:
        raise ValueError("no embedded JD skills from extracted jobs yet -- run extraction and graph sync first")

    mine_matrix = np.array([s.vector for s in mine], dtype=np.float32)
    width = (1.0 - low) / bands
    by_band: dict[int, list[dict]] = {}
    for name in jd:
        sims = mine_matrix @ by_name[name].vector
        best = int(np.argmax(sims))
        cosine = float(sims[best])
        if cosine < low - _EPS:
            continue
        band = min(int((cosine - low) / width), bands - 1)
        by_band.setdefault(band, []).append({"a": name, "b": mine[best].name, "cosine": round(cosine, 6), "band": band})

    rng = random.Random(seed)
    per_band = max(1, n // bands)
    picked: list[dict] = []
    for band in range(bands):
        pool = by_band.get(band, [])
        picked.extend(rng.sample(pool, min(per_band, len(pool))))
    picked.sort(key=lambda p: (p["a"], p["b"]))
    sidecar = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "n": n,
        "bands": [[round(low + i * width, 4), round(low + (i + 1) * width, 4)] for i in range(bands)],
        "embed_tag": cfg.embed_tag,
        "pairs": picked,
    }
    return [(p["a"], p["b"]) for p in picked], sidecar


# ─────────────────────────────────────────────────────────────────────────────
# 3. cosines
# ─────────────────────────────────────────────────────────────────────────────
def _read_cached(path: Path, embed_tag: str) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    cached: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("embed_tag") == embed_tag:
                cached[(row["a"], row["b"])] = float(row["cosine"])
    return cached


def write_cosines(path: Path, pairs: Sequence[Pair], cosines: dict[tuple[str, str], float], embed_tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["a", "b", "cosine", "embed_tag"])
        for p in sorted(pairs, key=lambda p: p.key):
            if p.key in cosines:
                w.writerow([p.key[0], p.key[1], f"{cosines[p.key]:.6f}", embed_tag])


def pair_cosines(
    conn,
    cfg: GraphConfig,
    pairs: Sequence[Pair],
    embed_factory: Callable[[], Embedder] | None = None,
    *,
    cache_path: Path | str = DEFAULT_COSINES_PATH,
) -> dict[tuple[str, str], float]:
    """Cosine of every pair, keyed by ``Pair.key``. Cached rows (same ``embed_tag``) are
    reused; node vectors are used as stored; only the leftover names are embedded, in one call.
    The node-based cosines are written to the cache BEFORE that call, so a budget stop loses
    nothing. ``embed_factory`` builds the embedder only when a name actually needs one."""
    cache_path = Path(cache_path)
    cosines = {k: v for k, v in _read_cached(cache_path, cfg.embed_tag).items() if k in {p.key for p in pairs}}
    nodes = {s.name: s for s in load_skills(conn, cfg) if s.vector is not None}

    def vector(name: str) -> np.ndarray | None:
        node = nodes.get(resolve_alias(conn, name) or name)
        return node.vector if node is not None else None

    todo = [p for p in pairs if p.key not in cosines]
    missing = sorted({n for p in todo for n in p.key if vector(n) is None})
    for p in todo:
        va, vb = vector(p.a), vector(p.b)
        if va is not None and vb is not None:
            cosines[p.key] = round(float(va @ vb), 6)  # 6 places: a cached re-run gives the same value
    write_cosines(cache_path, pairs, cosines, cfg.embed_tag)

    if missing:
        if embed_factory is None:
            raise ValueError(f"{len(missing)} name(s) are not graph nodes and need embedding: {', '.join(missing)}")
        matrix = embed_factory()(missing)
        fresh = {name: matrix[i] for i, name in enumerate(missing)}
        for p in todo:
            if p.key in cosines:
                continue
            va = vector(p.a) if p.a not in fresh else fresh[p.a]
            vb = vector(p.b) if p.b not in fresh else fresh[p.b]
            cosines[p.key] = round(float(np.asarray(va, dtype=np.float32) @ np.asarray(vb, dtype=np.float32)), 6)
        write_cosines(cache_path, pairs, cosines, cfg.embed_tag)
    return cosines


# ─────────────────────────────────────────────────────────────────────────────
# 4. sweep, plateau, holdout, gates
# ─────────────────────────────────────────────────────────────────────────────
#: Which labels count as POSITIVE for each boundary.
BOUNDARIES = {"have": {"same"}, "adj": {"same", "related"}}


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def confusion(points: Iterable[tuple[float, bool]], threshold: float) -> Confusion:
    """Predict positive when cosine >= threshold; count against the truth."""
    tp = fp = fn = tn = 0
    for cosine, positive in points:
        predicted = cosine >= threshold - _EPS
        if predicted and positive:
            tp += 1
        elif predicted:
            fp += 1
        elif positive:
            fn += 1
        else:
            tn += 1
    return Confusion(tp, fp, fn, tn)


def sweep(points: Sequence[tuple[float, bool]]) -> list[tuple[float, Confusion]]:
    """Every threshold 0.30-0.99 (step 0.01) with its confusion matrix."""
    return [(i / 100, confusion(points, i / 100)) for i in SWEEP]


def plateau_centre(rows: Sequence[tuple[float, Confusion]]) -> tuple[float, float, float]:
    """``(threshold, plateau_low, plateau_high)``. The plateau is the contiguous run of
    thresholds within PLATEAU_F1 of the best F1, around the lowest best threshold; the pick is
    its centre, rounded half-up to 0.01."""
    f1s = [c.f1 for _, c in rows]
    best = max(f1s)
    i = f1s.index(best)
    lo = hi = i
    while lo > 0 and f1s[lo - 1] >= best - PLATEAU_F1:
        lo -= 1
    while hi < len(f1s) - 1 and f1s[hi + 1] >= best - PLATEAU_F1:
        hi += 1
    centre = (lo + hi + 1) // 2  # half-up on the index, so the result is exact hundredths
    return rows[centre][0], rows[lo][0], rows[hi][0]


def _points(pairs: Sequence[Pair], cosines: dict, boundary: str, split: str, source: str | None = None):
    positive = BOUNDARIES[boundary]
    return [
        (cosines[p.key], p.label in positive)
        for p in pairs
        if p.split == split and (source is None or p.source == source)
    ]


@dataclass
class CalibrationResult:
    have: float
    adj: float
    plateaus: dict[str, tuple[float, float, float]]
    tune_f1: dict[str, float]
    holdout: dict[str, dict[str, Confusion]]  # boundary -> {"all" | "trap" | "real" -> Confusion}
    flags: list[str] = field(default_factory=list)
    broken: bool = False


def calibrate(pairs: Sequence[Pair], cosines: dict[tuple[str, str], float]) -> CalibrationResult:
    """Sweep each boundary on TUNE, pick the plateau centre, score once on HOLDOUT, apply gates."""
    plateaus, tune_f1, holdout = {}, {}, {}
    for boundary in BOUNDARIES:
        rows = sweep(_points(pairs, cosines, boundary, "tune"))
        plateaus[boundary] = plateau_centre(rows)
        t = plateaus[boundary][0]
        tune_f1[boundary] = confusion(_points(pairs, cosines, boundary, "tune"), t).f1
        holdout[boundary] = {
            group: confusion(_points(pairs, cosines, boundary, "holdout", None if group == "all" else group), t)
            for group in ("all", *SOURCES)
        }
    result = CalibrationResult(
        have=plateaus["have"][0], adj=plateaus["adj"][0], plateaus=plateaus, tune_f1=tune_f1, holdout=holdout
    )
    if result.have <= result.adj:
        result.broken = True
        result.flags.append(f"STOP: HAVE ({result.have:.2f}) <= ADJ ({result.adj:.2f}) -- a broken result, not written")
    elif result.have - result.adj < MIN_GAP - _EPS:
        result.flags.append(
            f"CHECK-IN: HAVE - ADJ = {result.have - result.adj:.2f} < {MIN_GAP:.2f} -- owner review before S6.3"
        )
    for boundary in BOUNDARIES:
        f1 = holdout[boundary]["all"].f1
        if f1 < MIN_HOLDOUT_F1:
            result.flags.append(f"S5.3: holdout F1 for {boundary.upper()} is {f1:.2f} < {MIN_HOLDOUT_F1:.2f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# write-back + report
# ─────────────────────────────────────────────────────────────────────────────
def write_fit_config(path: Path | str, have: float, adj: float, calibration_id: str) -> None:
    """Set ``fit.have`` / ``fit.adj`` / ``fit.calibration_id`` in config/graph.yml in place,
    keeping every comment. Only lines inside the ``fit:`` section are touched."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out, in_fit, done = [], False, set()
    values = {"have": f"{have:.2f}", "adj": f"{adj:.2f}", "calibration_id": calibration_id}
    for line in lines:
        if re.match(r"^\S", line):
            if in_fit and "calibration_id" not in done:
                out.append(f"  calibration_id: {calibration_id}\n")
                done.add("calibration_id")
            in_fit = line.startswith("fit:")
        m = re.match(r"^(\s+)(have|adj|calibration_id):\s*([^#\n]*?)(\s*#.*)?$", line.rstrip("\n")) if in_fit else None
        if m:
            indent, key, _, comment = m.groups()
            out.append(f"{indent}{key}: {values[key]}{comment or ''}\n")
            done.add(key)
            continue
        out.append(line)
    if in_fit and "calibration_id" not in done:
        out.append(f"  calibration_id: {calibration_id}\n")
        done.add("calibration_id")
    if done != set(values):
        raise ValueError(f"{path}: could not find fit key(s) {sorted(set(values) - done)}")
    path.write_text("".join(out), encoding="utf-8")


def _cm(c: Confusion) -> str:
    return f"tp {c.tp} / fp {c.fp} / fn {c.fn} / tn {c.tn}, F1 {c.f1:.3f}"


def render_report(
    result: CalibrationResult, pairs: Sequence[Pair], cosines: dict, cfg: GraphConfig, calibration_id: str
) -> str:
    """The public-safe report: counts, thresholds, F1 and confusion matrices, and the
    misclassified holdout pairs (skill names only, no personal data)."""
    counts = Counter((p.label, p.split) for p in pairs)
    lines = [
        f"# Step 5 threshold calibration -- {calibration_id}",
        "",
        f"Embedding: `{cfg.embed_tag}`. Pairs: {len(pairs)} "
        f"({sum(p.split == 'tune' for p in pairs)} tune, {sum(p.split == 'holdout' for p in pairs)} holdout; "
        f"{sum(p.source == 'trap' for p in pairs)} trap, {sum(p.source == 'real' for p in pairs)} real).",
        "",
        "| label | tune | holdout |",
        "|---|---|---|",
        *(f"| {label} | {counts[(label, 'tune')]} | {counts[(label, 'holdout')]} |" for label in LABELS),
        "",
        "## Result",
        "",
        f"- **HAVE = {result.have:.2f}** (plateau {result.plateaus['have'][1]:.2f}-{result.plateaus['have'][2]:.2f},"
        f" tune F1 {result.tune_f1['have']:.3f}). Positive = same.",
        f"- **ADJ = {result.adj:.2f}** (plateau {result.plateaus['adj'][1]:.2f}-{result.plateaus['adj'][2]:.2f},"
        f" tune F1 {result.tune_f1['adj']:.3f}). Positive = same + related.",
        "",
        "## Holdout (scored once)",
        "",
    ]
    for boundary in BOUNDARIES:
        for group, c in result.holdout[boundary].items():
            lines.append(f"- {boundary.upper()} / {group}: {_cm(c)}")
    lines += ["", "## Gates", ""]
    lines += [f"- {f}" for f in result.flags] or ["- all passed"]
    lines += ["", "## Misclassified holdout pairs", ""]
    wrong = []
    for p in sorted((p for p in pairs if p.split == "holdout"), key=lambda p: p.key):
        cosine = cosines[p.key]
        predicted = (
            "same" if cosine >= result.have - _EPS else "related" if cosine >= result.adj - _EPS else "unrelated"
        )
        if predicted != p.label:
            wrong.append(f"- {p.a} / {p.b}: labeled {p.label}, cosine {cosine:.3f} -> {predicted} ({p.source})")
    lines += wrong or ["- none"]
    return "\n".join(lines) + "\n"


def calibration_id_for(day: date) -> str:
    return f"step5-{day.isoformat()}"
