"""Step 7 (plan unit S7.1): a stratified, blind sample of scored jobs for the owner to rate.

    python -m hiredra.scoring rate --sample    # write the blind checklist sheet (30 jobs, two sittings of 15)
    python -m hiredra.scoring rate --ingest    # read the ticked sheet into data/calibration/job_ratings.csv
    python -m hiredra.scoring rate --add       # append a further sitting of 15 jobs not yet on the sheet

The plan specified an interactive rating CLI; the owner rates by ticking a checklist file instead (as for
the Step 5 skill pairs), so the same design is delivered as a sheet:
- pool: scored jobs not on the review list (QA6);
- 10 score deciles, 3 jobs each with a fixed seed, 2 tune + 1 holdout per decile -> 20/10 (D3);
- the sheet NEVER shows a score or any score feature, and lists jobs in a seeded random order, not score
  order (D3 "blind"); the split and the score at sampling live in a key file the owner never opens;
- each job's description goes in a companion file (job_ratings_descriptions.md, one ``## Jnn`` section per job,
  linked from the sheet). A long description folded inside the sheet put the editor and the Markdown preview
  out of step, since the preview maps one folded line to hundreds of source lines;
- ingest validates every ticked job (exactly one rating, a non-empty reason) and APPENDS rows; a re-rating
  is a new row and the latest wins (contracts 3.8). Untouched jobs are skipped, so a sitting can be
  ingested half done.
"""

from __future__ import annotations

import csv
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from hiredra.storage import DATA_DIR, get_job

CALIBRATION_DIR = DATA_DIR / "calibration"
SHEET_PATH = CALIBRATION_DIR / "job_ratings_sheet.md"
DESCRIPTIONS_PATH = CALIBRATION_DIR / "job_ratings_descriptions.md"
KEY_PATH = CALIBRATION_DIR / "job_ratings_key.json"
RATINGS_PATH = CALIBRATION_DIR / "job_ratings.csv"

SAMPLER_VERSION = "s7.1-sheet-1"
SEED = 20261028
STRATA = 10
PER_STRATUM = 3  # 2 tune + 1 holdout
SITTING_SIZE = 15
#: ``rate --add``: a further sitting drawn like sitting 2's redraw, 5 score strata x 3 (2 tune + 1 holdout).
ADD_STRATA = 5
ADD_SEED = 20261126
ADD_SAMPLER_VERSION = "s7.4-add"
RATINGS = ("apply", "maybe", "skip")
CSV_FIELDS = (
    "job_id",
    "split",
    "rating",
    "reason",
    "sitting",
    "sampled_at",
    "seed",
    "sampler_version",
    "score_at_sampling",
)


class SheetError(ValueError):
    """The sheet cannot be ingested as it stands; the message lists every problem."""


def sample(conn, *, seed: int = SEED) -> list[dict]:
    """30 jobs across the full score range: ``{job_id, split, sitting, score_at_sampling}`` in sheet order."""
    pool = conn.execute(
        "SELECT job_key, score FROM jobs WHERE status = 'scored' AND review_reason IS NULL ORDER BY score, job_key"
    ).fetchall()
    if len(pool) < STRATA * PER_STRATUM:
        raise SheetError(f"only {len(pool)} jobs in the rating pool; need at least {STRATA * PER_STRATUM}")
    rng = random.Random(seed)
    picks = []
    for i in range(STRATA):
        stratum = pool[i * len(pool) // STRATA : (i + 1) * len(pool) // STRATA]
        for k, (job_id, score) in enumerate(rng.sample(stratum, PER_STRATUM)):
            picks.append(
                {"job_id": job_id, "split": "holdout" if k == PER_STRATUM - 1 else "tune", "score_at_sampling": score}
            )
    rng.shuffle(picks)  # sheet order is random, never score order
    for n, p in enumerate(picks):
        p["sitting"] = 1 if n < SITTING_SIZE else 2
    return picks


def _skills(job: dict) -> str:
    names = sorted({s["name"] for s in job["skills"] or []})
    return ", ".join(names) or "(none extracted)"


def description_link(n: int, descriptions_name: str = DESCRIPTIONS_PATH.name) -> str:
    """The sheet's line pointing at job J<n>'s section in the descriptions file."""
    return f"**Description:** [read it]({descriptions_name}#j{n:02d})"


def description_section(
    n: int, title: str, company: str, description: str, sheet_name: str = SHEET_PATH.name
) -> list[str]:
    """Job J<n>'s section of the descriptions file. The bare ``## Jnn`` heading gives the link target ``#jnn``."""
    return [
        f"## J{n:02d}",
        "",
        f"**{title}** at {company}  ·  [back to the sheet]({sheet_name})",
        "",
        *[line.rstrip() for line in description.strip().splitlines()],
        "",
        "---",
        "",
    ]


DESCRIPTIONS_HEADER = [
    "# Step 7: job descriptions",
    "",
    "The full description of every job on the rating sheet, one section per job. Rate on the sheet, not here.",
    "",
]


def description_sections(conn, picks: list[dict], sheet_name: str = SHEET_PATH.name, start: int = 1) -> list[str]:
    out: list[str] = []
    for n, p in enumerate(picks, start):
        job = get_job(conn, p["job_id"])
        out += description_section(n, job["title"], job["company"], job["description"] or "", sheet_name)
    return out


def render_descriptions(conn, picks: list[dict], sheet_name: str = SHEET_PATH.name) -> str:
    """The companion file: each job's description under its own ``## Jnn`` heading. Blind like the sheet."""
    return "\n".join(DESCRIPTIONS_HEADER + description_sections(conn, picks, sheet_name))


def job_blocks(
    conn, picks: list[dict], descriptions_name: str = DESCRIPTIONS_PATH.name, *, start: int = 1, sitting: int = 0
) -> list[str]:
    """The sheet blocks for ``picks``, numbered from J<start>, with a ``## Sitting n`` heading wherever the sitting
    changes from ``sitting`` (the one before the first pick)."""
    out: list[str] = []
    for n, p in enumerate(picks, start):
        if p["sitting"] != sitting:
            sitting = p["sitting"]
            out += [f"## Sitting {sitting}", ""]
        job = get_job(conn, p["job_id"])
        out += [
            f"### J{n:02d}. {job['title']}",
            "",
            f"**Company:** {job['company']}  ·  **Location:** {job['location'] or 'not stated'}  ·  "
            f"**Posted:** {job['date_posted'] or 'unknown'}  ·  **Source:** {job['source']}",
            "",
            f"**Link:** {job['url']}",
            "",
            f"**Skills it asks for:** {_skills(job)}",
            "",
            description_link(n, descriptions_name),
            "",
            "- [ ] apply",
            "- [ ] maybe",
            "- [ ] skip",
            "",
            "Reason: ",
            "",
            f"job_id: `{p['job_id']}`",
            "",
            "---",
            "",
        ]
    return out


def render_sheet(conn, picks: list[dict], descriptions_name: str = DESCRIPTIONS_PATH.name) -> str:
    """The blind sheet: job facts and the extracted skills, a link to the description, a tick line per job."""
    out = [
        "# Step 7: rate these 30 jobs",
        "",
        "For each job tick ONE box (click it, or type x inside the brackets) and write one line saying why. Your answers teach the scorer your",
        "judgment, so rate them yourself: no AI help, and don't look anything up about how Hiredra ranked them.",
        "",
        "- **apply**: I would apply to this today.",
        "- **maybe**: borderline; I would apply if the batch were thin.",
        "- **skip**: I would not apply.",
        "",
        "Two sittings of 15 (about 75 minutes each). You can stop part way: run",
        "`python -m hiredra.scoring rate --ingest` and only the jobs you have ticked are saved.",
        "Rate the job as described, even if the posting may since have closed.",
        f"Each job's full description is in `{descriptions_name}`: follow the link under the job.",
        "",
    ]
    out += job_blocks(conn, picks, descriptions_name)
    return "\n".join(out)


def write_sample(conn, *, sheet: Path = SHEET_PATH, key: Path = KEY_PATH, seed: int = SEED, force: bool = False) -> int:
    """Draw the sample; write the sheet, its descriptions file (next to the sheet) and its key. Refuses to
    overwrite a sheet unless ``force``."""
    if sheet.exists() and not force:
        raise SheetError(f"{sheet} already exists; ingest it first, or pass --force to draw a new sample")
    picks = sample(conn, seed=seed)
    sampled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    descriptions = sheet.with_name(DESCRIPTIONS_PATH.name)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.write_text(render_sheet(conn, picks, descriptions.name), encoding="utf-8")
    descriptions.write_text(render_descriptions(conn, picks, sheet.name), encoding="utf-8")
    key.write_text(
        json.dumps(
            {
                "sampled_at": sampled_at,
                "seed": seed,
                "sampler_version": SAMPLER_VERSION,
                "jobs": {p["job_id"]: {k: p[k] for k in ("split", "sitting", "score_at_sampling")} for p in picks},
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    return len(picks)


def add_sample(conn, exclude: set[str], *, seed: int = ADD_SEED) -> list[dict]:
    """A further sitting of ``ADD_STRATA`` x ``PER_STRATUM`` jobs from today's rating pool, none already keyed.
    Stratified by today's score and split 2 tune + 1 holdout per stratum, like the first draw; random order."""
    pool = [
        r
        for r in conn.execute(
            "SELECT job_key, score FROM jobs WHERE status = 'scored' AND review_reason IS NULL ORDER BY score, job_key"
        ).fetchall()
        if r["job_key"] not in exclude
    ]
    if len(pool) < ADD_STRATA * PER_STRATUM:
        raise SheetError(f"only {len(pool)} unrated jobs in the rating pool; need {ADD_STRATA * PER_STRATUM}")
    rng = random.Random(seed)
    picks = []
    for i in range(ADD_STRATA):
        stratum = pool[i * len(pool) // ADD_STRATA : (i + 1) * len(pool) // ADD_STRATA]
        for k, row in enumerate(rng.sample(stratum, PER_STRATUM)):
            split = "holdout" if k == PER_STRATUM - 1 else "tune"
            picks.append({"job_id": row["job_key"], "split": split, "score_at_sampling": row["score"]})
    rng.shuffle(picks)
    return picks


def write_addition(conn, *, sheet: Path = SHEET_PATH, key: Path = KEY_PATH, seed: int = ADD_SEED) -> int:
    """Append a further sitting to the sheet, its descriptions file and its key. Returns the sitting number."""
    meta = json.loads(key.read_text(encoding="utf-8"))
    sitting = max(j["sitting"] for j in meta["jobs"].values()) + 1
    start = len(meta["jobs"]) + 1
    picks = add_sample(conn, set(meta["jobs"]), seed=seed)
    for p in picks:
        p["sitting"] = sitting
    descriptions = sheet.with_name(DESCRIPTIONS_PATH.name)
    blocks = job_blocks(conn, picks, descriptions.name, start=start)
    sheet.write_text(sheet.read_text(encoding="utf-8").rstrip("\n") + "\n\n" + "\n".join(blocks), encoding="utf-8")
    sections = description_sections(conn, picks, sheet.name, start=start)
    descriptions.write_text(
        descriptions.read_text(encoding="utf-8").rstrip("\n") + "\n\n" + "\n".join(sections), encoding="utf-8"
    )
    meta["jobs"].update({p["job_id"]: {k: p[k] for k in ("split", "sitting", "score_at_sampling")} for p in picks})
    meta[f"sitting{sitting}"] = {
        "sampled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "sampler_version": ADD_SAMPLER_VERSION,
    }
    key.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return sitting


_BLOCK_RE = re.compile(r"^### (J\d+)\.", re.M)
_TICK_RE = re.compile(r"^- \[([ xX])\] (apply|maybe|skip)\s*$", re.M)  # one task-list line per option


def parse_sheet(text: str) -> list[tuple[str, str, str]]:
    """``(job_id, rating, reason)`` for every ticked job. Raises ``SheetError`` listing every bad block."""
    starts = [m.start() for m in _BLOCK_RE.finditer(text)] + [len(text)]
    rows, problems = [], []
    for a, b in zip(starts, starts[1:]):
        block = text[a:b]
        label = _BLOCK_RE.match(block).group(1)
        ticked = [r for mark, r in _TICK_RE.findall(block) if mark != " "]
        reason = next((ln[len("Reason:") :].strip() for ln in block.splitlines() if ln.startswith("Reason:")), "")
        jid = re.search(r"^job_id: `([^`]+)`", block, re.M)
        if not ticked and not reason:
            continue  # not rated yet
        if jid is None:
            problems.append(f"{label}: the job_id line is missing")
        elif len(ticked) != 1:
            problems.append(f"{label}: tick exactly one of apply / maybe / skip (found {len(ticked)})")
        elif not reason:
            problems.append(f"{label}: a one-line reason is required")
        else:
            rows.append((jid.group(1), ticked[0], reason))
    if problems:
        raise SheetError("; ".join(problems))
    return rows


def _latest(path: Path) -> dict[str, tuple[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        return {r["job_id"]: (r["rating"], r["reason"]) for r in csv.DictReader(fh)}  # later rows win


def ingest(*, sheet: Path = SHEET_PATH, key: Path = KEY_PATH, ratings: Path = RATINGS_PATH) -> tuple[int, int]:
    """Append the sheet's ratings to ``job_ratings.csv``. Returns (new rows, unchanged)."""
    meta = json.loads(key.read_text(encoding="utf-8"))
    rows = parse_sheet(sheet.read_text(encoding="utf-8"))
    unknown = [jid for jid, _, _ in rows if jid not in meta["jobs"]]
    if unknown:
        raise SheetError(f"job ids not in this sample's key: {', '.join(unknown)}")
    latest = _latest(ratings)
    fresh = [(jid, r, why) for jid, r, why in rows if latest.get(jid) != (r, why)]
    new_file = not ratings.exists()
    ratings.parent.mkdir(parents=True, exist_ok=True)
    with ratings.open("a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(CSV_FIELDS)
        for jid, rating, why in fresh:
            j = meta["jobs"][jid]
            m = meta.get(f"sitting{j['sitting']}", meta)  # a later sitting carries its own draw record
            w.writerow([jid, j["split"], rating, why, j["sitting"], m["sampled_at"], m["seed"],
                        m["sampler_version"], j["score_at_sampling"]])  # fmt: skip
    return len(fresh), len(rows) - len(fresh)
