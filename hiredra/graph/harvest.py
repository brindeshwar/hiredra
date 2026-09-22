"""Sub-slice 4.3: the market-vocabulary harvest (docs/graph.md sec 4 step 1-3,
docs/repos.md sec 7).

Market nodes come from open-jobs' OWN skill fields (CC0), not from Gemini
extraction calls. Each enriched posting in a full per-ATS parquet export
carries ``enrichment_json.data.skills_required`` / ``skills_preferred``
(canonical-ish lowercase tokens). Only ~1-3% of rows are enriched, and the
``lite/`` diffs strip the column, so this reads the full files -- over HTTPS
with DuckDB column projection, so only ``enrichment_json`` transfers.

The funnel (each stage is reported, so the yield is inspectable):

  enriched jobs -> tech role-family jobs -> distinct folded tokens
  -> document frequency >= min_df -> light sanity filter -> new market nodes

A skill's document frequency is the number of DISTINCT jobs that mention it
(counted once per job). Survivors land in ``skill_vocab`` as ``source='market'``
with ``seen_count=0`` (never seen in OUR jobs); the frequencies go to a small
sidecar file used only to order the embedding queue and to break ties / inform
review when proposing merges. This is an offline, periodic batch step -- never
a runtime dependency.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from hiredra.graph.config import HarvestConfig
from hiredra.storage import add_market_skill
from hiredra.vocab import normalize

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DF_PATH = _REPO_ROOT / "data" / "market_df.json"

#: url -> the ``enrichment_json.data`` dict of each enriched job in that file.
Reader = Callable[[str], Iterable[dict]]

_DATE_DIR = re.compile(r"\d{4}-\d{2}-\d{2}")


def latest_export_date(base_url: str, *, client: httpx.Client | None = None) -> str:
    """Newest ``YYYY-MM-DD`` export directory the Worker lists."""
    resp = (client or httpx).get(f"{base_url}/", timeout=30.0)
    resp.raise_for_status()
    dates = sorted(d.rstrip("/") for d in resp.json().get("dirs", []) if _DATE_DIR.fullmatch(d.rstrip("/")))
    if not dates:
        raise RuntimeError(f"no dated export directories listed at {base_url}/")
    return dates[-1]


def export_url(cfg: HarvestConfig, export_date: str, ats: str) -> str:
    return f"{cfg.export_base_url}/{export_date}/jobs/{ats}.parquet"


def read_enrichment_data(url: str) -> Iterator[dict]:
    """Stream the ``data`` object of every enriched job in one parquet file.

    DuckDB projects just the ``enrichment_json`` column, so an HTTPS read moves
    that column's row groups, not the multi-hundred-MB file. Malformed JSON rows
    are skipped. (Lives here, behind one function, so tests feed synthetic rows.)
    """
    import duckdb

    con = duckdb.connect()
    try:
        if url.startswith(("http://", "https://")):
            con.execute("INSTALL httpfs; LOAD httpfs;")
        safe = url.replace("'", "''")
        cur = con.execute(
            f"SELECT enrichment_json FROM read_parquet('{safe}') WHERE enrichment_json IS NOT NULL"
        )
        while True:
            rows = cur.fetchmany(2000)
            if not rows:
                break
            for (raw,) in rows:
                try:
                    data = json.loads(raw).get("data")
                except (TypeError, ValueError, AttributeError):
                    continue
                if isinstance(data, dict):
                    yield data
    finally:
        con.close()


def sanity_reject_reason(name: str, cfg: HarvestConfig) -> str | None:
    """Why a folded token is not a usable skill name, or None if it is."""
    if not name:
        return "empty"
    if len(name) > cfg.max_chars:
        return "too_long"
    if len(name.split()) > cfg.max_words:
        return "too_many_words"
    if not any(c.isalpha() for c in name):
        return "no_letters"
    if "://" in name or name.startswith("www.") or "@" in name:  # NOT startswith("http"): "http", "http/2" are skills
        return "url_or_email"
    return None


def _job_skills(data: dict) -> set[str]:
    """The folded, de-duplicated skill tokens one enriched job lists."""
    out: set[str] = set()
    for key in ("skills_required", "skills_preferred"):
        for item in data.get(key) or []:
            if isinstance(item, str):
                folded = normalize(item)
                if folded:
                    out.add(folded)
    return out


@dataclass
class HarvestReport:
    export_date: str
    dry_run: bool
    per_ats: dict[str, dict] = field(default_factory=dict)  # ats -> counts, or {"error": ...}
    failed: list[str] = field(default_factory=list)
    enriched_jobs: int = 0
    in_family_jobs: int = 0
    distinct_tokens: int = 0
    after_min_df: int = 0
    rejected_by_sanity: Counter = field(default_factory=Counter)
    rejected_samples: dict[str, list[str]] = field(default_factory=dict)
    candidates: int = 0  # survived every filter
    inserted: int = 0  # new market nodes written
    already_known: int = 0  # candidate already a node or a merged alias
    df: dict[str, int] = field(default_factory=dict)  # candidate -> document frequency
    written: bool = False  # nodes + sidecar were actually written
    not_written_reason: str | None = None


def _read_file(reader: Reader, url: str, families: set[str]) -> tuple[int, int, Counter]:
    """One file's (enriched, in-family, per-skill document frequency). Returns
    only on a COMPLETE read: an exception mid-file discards everything from it,
    so a failed file can never leave partial counts behind."""
    enriched = in_family = 0
    df: Counter = Counter()
    for data in reader(url):
        enriched += 1
        if data.get("role_family") not in families:
            continue
        in_family += 1
        df.update(_job_skills(data))
    return enriched, in_family, df


def harvest(
    conn,
    cfg: HarvestConfig,
    *,
    reader: Reader = read_enrichment_data,
    export_date: str | None = None,
    df_path: Path | str | None = DEFAULT_DF_PATH,
    dry_run: bool = False,
    allow_partial: bool = False,
    attempts: int = 3,
    retry_wait_s: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    client: httpx.Client | None = None,
    log: Callable[[str], None] = lambda _msg: None,
) -> HarvestReport:
    """Run the harvest.

    ``dry_run=True`` computes the whole funnel and writes nothing (no nodes, no
    sidecar). Each file is retried up to ``attempts`` times (HTTPS reads do time
    out) and counts only if a read completes. If any file still fails, NOTHING is
    written unless ``allow_partial`` -- document frequencies over a subset of the
    files are not what the min_df filter was tuned for, and a half-harvest is
    harder to reason about than a re-run.
    """
    date = export_date or (
        latest_export_date(cfg.export_base_url, client=client) if cfg.export_date == "latest" else cfg.export_date
    )
    report = HarvestReport(export_date=date, dry_run=dry_run)
    families = set(cfg.role_families)
    df: Counter = Counter()

    for ats in cfg.ats:
        url = export_url(cfg, date, ats)
        for attempt in range(1, attempts + 1):
            log(f"reading {url}" + (f"  (attempt {attempt}/{attempts})" if attempt > 1 else ""))
            started = time.monotonic()
            try:
                enriched, in_family, file_df = _read_file(reader, url, families)
            except Exception as exc:  # a bad file must never silently shrink the yield
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                log(f"  {ats}: failed -- {error}")
                if attempt < attempts:
                    sleep(retry_wait_s)
                    continue
                report.per_ats[ats] = {"error": error, "attempts": attempts}
                report.failed.append(ats)
                break
            df.update(file_df)
            report.per_ats[ats] = {"enriched": enriched, "in_family": in_family}
            report.enriched_jobs += enriched
            report.in_family_jobs += in_family
            log(f"  {ats}: {enriched} enriched, {in_family} in target role families ({time.monotonic() - started:.0f}s)")
            break

    report.distinct_tokens = len(df)
    over_min = {name: n for name, n in df.items() if n >= cfg.min_df}
    report.after_min_df = len(over_min)

    kept: dict[str, int] = {}
    for name, n in over_min.items():
        reason = sanity_reject_reason(name, cfg)
        if reason is None:
            kept[name] = n
        else:
            report.rejected_by_sanity[reason] += 1
            report.rejected_samples.setdefault(reason, []).append(name)
    report.candidates = len(kept)
    report.df = dict(sorted(kept.items(), key=lambda t: (-t[1], t[0])))

    if dry_run:
        report.not_written_reason = "dry run"
    elif report.failed and not allow_partial:
        report.not_written_reason = (
            f"{len(report.failed)} file(s) failed ({', '.join(report.failed)}); re-run, or pass --allow-partial"
        )
    else:
        for name in report.df:
            if add_market_skill(conn, name):
                report.inserted += 1
            else:
                report.already_known += 1
        if df_path is not None:
            path = Path(df_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"export_date": date, "df": report.df}, indent=0, sort_keys=False),
                encoding="utf-8",
            )
        report.written = True
    return report


def load_df(path: Path | str = DEFAULT_DF_PATH) -> dict[str, int]:
    """Document frequencies from the last harvest, or {} if none was written."""
    p = Path(path)
    if not p.exists():
        return {}
    return {str(k): int(v) for k, v in json.loads(p.read_text(encoding="utf-8")).get("df", {}).items()}
