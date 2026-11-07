"""YAML config loading for the sourcing orchestrator (docs/sourcing.md §7.5, §9.5).

* ``config/companies.yml`` -> ``list[CompanyEntry]``: an IMPORT list. ``python -m hiredra.sourcing boards import``
  writes it into the ``boards`` table, which is what a run crawls (decision D2.1, build slice 2.1).
* ``config/scoring.yml`` / ``config/scoring.wide.yml`` -> ``RunConfig``; its ``boards`` key picks which boards a
  run crawls, by provenance (``[curated]``, or ``all``).
* a ``boards`` row -> ``CompanyEntry`` (``board_entry``), so the orchestrator stays free of any database code.

**Strict**: an unknown key -- at the top level OR inside any nested section
(``jobspy``, ``jobspy.default_query``, ``pacing``, ``http`` ...) -- is a
``ConfigError`` naming its full dotted path, never a silent fall-through to a
default.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from hiredra.sourcing.belt import BeltConfig, Platform
from hiredra.sourcing.http import HttpConfig
from hiredra.sourcing.jobspy_source import JobSpyConfig, SiteQuery
from hiredra.sourcing.provider import CompanyEntry

_T = TypeVar("_T")


class ConfigError(ValueError):
    """A malformed config file."""


@dataclass(frozen=True, slots=True)
class Pacing:
    between_companies: float = 1.0  # base seconds between company fetches
    jitter: float = 0.5  # + up to this many random seconds (between companies, one worker only)
    workers: int = 1  # boards fetched at once (D4.1); past 1, the per-host limit (http.per_host) paces


@dataclass(frozen=True, slots=True)
class RunConfig:
    kind: str  # "scoring" (the backfill kind was removed, R1a)
    config_name: str  # basename, recorded on the runs row
    boards: tuple[str, ...]  # provenances to crawl; ("all",) = every provenance; () = no boards (JobSpy only)
    jobspy: JobSpyConfig
    http: HttpConfig
    pacing: Pacing
    path2_enabled: bool = False  # the Path-2 discovery slot
    belt: BeltConfig = BeltConfig()  # the conveyor belt's rules (no platform held to a rate when absent)


# ── strict dataclass construction ──────────────────────────────────────────
def _build(cls: type[_T], data: Any, path: str) -> _T:
    """Construct a (frozen) dataclass from a dict, rejecting unknown keys."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(data).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {sorted(unknown)} (allowed: {sorted(allowed)})")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        raw = data[f.name]
        sub = f"{path}.{f.name}" if path else f.name
        ftype = str(f.type)
        if raw is None:
            kwargs[f.name] = None  # e.g. `linkedin_query: null`
        elif "SiteQuery" in ftype:
            kwargs[f.name] = _build(SiteQuery, raw, sub)
        elif ftype.startswith("tuple"):
            if not isinstance(raw, (list, tuple)):
                raise ConfigError(f"{sub}: expected a list")
            kwargs[f.name] = tuple(raw)
        else:
            kwargs[f.name] = raw
    _require(cls, kwargs, path)
    return cls(**kwargs)  # type: ignore[return-value]


def _require(cls, kwargs, path) -> None:
    for f in fields(cls):
        if f.default is MISSING and f.default_factory is MISSING and f.name not in kwargs:  # type: ignore[misc]
            raise ConfigError(f"{path or cls.__name__}: missing required key {f.name!r}")


# ── companies.yml ──────────────────────────────────────────────────────────
_COMPANY_KEYS = {"name", "careers_url", "api", "provider", "slug", "enabled", "extra"}


#: Which company list a board came from (``boards.provenance``), by the list's file name.
LIST_PROVENANCE = {"companies.yml": "curated", "companies.wide.yml": "openjobs"}


def list_provenance(path: str | Path | None) -> str:
    """``curated`` / ``openjobs`` for the two known lists, another list's file stem, ``orphan`` for none."""
    if not path:
        return "orphan"
    return LIST_PROVENANCE.get(Path(path).name, Path(path).stem)


def load_companies(
    path: str | Path, provenance: str | None = None, *, include_disabled: bool = False
) -> list[CompanyEntry]:
    """The list's entries, each carrying ``extra["provenance"]`` (``provenance``, else ``list_provenance(path)``).
    ``enabled: false`` entries are left out, or kept with ``extra["enabled"] = False`` (``boards import`` retires
    them)."""
    provenance = provenance or list_provenance(path)
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw = doc.get("companies") or []
    if not isinstance(raw, list):
        raise ConfigError(f"{path}: 'companies' must be a list")

    entries: list[CompanyEntry] = []
    for i, item in enumerate(raw):
        where = f"companies[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{where}: expected a mapping")
        name = item.get("name")
        if not name or not isinstance(name, str):
            raise ConfigError(f"{where}: missing/invalid 'name'")
        unknown = set(item) - _COMPANY_KEYS
        if unknown:
            raise ConfigError(f"{where} ({name}): unknown key(s) {sorted(unknown)}")
        if item.get("enabled") is False and not include_disabled:
            continue
        extra = dict(item.get("extra") or {})
        if item.get("enabled") is False:
            extra["enabled"] = False
        if item.get("slug"):
            extra["slug"] = item["slug"]
        extra["provenance"] = provenance
        entries.append(
            CompanyEntry(
                name=name,
                careers_url=item.get("careers_url"),
                api=item.get("api"),
                provider=item.get("provider"),
                extra=extra,
            )
        )
    return entries


#: Where a public ATS-company list came from (decision D2.2), the value ``boards.provenance`` records.
INVENTORY_PROVENANCE = "ats-inventory"


def load_inventory(
    path: str | Path, provider: str, provenance: str = INVENTORY_PROVENANCE, *, board_from_url: bool = False
) -> list[CompanyEntry]:
    """A public ATS-company list (build slice 2.3): one ``name,slug,url`` CSV per ATS, e.g. kalil0321/ats-scrapers'
    ``ats-companies/<provider>.csv`` (MIT). Every row is that one ``provider``; the slug is used as given, so nothing
    is fetched or detected. Rows without a name or slug are skipped.

    ``board_from_url`` (the lists for the step-3 ATSs, whose ``slug`` is not our board: Workday's lacks its data
    centre, Oracle's its site, Keka's list has none): the board is read from the row's URL by
    ``discover.board_from_url``, the same detector apply links go through; a row whose URL names no board of this
    provider is skipped. The ``slug`` column is then optional (an Eightfold ``domain`` column is read)."""
    from hiredra.sourcing.discover import board_from_url as from_url  # discover imports this module's neighbours

    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"name", "url"} if board_from_url else {"name", "slug", "url"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ConfigError(f"{path}: missing column(s) {sorted(missing)} (expected {','.join(sorted(required))})")
        entries = []
        for row in reader:
            name, url = (row.get("name") or "").strip(), (row.get("url") or "").strip()
            slug = from_url(provider, url, (row.get("domain") or "").strip() or None) if board_from_url else (
                row.get("slug") or ""
            ).strip()  # fmt: skip
            if name and slug:
                entries.append(CompanyEntry(name=name, careers_url=url or None, provider=provider,
                                            extra={"slug": slug, "provenance": provenance}))  # fmt: skip
        return entries


def board_entry(row: Mapping[str, Any]) -> CompanyEntry:
    """One ``boards`` row (``storage.db.load_boards``) as the entry the registry routes: the stored provider and
    slug, so nothing is re-detected; the careers URL when one is stored (e.g. a Lever EU host)."""
    extra: dict[str, Any] = {"slug": row["board"], "provenance": row["provenance"], "board_id": row["id"]}
    if row["list_tag"]:
        extra["tag"] = row["list_tag"]
    if "last_ms" in row.keys() and row["last_ms"] is not None:
        extra["last_seconds"] = row["last_ms"] / 1000.0  # its last crawl's time: its place on the belt
    return CompanyEntry(
        name=row["name"] or row["board"], careers_url=row["careers_url"], provider=row["provider"], extra=extra
    )


# ── scoring.yml / scoring.wide.yml ─────────────────────────────────────────────
_RUN_KEYS = {"kind", "boards", "jobspy", "http", "pacing", "path2", "belt"}


def _boards(raw: Any, path: Path) -> tuple[str, ...]:
    if raw is None:
        return ()
    if raw == "all":
        return ("all",)
    if isinstance(raw, list) and raw and all(isinstance(p, str) and p and p != "all" for p in raw):
        return tuple(raw)
    raise ConfigError(f"{path}: 'boards' must be 'all' or a non-empty list of provenances, got {raw!r}")


def _belt(raw: Any) -> BeltConfig:
    """``belt:`` -- ``fill``, ``tries``, ``watchdog_minutes``, ``status_minutes`` and ``platforms: {name: {limit,
    at_once}}``, strictly."""
    if raw is None:
        return BeltConfig()
    if not isinstance(raw, dict):
        raise ConfigError("belt: expected a mapping")
    platforms = raw.get("platforms") or {}
    if not isinstance(platforms, dict):
        raise ConfigError("belt.platforms: expected a mapping of platform name to {limit, at_once}")
    built = {str(name): _build(Platform, spec, f"belt.platforms.{name}") for name, spec in platforms.items()}
    for name, plat in built.items():
        if not (isinstance(plat.limit, (int, float)) and plat.limit > 0 and isinstance(plat.at_once, int)
                and plat.at_once > 0):  # fmt: skip
            raise ConfigError(f"belt.platforms.{name}: limit must be > 0 and at_once a whole number > 0")
    belt = _build(BeltConfig, {k: v for k, v in raw.items() if k != "platforms"}, "belt")
    if not 0 < belt.fill <= 1:
        raise ConfigError(f"belt.fill: must be above 0 and at most 1, got {belt.fill!r}")
    return BeltConfig(fill=belt.fill, platforms=built, tries=belt.tries, watchdog_minutes=belt.watchdog_minutes,
                      status_minutes=belt.status_minutes)  # fmt: skip


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    unknown = set(doc) - _RUN_KEYS
    if unknown:
        raise ConfigError(f"{path}: unknown top-level key(s) {sorted(unknown)}")

    kind = doc.get("kind")
    if kind != "scoring":
        raise ConfigError(f"{path}: 'kind' must be 'scoring', got {kind!r}")

    path2 = doc.get("path2") or {}
    if not isinstance(path2, dict) or set(path2) - {"enabled"}:
        raise ConfigError(f"{path}: 'path2' may only contain 'enabled'")

    return RunConfig(
        kind=kind,
        config_name=path.name,
        boards=_boards(doc.get("boards"), path),
        jobspy=_build(JobSpyConfig, doc.get("jobspy"), "jobspy"),
        http=_build(HttpConfig, doc.get("http"), "http"),
        pacing=_build(Pacing, doc.get("pacing"), "pacing"),
        path2_enabled=bool(path2.get("enabled", False)),
        belt=_belt(doc.get("belt")),
    )
