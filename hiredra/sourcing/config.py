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

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

from hiredra.sourcing.http import HttpConfig
from hiredra.sourcing.jobspy_source import JobSpyConfig, SiteQuery
from hiredra.sourcing.provider import CompanyEntry

_T = TypeVar("_T")


class ConfigError(ValueError):
    """A malformed config file."""


@dataclass(frozen=True, slots=True)
class Pacing:
    between_companies: float = 1.0  # base seconds between company fetches
    jitter: float = 0.5  # + up to this many random seconds


@dataclass(frozen=True, slots=True)
class RunConfig:
    kind: str  # "scoring" (the backfill kind was removed, R1a)
    config_name: str  # basename, recorded on the runs row
    boards: tuple[str, ...]  # provenances to crawl; ("all",) = every provenance; () = no boards (JobSpy only)
    jobspy: JobSpyConfig
    http: HttpConfig
    pacing: Pacing
    path2_enabled: bool = False  # the Path-2 discovery slot


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


def board_entry(row: Mapping[str, Any]) -> CompanyEntry:
    """One ``boards`` row (``storage.db.load_boards``) as the entry the registry routes: the stored provider and
    slug, so nothing is re-detected; the careers URL when one is stored (e.g. a Lever EU host)."""
    extra: dict[str, Any] = {"slug": row["board"], "provenance": row["provenance"], "board_id": row["id"]}
    if row["list_tag"]:
        extra["tag"] = row["list_tag"]
    return CompanyEntry(
        name=row["name"] or row["board"], careers_url=row["careers_url"], provider=row["provider"], extra=extra
    )


# ── scoring.yml / scoring.wide.yml ─────────────────────────────────────────────
_RUN_KEYS = {"kind", "boards", "jobspy", "http", "pacing", "path2"}


def _boards(raw: Any, path: Path) -> tuple[str, ...]:
    if raw is None:
        return ()
    if raw == "all":
        return ("all",)
    if isinstance(raw, list) and raw and all(isinstance(p, str) and p and p != "all" for p in raw):
        return tuple(raw)
    raise ConfigError(f"{path}: 'boards' must be 'all' or a non-empty list of provenances, got {raw!r}")


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
    )
