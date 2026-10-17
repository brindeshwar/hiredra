"""Step 4 config: ``config/graph.yml`` loaded into one frozen dataclass.

Strict on purpose (an unknown key is an error, same posture as the sourcing
run configs) so a typo'd knob fails loudly instead of silently keeping its
default.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "graph.yml"

_SECTIONS: dict[str, set[str]] = {
    "embedding": {"model", "dimensions", "prefix", "inputs_per_request"},
    "limits": {"requests_per_minute", "tokens_per_minute", "requests_per_day", "headroom"},
    "dedup": {"threshold", "review_cosine", "auto_cosine"},
    "fit": {"have", "adj"},
    "harvest": {
        "export_base_url",
        "export_date",
        "ats",
        "role_families",
        "min_df",
        "max_words",
        "max_chars",
    },
    "gate": {"canary_skills", "canary_min_cosine"},
}

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class HarvestConfig:
    """The open-jobs market-vocabulary harvest (docs/repos.md sec 7)."""

    export_base_url: str
    export_date: str  # "latest" or a YYYY-MM-DD export directory
    ats: tuple[str, ...]
    role_families: tuple[str, ...]
    min_df: int  # a skill must appear in at least this many enriched jobs
    max_words: int
    max_chars: int


@dataclass(frozen=True)
class GraphConfig:
    model: str
    dimensions: int
    prefix: str
    inputs_per_request: int
    requests_per_minute: int
    dedup_threshold: float
    have: float
    adj: float
    harvest: HarvestConfig | None = None
    canary_skills: tuple[str, ...] = ()
    canary_min_cosine: float = 0.999
    tokens_per_minute: int = 30_000
    requests_per_day: int = 1_000
    headroom: float = 0.90
    review_cosine: float = 0.93
    auto_cosine: float = 0.96

    @property
    def rpm_budget(self) -> int:
        """Requests/minute we allow ourselves (Google's limit x headroom). Google
        counts every input inside a batch as one request."""
        return int(self.requests_per_minute * self.headroom)

    @property
    def tpm_budget(self) -> int:
        return int(self.tokens_per_minute * self.headroom)

    @property
    def rpd_budget(self) -> int:
        return int(self.requests_per_day * self.headroom)

    @property
    def embed_tag(self) -> str:
        """Version tag stored beside every vector. Covers everything in this
        config that changes what a vector *is*: model, dimensionality, prefix.
        (Provider-side weight drift under the same model id is what the canary
        check catches -- a config tag can't see that.)"""
        prefix = hashlib.sha256(self.prefix.encode("utf-8")).hexdigest()[:8] if self.prefix else "none"
        return f"{self.model}|d{self.dimensions}|prefix={prefix}"


def _str_list(value, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"graph config: {where} must be a non-empty list of strings")
    return tuple(value)


def load_graph_config(path: Path | str = DEFAULT_CONFIG_PATH) -> GraphConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    unknown_sections = set(data) - set(_SECTIONS)
    if unknown_sections:
        raise ValueError(f"graph config: unknown section(s) {sorted(unknown_sections)}")
    for section, allowed in _SECTIONS.items():
        body = data.get(section)
        if not isinstance(body, dict):
            raise ValueError(f"graph config: section {section!r} is missing or not a mapping")
        unknown = set(body) - allowed
        if unknown:
            raise ValueError(f"graph config: unknown key(s) {sorted(unknown)} in {section!r}")
        missing = allowed - set(body)
        if missing:
            raise ValueError(f"graph config: missing key(s) {sorted(missing)} in {section!r}")

    emb, lim, dedup, fit = data["embedding"], data["limits"], data["dedup"], data["fit"]
    hv, gate = data["harvest"], data["gate"]
    harvest = HarvestConfig(
        export_base_url=str(hv["export_base_url"]).rstrip("/"),
        export_date=str(hv["export_date"]),
        ats=_str_list(hv["ats"], "harvest.ats"),
        role_families=_str_list(hv["role_families"], "harvest.role_families"),
        min_df=int(hv["min_df"]),
        max_words=int(hv["max_words"]),
        max_chars=int(hv["max_chars"]),
    )
    cfg = GraphConfig(
        model=str(emb["model"]),
        dimensions=int(emb["dimensions"]),
        prefix=str(emb["prefix"] or ""),
        inputs_per_request=int(emb["inputs_per_request"]),
        requests_per_minute=int(lim["requests_per_minute"]),
        tokens_per_minute=int(lim["tokens_per_minute"]),
        requests_per_day=int(lim["requests_per_day"]),
        headroom=float(lim["headroom"]),
        dedup_threshold=float(dedup["threshold"]),
        review_cosine=float(dedup["review_cosine"]),
        auto_cosine=float(dedup["auto_cosine"]),
        have=float(fit["have"]),
        adj=float(fit["adj"]),
        harvest=harvest,
        canary_skills=_str_list(gate["canary_skills"], "gate.canary_skills"),
        canary_min_cosine=float(gate["canary_min_cosine"]),
    )

    if not cfg.model:
        raise ValueError("graph config: embedding.model must be non-empty")
    if not 128 <= cfg.dimensions <= 3072:
        raise ValueError(f"graph config: embedding.dimensions must be 128-3072, got {cfg.dimensions}")
    if not 1 <= cfg.inputs_per_request <= 100:
        raise ValueError(f"graph config: embedding.inputs_per_request must be 1-100, got {cfg.inputs_per_request}")
    if min(cfg.requests_per_minute, cfg.tokens_per_minute, cfg.requests_per_day) < 1:
        raise ValueError("graph config: limits.requests_per_minute / tokens_per_minute / requests_per_day must be >= 1")
    if not 0.0 < cfg.headroom <= 1.0:
        raise ValueError(f"graph config: limits.headroom must be in (0, 1], got {cfg.headroom}")
    if cfg.inputs_per_request > cfg.rpm_budget:
        raise ValueError(
            f"graph config: one request of {cfg.inputs_per_request} inputs can never fit the "
            f"per-minute budget of {cfg.rpm_budget} (Google counts each input as a request)"
        )
    if cfg.rpm_budget > cfg.rpd_budget:
        raise ValueError("graph config: the per-minute budget cannot exceed the per-day budget")
    if not 0.0 < cfg.dedup_threshold <= cfg.review_cosine <= cfg.auto_cosine <= 1.0:
        raise ValueError(
            "graph config: need 0 < dedup.threshold <= dedup.review_cosine <= dedup.auto_cosine <= 1, got "
            f"{cfg.dedup_threshold} / {cfg.review_cosine} / {cfg.auto_cosine}"
        )
    if not 0.0 < cfg.adj < cfg.have <= 1.0:
        raise ValueError(f"graph config: need 0 < fit.adj < fit.have <= 1, got adj={cfg.adj} have={cfg.have}")
    if not 0.0 < cfg.canary_min_cosine <= 1.0:
        raise ValueError(f"graph config: gate.canary_min_cosine must be in (0, 1], got {cfg.canary_min_cosine}")
    # ats names go into a URL path; keep them to a safe charset
    for token in (*harvest.ats, harvest.export_date):
        if token != "latest" and not _SAFE_TOKEN.match(token):
            raise ValueError(f"graph config: harvest value {token!r} has unsafe characters")
    if harvest.min_df < 1 or harvest.max_words < 1 or harvest.max_chars < 1:
        raise ValueError("graph config: harvest.min_df / max_words / max_chars must be >= 1")
    return cfg
