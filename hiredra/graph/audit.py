"""The graph's health check (owner, 2026-11-12): the last step of every pipeline run, so a step that silently
stops -- merging stalled for 12 days before the 2026-11-12 audit found it -- shows up in the next run's log.

    python -m hiredra.graph audit            # print the checks, write data/graph-audit/<date>.md
    python -m hiredra.graph audit --strict   # exit 1 when a check fails (for a scripted pipeline)

Read-only. Two kinds of line:

* CHECKS -- each has a limit and a fix; a check over its limit prints WARNING with the command that fixes it:
  every node embedded from its current input; no skill waiting for a description; no extracted job held back by
  one; nothing `graph clean` would still change (spelling variants, lists, junk); no job row on an old spelling or
  on a skill that no longer exists.
* MATCHING -- numbers to watch, no limit yet (the matching experiment sets them): how job skills split into
  missing / partial / full credit, the different skills that most often get full credit for each other, and the
  owner's skills with no close neighbour. The 2026-11-12 audit found 3.3% missing and gcp getting full credit for
  aws 473 times: these lines are where that would show again.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from hiredra.graph.config import GraphConfig
from hiredra.graph.descriptions import embedding_descriptions
from hiredra.graph.index import load_skills
from hiredra.graph.sync import input_texts
from hiredra.storage import skill_index_status

DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "data" / "graph-audit"
#: A mine skill whose nearest other skill is below this has nothing in the graph to match against.
LONELY_BELOW = 0.70


@dataclass(frozen=True)
class Check:
    name: str
    value: int
    fix: str  # the command that brings it back to 0

    @property
    def ok(self) -> bool:
        return self.value == 0


@dataclass
class Audit:
    checks: list[Check] = field(default_factory=list)
    credit: dict[str, int] = field(default_factory=dict)  # missing / partial / full job-skill instances
    siblings: list[tuple[str, str, int]] = field(default_factory=list)  # (job skill, mine skill, times) at full
    lonely: list[tuple[str, float]] = field(default_factory=list)  # mine skills with no close neighbour

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]


def _count(conn, sql: str, params=()) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


def run_audit(conn, cfg: GraphConfig, *, clean_plan=None) -> Audit:
    """Every check and number. ``clean_plan`` is ``graph clean``'s dry run (computed by the caller)."""
    from hiredra.scoring.run import waiting_for_descriptions

    a = Audit()
    status = skill_index_status(conn, cfg.embed_tag, input_texts(embedding_descriptions(conn, cfg)))
    a.checks += [
        Check("skills not embedded from their current input", status["unembedded"] + status["stale"],
              "python -m hiredra.graph sync"),
        Check("skills with no description yet", _count(conn, "SELECT count(*) FROM skill_vocab WHERE description_source IS NULL"),
              "python -m hiredra.graph.describe"),
        Check("extracted jobs waiting for skill descriptions", waiting_for_descriptions(conn),
              "python -m hiredra.graph.describe, then python -m hiredra.scoring run"),
        Check("names graph clean would still change", len(clean_plan or []),
              "python -m hiredra.graph clean (review), then clean --apply"),
        Check("job rows on an old spelling (an alias)",
              _count(conn, "SELECT count(*) FROM job_skills WHERE skill IN (SELECT alias FROM skill_alias)"),
              "python -m hiredra.graph clean --apply rewrites them"),
        Check("job rows on a skill that does not exist",
              _count(conn, "SELECT count(*) FROM job_skills WHERE skill NOT IN (SELECT name FROM skill_vocab) "
                           "AND skill NOT IN (SELECT alias FROM skill_alias)"),
              "python -m hiredra.graph sync (a new name), or look at the job"),
    ]  # fmt: skip

    bands, pairs = Counter(), Counter()
    for (rows,) in conn.execute(
        "SELECT score_features->'skills' FROM jobs WHERE status = 'scored' AND score_features ? 'skills'"
    ):
        for r in rows or []:
            fit = float(r.get("fit") or 0)
            bands["full" if fit >= 1.0 else "partial" if fit > 0 else "missing"] += 1
            if fit >= 1.0 and r.get("nearest") and r.get("skill") != r.get("nearest"):
                pairs[(r["skill"], r["nearest"])] += 1
    a.credit = {k: bands[k] for k in ("missing", "partial", "full")}
    a.siblings = [(s, n, c) for (s, n), c in pairs.most_common(10)]

    skills = [s for s in load_skills(conn, cfg) if s.vector is not None]
    others = [s for s in skills if not s.mine]
    if others:
        matrix = np.stack([s.vector for s in others])
        for s in (s for s in skills if s.mine):
            best = float((matrix @ s.vector).max())
            if best < LONELY_BELOW:
                a.lonely.append((s.name, round(best, 3)))
    return a


def render(a: Audit, today: date) -> str:
    total = sum(a.credit.values()) or 1
    lines = [f"# Graph audit {today}", "", "## Checks", "", "| check | value | fix |", "|---|---|---|"]
    lines += [f"| {'ok' if c.ok else 'WARNING'}: {c.name} | {c.value} | {'' if c.ok else c.fix} |" for c in a.checks]
    lines += ["", "## Matching (no limits until the matching experiment)", ""]
    lines += ["- job skills: " + ", ".join(f"{k} {v} ({100 * v / total:.1f}%)" for k, v in a.credit.items())]
    lines += ["- full credit between different names (top 10): "
              + "; ".join(f"{s} <- {n} ({c})" for s, n, c in a.siblings)]  # fmt: skip
    lines += [f"- owner's skills with no neighbour >= {LONELY_BELOW}: "
              + (", ".join(f"{n} ({b})" for n, b in a.lonely) or "none")]  # fmt: skip
    return "\n".join(lines) + "\n"


def write_report(a: Audit, report_dir: Path = DEFAULT_REPORT_DIR, today: date | None = None) -> Path:
    today = today or date.today()
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{today}.md"
    path.write_text(render(a, today), encoding="utf-8")
    return path
