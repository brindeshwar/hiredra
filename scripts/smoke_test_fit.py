"""Real-data smoke test for hiredra.graph.fit -- NOT a pytest test (kept outside
tests/ because it reads the real database, not a hermetic fixture).

Makes no model call. It reuses the already-stored vectors for the nine
skill names in docs/scoring.md sec 4's worked example -- they are real
market nodes, already embedded by `graph sync` -- as a "cache" that
`fit_skills` reads from, so its embedder callable is never actually invoked.

Run it manually:

    python scripts/smoke_test_fit.py

Owner skills (a fixed stand-in for `mine`): python, fastapi, postgresql, langchain, mcp, docker, aws.
JD skills: python, fastapi, postgresql, kubernetes, kafka.

What it checks holds for any embedding model: exact matches fit 1.0, and every other fit follows the band rule
(s >= HAVE -> 1.0, ADJ <= s < HAVE -> s, s < ADJ -> 0.0). Which owner skill is nearest, and whether kubernetes or
kafka earns adjacent credit, depends on the model: it is printed for a human to judge, not asserted. (Under Gemini,
kubernetes~docker was adjacent and kafka floored to 0; under Qwen, kubernetes~aws 0.66 and kafka~aws 0.58 are both
adjacent -- a watch item for weight calibration.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hiredra.graph.config import load_graph_config  # noqa: E402
from hiredra.graph.fit import fit_skills  # noqa: E402
from hiredra.graph.index import load_skills  # noqa: E402
from hiredra.storage import connect  # noqa: E402

OWNER_SKILLS = ["python", "fastapi", "postgresql", "langchain", "mcp", "docker", "aws"]
JD_SKILLS = ["python", "fastapi", "postgresql", "kubernetes", "kafka"]


def _unavailable_embedder(texts):
    raise AssertionError(
        f"embed() was called for {texts!r} -- every JD skill name in this smoke test "
        "should already be cached from the stored DB vector; a real call means the "
        "cache-seeding below is missing a name, not that a live call is needed."
    )


def main() -> int:
    cfg = load_graph_config()
    conn = connect()
    try:
        skills = {s.name: s for s in load_skills(conn, cfg)}
    finally:
        conn.close()

    missing = [n for n in OWNER_SKILLS + JD_SKILLS if n not in skills or skills[n].vector is None]
    if missing:
        print(f"SKIPPED: these names are not currently-embedded nodes in the database: {missing}")
        print("Run `python -m hiredra.graph sync` first.")
        return 1

    reference = [skills[n] for n in OWNER_SKILLS]
    cache = {n: skills[n].vector for n in JD_SKILLS}  # pre-seeded: embed() must never fire

    results = fit_skills(JD_SKILLS, reference, cfg, _unavailable_embedder, cache=cache)

    print(f"HAVE={cfg.have}  ADJ={cfg.adj}  (calibration {cfg.calibration_id or 'none'})\n")
    print(f"{'skill':<12} {'nearest':<12} {'s':>7} {'fit':>7}")
    for r in results:
        print(f"{r.skill:<12} {r.nearest or '-':<12} {r.similarity:>7.4f} {r.fit:>7.4f}")

    by_skill = {r.skill: r for r in results}
    ok = True

    def check(cond, label):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print()
    for exact in ("python", "fastapi", "postgresql"):
        r = by_skill[exact]
        check(r.nearest == exact and r.fit == 1.0, f"{exact}: exact match, fit = 1.0")
    for r in results:
        if r.skill in ("python", "fastapi", "postgresql"):
            continue
        want = 1.0 if r.similarity >= cfg.have else r.similarity if r.similarity >= cfg.adj else 0.0
        band = "have" if r.similarity >= cfg.have else "adjacent" if r.similarity >= cfg.adj else "missing"
        check(r.fit == want, f"{r.skill}: nearest {r.nearest} at {r.similarity:.4f} -> {band}, fit {r.fit:.4f}")
    print("\nFIT SMOKE", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
