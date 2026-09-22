"""Real-data smoke test for hiredra.graph.fit -- NOT a pytest test (kept outside
tests/ because it reads the real `data/hiredra.db`, not a hermetic fixture).

Makes ZERO API calls. It reuses the already-stored vectors for the nine
skill names in docs/scoring.md sec 4's worked example -- they are real
market nodes, already embedded by the reset-day sync -- as a "cache" that
`fit_skills` reads from, so its embedder callable is never actually invoked.

Run it manually:

    python scripts/smoke_test_fit.py

Owner skills (the `reference` stand-in for `mine`, since the master-profile
overlay is deferred past Step 4): python, fastapi, postgresql, langchain,
mcp, docker, aws.
JD skills: python, fastapi, postgresql, kubernetes, kafka.

docs/scoring.md's table is illustrative (kubernetes~docker = 0.82); the real
embedding model gives its own number here. Either way the SHAPE must hold:
exact matches at 1.0, kubernetes lands adjacent (an ADJ..HAVE fit, not 0 or
1), kafka floors to 0.
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
    conn = connect("data/hiredra.db")
    try:
        skills = {s.name: s for s in load_skills(conn, cfg)}
    finally:
        conn.close()

    missing = [n for n in OWNER_SKILLS + JD_SKILLS if n not in skills or skills[n].vector is None]
    if missing:
        print(f"SKIPPED: these names are not currently-embedded nodes in data/hiredra.db: {missing}")
        print("Run the reset-day runbook (status/consistency/sync) first.")
        return 1

    reference = [skills[n] for n in OWNER_SKILLS]
    cache = {n: skills[n].vector for n in JD_SKILLS}  # pre-seeded: embed() must never fire

    results = fit_skills(JD_SKILLS, reference, cfg, _unavailable_embedder, cache=cache)

    print(f"HAVE={cfg.have}  ADJ={cfg.adj}  (PLACEHOLDER thresholds, not yet calibrated -- Step 5)\n")
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
    k8s = by_skill["kubernetes"]
    check(k8s.nearest == "docker", "kubernetes: nearest owner skill is docker")
    check(cfg.adj <= k8s.similarity < cfg.have, f"kubernetes: similarity {k8s.similarity:.4f} lands in the adjacent band")
    check(k8s.fit == k8s.similarity, "kubernetes: adjacent fit equals the similarity itself (no flattening)")
    kafka = by_skill["kafka"]
    check(kafka.fit == 0.0, f"kafka: fit floors to 0.0 (similarity {kafka.similarity:.4f} was below ADJ)")

    print("\nFIT SMOKE", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
