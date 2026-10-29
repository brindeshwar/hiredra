"""Score every embedded (model, variant) and write the comparison table (docs/model-selection-embedding.md).

Project venv:  PYTHONPATH=. uv run python scripts/embed_eval/report.py
Gold: thresholds tuned and holdout-scored with the SAME code as Step 5 (hiredra.graph.calibrate.calibrate).
Silver: ROC-AUC for HAVE (same vs the rest) and ADJ (same + related vs unrelated), threshold-free.
Ranking rule (fixed before running): gate gold ADJ holdout F1 >= 0.80; primary = mean(gold ADJ F1, silver ADJ AUC);
tie-break = mean(gold HAVE F1, silver HAVE AUC); then CPU time on the 2,223 graph names.
"""

from __future__ import annotations

import json
from pathlib import Path

from hiredra.graph.calibrate import Pair, calibrate

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "calibration" / "embed-eval"


def auc(pos: list[float], neg: list[float]) -> float:
    """Probability a random positive outranks a random negative (ties count half)."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        rank_sum += avg_rank * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def score(cos: dict[str, float], sets: dict) -> dict:
    gold = [Pair(p["a"], p["b"], p["label"], p["split"], p["source"]) for p in sets["gold"]]
    gcos = {p.key: cos[f"{p.a}\t{p.b}"] for p in gold}  # calibrate() looks pairs up by the sorted key
    res = calibrate(gold, gcos)
    sil = sets["silver"]
    c = lambda p: cos[f"{p['a']}\t{p['b']}"]  # noqa: E731
    same = [c(p) for p in sil if p["label"] == "same"]
    rel = [c(p) for p in sil if p["label"] == "related"]
    unr = [c(p) for p in sil if p["label"] == "unrelated"]
    out = {
        "have": res.have,
        "adj": res.adj,
        "gold_have_f1": res.holdout["have"]["all"].f1 if hasattr(res.holdout["have"]["all"], "f1") else None,
        "gold_adj_f1": res.holdout["adj"]["all"].f1 if hasattr(res.holdout["adj"]["all"], "f1") else None,
        "silver_have_auc": auc(same, rel + unr),
        "silver_adj_auc": auc(same + rel, unr),
    }
    out["primary"] = (out["gold_adj_f1"] + out["silver_adj_auc"]) / 2
    out["tiebreak"] = (out["gold_have_f1"] + out["silver_have_auc"]) / 2
    return out


def main() -> None:
    sets = json.loads((EVAL / "sets.json").read_text(encoding="utf-8"))
    rows, failed = [], []
    for f in sorted((EVAL / "cos").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if not r.get("ok"):
            failed.append((r["key"], r.get("error", "")[:160]))
            continue
        for variant in ("bare", "described"):
            keys = {f"{p['a']}\t{p['b']}" for p in sets["gold"] + sets["silver"]}
            if variant in r and keys <= set(r[variant]["cosines"]):  # every pair present (dict keys dedupe repeats)
                s = score(r[variant]["cosines"], sets)
                rows.append(
                    {
                        "key": r["key"],
                        "variant": variant,
                        "dims": r.get("dims"),
                        "params_m": r.get("params_m"),
                        "graph_s": r.get("graph_2223_bare_s"),
                        **s,
                    }
                )
    rows.sort(
        key=lambda x: (
            x["gold_adj_f1"] >= 0.80,
            round(x["primary"], 4),
            round(x["tiebreak"], 4),
            -(x["graph_s"] or 1e9),
        ),
        reverse=True,
    )
    lines = [
        "| # | model | input | gold ADJ F1 | silver ADJ AUC | **primary** | gold HAVE F1 | silver HAVE AUC | tie-break | HAVE / ADJ | dims | params | CPU s (2,223 names) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, x in enumerate(rows, 1):
        gate = "" if x["gold_adj_f1"] >= 0.80 else " ✗"
        lines.append(
            f"| {i} | {x['key']} | {x['variant']} | {x['gold_adj_f1']:.3f}{gate} | {x['silver_adj_auc']:.3f} | **{x['primary']:.3f}** | "
            f"{x['gold_have_f1']:.3f} | {x['silver_have_auc']:.3f} | {x['tiebreak']:.3f} | {x['have']:.2f} / {x['adj']:.2f} | "
            f"{x['dims']} | {x['params_m']}M | {x['graph_s'] if x['graph_s'] is not None else '—'} |"
        )
    if failed:
        lines += ["", "Failed to run:"] + [f"- {k}: {e}" for k, e in failed]
    (EVAL / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
