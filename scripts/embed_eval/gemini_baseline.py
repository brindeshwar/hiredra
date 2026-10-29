"""Gemini Embedding 2 baseline row for the local-embedding comparison (bare names only).

Project venv:  PYTHONPATH=. uv run python scripts/embed_eval/gemini_baseline.py
Uses stored graph vectors where a string is a node (0 API), embeds the rest with the project's embedder under
its daily budget (counted, paced), and never writes to the graph. Output: data/calibration/embed-eval/cos/
gemini-embedding-2.json, the same shape the local runner writes.

Kept as the record of the DB4 model choice; it no longer runs. It was written for the SQLite store and the Gemini
graph vectors, which DB3 and DB4 replaced (Postgres, Qwen vectors, quota counters in api_usage).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hiredra.graph.config import load_graph_config
from hiredra.graph.embedder import build_embedder
from hiredra.graph.index import load_skills
from hiredra.graph.usage import DEFAULT_USAGE_PATH, DailyUsage
from hiredra.storage import connect, resolve_alias
from hiredra.vocab import normalize

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "calibration" / "embed-eval"


def main() -> None:
    cfg = load_graph_config()
    conn = connect(ROOT / "data" / "hiredra.db")
    stored = {s.name: s.vector for s in load_skills(conn, cfg) if s.vector is not None}
    sets = json.loads((EVAL / "sets.json").read_text(encoding="utf-8"))
    strings = json.loads((EVAL / "strings.json").read_text(encoding="utf-8"))["bare"]
    vecs: dict[str, np.ndarray] = {}
    missing = []
    for s in strings:
        node = resolve_alias(conn, normalize(s)) or normalize(s)
        if node in stored:
            vecs[s] = stored[node]
        else:
            missing.append(s)
    usage = DailyUsage(cfg.rpd_budget, path=DEFAULT_USAGE_PATH)
    embed = build_embedder(cfg, usage=usage)
    t0 = time.perf_counter()
    if missing:
        for s, v in zip(missing, embed(missing)):
            vecs[s] = v
    secs = time.perf_counter() - t0
    norm = {s: v / np.linalg.norm(v) for s, v in vecs.items()}
    pairs = sets["gold"] + sets["silver"]
    cos = {f"{p['a']}\t{p['b']}": float(norm[p["a"]] @ norm[p["b"]]) for p in pairs}
    out = {
        "key": "gemini-embedding-2",
        "model": cfg.model,
        "prompt": None,
        "device": "google api",
        "ok": True,
        "dims": cfg.dimensions,
        "params_m": None,
        "graph_2223_bare_s": None,
        "bare": {"embed_s": round(secs, 1), "cosines": cos, "api_calls": len(missing)},
    }
    (EVAL / "cos").mkdir(exist_ok=True)
    (EVAL / "cos" / "gemini-embedding-2.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"gemini baseline: {len(vecs)} strings, {len(missing)} embedded via API in {secs:.0f}s")


if __name__ == "__main__":
    main()
