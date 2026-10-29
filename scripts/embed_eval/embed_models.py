"""Embed the evaluation strings with each local candidate model, on the CPU only, and store cosines.

Runs in the isolated eval venv (data/embed-eval/.venv: sentence-transformers + CPU torch), NOT the project
venv, so the comparison never changes the project's dependencies:
    data/embed-eval/.venv/Scripts/python.exe scripts/embed_eval/embed_models.py [model-key ...]

Per (model, variant) it writes data/calibration/embed-eval/cos/<key>.json with every gold + silver pair's
cosine, plus timing for embedding all 2,223 graph skill names (bare). Models download into
data/embed-eval/hf (git-ignored). A model that fails to load is recorded with its error, never retried here.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "data" / "calibration" / "embed-eval"
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "embed-eval" / "hf"))
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: the owner's GPU stays free

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))

SKILL_TASK = "Instruct: Given a job skill, retrieve the same or a closely related job skill\nQuery: "

# key -> (hf id, prompt or None, kwargs, special)
MODELS = {
    "harrier-0.6b": (
        "microsoft/harrier-oss-v1-0.6b",
        "Instruct: Retrieve semantically similar text\nQuery: ",
        {},
        None,
    ),
    "harrier-270m": (
        "microsoft/harrier-oss-v1-270m",
        "Instruct: Retrieve semantically similar text\nQuery: ",
        {},
        None,
    ),
    "qwen3-0.6b": ("Qwen/Qwen3-Embedding-0.6B", None, {}, None),
    "qwen3-0.6b-skilltask": ("Qwen/Qwen3-Embedding-0.6B", SKILL_TASK, {}, None),
    "embeddinggemma-2": ("google/embeddinggemma-2", "task: sentence similarity | query: ", {}, None),
    "jobbert-v3-skill": ("TechWolf/JobBERT-v3", None, {}, "positive"),
    "jobbert-v3-title": ("TechWolf/JobBERT-v3", None, {}, "anchor"),
    "gte-modernbert": ("Alibaba-NLP/gte-modernbert-base", None, {}, None),
    # xformers has no Windows/CPU build: the model card's switch turns its memory-efficient attention off
    "arctic-m-v2": (
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        "query: ",
        {"trust_remote_code": True, "config_kwargs": {"use_memory_efficient_attention": False, "unpad_inputs": False}},
        None,
    ),
    "e5-multilingual-base": ("intfloat/multilingual-e5-base", "query: ", {}, None),
    "bge-m3": ("BAAI/bge-m3", None, {}, None),
}


def encode(model, texts, prompt, asym_key):
    if asym_key:  # JobBERT-v3: an Asym head per side ("anchor" = job titles, "positive" = skills)
        from sentence_transformers.util import batch_to_device

        out = []
        for i in range(0, len(texts), 32):
            feats = batch_to_device(model.tokenize(texts[i : i + 32]), model.device)
            feats["text_keys"] = [asym_key]
            with torch.no_grad():
                out.append(model.forward(feats)["sentence_embedding"].cpu().numpy())
        v = np.concatenate(out)
    else:
        v = model.encode(texts, prompt=prompt, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def run(key: str) -> None:
    hf_id, prompt, kwargs, asym = MODELS[key]
    sets = json.loads((EVAL / "sets.json").read_text(encoding="utf-8"))
    strings = json.loads((EVAL / "strings.json").read_text(encoding="utf-8"))
    graph = json.loads((EVAL / "graph_names.json").read_text(encoding="utf-8"))
    pairs = sets["gold"] + sets["silver"]
    result = {"key": key, "model": hf_id, "prompt": prompt, "asym": asym, "device": "cpu"}
    try:
        t0 = time.perf_counter()
        # float32: half precision is slow on most CPUs, and every model gets the same treatment
        model = SentenceTransformer(hf_id, device="cpu", model_kwargs={"dtype": torch.float32}, **kwargs)
        result["load_s"] = round(time.perf_counter() - t0, 1)
        result["dims"] = int(model.get_sentence_embedding_dimension() or 0)
        result["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
        for variant in ("bare", "described"):
            texts = strings["bare"] if variant == "bare" else [strings["described"][s] for s in strings["bare"]]
            t0 = time.perf_counter()
            vecs = encode(model, texts, prompt, asym)
            secs = time.perf_counter() - t0
            idx = {s: i for i, s in enumerate(strings["bare"])}
            cos = {f"{p['a']}\t{p['b']}": float(vecs[idx[p["a"]]] @ vecs[idx[p["b"]]]) for p in pairs}
            result[variant] = {"embed_s": round(secs, 1), "cosines": cos}
        t0 = time.perf_counter()
        encode(model, graph, prompt, asym)
        result["graph_2223_bare_s"] = round(time.perf_counter() - t0, 1)
        result["ok"] = True
    except Exception as exc:  # recorded, never hidden
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["trace"] = traceback.format_exc()[-2000:]
    out = EVAL / "cos"
    out.mkdir(exist_ok=True)
    (out / f"{key}.json").write_text(json.dumps(result), encoding="utf-8")
    print(
        f"{key}: {'ok' if result['ok'] else 'FAILED ' + result['error'][:200]}"
        + (
            f" dims={result.get('dims')} params={result.get('params_m')}M graph={result.get('graph_2223_bare_s')}s"
            if result["ok"]
            else ""
        ),
        flush=True,
    )


if __name__ == "__main__":
    for k in sys.argv[1:] or list(MODELS):
        if (EVAL / "cos" / f"{k}.json").exists() and "--force" not in sys.argv:
            print(f"{k}: already done (use --force to redo)", flush=True)
            continue
        if k.startswith("--"):
            continue
        run(k)
