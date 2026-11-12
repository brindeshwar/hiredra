"""Sub-slice 4.1: the embedder (build step 4, docs/graph.md sec 4).

``build_embedder`` returns one plain callable, ``embed(texts) -> ndarray``:
an ``(n, dimensions)`` float32 matrix, one L2-normalized row per input, so
nearest-skill is a dot product everywhere downstream. Everything else in the
graph package takes that callable as an argument, so tests hand in a fake and
never need the model.

The model is Qwen/Qwen3-Embedding-0.6B, run locally on the CPU (plan unit DB4, decision D1.15): no network, no
key and no quota. It replaced Google's gemini-embedding-2 on 2026-10-29; the Gemini client, its pacing and its
daily request counter were removed afterwards (the comparison is docs/model-selection-embedding.md).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from hiredra.graph.config import GraphConfig
from hiredra.graph.descriptions import described
from hiredra.quota import ProviderError

#: text list in, (n, dimensions) float32 L2-normalized matrix out.
Embedder = Callable[[Sequence[str]], np.ndarray]


#: The model returned something unusable (wrong shape, a zero vector). ``hiredra.quota.classify_failure`` reports
#: it as "bad_response", so a run stops cleanly instead of crashing.
EmbeddingError = ProviderError

#: Where the model's weights live (git-ignored). HF_HOME, when set, wins.
MODEL_CACHE = Path(__file__).resolve().parents[2] / "data" / "models" / "hf"


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise unit-length copy, float32. Raises on a zero/non-finite row."""
    m = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise EmbeddingError("embedding has a zero or non-finite norm")
    return m / norms


def load_local_model(cfg: GraphConfig):
    """``cfg.model`` as a sentence-transformers model on the CPU, in float32.

    float32, not the bfloat16 the library defaults to: measured 2026-10-29, float32 vectors are identical across
    library versions (max cosine difference 7e-7 between transformers 5.3 and 5.20) and do not depend on which
    other skills share a batch; bfloat16 moved cosines by up to 0.014. Loads from the local cache; downloads only
    if the weights are missing. Needs the optional dependency group: ``uv sync --group embed``.
    """
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE))
    hub = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + cfg.model.replace("/", "--"))
    if any(hub.glob("snapshots/*/config.json")):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # cached: no network call at all (the hub probes otherwise)
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "the local embedding model needs the optional libraries: run `uv sync --group embed`"
        ) from exc
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))  # leave the machine usable
    kwargs = {"device": "cpu", "model_kwargs": {"dtype": torch.float32}}
    try:
        return SentenceTransformer(cfg.model, local_files_only=True, **kwargs)
    except OSError:
        return SentenceTransformer(cfg.model, **kwargs)  # first use: fetch the weights into the cache


def build_embedder(cfg: GraphConfig, *, model=None, descriptions: dict[str, str] | None = None) -> Embedder:
    """``embed(texts)`` on the local model.

    ``cfg.prefix`` is the model's instruction (the ``prompt``); with ``inputs: described`` each name becomes
    "name: description" where ``descriptions`` has one -- pass ``embedding_descriptions(conn, cfg)``, the same
    lookup the stored vectors were made with (required for described inputs). ``model`` is anything with
    ``encode(texts, prompt=..., batch_size=..., convert_to_numpy=True, show_progress_bar=False)``; the default
    loads the real one on first use.
    """
    if model is None:
        model = load_local_model(cfg)
    if descriptions is None:
        if cfg.inputs == "described":
            raise ValueError("described inputs need descriptions=embedding_descriptions(conn, cfg)")
        descriptions = {}

    def embed(texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, cfg.dimensions), dtype=np.float32)
        inputs = [described(t, descriptions) for t in texts]
        vectors = model.encode(
            inputs,
            prompt=cfg.prefix or None,
            batch_size=cfg.inputs_per_request,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.shape != (len(texts), cfg.dimensions):
            raise EmbeddingError(f"expected {(len(texts), cfg.dimensions)} vectors, got {matrix.shape}")
        return l2_normalize(matrix)

    return embed
