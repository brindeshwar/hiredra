"""The local embedder (hiredra/graph/embedder.py, plan unit DB4, D1.15). The fast tests use a fake model; the
real-model test runs only where the optional `embed` group is installed and the weights are cached."""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pytest

from hiredra.graph import embedder as emb
from hiredra.graph.config import load_graph_config
from hiredra.graph.descriptions import described, load_descriptions

CFG = load_graph_config()


class FakeModel:
    """Records what it was asked to encode; returns a fixed vector per text (not unit length)."""

    def __init__(self, dims=CFG.dimensions):
        self.dims, self.calls = dims, []

    def encode(self, texts, *, prompt, batch_size, convert_to_numpy, show_progress_bar):
        self.calls.append({"texts": list(texts), "prompt": prompt, "batch_size": batch_size})
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, hash(t) % self.dims] = 3.0
        return out


def test_names_become_name_colon_description_where_one_exists():
    fake = FakeModel()
    embed = emb.build_local_embedder(CFG, model=fake, descriptions={"kafka": "A streaming platform."})
    vecs = embed(["kafka", "frobnicate"])
    assert fake.calls[0]["texts"] == ["kafka: A streaming platform.", "frobnicate"]
    assert fake.calls[0]["prompt"] == CFG.prefix and fake.calls[0]["batch_size"] == CFG.inputs_per_request
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-6)  # unit length, float32
    assert vecs.dtype == np.float32 and vecs.shape == (2, CFG.dimensions)


def test_bare_inputs_ignore_the_descriptions():
    fake = FakeModel()
    emb.build_local_embedder(dataclasses.replace(CFG, inputs="bare"), model=fake)(["kafka"])
    assert fake.calls[0]["texts"] == ["kafka"]


def test_empty_input_makes_no_call_and_a_wrong_shape_is_an_error():
    fake = FakeModel()
    embed = emb.build_local_embedder(CFG, model=fake, descriptions={})
    assert embed([]).shape == (0, CFG.dimensions) and fake.calls == []
    with pytest.raises(emb.EmbeddingError, match="expected"):
        emb.build_local_embedder(CFG, model=FakeModel(dims=8), descriptions={})(["x"])


def test_build_embedder_dispatches_to_the_local_model_and_never_counts_quota(monkeypatch):
    built = []
    monkeypatch.setattr(emb, "load_local_model", lambda cfg: built.append(cfg) or FakeModel())

    class NoQuota:
        def __getattr__(self, name):
            raise AssertionError("the local model must not touch the daily counter")

    embed = emb.build_embedder(CFG, usage=NoQuota())
    assert embed(["python"]).shape == (1, CFG.dimensions) and built == [CFG]


def test_a_missing_optional_group_says_how_to_install_it(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name in ("torch", "sentence_transformers"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(RuntimeError, match="uv sync --group embed"):
        emb.load_local_model(CFG)


def test_descriptions_match_exact_names_only():
    d = load_descriptions()
    assert described("c++", d).startswith("c++: ") and described("node.js", d).startswith("node.js: ")
    assert described("entity framework", d) == "entity framework"  # an alias of .NET in freehire: not used


_HAVE_MODEL = importlib.util.find_spec("sentence_transformers") is not None and any(
    (emb.MODEL_CACHE / "hub" / "models--Qwen--Qwen3-Embedding-0.6B").glob("snapshots/*/model.safetensors")
)


@pytest.mark.skipif(not _HAVE_MODEL, reason="needs `uv sync --group embed` and the cached Qwen3 weights")
def test_the_real_model_is_deterministic_batch_independent_and_sensible():
    embed = emb.build_embedder(CFG)
    names = ["python", "java", "kubernetes", "k8s", "docker", "excel", "react", "reactjs"]
    batch = embed(names)
    alone = np.stack([embed([n])[0] for n in names])
    assert float(np.max(1 - np.sum(batch * alone, axis=1))) < 1e-5  # float32: a batch never changes a vector
    np.testing.assert_array_equal(embed(names), batch)  # deterministic
    s = {(a, b): float(batch[names.index(a)] @ batch[names.index(b)]) for a in names for b in names}
    assert s[("kubernetes", "k8s")] > s[("kubernetes", "excel")] and s[("react", "reactjs")] > s[("react", "java")]


def test_the_local_model_is_unmetered_and_gemini_keeps_its_daily_counter():
    from hiredra.graph.usage import DailyUsage, Unmetered, usage_for

    local = usage_for(None, CFG)
    assert isinstance(local, Unmetered) and local.remaining() >= 2566 and not local.exhausted()
    local.reserve(10_000)  # never refuses, never counts
    local.record(10_000)
    assert local.used() == 0
    assert isinstance(usage_for(None, dataclasses.replace(CFG, provider="gemini")), DailyUsage)


def test_the_descriptions_tag_ignores_line_endings(tmp_path):
    from hiredra.graph.descriptions import source_tag

    lf, crlf = tmp_path / "lf.tsv", tmp_path / "crlf.tsv"
    lf.write_bytes(b"python\tPython\tA language.\n")
    crlf.write_bytes(b"python\tPython\tA language.\r\n")
    assert source_tag(lf) == source_tag(crlf)
