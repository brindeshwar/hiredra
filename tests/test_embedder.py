"""The embedder (hiredra/graph/embedder.py): Qwen3 on the CPU since plan unit DB4 (D1.15). The fast tests use a
fake model; the real-model test runs only where the optional `embed` group is installed and the weights are cached."""

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
    embed = emb.build_embedder(CFG, model=fake, descriptions={"kafka": "A streaming platform."})
    vecs = embed(["kafka", "frobnicate"])
    assert fake.calls[0]["texts"] == ["kafka: A streaming platform.", "frobnicate"]
    assert fake.calls[0]["prompt"] == CFG.prefix and fake.calls[0]["batch_size"] == CFG.inputs_per_request
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-6)  # unit length, float32
    assert vecs.dtype == np.float32 and vecs.shape == (2, CFG.dimensions)


def test_bare_inputs_ignore_the_descriptions():
    fake = FakeModel()
    emb.build_embedder(dataclasses.replace(CFG, inputs="bare"), model=fake)(["kafka"])
    assert fake.calls[0]["texts"] == ["kafka"]


def test_empty_input_makes_no_call_and_a_wrong_shape_is_an_error():
    fake = FakeModel()
    embed = emb.build_embedder(CFG, model=fake, descriptions={})
    assert embed([]).shape == (0, CFG.dimensions) and fake.calls == []
    with pytest.raises(emb.EmbeddingError, match="expected"):
        emb.build_embedder(CFG, model=FakeModel(dims=8), descriptions={})(["x"])


def test_described_inputs_must_be_given_their_descriptions():
    with pytest.raises(ValueError, match="embedding_descriptions"):
        emb.build_embedder(CFG, model=FakeModel())


def test_without_a_model_the_real_one_is_loaded_once(monkeypatch):
    built = []
    monkeypatch.setattr(emb, "load_local_model", lambda cfg: built.append(cfg) or FakeModel())
    embed = emb.build_embedder(CFG, descriptions={})
    assert embed(["python"]).shape == (1, CFG.dimensions) and built == [CFG]
    embed(["java"])
    assert built == [CFG]


def test_a_zero_or_nan_vector_is_an_error():
    class Zeros(FakeModel):
        def encode(self, texts, **kw):
            return np.zeros((len(texts), self.dims), dtype=np.float32)

    with pytest.raises(emb.EmbeddingError, match="zero or non-finite"):
        emb.build_embedder(CFG, model=Zeros(), descriptions={})(["a"])
    with pytest.raises(emb.EmbeddingError):
        emb.l2_normalize(np.array([[1.0, float("nan")]]))


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
    embed = emb.build_embedder(CFG, descriptions=load_descriptions())
    names = ["python", "java", "kubernetes", "k8s", "docker", "excel", "react", "reactjs"]
    batch = embed(names)
    alone = np.stack([embed([n])[0] for n in names])
    assert float(np.max(1 - np.sum(batch * alone, axis=1))) < 1e-5  # float32: a batch never changes a vector
    np.testing.assert_array_equal(embed(names), batch)  # deterministic
    s = {(a, b): float(batch[names.index(a)] @ batch[names.index(b)]) for a in names for b in names}
    assert s[("kubernetes", "k8s")] > s[("kubernetes", "excel")] and s[("react", "reactjs")] > s[("react", "java")]


def test_the_descriptions_tag_ignores_line_endings(tmp_path):
    from hiredra.graph.descriptions import source_tag

    lf, crlf = tmp_path / "lf.tsv", tmp_path / "crlf.tsv"
    lf.write_bytes(b"python\tPython\tA language.\n")
    crlf.write_bytes(b"python\tPython\tA language.\r\n")
    assert source_tag(lf) == source_tag(crlf)
