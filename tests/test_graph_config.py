"""Tests for hiredra.graph.config (build sub-slice 4.1)."""

from __future__ import annotations

import textwrap

import pytest

from hiredra.graph.config import DEFAULT_CONFIG_PATH, GraphConfig, load_graph_config

_BASE = """
embedding:
  model: gemini-embedding-2
  dimensions: 768
  prefix: ""
  inputs_per_request: 50
dedup:
  threshold: 0.90
  review_cosine: 0.93
  auto_cosine: 0.96
fit:
  have: 0.90
  adj: 0.70
harvest:
  export_base_url: https://example.test/data/exports/
  export_date: latest
  ats: [ashby, lever]
  role_families: [software_engineering, data]
  min_df: 3
  max_words: 4
  max_chars: 40
gate:
  canary_skills: [python, kubernetes]
  canary_min_cosine: 0.999
"""


def _load(tmp_path, text: str = _BASE) -> GraphConfig:
    p = tmp_path / "graph.yml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return load_graph_config(p)


def test_real_config_file_loads_and_matches_the_approved_design():
    """D1.15 (decided 2026-10-26, switched in DB4): Qwen3-Embedding-0.6B on the CPU, the skill instruction,
    "name: description" inputs, 1024 dimensions."""
    cfg = load_graph_config(DEFAULT_CONFIG_PATH)
    assert (cfg.model, cfg.dimensions, cfg.inputs) == ("Qwen/Qwen3-Embedding-0.6B", 1024, "described")
    assert cfg.prefix == "Instruct: Given a job skill, retrieve the same or a closely related job skill\nQuery: "
    assert cfg.dedup_threshold == 0.90
    assert cfg.adj < cfg.have


def _with(extra: str) -> str:
    """The base config with extra embedding keys."""
    return _BASE.replace('prefix: ""', 'prefix: ""' + "".join(f"\n  {line}" for line in extra.split(";")))


def test_inputs_default_to_bare_names_and_are_checked(tmp_path):
    assert _load(tmp_path).inputs == "bare"
    assert _load(tmp_path, _with("inputs: described")).inputs == "described"
    with pytest.raises(ValueError, match="inputs"):
        _load(tmp_path, _with("inputs: everything"))


def test_the_removed_gemini_keys_are_rejected(tmp_path):
    """The Gemini embedding path is gone (after DB4): its provider switch and rate limits are unknown keys now."""
    with pytest.raises(ValueError, match="unknown key"):
        _load(tmp_path, _with("provider: gemini"))
    with pytest.raises(ValueError, match="unknown section"):
        _load(tmp_path, _BASE + "limits:\n  requests_per_minute: 100\n")


def test_described_inputs_put_the_descriptions_file_in_the_tag(tmp_path, monkeypatch):
    from hiredra.graph import descriptions

    text = _with("inputs: described")
    tag = _load(tmp_path, text).embed_tag
    assert tag.startswith("gemini-embedding-2|d768|prefix=none|desc=")
    other = tmp_path / "other.tsv"
    other.write_text("python\tPython\tA language.\n", encoding="utf-8")
    monkeypatch.setattr(descriptions.source_tag, "__defaults__", (other,))
    assert _load(tmp_path, text).embed_tag != tag  # edited descriptions -> every vector is stale


def test_tag_for_bare_prefix_is_readable(tmp_path):
    assert _load(tmp_path).embed_tag == "gemini-embedding-2|d768|prefix=none"


def test_tag_changes_with_prefix_dimensions_and_model(tmp_path):
    base = _load(tmp_path).embed_tag
    with_prefix = _load(tmp_path, _BASE.replace('prefix: ""', 'prefix: "task: x | query: "')).embed_tag
    other_dim = _load(tmp_path, _BASE.replace("dimensions: 768", "dimensions: 1536")).embed_tag
    other_model = _load(tmp_path, _BASE.replace("gemini-embedding-2", "gemini-embedding-001")).embed_tag
    assert len({base, with_prefix, other_dim, other_model}) == 4


def test_unknown_key_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        _load(tmp_path, _BASE.replace("dimensions: 768", "dimensions: 768\n  taskType: x"))


def test_unknown_section_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="unknown section"):
        _load(tmp_path, _BASE + "\nextra:\n  a: 1\n")


def test_missing_key_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="missing key"):
        _load(tmp_path, _BASE.replace("  inputs_per_request: 50\n", ""))


@pytest.mark.parametrize(
    "old,new",
    [
        ("dimensions: 768", "dimensions: 64"),
        ("dimensions: 768", "dimensions: 4096"),
        ("inputs_per_request: 50", "inputs_per_request: 101"),
        ("inputs_per_request: 50", "inputs_per_request: 0"),
        ("threshold: 0.90", "threshold: 0"),
        ("have: 0.90", "have: 0.60"),  # have must exceed adj
        ("adj: 0.70", "adj: 0"),
    ],
)
def test_out_of_range_values_are_rejected(tmp_path, old, new):
    with pytest.raises(ValueError):
        _load(tmp_path, _BASE.replace(old, new))


def test_harvest_and_gate_sections_load(tmp_path):
    cfg = _load(tmp_path)
    assert cfg.harvest.export_base_url == "https://example.test/data/exports"  # trailing / stripped
    assert cfg.harvest.export_date == "latest"
    assert cfg.harvest.ats == ("ashby", "lever")
    assert cfg.harvest.role_families == ("software_engineering", "data")
    assert (cfg.harvest.min_df, cfg.harvest.max_words, cfg.harvest.max_chars) == (3, 4, 40)
    assert cfg.canary_skills == ("python", "kubernetes")
    assert cfg.canary_min_cosine == 0.999


def test_real_config_has_the_owner_set_harvest_levers():
    cfg = load_graph_config(DEFAULT_CONFIG_PATH)
    assert cfg.harvest.ats == ("ashby", "lever", "greenhouse")  # the 3 sampled files
    assert cfg.harvest.min_df == 3
    assert set(cfg.harvest.role_families) == {"software_engineering", "data", "ml_ai", "devops_sre"}
    assert cfg.canary_min_cosine == 0.999


@pytest.mark.parametrize(
    "old,new",
    [
        ("ats: [ashby, lever]", "ats: []"),
        ("ats: [ashby, lever]", "ats: ashby"),
        ("ats: [ashby, lever]", "ats: [ashby, '../etc']"),  # goes into a URL path
        ("export_date: latest", "export_date: '2026/09/19'"),
        ("min_df: 3", "min_df: 0"),
        ("max_words: 4", "max_words: 0"),
        ("canary_skills: [python, kubernetes]", "canary_skills: []"),
        ("canary_min_cosine: 0.999", "canary_min_cosine: 1.5"),
        ("canary_min_cosine: 0.999", "canary_min_cosine: 0"),
    ],
)
def test_bad_harvest_and_gate_values_are_rejected(tmp_path, old, new):
    with pytest.raises(ValueError):
        _load(tmp_path, _BASE.replace(old, new))


def test_a_dated_export_directory_is_accepted(tmp_path):
    cfg = _load(tmp_path, _BASE.replace("export_date: latest", "export_date: 2026-09-19"))
    assert cfg.harvest.export_date == "2026-09-19"


def test_the_old_per_string_names_are_gone(tmp_path):
    for old in ("batch_size: 50", "items_per_minute: 90"):
        with pytest.raises(ValueError, match="unknown key"):
            _load(tmp_path, _BASE.replace("  inputs_per_request: 50", f"  inputs_per_request: 50\n  {old}"))


def test_changing_the_batch_size_never_changes_the_embedding_tag(tmp_path):
    """The batch size is how we SEND skills, not what a vector is: the stored
    vectors must stay valid when it changes."""
    base = _load(tmp_path).embed_tag
    changed = _load(tmp_path, _BASE.replace("inputs_per_request: 50", "inputs_per_request: 20")).embed_tag
    assert base == changed == "gemini-embedding-2|d768|prefix=none"


# ── the merge bands (build 4.3) ──────────────────────────────────────────────
def test_band_cutoffs_load(tmp_path):
    cfg = _load(tmp_path)
    assert (cfg.dedup_threshold, cfg.review_cosine, cfg.auto_cosine) == (0.90, 0.93, 0.96)


def test_real_config_bands_are_the_approved_ones():
    cfg = load_graph_config(DEFAULT_CONFIG_PATH)
    assert (cfg.dedup_threshold, cfg.review_cosine, cfg.auto_cosine) == (0.90, 0.93, 0.96)


@pytest.mark.parametrize(
    "old,new",
    [
        ("review_cosine: 0.93", "review_cosine: 0.85"),  # below the threshold
        ("review_cosine: 0.93", "review_cosine: 0.99"),  # above auto_cosine
        ("auto_cosine: 0.96", "auto_cosine: 1.2"),
        ("auto_cosine: 0.96", "auto_cosine: 0.92"),  # below review_cosine
        ("threshold: 0.90", "threshold: 0.95"),  # above review_cosine
    ],
)
def test_misordered_band_cutoffs_are_rejected(tmp_path, old, new):
    with pytest.raises(ValueError, match="dedup"):
        _load(tmp_path, _BASE.replace(old, new))


def test_missing_band_key_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="missing key"):
        _load(tmp_path, _BASE.replace("  auto_cosine: 0.96\n", ""))
