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
limits:
  requests_per_minute: 100
  tokens_per_minute: 30000
  requests_per_day: 1000
  headroom: 0.90
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
    cfg = load_graph_config(DEFAULT_CONFIG_PATH)
    assert cfg.model == "gemini-embedding-2"
    assert cfg.dimensions == 768
    assert cfg.prefix == ""
    assert cfg.dedup_threshold == 0.90
    assert cfg.adj < cfg.have


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
        ("requests_per_minute: 100", "requests_per_minute: 0"),
        ("tokens_per_minute: 30000", "tokens_per_minute: 0"),
        ("requests_per_day: 1000", "requests_per_day: 0"),
        ("headroom: 0.90", "headroom: 0"),
        ("headroom: 0.90", "headroom: 1.5"),
        ("requests_per_minute: 100", "requests_per_minute: 50"),  # 45/min budget < 50 inputs per request
        ("requests_per_day: 1000", "requests_per_day: 80"),  # 72/day budget < 90/min budget
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


# ── the real Google limits (build 4.3 fix) ───────────────────────────────────
def test_limits_load_and_budgets_apply_the_headroom(tmp_path):
    cfg = _load(tmp_path)
    assert (cfg.requests_per_minute, cfg.tokens_per_minute, cfg.requests_per_day) == (100, 30000, 1000)
    assert cfg.inputs_per_request == 50
    assert (cfg.rpm_budget, cfg.tpm_budget, cfg.rpd_budget) == (90, 27000, 900)  # 0.90 headroom on each


def test_real_config_matches_the_dashboard_numbers():
    cfg = load_graph_config(DEFAULT_CONFIG_PATH)
    assert (cfg.requests_per_minute, cfg.tokens_per_minute, cfg.requests_per_day) == (100, 30_000, 1_000)
    assert cfg.inputs_per_request <= 100 and cfg.inputs_per_request <= cfg.rpm_budget


def test_the_old_per_string_names_are_gone(tmp_path):
    for old in ("batch_size: 50", "items_per_minute: 90"):
        with pytest.raises(ValueError, match="unknown key"):
            _load(tmp_path, _BASE.replace("  inputs_per_request: 50", f"  inputs_per_request: 50\n  {old}"))


def test_changing_rate_settings_never_changes_the_embedding_tag(tmp_path):
    """Rate limits are how we SEND requests, not what a vector is: the 950 stored
    vectors must stay valid when a limit or the chunk size changes."""
    base = _load(tmp_path).embed_tag
    changed = _load(
        tmp_path,
        _BASE.replace("requests_per_minute: 100", "requests_per_minute: 60")
        .replace("inputs_per_request: 50", "inputs_per_request: 20")
        .replace("headroom: 0.90", "headroom: 0.5"),
    ).embed_tag
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
