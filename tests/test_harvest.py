"""Tests for hiredra.graph.harvest (build sub-slice 4.3). No network."""

from __future__ import annotations

import json

import duckdb
import httpx
import pytest

from hiredra.graph.config import HarvestConfig
from hiredra.graph.harvest import (
    _job_skills,
    export_url,
    harvest,
    latest_export_date,
    load_df,
    read_enrichment_data,
    sanity_reject_reason,
)
from hiredra.storage import add_alias, add_market_skill, connect, get_skill_vocab, init_db, iter_skill_vocab


def _cfg(**over) -> HarvestConfig:
    base = dict(
        export_base_url="https://x.test/exports",
        export_date="2026-09-19",
        ats=("ashby", "lever"),
        role_families=("software_engineering", "data"),
        min_df=3,
        max_words=4,
        max_chars=40,
    )
    base.update(over)
    return HarvestConfig(**base)


def _job(family="software_engineering", req=(), pref=()):
    return {"role_family": family, "skills_required": list(req), "skills_preferred": list(pref)}


class FakeReader:
    """url -> list of `data` dicts (or an exception to raise)."""

    def __init__(self, by_ats):
        self.by_ats = by_ats
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        for ats, payload in self.by_ats.items():
            if url.endswith(f"/{ats}.parquet"):
                if isinstance(payload, Exception):
                    raise payload
                return iter(payload)
        raise AssertionError(f"unexpected url {url}")


@pytest.fixture()
def conn():
    c = connect(":memory:")
    init_db(c)
    yield c
    c.close()


# ── per-job token extraction ─────────────────────────────────────────────────
def test_job_skills_folds_dedupes_and_ignores_junk():
    data = _job(req=["Python", " python ", "Kubernetes", 7, None, ""], pref=["PYTHON", "  React   Native "])
    assert _job_skills(data) == {"python", "kubernetes", "react native"}


def test_job_skills_tolerates_missing_and_null_lists():
    assert _job_skills({}) == set()
    assert _job_skills({"skills_required": None, "skills_preferred": None}) == set()


# ── sanity filter ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "name,expected",
    [
        ("python", None),
        ("ci/cd", None),
        ("node.js", None),
        ("distributed systems", None),
        ("a b c d", None),                       # exactly max_words
        ("a b c d e", "too_many_words"),
        ("x" * 41, "too_long"),
        ("x" * 40, None),
        ("", "empty"),
        ("2024", "no_letters"),
        ("++", "no_letters"),
        ("https://example.com/skill", "url_or_email"),
        ("www.example.com", "url_or_email"),
        ("me@example.com", "url_or_email"),
        ("http", None),                          # a real skill, not a URL
        ("https", None),
        ("http/2", None),
    ],
)
def test_sanity_reject_reason(name, expected):
    assert sanity_reject_reason(name, _cfg()) == expected


# ── the funnel ───────────────────────────────────────────────────────────────
def test_funnel_counts_and_filters(conn, tmp_path):
    reader = FakeReader({
        "ashby": [
            _job(req=["python", "kubernetes"], pref=["docker"]),
            _job(req=["Python", "kubernetes"]),
            _job(req=["python", "kubernetes", "rare-skill"]),
            _job(family="sales", req=["salesforce", "salesforce", "salesforce"]),   # wrong family: ignored
        ],
        "lever": [
            _job(family="data", req=["python", "docker", "sql"]),
            _job(req=["docker", "a b c d e f"]),                                       # too many words
            _job(req=["docker"]),
        ],
    })
    r = harvest(conn, _cfg(), reader=reader, df_path=tmp_path / "df.json")

    assert r.per_ats == {"ashby": {"enriched": 4, "in_family": 3}, "lever": {"enriched": 3, "in_family": 3}}
    assert (r.enriched_jobs, r.in_family_jobs) == (7, 6)
    # python x4, kubernetes x3, docker x4, sql x1, rare-skill x1, 'a b c d e f' x1   (salesforce never counted)
    assert r.distinct_tokens == 6
    assert r.after_min_df == 3
    assert r.candidates == 3
    assert r.df == {"docker": 4, "python": 4, "kubernetes": 3}
    assert list(r.df) == ["docker", "python", "kubernetes"]      # ordered by df desc, then name
    assert "salesforce" not in r.df
    assert r.inserted == 3 and r.already_known == 0 and r.failed == []


def test_document_frequency_counts_a_job_once_however_often_it_repeats_a_skill(conn, tmp_path):
    reader = FakeReader({
        "ashby": [_job(req=["python", "python", "Python"], pref=["python"])],   # one job, four mentions
        "lever": [],
    })
    r = harvest(conn, _cfg(min_df=1), reader=reader, df_path=tmp_path / "df.json")
    assert r.df == {"python": 1}


def test_min_df_is_the_breadth_lever(conn, tmp_path):
    jobs = [_job(req=["python"]), _job(req=["python", "go"]), _job(req=["python", "go", "rust"])]
    reader = FakeReader({"ashby": jobs, "lever": []})
    assert set(harvest(conn, _cfg(min_df=1), reader=reader, dry_run=True, df_path=None).df) == {"python", "go", "rust"}
    reader2 = FakeReader({"ashby": jobs, "lever": []})
    assert set(harvest(conn, _cfg(min_df=2), reader=reader2, dry_run=True, df_path=None).df) == {"python", "go"}
    reader3 = FakeReader({"ashby": jobs, "lever": []})
    assert set(harvest(conn, _cfg(min_df=3), reader=reader3, dry_run=True, df_path=None).df) == {"python"}


def test_role_families_are_the_other_breadth_lever(conn, tmp_path):
    reader = FakeReader({"ashby": [_job(family="ml_ai", req=["pytorch"])] * 3, "lever": []})
    assert harvest(conn, _cfg(), reader=reader, dry_run=True, df_path=None).df == {}   # ml_ai not in the cfg
    reader = FakeReader({"ashby": [_job(family="ml_ai", req=["pytorch"])] * 3, "lever": []})
    wider = _cfg(role_families=("software_engineering", "ml_ai"))
    assert harvest(conn, wider, reader=reader, dry_run=True, df_path=None).df == {"pytorch": 3}


def test_sanity_rejections_are_reported_with_samples(conn, tmp_path):
    long_phrase = "one two three four five"
    reader = FakeReader({"ashby": [_job(req=[long_phrase, "2024"])] * 3, "lever": []})
    r = harvest(conn, _cfg(), reader=reader, df_path=tmp_path / "df.json")
    assert r.after_min_df == 2 and r.candidates == 0
    assert r.rejected_by_sanity == {"too_many_words": 1, "no_letters": 1}
    assert r.rejected_samples["too_many_words"] == [long_phrase]


# ── writes ───────────────────────────────────────────────────────────────────
def test_survivors_become_unseen_market_nodes_and_df_sidecar_is_written(conn, tmp_path):
    reader = FakeReader({"ashby": [_job(req=["python", "docker"])] * 3, "lever": [_job(req=["python"])]})
    df_path = tmp_path / "sub" / "market_df.json"
    r = harvest(conn, _cfg(), reader=reader, df_path=df_path)

    py = get_skill_vocab(conn, "python")
    assert py["source"] == "market" and py["seen_count"] == 0 and py["vector"] is None
    assert {row["name"] for row in iter_skill_vocab(conn)} == {"python", "docker"}
    saved = json.loads(df_path.read_text(encoding="utf-8"))
    assert saved == {"export_date": "2026-09-19", "df": {"python": 4, "docker": 3}}
    assert load_df(df_path) == {"python": 4, "docker": 3}
    assert r.inserted == 2


def test_existing_nodes_and_merged_aliases_are_not_touched_or_resurrected(conn, tmp_path):
    add_market_skill(conn, "python")
    conn.execute("UPDATE skill_vocab SET seen_count = 9, source = 'jd' WHERE name = 'python'")
    add_market_skill(conn, "fastapi")
    add_market_skill(conn, "fast api")
    conn.commit()
    from hiredra.storage import merge_skill
    merge_skill(conn, "fast api", "fastapi")

    reader = FakeReader({"ashby": [_job(req=["python", "fast api", "docker"])] * 3, "lever": []})
    r = harvest(conn, _cfg(), reader=reader, df_path=tmp_path / "df.json")

    assert r.inserted == 1 and r.already_known == 2          # docker new; python node + 'fast api' alias known
    assert get_skill_vocab(conn, "python")["seen_count"] == 9 and get_skill_vocab(conn, "python")["source"] == "jd"
    assert get_skill_vocab(conn, "fast api") is None          # alias not resurrected as a node


def test_dry_run_writes_nothing(conn, tmp_path):
    reader = FakeReader({"ashby": [_job(req=["python"])] * 3, "lever": []})
    df_path = tmp_path / "df.json"
    r = harvest(conn, _cfg(), reader=reader, df_path=df_path, dry_run=True)
    assert r.candidates == 1 and r.inserted == 0 and r.dry_run
    assert iter_skill_vocab(conn) == []
    assert not df_path.exists()


def test_rerunning_is_idempotent(conn, tmp_path):
    def run():
        reader = FakeReader({"ashby": [_job(req=["python"])] * 3, "lever": []})
        return harvest(conn, _cfg(), reader=reader, df_path=tmp_path / "df.json")

    assert run().inserted == 1
    second = run()
    assert second.inserted == 0 and second.already_known == 1
    assert len(iter_skill_vocab(conn)) == 1


def _run(conn, tmp_path, reader, **kw):
    kw.setdefault("retry_wait_s", 0.0)
    return harvest(conn, _cfg(), reader=reader, df_path=tmp_path / "df.json", **kw)


def test_a_failing_file_is_reported_and_others_still_count(conn, tmp_path):
    reader = FakeReader({"ashby": RuntimeError("HTTP 503 from worker"), "lever": [_job(req=["python"])] * 3})
    r = _run(conn, tmp_path, reader)
    assert r.failed == ["ashby"]
    assert "RuntimeError" in r.per_ats["ashby"]["error"] and "503" in r.per_ats["ashby"]["error"]
    assert r.per_ats["ashby"]["attempts"] == 3
    assert r.per_ats["lever"] == {"enriched": 3, "in_family": 3}
    assert r.df == {"python": 3}                       # computed from what did read...


def test_any_failed_file_means_nothing_is_written_by_default(conn, tmp_path):
    """A half-harvest is harder to reason about than a re-run: df over a subset of
    the files is not what min_df was tuned for."""
    reader = FakeReader({"ashby": RuntimeError("timeout"), "lever": [_job(req=["python"])] * 3})
    r = _run(conn, tmp_path, reader)
    assert r.written is False and r.inserted == 0 and "ashby" in r.not_written_reason
    assert iter_skill_vocab(conn) == [] and not (tmp_path / "df.json").exists()


def test_allow_partial_writes_what_did_read(conn, tmp_path):
    reader = FakeReader({"ashby": RuntimeError("timeout"), "lever": [_job(req=["python"])] * 3})
    r = _run(conn, tmp_path, reader, allow_partial=True)
    assert r.written is True and r.inserted == 1 and r.failed == ["ashby"]
    assert get_skill_vocab(conn, "python") is not None


def test_a_file_that_fails_midway_contributes_nothing(conn, tmp_path):
    def flaky(url):
        if url.endswith("/ashby.parquet"):
            def rows():
                yield _job(req=["python", "go"])
                yield _job(req=["python", "go"])
                yield _job(req=["python", "go"])
                raise ConnectionError("connection reset half way through the file")
            return rows()
        return iter([_job(req=["rust"])] * 3)

    r = _run(conn, tmp_path, flaky, attempts=1)
    assert r.failed == ["ashby"]
    assert r.df == {"rust": 3}                          # NOT python/go: the 3 rows read before the reset are discarded
    assert (r.enriched_jobs, r.in_family_jobs) == (3, 3)


def test_a_transient_failure_is_retried_and_counted_once(conn, tmp_path):
    calls = {"ashby": 0}

    def reader(url):
        if url.endswith("/ashby.parquet"):
            calls["ashby"] += 1
            if calls["ashby"] == 1:
                raise TimeoutError("Timeout was reached error for HTTP GET")
            return iter([_job(req=["python"])] * 3)
        return iter([])

    slept: list[float] = []
    lines: list[str] = []
    r = harvest(conn, _cfg(), reader=reader, df_path=tmp_path / "df.json", retry_wait_s=7.0,
                sleep=slept.append, log=lines.append)
    assert r.failed == [] and r.written and calls["ashby"] == 2
    assert r.df == {"python": 3}                        # not doubled by the retry
    assert slept == [7.0] and any("attempt 2/3" in ln for ln in lines)


def test_it_gives_up_after_the_configured_attempts(conn, tmp_path):
    calls: list[str] = []

    def reader(url):
        calls.append(url)
        raise TimeoutError("nope")

    slept: list[float] = []
    r = harvest(conn, _cfg(ats=("ashby",)), reader=reader, df_path=tmp_path / "df.json",
                attempts=4, retry_wait_s=2.0, sleep=slept.append)
    assert len(calls) == 4 and slept == [2.0, 2.0, 2.0]   # no sleep after the final attempt
    assert r.failed == ["ashby"] and r.per_ats["ashby"]["attempts"] == 4


# ── export discovery ─────────────────────────────────────────────────────────
def _client(payload, status=200):
    return httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(status, json=payload)))


def test_latest_export_date_picks_the_newest_dated_directory():
    payload = {"dirs": ["2026-09-18/", "2026-09-19/", "scratch/"], "files": [{"file": "open-jobs-latest.tar"}]}
    assert latest_export_date("https://x.test/exports", client=_client(payload)) == "2026-09-19"


def test_latest_export_date_errors_when_nothing_is_dated():
    with pytest.raises(RuntimeError, match="no dated export"):
        latest_export_date("https://x.test/exports", client=_client({"dirs": ["scratch/"], "files": []}))


def test_latest_export_date_raises_on_http_error():
    with pytest.raises(httpx.HTTPStatusError):
        latest_export_date("https://x.test/exports", client=_client({}, status=503))


def test_export_date_latest_is_resolved_through_the_worker_listing(conn, tmp_path):
    reader = FakeReader({"ashby": [_job(req=["python"])] * 3, "lever": []})
    client = _client({"dirs": ["2026-09-18/", "2026-09-20/"]})
    r = harvest(conn, _cfg(export_date="latest"), reader=reader, client=client, df_path=tmp_path / "df.json")
    assert r.export_date == "2026-09-20"
    assert reader.urls[0] == "https://x.test/exports/2026-09-20/jobs/ashby.parquet"


def test_an_explicit_export_date_skips_the_listing(conn, tmp_path):
    reader = FakeReader({"ashby": [], "lever": []})
    r = harvest(conn, _cfg(), reader=reader, df_path=tmp_path / "df.json")   # no client passed: would fail if it listed
    assert r.export_date == "2026-09-19"


def test_export_url_shape():
    assert export_url(_cfg(), "2026-09-19", "ashby") == "https://x.test/exports/2026-09-19/jobs/ashby.parquet"


def test_load_df_returns_empty_when_no_harvest_has_run(tmp_path):
    assert load_df(tmp_path / "missing.json") == {}


# ── the real DuckDB reader, against a local parquet file ─────────────────────
def _write_parquet(path, rows):
    con = duckdb.connect()
    con.execute("CREATE TABLE t (id INTEGER, title VARCHAR, enrichment_json VARCHAR)")
    for i, raw in enumerate(rows):
        con.execute("INSERT INTO t VALUES (?, ?, ?)", [i, f"job {i}", raw])
    con.execute(f"COPY t TO '{path.as_posix()}' (FORMAT PARQUET)")
    con.close()


def test_read_enrichment_data_streams_only_enriched_rows_with_a_data_object(tmp_path):
    good = json.dumps({"schema_version": 1, "data": {"role_family": "data", "skills_required": ["sql"]}})
    p = tmp_path / "x.parquet"
    _write_parquet(p, [
        good,
        None,                                              # not enriched -> filtered in SQL
        "{not valid json",                                 # malformed -> skipped
        json.dumps({"schema_version": 1}),                 # no `data` -> skipped
        json.dumps({"data": "not-a-dict"}),                # wrong shape -> skipped
        json.dumps({"data": {"role_family": "data", "skills_required": ["python"]}}),
    ])
    out = list(read_enrichment_data(str(p)))
    assert out == [
        {"role_family": "data", "skills_required": ["sql"]},
        {"role_family": "data", "skills_required": ["python"]},
    ]


def test_harvest_end_to_end_through_the_real_reader_on_a_local_file(conn, tmp_path):
    rows = [json.dumps({"data": _job(req=["python", "kubernetes"])}) for _ in range(3)]
    for name in ("ashby", "lever"):
        (tmp_path / "2026-09-19" / "jobs").mkdir(parents=True, exist_ok=True)
        _write_parquet(tmp_path / "2026-09-19" / "jobs" / f"{name}.parquet", rows if name == "ashby" else [])
    cfg = _cfg(export_base_url=(tmp_path).as_posix())
    r = harvest(conn, cfg, df_path=tmp_path / "df.json")          # default reader = the real duckdb one
    assert r.df == {"kubernetes": 3, "python": 3} and r.inserted == 2
