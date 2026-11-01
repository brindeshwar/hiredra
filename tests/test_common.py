"""Tests for the shared provider helpers (build sub-slice 2a.5)."""

from __future__ import annotations

import pytest

from hiredra.sourcing.http import FetchError
from hiredra.sourcing.providers._common import (
    assert_allowed_url,
    clean,
    infer_remote,
    iso_date,
    sitemap_locs,
)

_HOSTS = frozenset({"api.lever.co"})


# ── iso_date ───────────────────────────────────────────────────────────────
def test_iso_date_epoch_ms():
    assert iso_date(1_726_000_000_000) == "2024-09-10"


def test_iso_date_iso_string():
    assert iso_date("2026-08-20T12:00:00.000Z") == "2026-08-20"
    assert iso_date("2026-08-20T12:00:00+02:00") == "2026-08-20"


def test_iso_date_bare_date():
    assert iso_date("2026-08-20") == "2026-08-20"


def test_iso_date_date_and_datetime_objects():
    import datetime as _dt

    assert iso_date(_dt.date(2026, 8, 20)) == "2026-08-20"
    assert iso_date(_dt.datetime(2026, 8, 20, 13, 30)) == "2026-08-20"


def test_iso_date_pandas_timestamp_and_nat():
    import pandas as pd

    assert iso_date(pd.Timestamp("2026-08-20T09:00:00Z")) == "2026-08-20"
    assert iso_date(pd.NaT) is None


@pytest.mark.parametrize("value", [None, "", "not a date", True, False, [], {}, "2026-13-40"])
def test_iso_date_unparseable_is_none(value):
    assert iso_date(value) is None


# ── infer_remote ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "wt, expected",
    [
        ("remote", True),
        ("Remote", True),
        ("hybrid", False),
        ("Hybrid", False),
        ("on-site", False),
        ("Onsite", False),
        ("unspecified", None),
        (None, None),
        ("something-new", None),
    ],
)
def test_infer_remote_workplace_type(wt, expected):
    assert infer_remote("New York Engineer", workplace_type=wt) is expected


def test_infer_remote_keyword_yields_true_or_none_only():
    assert infer_remote("Remote - US, Backend Engineer") is True
    assert infer_remote("work from home") is True
    assert infer_remote("New York, Backend Engineer") is None  # NOT False


def test_infer_remote_keyword_fallback_when_workplace_type_unknown():
    assert infer_remote("fully remote role", workplace_type="unspecified") is True
    assert infer_remote("office in Berlin", workplace_type="unspecified") is None


# ── assert_allowed_url ─────────────────────────────────────────────────────
def test_assert_allowed_url_accepts_good():
    assert_allowed_url("https://api.lever.co/v0/postings/x", _HOSTS, "lever")  # no raise


def test_assert_allowed_url_rejects_non_https():
    with pytest.raises(FetchError):
        assert_allowed_url("http://api.lever.co/x", _HOSTS, "lever")


def test_assert_allowed_url_rejects_bad_host():
    with pytest.raises(FetchError):
        assert_allowed_url("https://evil.example.com/x", _HOSTS, "lever")


# ── clean ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value, expected",
    [("  hi  ", "hi"), ("", None), ("   ", None), (None, None), (123, None), ([], None)],
)
def test_clean(value, expected):
    assert clean(value) == expected


# ── sitemap_locs ───────────────────────────────────────────────────────────
def test_sitemap_locs_reads_both_shapes():
    ns = 'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"'
    urlset = (
        f'\ufeff<?xml version="1.0"?><urlset {ns}><url><loc> https://a/jobs/1?x=1&amp;y=2 </loc>'
        "<lastmod>2026-10-01</lastmod></url><url><loc>https://a/jobs/2</loc></url><url/></urlset>"
    )
    assert sitemap_locs(urlset) == ([("https://a/jobs/1?x=1&y=2", "2026-10-01"), ("https://a/jobs/2", None)], [])
    index = f"<sitemapindex {ns}><sitemap><loc>https://a/s1.xml</loc></sitemap></sitemapindex>"
    assert sitemap_locs(index) == ([], ["https://a/s1.xml"])


@pytest.mark.parametrize("body", ["<html><body>maintenance</body></html>", "not xml at all"])
def test_sitemap_locs_rejects_anything_else(body):
    with pytest.raises(FetchError):
        sitemap_locs(body)
