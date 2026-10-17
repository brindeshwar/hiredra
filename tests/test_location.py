"""Tests for hiredra.location (ported from open-jobs' locparse.py, CC0-1.0)
and hiredra.filter.passes_location -- the four India-eligibility rules plus
the unknown-location "never guess" case."""

from __future__ import annotations

from hiredra import filter as funnel
from hiredra.location import eligibility, parse

CONFIG = funnel.load_filter_config()


# --- the four required rules, against eligibility("India", ...) directly ---
def test_india_city_is_eligible():
    ok, reason = eligibility("India", "Bengaluru, India")
    assert ok is True and "matches" in reason


def test_india_country_name_alone_is_eligible():
    ok, _ = eligibility("India", "India")
    assert ok is True


def test_worldwide_remote_with_no_restriction_is_eligible():
    ok, reason = eligibility("India", "Remote")
    assert ok is True and "no stated region" in reason


def test_onsite_outside_india_is_not_eligible():
    ok, reason = eligibility("India", "London, United Kingdom")
    assert ok is False and "GB" in reason


def test_remote_restricted_to_another_country_is_not_eligible():
    ok, reason = eligibility("India", "Remote (US)")
    assert ok is False and "US" in reason


def test_remote_restricted_via_jd_text_is_not_eligible():
    ok, reason = eligibility("India", "Remote", "This role is US-based only. Must be located in the United States.")
    assert ok is False and "US" in reason


def test_no_location_signal_at_all_is_unknown():
    ok, reason = eligibility("India", "")
    assert ok is None and "no location" in reason


# --- parse() sanity: India cities resolve, remote/onsite detected ----------
def test_parse_recognizes_common_india_cities():
    for city in ("Bengaluru", "Bangalore", "Hyderabad", "Pune", "Mumbai", "Gurugram"):
        p = parse(city)
        assert "IN" in p["countries"], city


def test_parse_bare_remote_is_remote_no_countries():
    p = parse("Remote")
    assert p["remote"] == "remote" and p["countries"] == []


def test_parse_splits_ashby_middle_dot_joined_locations():
    # Hiredra's own Ashby provider joins multiple locations with " · " (middle
    # dot, U+00B7) -- see the location-format-not-uniform memory note. The
    # upstream "•" (bullet) alone in _split() missed this; real bug, found via
    # the max-scale sourcing test (a US-onsite Ashby posting reading as
    # "unknown location" and slipping through the funnel).
    p = parse("San Francisco · New York · United States")
    assert p["countries"] == ["US"]


def test_eligibility_drops_ashby_style_us_only_posting():
    ok, reason = eligibility("India", "San Francisco · New York · United States")
    assert ok is False and "US" in reason


# --- two cheap US-parse fixes found via the max-scale sourcing test ---------
def test_remote_us_with_trailing_period_is_restricted():
    # over-eager punctuation stripping turned "u.s." into "u.s", missing the
    # dictionary key; "Remote U.S." read as unknown instead of US-restricted.
    ok, reason = eligibility("India", "Remote U.S.")
    assert ok is False and "US" in reason


def test_uppercase_or_splits_like_lowercase_or():
    # real posting text: "Beijing OR Shanghai" -- the split wasn't case-insensitive.
    p = parse("Beijing OR Shanghai")
    assert p["countries"] == ["CN"]
    assert set(p["cities"]) == {"Beijing", "Shanghai"}


def test_uppercase_and_splits_like_lowercase_and():
    p = parse("Bengaluru AND Mumbai")
    assert p["countries"] == ["IN"]


# --- hiredra.filter.passes_location wiring ----------------------------------
def test_passes_location_keeps_india():
    r = funnel.passes_location("Backend Engineer", "Bengaluru, India", "", CONFIG)
    assert r.keep and r.reason is None


def test_passes_location_keeps_unrestricted_remote():
    r = funnel.passes_location("Backend Engineer", "Remote", "", CONFIG)
    assert r.keep and r.reason is None  # a definite match ("no stated region"), not "unknown"


def test_passes_location_drops_onsite_outside_india():
    r = funnel.passes_location("Backend Engineer", "London, United Kingdom", "", CONFIG)
    assert not r.keep and "GB" in r.reason


def test_passes_location_drops_us_restricted_remote():
    r = funnel.passes_location("Backend Engineer", "Remote (US)", "", CONFIG)
    assert not r.keep and "US" in r.reason


def test_passes_location_keeps_and_flags_unknown_when_no_signal():
    r = funnel.passes_location("Backend Engineer", None, None, CONFIG)
    assert r.keep and r.reason == "unknown" and r.note == "location: unknown"


def test_passes_location_keeps_and_notes_hybrid_with_no_place():
    # the owner wants these retained and distinguishable, not dropped and not
    # silently folded into the ordinary "unknown" bucket.
    r = funnel.passes_location("Backend Engineer", "Hybrid", "", CONFIG)
    assert r.keep and r.reason == "unknown"
    assert r.note == "location: hybrid, place unknown"


def test_passes_location_keeps_and_notes_in_office_with_no_place():
    r = funnel.passes_location("Backend Engineer", "In-Office", "", CONFIG)
    assert r.keep and r.reason == "unknown"
    assert r.note == "location: onsite, place unknown"


def test_passes_location_no_note_when_hybrid_names_a_place():
    # the flag is specifically for "we know it's not remote but not where" --
    # a stated place (even outside India, even if later dropped) isn't that.
    r = funnel.passes_location("Backend Engineer", "Hybrid - Prague", "", CONFIG)
    assert r.note is None


# --- wiring into run_funnel: location runs first, unknown_location counted -
def test_run_funnel_location_stage_runs_first_and_counts_unknown(monkeypatch):
    from hiredra.storage import connect, get_job, init_db, upsert_job

    conn = connect(":memory:")
    init_db(conn)
    upsert_job(
        conn,
        {
            "job_id": "in1",
            "url": "u1",
            "title": "Backend Engineer",
            "company": "Acme",
            "source": "greenhouse",
            "location": "Bengaluru, India",
            "description": "No years stated.",
        },
    )
    upsert_job(
        conn,
        {
            "job_id": "us1",
            "url": "u2",
            "title": "Backend Engineer",
            "company": "Acme",
            "source": "greenhouse",
            "location": "Remote (US)",
            "description": "No years stated.",
        },
    )
    upsert_job(
        conn,
        {
            "job_id": "unk1",
            "url": "u3",
            "title": "Backend Engineer",
            "company": "Acme",
            "source": "greenhouse",
            "location": None,
            "description": "No years stated.",
        },
    )

    stats = funnel.run_funnel(conn, CONFIG)

    assert stats.seen == 3
    assert stats.dropped_location == 1
    assert stats.unknown_location == 1
    assert stats.kept == 2  # the India job and the unknown-location job both survive to 'filtered'

    us_job = get_job(conn, "us1")
    assert us_job["status"] == "dropped" and us_job["status_detail"].startswith("location:")
    # dropped on location BEFORE the seniority gate ever ran -- years_required stays unset
    assert us_job["years_required"] is None

    conn.close()


def test_run_funnel_flags_onsite_unknown_place_via_job_events():
    from hiredra.storage import connect, get_events, init_db, upsert_job

    conn = connect(":memory:")
    init_db(conn)
    upsert_job(
        conn,
        {
            "job_id": "hyb1",
            "url": "u1",
            "title": "Backend Engineer",
            "company": "Acme",
            "source": "ashby",
            "location": "Hybrid",
            "description": "No years stated.",
        },
    )
    # a genuinely place-less AND arrangement-less unknown -- real bug found via the
    # discovery harvest run: onsite_unknown_place was incrementing for every
    # "location: unknown" note, not just this onsite/hybrid sub-case, because
    # both share a truthy `note` field.
    upsert_job(
        conn,
        {
            "job_id": "unk1",
            "url": "u2",
            "title": "Backend Engineer",
            "company": "Acme",
            "source": "ashby",
            "location": None,
            "description": "No years stated.",
        },
    )

    stats = funnel.run_funnel(conn, CONFIG)

    assert stats.unknown_location == 2
    assert stats.onsite_unknown_place == 1  # only the hybrid one, not the generic unknown too
    assert stats.kept == 2  # both retained, not dropped

    events = [e for e in get_events(conn, "hyb1") if e["kind"] == "note"]
    assert len(events) == 1
    assert events[0]["detail"] == "location: hybrid, place unknown"

    events2 = [e for e in get_events(conn, "unk1") if e["kind"] == "note"]
    assert len(events2) == 1
    assert events2[0]["detail"] == "location: unknown"

    conn.close()
