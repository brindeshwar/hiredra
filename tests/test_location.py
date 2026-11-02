"""Tests for hiredra.location (ported from open-jobs' locparse.py, CC0-1.0)
and hiredra.filter.passes_location -- the four India-eligibility rules plus
the unknown-location "never guess" case."""

from __future__ import annotations

import json
from pathlib import Path

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
def test_run_funnel_location_stage_runs_first_and_counts_unknown(monkeypatch, pg_url):
    from hiredra.storage import connect, get_job, init_db, upsert_job

    conn = connect(pg_url)
    init_db(conn)
    upsert_job(
        conn,
        {
            "job_id": "greenhouse:in1",
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
            "job_id": "greenhouse:us1",
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
            "job_id": "greenhouse:unk1",
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

    us_job = get_job(conn, "greenhouse:us1")
    assert us_job["status"] == "dropped" and us_job["status_detail"].startswith("location:")
    # dropped on location BEFORE the seniority gate ever ran -- years_required stays unset
    assert us_job["years_required"] is None

    conn.close()


def test_run_funnel_flags_onsite_unknown_place_via_job_events(pg_url):
    from hiredra.storage import connect, get_events, init_db, upsert_job

    conn = connect(pg_url)
    init_db(conn)
    upsert_job(
        conn,
        {
            "job_id": "ashby:hyb1",
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
            "job_id": "ashby:unk1",
            "url": "u2",
            "title": "Backend Engineer",
            "company": "Acme",
            "source": "ashby",
            "location": None,
            "description": "No years stated here either.",  # its own role: one text is one role (D6.3)
        },
    )

    stats = funnel.run_funnel(conn, CONFIG)

    assert stats.unknown_location == 2
    assert stats.onsite_unknown_place == 1  # only the hybrid one, not the generic unknown too
    assert stats.kept == 2  # both retained, not dropped

    events = [e for e in get_events(conn, "ashby:hyb1") if e["kind"] == "note"]
    assert len(events) == 1
    assert events[0]["detail"] == "location: hybrid, place unknown"

    events2 = [e for e in get_events(conn, "ashby:unk1") if e["kind"] == "note"]
    assert len(events2) == 1
    assert events2[0]["detail"] == "location: unknown"

    conn.close()


# --- Hiredra additions (M0.4 finding): workplace words, flag emoji, Washington D.C. ---
import pytest  # noqa: E402


@pytest.mark.parametrize(
    ("location", "country"),
    [
        ("San Francisco Office", "US"),
        ("New York HQ", "US"),
        ("New York, NY (HQ)", "US"),
        ("Palo Alto Office", "US"),
        ("Emeryville HQ", "US"),
        ("London office", "GB"),
        ("Zurich HQ", "CH"),
        ("Sofia Office", "BG"),
        ("Lisbon Offices", "PT"),
        ("Berlin Headquarters", "DE"),
        ("Bengaluru Office", "IN"),
        ("Pune Campus", "IN"),
        ("Hyderabad Hub", "IN"),
        ("SF Bay Area", "US"),
        ("San Francisco Bay Area", "US"),
    ],
)
def test_a_trailing_workplace_word_no_longer_hides_the_city(location, country):
    assert parse(location)["countries"] == [country]


def test_workplace_cities_get_the_right_india_verdict():
    assert eligibility("India", "Bengaluru Office")[0] is True
    assert eligibility("India", "San Francisco Office") == (False, "restricted to US")


@pytest.mark.parametrize("location", ["HQ", "Head Office", "Office", "Main Campus"])
def test_a_workplace_word_alone_stays_unknown(location):
    assert parse(location)["countries"] == []
    assert eligibility("India", location)[0] is None  # never guessed: kept as unknown by the funnel


def test_washington_dc_records_the_country():
    assert parse("Washington, D.C.")["countries"] == ["US"]
    assert parse("Washington, DC")["countries"] == ["US"]
    assert eligibility("India", "Washington, D.C.")[0] is False


@pytest.mark.parametrize(
    ("location", "countries"),
    [
        ("Lyon \U0001f1eb\U0001f1f7", ["FR"]),
        ("Berlin \U0001f1e9\U0001f1ea, Remote", ["DE"]),
        ("\U0001f1ec\U0001f1e7", ["GB"]),
    ],
)
def test_a_flag_emoji_is_an_explicit_country(location, countries):
    assert parse(location)["countries"] == countries


def test_a_flag_beats_a_city_inferred_elsewhere():
    # Explicit beats inferred, as for a written country: "Paris" alone infers FR, the flag says US.
    assert parse("Paris \U0001f1fa\U0001f1f8")["countries"] == ["US"]


def test_a_workplace_word_must_be_its_own_word():
    assert parse("Parishq")["countries"] == []  # not "Paris" + "hq"


def test_the_flag_is_removed_from_the_city_name():
    assert parse("Lyon \U0001f1eb\U0001f1f7")["cities"] == ["Lyon"]
    assert parse("\U0001f1ec\U0001f1e7")["cities"] == []


# --- recognition fixes (2026-10-27, docs/freehire-merge/04-verification.md sec 7) -------------------
# Every location below is a real string from our database. The first group were Indian jobs dropped as
# "restricted to US": a 2-letter "IN" was always read as Indiana.
@pytest.mark.parametrize(
    "location",
    ["KA, IN", "MH, IN", "TN, IN", "GA, IN", "Remote, IN", "Bangalore, IN", "Pune IN", "IN - Jaipur", "IN-Pune",
     "Mumbai, MH", "Chennai, TN", "Chennai, TN, India", "Indore MP", "Mhesana , Gujrat", "Kozhikode, Kerala"],
)  # fmt: skip
def test_indian_locations_with_codes_or_states_are_india(location):
    assert parse(location)["countries"] == ["IN"]
    assert eligibility("India", location)[0] is True


@pytest.mark.parametrize(
    ("location", "countries"),
    [
        ("Indianapolis, IN", ["US"]),  # a known US city confirms the state
        ("Athens, GA", ["US"]),  # a city of a third country: US towns reuse foreign names
        ("Munster, IN", ["US"]),
        ("Abernathy, TX · New Lebanon, IN · United States", ["US"]),  # the line names the US
        ("Tel Aviv, IL", ["IL"]),  # the city is in the clashing country
        ("Berlin, DE", ["DE"]),
        ("Bogota, CO", ["CO"]),
        ("Lima, PE", ["PE"]),  # PE also Prince Edward Island
        ("Pittsburgh, PA", ["US"]),
        ("Brisbane, CA", ["US"]),  # CA stays California
        ("Belo Horizonte, MG", ["BR"]),  # a local code after a city of another country: the city decides
        ("Adelaide, SA", ["AU"]),
        ("Whitehorse, YT", ["CA"]),
        ("IL-Cupertino", ["US"]),
        ("TX-Houston", ["US"]),
    ],
)
def test_two_letter_codes_that_clash(location, countries):
    assert parse(location)["countries"] == countries


@pytest.mark.parametrize(
    ("location", "countries"),
    [
        ("Massachusetts - Boston", ["US"]),  # a spaced dash always separates
        ("London - The River Building HQ", ["GB"]),
        ("San Francisco-HQ", ["US"]),
        ("SF Office", ["US"]),
        ("Olathe KS", ["US"]),  # a trailing state code inside a token
        ("Rancho Bernardo, CA.", ["US"]),
        ("London SW1A 2HB", ["GB"]),  # a full UK postcode
        ("Durham, North East England, DH1 3HU", ["GB"]),
        ("Elmshorn, Schleswig-Holstein", ["DE"]),  # a region name states its country
        ("Belgrade", ["RS"]),  # freehire's curated names
        ("Bristol", ["GB"]),
        ("Ludhiana", ["IN"]),  # GeoNames, the last resort
        ("Düsseldorf", ["DE"]),
        ("г Москва", ["RU"]),
    ],
)
def test_places_the_tokenizer_now_reads(location, countries):
    assert parse(location)["countries"] == countries


@pytest.mark.parametrize(
    "location",
    ["In-Office", "In Office", "Field - Based", "Home", "Nationwide", "B2 2nd floor", "Valencia, Venezuela or Spain"],
)
def test_words_and_stubs_are_not_places(location):
    assert "US" not in parse(location)["countries"]


def test_a_country_code_must_be_upper_case():
    assert parse("NYC or Palo Alto (in office)")["countries"] == ["US"]  # "in" is not India
    assert parse("na")["countries"] == []


@pytest.mark.parametrize(
    "location", ["Home based - EMEA", "Home Based - Americas", "Latin America", "Remote: MENA", "Remote NA"]
)
def test_regions_that_exclude_india_drop(location):
    assert eligibility("India", location)[0] is False


@pytest.mark.parametrize(("location", "ok"), [("Home Based - APAC", True), ("Americas; APAC", True)])
def test_regions_that_include_india_keep(location, ok):
    assert eligibility("India", location)[0] is ok


def test_a_contested_city_with_no_curated_answer_states_nothing():
    assert parse("Cambridge")["countries"] == []  # GB and US both claim it


def test_geonames_cities_must_agree():
    assert parse("Colorado Springs, Eden Prairie")["countries"] == ["US"]


# freehire's RestrictionFromTitle cases (title_restriction_test.go, MIT) plus a dash suffix of our own.
@pytest.mark.parametrize(
    ("title", "ok"),
    [
        ("Senior Back End Engineer [Remote-US]", False),
        ("Senior Backend Engineer (Go) (Location - Australia or New Zealand)", False),
        ("Data Engineer - US Remote", False),
        ("SecOps Engineer - North Central region (Remote in the U.S.)", False),
        ("Backend Engineer [Remote-India]", True),
        ("Frontend Engineer (React)", True),  # no anchor word
        ("Product Manager (Contract)", True),
        ("Engineer [Remote-EMEA-ish]", True),  # anchor but no place
        ("Generative AI Analyst | Simplified Chinese (US)", True),  # "(US)" without "remote" is not a restriction
        ("", True),
    ],
)
def test_a_restriction_in_the_title_counts_when_the_location_is_bare(title, ok):
    assert eligibility("India", "Remote", "", title)[0] is ok


@pytest.mark.parametrize(
    ("location", "ok"),
    [
        ("USA Only", False),  # was unknown, so kept: "only" made the segment match nothing
        ("Remote - US Only", False),
        ("Remote (US-only)", False),
        ("Remote, California Only", False),
        ("Remote - Europe Only", False),
        ("Remote (US + Canada Only)", False),  # a spaced "+" separates, like "&"
        ("Remote (India + Singapore)", True),
        ("Remote (India - Bangalore ONLY)", True),
        ("India only", True),
    ],
)
def test_the_word_only_does_not_hide_the_place(location, ok):
    assert eligibility("India", location, "", "Engineer")[0] is ok


@pytest.mark.parametrize(
    "location", ["Remote (EU)", "EU - Remote", "Remote-EU", "EU", "European Union", "Remote - European Union"]
)
def test_eu_is_europe(location):
    assert eligibility("India", location, "", "Engineer")[0] is False
    assert eligibility("India", "Remote (EU or India)", "", "Engineer")[0] is True


def test_the_title_never_overrides_a_stated_location():
    assert eligibility("India", "Bengaluru, India", "", "Backend Engineer [Remote-US]")[0] is True


_FREEHIRE_CASES = json.loads((Path(__file__).parent / "fixtures" / "freehire_location_cases.json").read_text("utf-8"))


@pytest.mark.parametrize("case", _FREEHIRE_CASES["cases"], ids=lambda c: c["location"])
def test_freehire_parse_cases(case):
    # freehire's own Parse cases, countries only. Where we deliberately differ the fixture records our answer
    # and why ("hiredra_countries", "why").
    assert parse(case["location"])["countries"] == case.get("hiredra_countries", case["countries"])
