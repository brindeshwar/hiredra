"""Tests for hiredra/sourcing/identity.py. Cases ported from freehire's company_slug_test.go, slug_test.go,
rolefingerprint_test.go and normalizedroletitle_test.go (MIT); the two deliberate differences are marked."""

from __future__ import annotations

import pytest

from hiredra.sourcing.identity import company_slug, content_hash, role_fingerprint, slug


@pytest.mark.parametrize(
    "name, want",
    [
        ("RingCentral, Inc.", "ringcentral"),
        ("Sun Technologies,Inc.", "sun-technologies"),
        ("Hapag-Lloyd AG", "hapag-lloyd"),
        ("Booking B.V.", "booking"),
        ("Acme S.A.", "acme"),
        ("Trafalgar A/S", "trafalgar"),
        ("Acme GmbH & Co. KG", "acme"),
        ("Tiffany & Co.", "tiffany"),
        ("Acme Holdings Ltd", "acme-holdings"),
        ("Limited Brands", "limited-brands"),
        ("Limited", "limited"),
        ("Yandex", "yandex"),
        ("Kohl's", "kohls"),
        ("Domino's", "dominos"),
        ("Brink's Incorporated", "brinks"),
        ("Macy’s", "macys"),
        ("Acme ™", "acme"),
        ("Яндекс ООО", "яндекс-ооо"),  # freehire: "iandeks-ooo" (no unidecode table here, see the module docstring)
        ("", ""),
        # India's Private Limited (ours, not freehire's; build slice 2.2)
        ("Razorpay Software Pvt Ltd", "razorpay-software"),
        ("Razorpay Software Private Limited", "razorpay-software"),
        ("ALPONENT DIGITAL SOLUTIONS PVT.LTD", "alponent-digital-solutions"),
        ("Arrise Solutions (India) Pvt. Ltd.", "arrise-solutions-india"),
        ("TechUnity Software Systems India Pvt.Ltd.", "techunity-software-systems-india"),
        ("Acme Private Equity", "acme-private-equity"),  # "Private" alone is a word, not a form
        ("Private Ltd", "private"),  # never strips a name down to nothing
    ],
)
def test_company_slug(name, want):
    assert company_slug(name) == want


@pytest.mark.parametrize(
    "name, want",
    [
        ("Yandex", "yandex"),
        ("GitHub", "github"),
        ("Yandex LLC", "yandex-llc"),
        ("  Acme,  Inc. ", "acme-inc"),
        ("3M", "3m"),
        ("Köln", "koln"),
        ("Яндекс", "яндекс"),  # freehire: "iandeks"
        ("", ""),
        ("   ", ""),
        ("!!!", ""),
    ],
)
def test_slug(name, want):
    assert slug(name) == want


def fp(title="Senior Backend Engineer", description="Build things.", company="acme", countries=("IN",)):
    return role_fingerprint(company, title, description, countries)


@pytest.mark.parametrize(
    "variant",
    [
        "senior backend engineer",
        "Senior   Backend Engineer",
        "Senior Backend Engineer, Krakow",
        "Senior Backend Engineer - Remote",
        "Senior Backend Engineer — Bengaluru",
        "Senior Backend Engineer | Pune",
        "<b>Senior</b> Backend Engineer",
    ],
)
def test_fingerprint_ignores_cosmetic_title_variation_and_a_city_suffix(variant):
    assert fp(title=variant) == fp()


def test_fingerprint_keeps_what_identifies_the_role():
    assert fp(title="Junior Backend Engineer") != fp()  # a leading grade is never stripped
    assert fp(description="Build other things.") != fp()
    assert fp(company="globex") != fp()
    assert fp(title="Front-end Engineer") != fp(title="Front Engineer")  # an in-word hyphen is no separator


def test_fingerprint_keeps_a_title_the_strip_would_leave_too_short():
    assert fp(title="Engineer, Krakow") != fp(title="Engineer, Warsaw")


def test_fingerprint_reads_description_text_not_markup():
    assert fp(description="<p>Build</p><p>things.</p>") == fp(description="Build things.")
    assert fp(description="Tom &amp; Jerry") == fp(description="Tom & Jerry")
    assert fp(description="") == fp(description=None) == fp(description="<p></p>")


def test_fingerprint_separates_countries_q3():
    assert fp(countries=("GB",)) != fp(countries=("IN",))
    assert fp(countries=("IN", "US")) == fp(countries=("US", "IN"))  # a set: order is irrelevant


def test_fingerprint_fields_are_delimited():
    assert role_fingerprint("ab", "c d", "", ()) != role_fingerprint("a", "bc d", "", ())


def test_content_hash_moves_with_each_input_and_only_them():
    base = content_hash("Engineer", "Acme", "Pune", "Build things.")
    assert base == content_hash("Engineer", "Acme", "Pune", "Build things.")
    for changed in (
        content_hash("Engineer II", "Acme", "Pune", "Build things."),
        content_hash("Engineer", "Acme Inc", "Pune", "Build things."),
        content_hash("Engineer", "Acme", "Mumbai", "Build things."),
        content_hash("Engineer", "Acme", "Pune", "Build more things."),
    ):
        assert changed != base
    assert content_hash("Engineer", "Acme", None, None) == content_hash("Engineer", "Acme", "", "")
