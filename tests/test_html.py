"""Tests for the JD-body HTML -> text helper (build sub-slice 2a.4)."""

from __future__ import annotations

import pytest

from hiredra.sourcing._html import html_to_text, itemprop_html, ld_job_posting, meta_property


@pytest.mark.parametrize("value", [None, "", "   "])
def test_falsy_or_blank_returns_empty(value):
    assert html_to_text(value).strip() == ""


def test_basic_tags_stripped():
    assert html_to_text("<p>Hello <b>world</b></p>") == "Hello world"


def test_double_encoded_markup_and_entities():
    # &lt;p&gt; -> <p>, and &amp;amp; -> &amp; -> & once the tags are gone
    assert html_to_text("&lt;p&gt;Hi &amp;amp; bye&lt;/p&gt;") == "Hi & bye"


def test_escaped_angle_brackets_in_single_encoded_html_are_text():
    # real tags present: one encoding level, so "&lt;String&gt;" is the text "<String>" (Oracle, Lever, Workday)
    assert html_to_text("<p>Java: List&lt;String&gt;</p>") == "Java: List<String>"


def test_nbsp_becomes_space():
    assert html_to_text("<p>a&nbsp;b</p>") == "a b"


def test_script_and_style_removed():
    assert html_to_text("<style>x{}</style><script>evil()</script><p>Text</p>") == "Text"


def test_list_items_become_lines():
    assert html_to_text("<ul><li>one</li><li>two</li></ul>").splitlines() == ["one", "two"]


def test_br_becomes_newline():
    assert html_to_text("a<br>b<br/>c").splitlines() == ["a", "b", "c"]


def test_runs_of_blank_lines_collapse():
    assert "\n\n\n" not in html_to_text("<p>a</p><p></p><p></p><p>b</p>")


# ── ld_job_posting (freehire's jsonld.go cases) ────────────────────────────
@pytest.mark.parametrize(
    "block",
    [
        '{"@type":"JobPosting","title":"A"}',
        '[{"@type":"Organization"},{"@type":"JobPosting","title":"A"}]',
        '{"@graph":[{"@type":"WebPage"},{"@type":"jobposting","title":"A"}]}',
        '{"@type":"WebPage","mainEntity":{"@type":"JobPosting","title":"A"}}',
        '{"@type":"JobPosting","title":"A","description":"line\none"}',  # a raw newline inside a string
    ],
)
def test_ld_job_posting_finds_the_posting_in_any_shape(block):
    script = '<script type="application/ld+json">{}</script>'
    page = script.format('{"@type":"Organization"}') + script.format(block)
    assert ld_job_posting(page)["title"] == "A"


def test_ld_job_posting_none_without_one():
    assert ld_job_posting("<script type='application/ld+json'>{broken</script><p>x</p>") is None
    assert ld_job_posting(None) is None


# ── itemprop_html / meta_property (freehire's html.go cases) ───────────────
def test_itemprop_html_takes_each_element_to_its_matching_close():
    page = (
        '<div itemprop="description"><h2>Role</h2><div><p>Build it.</p></div></div>'
        '<span itemprop="title">Engineer</span><meta itemprop="title" content="x"/><div itemprop="description">open'
    )
    assert itemprop_html(page, "description") == ["<h2>Role</h2><div><p>Build it.</p></div>"]
    assert itemprop_html(page, "title") == ["Engineer"]
    assert itemprop_html(page, "missing") == [] and itemprop_html(None, "title") == []


def test_meta_property():
    page = '<meta name="x" content="no"><meta content="A &amp; B" property="og:title"><meta property="og:title" content="2">'
    assert meta_property(page, "og:title") == "A & B"
    assert meta_property(page, "og:image") is None
