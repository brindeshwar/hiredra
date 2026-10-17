"""Tests for hiredra.profile (plan unit PF1): every rule of the frozen Skills grammar, each with
its own failing fixture and the line number it must name, plus the hashes and Last updated."""

from __future__ import annotations

import pytest

from hiredra.profile import LEVELS, ProfileError, parse_skills, skills_hash

GOOD = """# Owner profile
**Last updated:** 2026-10-16

## Summary
Prose that is never parsed: python, ## Skills look-alike text, **Weird:**

## Skills (what I can defend in an interview)

**Strong (production use):**
python, fastapi, postgresql

**Working (built with it):**
docker, scikit-learn, c++, node.js

**Familiar (learned, not shipped):**
rust, c#

Explicitly NOT claimed: kubernetes in production

## Projects
More prose.
"""


def _write(tmp_path, text, name="profile.md"):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def _errors(tmp_path, text) -> str:
    with pytest.raises(ProfileError) as info:
        parse_skills(_write(tmp_path, text))
    return str(info.value)


def _section(body: str) -> str:
    """A file whose Skills header is line 3, so the body starts at line 4."""
    return "# P\nLast updated: today\n## Skills\n" + body


def test_the_good_profile_parses(tmp_path):
    p = parse_skills(_write(tmp_path, GOOD))
    assert p.tiers == {
        "strong": ("python", "fastapi", "postgresql"),
        "working": ("docker", "scikit-learn", "c++", "node.js"),
        "familiar": ("rust", "c#"),
    }
    assert p.tokens()[0] == ("python", "strong") and p.tokens()[-1] == ("c#", "familiar")
    assert p.last_updated == "2026-10-16" and p.warnings == ()
    assert len(p.skills_hash) == 64 and len(p.file_hash) == 64
    assert LEVELS == {"strong": "production", "working": "built-with", "familiar": "learned"}


def test_tiers_in_any_order_and_the_section_ends_at_eof(tmp_path):
    p = parse_skills(_write(tmp_path, _section("**Familiar:**\nrust\n**Strong:**\npython\n**Working:**\ndocker\n")))
    assert p.tiers == {"strong": ("python",), "working": ("docker",), "familiar": ("rust",)}


def test_skills_hash_ignores_token_order_and_prose_but_file_hash_does_not(tmp_path):
    a = parse_skills(_write(tmp_path, GOOD, "a.md"))
    reordered = GOOD.replace("python, fastapi, postgresql", "postgresql, python, fastapi")
    b = parse_skills(_write(tmp_path, reordered.replace("More prose.", "Other prose."), "b.md"))
    assert a.skills_hash == b.skills_hash and a.file_hash != b.file_hash
    moved = GOOD.replace("python, fastapi, postgresql", "fastapi, postgresql").replace("rust, c#", "rust, c#, python")
    assert parse_skills(_write(tmp_path, moved, "c.md")).skills_hash != a.skills_hash  # a tier change rescores
    assert skills_hash({"strong": ("b", "a"), "working": (), "familiar": ()}) == skills_hash(
        {"strong": ("a", "b"), "working": (), "familiar": ()}
    )


def test_missing_last_updated_warns_but_parses(tmp_path):
    p = parse_skills(_write(tmp_path, GOOD.replace("**Last updated:** 2026-10-16\n", "")))
    assert p.last_updated is None and "Last updated" in p.warnings[0]


@pytest.mark.parametrize(
    "line",
    [
        "Last updated: 2026-10-16",
        "**Last updated:** 2026-10-16",
        "  last UPDATED 2026-10-16",
        "**Last updated** 2026-10-16",
    ],
)
def test_last_updated_forms(tmp_path, line):
    text = GOOD.replace("**Last updated:** 2026-10-16", line)
    assert parse_skills(_write(tmp_path, text)).last_updated == "2026-10-16"


# ── one fixture per rule, each naming its line ──────────────────────────────
def test_rule_1_no_skills_header(tmp_path):
    assert "rule 1" in _errors(tmp_path, "# P\n## Summary\ntext\n")


def test_rule_1_two_skills_headers(tmp_path):
    msg = _errors(
        tmp_path, _section("**Strong:**\npython\n**Working:**\ndocker\n**Familiar:**\nrust\n## Skills again\n")
    )
    assert "rule 1" in msg and "line 3, line 10" in msg


def test_rule_2_missing_tier(tmp_path):
    msg = _errors(tmp_path, _section("**Strong:**\npython\n**Working:**\ndocker\n"))
    assert "line 3: rule 2: the Familiar tier is missing" in msg


def test_rule_2_tier_twice(tmp_path):
    msg = _errors(
        tmp_path, _section("**Strong:**\npython\n**Working:**\ndocker\n**Strong again:**\nrust\n**Familiar:**\nc\n")
    )
    assert "line 8: rule 2: Strong appears again (first at line 4)" in msg


def test_rule_3_label_without_token_line(tmp_path):
    msg = _errors(tmp_path, _section("**Strong:**\n\n**Working:**\ndocker\n**Familiar:**\nrust\n"))
    assert "line 4: rule 3: the Strong label has no token line before line 6" in msg


def test_rule_3_label_at_the_end_of_the_section(tmp_path):
    msg = _errors(tmp_path, _section("**Strong:**\npython\n**Working:**\ndocker\n**Familiar:**\n\n## Next\n"))
    assert "line 8: rule 3: the Familiar label has no token line" in msg


def test_rule_3_second_token_line(tmp_path):
    msg = _errors(tmp_path, _section("**Strong:**\npython\nfastapi\n**Working:**\ndocker\n**Familiar:**\nrust\n"))
    assert "line 6: rule 3: a token line must come right after a tier label" in msg


def test_rule_4_empty_tokens(tmp_path):
    msg = _errors(tmp_path, _section("**Strong:**\npython,,fastapi\n**Working:**\ndocker,\n**Familiar:**\nrust\n"))
    assert "line 5: rule 4: empty token" in msg and "line 7: rule 4: empty token" in msg


@pytest.mark.parametrize("bad", ["Python", "fast api", "-go", "c%", "_x"])
def test_rule_4_token_regex(tmp_path, bad):
    msg = _errors(tmp_path, _section(f"**Strong:**\npython, {bad}\n**Working:**\ndocker\n**Familiar:**\nrust\n"))
    assert f"line 5: rule 4: token {bad!r}" in msg


def test_rule_5_token_in_two_tiers(tmp_path):
    msg = _errors(tmp_path, _section("**Strong:**\npython\n**Working:**\ndocker, python\n**Familiar:**\nrust\n"))
    assert "line 7: rule 5: token 'python' is already listed under Strong (line 5)" in msg


def test_rule_5_token_twice_in_one_tier(tmp_path):
    msg = _errors(tmp_path, _section("**Strong:**\npython, python\n**Working:**\ndocker\n**Familiar:**\nrust\n"))
    assert "line 5: rule 5" in msg


def test_rule_6_unknown_label(tmp_path):
    msg = _errors(
        tmp_path, _section("**Strong:**\npython\n**Expert:**\ngo\n**Working:**\ndocker\n**Familiar:**\nrust\n")
    )
    assert "line 6: rule 6: unknown label '**Expert:**'" in msg


def test_rule_6_stray_prose(tmp_path):
    msg = _errors(
        tmp_path, _section("**Strong:**\npython\nI also like Go.\n**Working:**\ndocker\n**Familiar:**\nrust\n")
    )
    assert "line 6: rule 6: unexpected line" in msg


def test_rule_6_subheading_inside_the_section(tmp_path):
    msg = _errors(tmp_path, _section("**Strong:**\npython\n### Notes\n**Working:**\ndocker\n**Familiar:**\nrust\n"))
    assert "line 6: rule 6" in msg


def test_rule_7_not_claimed_twice(tmp_path):
    body = "**Strong:**\npython\nExplicitly NOT claimed: a\n**Working:**\ndocker\n**Familiar:**\nrust\nExplicitly NOT claimed: b\n"
    msg = _errors(tmp_path, _section(body))
    assert "line 11: rule 7" in msg and "first at line 6" in msg


def test_every_problem_is_listed_at_once(tmp_path):
    msg = _errors(tmp_path, _section("**Strong:**\nPython\n**Working:**\ndocker,\n"))
    assert "3 problem(s)" in msg  # bad token, empty token, missing Familiar
