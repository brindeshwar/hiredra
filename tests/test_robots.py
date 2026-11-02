"""robots.txt matching (decision D4.2), RFC 9309: group choice, path AND query, ``*`` and ``$``, longest match.
The Google cases are the lines of www.google.com/robots.txt that ``urllib.robotparser`` misread (2026-11-02)."""

from __future__ import annotations

import pytest

from hiredra.sourcing.providers._common import Robots, robots_allows, robots_rules

AGENT = "Hiredra/0.1 (+https://github.com/brindeshwar)"
RESULTS = "https://www.google.com/about/careers/applications/jobs/results"
GOOGLE = """User-agent: *
User-agent: Yandex
Disallow: /search
Allow: /search/about
Disallow: /?
Allow: /?hl=
Disallow: /?hl=*&
Disallow: /about/careers/applications/jobs/results?page=
Disallow: /about/careers/applications/jobs/results?*&page=

User-agent: Yandex
Disallow: /about/careers/applications/jobs/results
"""


def _allowed(text: str, url: str, agent: str = AGENT) -> bool:
    return robots_allows(robots_rules(text, agent), url)


@pytest.mark.parametrize(
    ("url", "want"),
    [
        (RESULTS, True),  # robotparser read ``Disallow: /?`` as ``Disallow: /`` and refused this
        (f"{RESULTS}/12345-engineer", True),
        (f"{RESULTS}?location=India", True),
        (f"{RESULTS}?page=2", False),  # robotparser dropped the query and allowed these
        (f"{RESULTS}?location=India&page=2", False),
        ("https://www.google.com/?q=x", False),
        ("https://www.google.com/?hl=en", True),  # the longer Allow wins
        ("https://www.google.com/?hl=en&q=x", False),  # and the longer Disallow beats it
        ("https://www.google.com/search?q=x", False),
        ("https://www.google.com/search/about", True),
        ("https://www.google.com/", True),
    ],
)
def test_google_careers(url, want):
    assert _allowed(GOOGLE, url) is want


def test_a_group_naming_us_replaces_star_and_groups_of_one_name_combine():
    assert not _allowed(GOOGLE, RESULTS, agent="Yandex")  # Yandex's two groups combine
    text = (
        "User-agent: *\nDisallow: /\n\nUser-agent: hiredra\nDisallow: /private\n\nUser-agent: HIREDRA\nDisallow: /x\n"
    )
    assert _allowed(text, "https://a.com/jobs")
    assert not _allowed(text, "https://a.com/private/1") and not _allowed(text, "https://a.com/x")
    assert not _allowed(text, "https://a.com/jobs", agent="OtherBot/1.0")
    assert _allowed("User-agent: hiredra\nDisallow:\n\nUser-agent: *\nDisallow: /\n", "https://a.com/jobs")  # ours
    # a blank line ends nothing: two User-agent lines around one are one group, and its rules are ours too
    assert not _allowed("User-agent: hiredra\n\nUser-agent: *\nDisallow: /\n", "https://a.com/jobs")


@pytest.mark.parametrize(
    ("rules", "url", "want"),
    [
        ("Disallow: /*.php$", "https://a.com/index.php", False),
        ("Disallow: /*.php$", "https://a.com/index.php?x=1", True),  # ``$`` ends the match
        ("Disallow: /fish*", "https://a.com/fish.html", False),
        ("Disallow: /fish", "https://a.com/Fish", True),  # paths are case-sensitive
        ("Disallow: /p\nAllow: /p", "https://a.com/p", True),  # a tie goes to Allow
        ("Disallow:", "https://a.com/anything", True),  # an empty Disallow sets nothing
        ("Disallow: /café", "https://a.com/caf%c3%a9", False),  # one spelling, encoded or not
        ("Disallow: / # everything", "https://a.com/x", False),  # comments are cut
        ("Disallow: /", "https://a.com/robots.txt", True),
    ],
)
def test_rule_matching(rules, url, want):
    assert _allowed(f"User-agent: *\n{rules}\n", url) is want


def test_rules_before_any_group_and_unknown_lines_are_ignored():
    text = "Disallow: /\nSitemap: https://a.com/s.xml\nUser-agent: *\nCrawl-delay: 5\nDisallow: /private\n"
    assert _allowed(text, "https://a.com/jobs") and not _allowed(text, "https://a.com/private")
    assert _allowed("<html>not a robots file</html>", "https://a.com/jobs")


def test_robots_is_read_once_per_origin():
    asked = []

    def fetch(url):
        asked.append(url)
        return GOOGLE if "google" in url else None

    robots = Robots(fetch, AGENT)
    assert robots.allowed(RESULTS) and not robots.allowed(f"{RESULTS}?page=2")
    assert robots.allowed("https://other.com/anything")
    assert asked == ["https://www.google.com/robots.txt", "https://other.com/robots.txt"]
