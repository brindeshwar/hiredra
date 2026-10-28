"""How a job and its employer are identified across crawls (decisions D6.2, D6.3, D6.5; Q3 in
docs/freehire-merge/05-schema.md). Ported from freehire (MIT, see THIRD_PARTY_NOTICES.md):
``internal/dict/normalize/company.go`` + ``slug.go`` and ``internal/job/jobhash/rolefingerprint.go``.

- ``company_slug``: the key one employer is filed under, whatever the source wrote ("Booking B.V.",
  "booking") -> ``booking``.
- ``role_fingerprint``: the IDENTITY of a role, so per-city copies and reposts of one role cluster.
  Unlike freehire it includes the parsed country set (Q3): a London copy and a Bangalore copy are
  two clusters, so hiding the non-canonical copy can never hide an India-eligible job.
- ``content_hash``: the CHANGE signal. It covers exactly what the funnel and the extraction read, so
  a re-crawl re-spends LLM quota only when one of those inputs moved (D6.2). Unlike freehire's it
  leaves out the posted date: a bumped date changes nothing the extraction sees.

Deliberate difference from freehire: transliteration is accent-stripping only (NFKD), with no
unidecode table. A non-Latin name keeps its own letters (``Яндекс`` -> ``яндекс``, freehire
``iandeks``), which is still one stable key per spelling and needs no extra dependency.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from collections.abc import Iterable

_RS = "\x1e"  # field separator inside a hash, as in freehire

# Trailing corporate forms, matched on a word's ASCII letters ("B.V." -> "bv"). freehire's list, as is.
_LEGAL_FORMS = frozenset({
    "corporation", "limited", "gmbh", "corp", "llc", "ltd", "inc", "incorporated", "plc", "llp", "lp",
    "cic", "cio", "srl", "pty", "pte", "bv", "nv", "aps", "ab", "ag", "kg", "oy", "sa", "as", "se", "co",
})  # fmt: skip


def _fold(text: str) -> str:
    """Accents off: NFKD, then drop the combining marks (``Köln`` -> ``Koln``)."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def slug(name: str) -> str:
    """Lowercase letters and digits, every other run turned into one hyphen, no hyphen at either end."""
    out: list[str] = []
    hyphen = False
    for c in _fold(name):
        if c.isalnum():
            out.append(c.lower())
            hyphen = False
        elif out and not hyphen:
            out.append("-")
            hyphen = True
    return "".join(out).rstrip("-")


def _letters(word: str) -> str:
    return "".join(c for c in word.lower() if "a" <= c <= "z")


def company_slug(name: str) -> str:
    """``slug`` of the name with trailing corporate forms removed, repeatedly ("Acme GmbH & Co. KG" -> ``acme``).

    Words split on anything but letters, digits, "." and "/" (those two live inside forms: "B.V.", "A/S").
    Apostrophes join ("Kohl's" -> ``kohls``). A one-word name is never stripped ("Limited" stays).
    """
    name = name.replace("'", "").replace("’", "").replace("ʼ", "")
    words = [w for w in re.split(r"[^\w./]+|_", name) if w]
    while len(words) > 1 and _letters(words[-1]) in _LEGAL_FORMS:
        words.pop()
    return slug(" ".join(words))


_HTML_TAG = re.compile(r"<[^>]*>")
# The last clause of a title: ", X" or " - X" / " | X" / " @ X" (dash, en dash, em dash), X free of separators.
_TRAILING_CLAUSE = re.compile(r"(\s*,\s*|\s+[|@]\s*|\s+[-–—]\s*)[^,|@\-–—]*$", re.ASCII)


def _role_text(text: str) -> str:
    """Visible text only: tags to spaces, entities decoded, lowercased, whitespace collapsed."""
    return " ".join(html.unescape(_HTML_TAG.sub(" ", text)).lower().split())


def _strip_trailing_clause(title: str) -> str:
    """Drop a trailing city or qualifier ("Senior Engineer, Krakow" -> "Senior Engineer"), unless under two words remain."""
    stripped = _TRAILING_CLAUSE.sub("", title, count=1)
    return stripped if len(stripped.split()) >= 2 else title


def _sha(parts: Iterable[str]) -> str:
    return hashlib.sha256(_RS.join(parts).encode("utf-8")).hexdigest()


def role_fingerprint(company_key: str, title: str, description: str | None, countries: Iterable[str]) -> str:
    """One role at one employer in one set of countries; ignores url, date, id and the raw location."""
    return _sha(
        (
            company_key,
            _role_text(_strip_trailing_clause(title)),
            _role_text(description or ""),
            ",".join(sorted(countries)),
        )
    )


def content_hash(title: str, company: str, location: str | None, description: str | None) -> str:
    """Moves exactly when an input of the funnel or the extraction moves (D6.2)."""
    return _sha((title, company, location or "", description or ""))
