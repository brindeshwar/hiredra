"""Deterministic skill tagger: a Python port of freehire's ``skilltag.Parse`` (decision D10.3).

A curated dictionary, not an extractor: it finds the ~1,100 skills it knows, under any of their
spellings, and emits nothing for anything else (it never guesses). The dictionary itself is
freehire's, exported unchanged into ``hiredra/skill_dict.json`` (MIT, see THIRD_PARTY_NOTICES.md).
The step-9 comparison showed the extraction LLM leaves out skills named inside long lists; this
tagger is the second producer that adds them (as good_to_have, see ``dictionary_extras``).

Matching rules, all as freehire's (``internal/dict/skilltag/skilltag.go``, ``internal/dict/wordmatch``):

- HTML tags and URLs are replaced by a space before matching.
- Acronym pass, case-sensitive: ``ML`` is machine learning, ``ml`` is not.
- Phrase pass on the lowercased text: a multi-word alias matches with any run of ``-``, ``_`` or
  whitespace between its words; a match preceded by ``.`` or ``-`` does not count, so the ``c`` of
  ``objective-c`` and the ``asp.net`` of ``foo.asp.net`` are not skills.
- Word pass: whole letter/digit tokens. An ambiguous word (react, swift, spring, ...) counts only
  when the text also has a strong match; the proper nouns among them must also be written with a
  capital somewhere ("react quickly" is not React).
- A discipline phrase (networking, content marketing, ...) tags itself but never vouches for an
  ambiguous word.

A tag is a freehire slug (``ci-cd``, ``nodejs``); ``vocab_names`` translates slugs into our names.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from hiredra.storage import resolve_alias
from hiredra.vocab import normalize

SKILL_DICT_PATH = Path(__file__).with_name("skill_dict.json")

#: Our domain bucket -> the freehire job category that unlocks its category-scoped acronyms
#: (``RAG`` is retrieval-augmented generation only inside an AI posting; elsewhere "RAG status").
DOMAIN_CATEGORY = {"agent_llm": "ai_engineering", "ml": "ml_ai"}

# Go's RE2 classes, spelled out where Python's differ: ``\s`` and ``\S`` are ASCII there.
_HTML_TAG_RE = re.compile(r"<[^>]*>")
_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\t\n\f\r ]+", re.IGNORECASE | re.ASCII)
_WORD_TOKEN_RE = re.compile(r"[^\W_]+")  # letters and digits of any script, as [\p{L}\p{N}]+
_SEP = r"[-_\t\n\f\r ]+"


def _is_word_char(ch: str) -> bool:
    return ch.isalpha() or ch.isdecimal()  # Go: unicode.IsLetter || unicode.IsDigit


def _unicode_boundary(s: str, start: int, end: int) -> bool:
    return not (start > 0 and _is_word_char(s[start - 1])) and not (end < len(s) and _is_word_char(s[end]))


def _tech_boundary(s: str, start: int, end: int) -> bool:
    return not (start > 0 and s[start - 1] in ".-") and _unicode_boundary(s, start, end)


def _contains(s: str, term: str, boundary) -> bool:
    """Does ``term`` occur in ``s`` at a position ``boundary`` accepts (every occurrence is tried)?"""
    i = s.find(term)
    while i >= 0:
        if boundary(s, i, i + len(term)):
            return True
        i = s.find(term, i + 1)
    return False


def _strip_markup(text: str) -> str:
    return _URL_RE.sub(" ", _HTML_TAG_RE.sub(" ", text))


class _Tables:
    def __init__(self, data: dict):
        self.word_aliases: dict[str, str] = data["word_aliases"]
        self.ambiguous = frozenset(data["ambiguous_words"])
        self.proper_nouns = frozenset(data["proper_noun_words"])
        self.non_corroborating = frozenset(data["non_corroborating_phrases"])
        self.shared_acronyms: dict[str, str] = data["shared_acronyms"]
        self.scoped_acronyms = {
            surface: (v["canonical"], frozenset(v["categories"]))
            for surface, v in data["category_scoped_acronyms"].items()
        }
        self.labels: dict[str, str] = data["labels"]
        self.aliases: dict[str, list[str]] = data["aliases"]
        # (canonical, single token or None, compiled multi-word pattern or None)
        self.phrases = []
        for canonical, alias in data["phrase_aliases"]:
            segs = [s for s in re.split(_SEP, alias.lower()) if s]
            if len(segs) <= 1:
                self.phrases.append((canonical, alias.lower(), None))
            else:
                self.phrases.append((canonical, None, re.compile(_SEP.join(re.escape(s) for s in segs))))


@lru_cache(maxsize=1)
def _tables() -> _Tables:
    return _Tables(json.loads(SKILL_DICT_PATH.read_text(encoding="utf-8")))


def _phrase_matches(norm: str, token: str | None, pattern: re.Pattern | None) -> bool:
    if pattern is None:
        return _contains(norm, token, _tech_boundary)
    return any(_tech_boundary(norm, m.start(), m.end()) for m in pattern.finditer(norm))


def _written_capitalized(cased: str, word: str) -> bool:
    return _contains(cased, word[:1].upper() + word[1:], _unicode_boundary) or _contains(
        cased, word.upper(), _unicode_boundary
    )


def parse(text: str, category: str | None = None) -> list[str]:
    """The dictionary skill slugs ``text`` names, sorted. ``category`` is a freehire job category
    (see ``DOMAIN_CATEGORY``) and only unlocks that category's acronyms."""
    t = _tables()
    strong: set[str] = set()
    weak: set[str] = set()
    standalone: set[str] = set()

    cased = _strip_markup(text)
    for surface, canonical in t.shared_acronyms.items():
        if _contains(cased, surface, _unicode_boundary):
            strong.add(canonical)
    if category:
        for surface, (canonical, categories) in t.scoped_acronyms.items():
            if category in categories and _contains(cased, surface, _unicode_boundary):
                strong.add(canonical)

    norm = cased.lower().strip()
    for canonical, token, pattern in t.phrases:
        if _phrase_matches(norm, token, pattern):
            (standalone if canonical in t.non_corroborating else strong).add(canonical)

    lowercase_only: dict[str, bool] = {}
    for tok in _WORD_TOKEN_RE.findall(norm):
        canonical = t.word_aliases.get(tok)
        if canonical is None:
            continue
        if tok not in t.ambiguous:
            strong.add(canonical)
            continue
        if tok in t.proper_nouns:
            if tok not in lowercase_only:
                lowercase_only[tok] = not _written_capitalized(cased, tok)
            if lowercase_only[tok]:
                continue
        weak.add(canonical)

    if strong:
        strong |= weak
    return sorted(strong | standalone)


def vocab_names(conn, slugs: Iterable[str] | None = None) -> dict[str, str]:
    """Slug -> our canonical skill name. Each of the slug's forms (the slug, its spaced form, its label,
    its aliases) is looked up in our alias table, then among our vocabulary names; the first hit wins.
    A slug none of whose forms we know becomes its lowercased label (a new name, like any new skill)."""
    t = _tables()
    names = {r["name"] for r in conn.execute("SELECT name FROM skill_vocab")}
    out = {}
    for slug in t.labels if slugs is None else slugs:
        label = t.labels.get(slug, slug)
        for form in (slug, slug.replace("-", " "), label, *t.aliases.get(slug, ())):
            n = normalize(form)
            hit = resolve_alias(conn, n) or (n if n in names else None)
            if hit:
                out[slug] = hit
                break
        else:
            out[slug] = normalize(label)
    return out
