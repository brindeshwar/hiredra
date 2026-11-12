"""The growing canonical skill vocabulary (build step 3) and the rules that clean a skill name on its way in.

A flat, growing list of canonical skill names (docs/graph.md's `market` nodes live in this same table) -- never an
enum, since the vocabulary is discovered from JD text and grows every run.

Every name extraction produces goes through ``Resolver.resolve`` (owner decisions, 2026-11-12, after the graph audit
found 124 groups of spelling variants waiting and compound names stored as single skills). It is deterministic,
offline, and never calls the embedder. In order:

1. fold case and whitespace; a name that is already a node or an alias resolves at once;
2. ``x (y)``: when one is the acronym of the other ("model context protocol (mcp)") it is ONE skill and every form
   becomes an alias of it; otherwise the bracket lists examples and each one is a skill;
3. split lists into the skills we recognise: a comma list keeps its known parts (none known: the name stays
   whole); "and", "&", "or" and "/" split only when every part is known (so "docker & kubernetes" splits,
   "trust and safety" does not). ``keep_whole`` names are never split;
4. each part, unless already known: a version of a known skill ("angular 7", "python3", ".net 6/7/8") or the same
   spelling as one ("java script", "node js", "panda") becomes an alias of it (decision D1: same-spelling variants
   are aliased automatically). Pairs in ``config/skill_never_merge.yml`` are never joined;
5. a part with non-Latin letters, too many words or too many characters is dropped (a phrase, not a skill name);
6. anything left is a new node.

An alias is written to ``skill_alias`` the first time, so the next sighting is an exact hit. The vector-based merge
between runs (``hiredra.graph.merge``) is untouched by this.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Collection

from hiredra.storage import add_alias, resolve_alias, touch_skill_vocab

_LIST_SEP = re.compile(r"\s*[,;]\s*(?:(?:and|or)\s+)?")
_WORD_SEPS = (re.compile(r"\s+(?:and|or)\s+|\s+&\s+"), re.compile(r"\s*/\s*"))
_BRACKET = re.compile(r"(.+?)\s*\((.+)\)")
_EXAMPLES_LEAD = re.compile(r"^(?:e\.g\.|eg\.|i\.e\.|such as|like|incl\.|including)\s*,?\s*")
#: a trailing version: "angular 7", "angular 16+", "python 3.11", ".net 6/7/8", "wcag 2.0", "odata v4", "gpt-4". At
#: most two digits for the major number, so "microsoft 365", "arinc653" and "iso 27001" are names, not versions.
_VERSION = re.compile(
    r"(?P<base>.*?[a-z+#.])(?P<sep>\s+|-)?v?\d{1,2}(?:\.(?:\d+|x))*(?:\s*[/,&]\s*v?\d{1,2}(?:\.(?:\d+|x))*)*\+?"
)
_STOPWORDS = frozenset({"of", "for", "and", "the", "in", "to", "with", "on", "&"})


def normalize(raw_name: str) -> str:
    """Case/whitespace fold only -- no synonym logic here."""
    return " ".join(raw_name.split()).lower()


def latin_only(name: str) -> bool:
    """Every letter in ``name`` is a Latin letter (accents allowed: "estatística" passes, "llamainデックス" fails)."""
    return all(not ch.isalpha() or "LATIN" in unicodedata.name(ch, "") for ch in name)


def version_base(name: str) -> str | None:
    """The name without its trailing version ("angular 7" -> "angular", "python3" -> "python"), or None. A version
    stuck to the name needs a base of 3+ characters ("db2", "ec2" and "s3" are names); a separated one needs 2."""
    m = _VERSION.fullmatch(name)
    if not m:
        return None
    base = m.group("base").strip()
    return base if len(base) >= (2 if m.group("sep") else 3) else None


def is_acronym(short: str, long: str) -> bool:
    """``short`` is the initials of ``long``: "mcp" / "model context protocol", "dfm" / "design for manufacturing",
    "w&b" / "weights & biases", "ddd" / "domain-driven design". Small words may be skipped; a plural "s" is fine."""
    letters = re.sub(r"[^a-z]", "", short)
    if not 2 <= len(letters) <= 8 or " " in short.strip():
        return False
    words = [w for w in re.split(r"[\s\-/]+", long) if w and w[0].isalnum()]
    if len(words) < 2:
        return False
    every = "".join(w[0] for w in words)
    content = "".join(w[0] for w in words if w not in _STOPWORDS)
    return letters in (every, content) or (letters.endswith("s") and letters[:-1] in (every, content))


def _spelling_keys(name: str) -> frozenset[str]:
    from hiredra.graph.merge import spelling_keys  # the graph package imports this module; keep the import lazy

    return spelling_keys(name)


class Resolver:
    """Resolves raw skill names to canonical ones (see the module docstring), with the vocabulary held in memory for
    one run. Make one per run: it reads every node and alias once, then keeps itself up to date as it adds them.

    ``skip`` names (the generic list) are dropped wherever they appear, including as a part of a split name.
    ``dropped`` collects ``(name, reason)`` for names refused as junk; the caller notes and clears it.
    """

    def __init__(
        self,
        conn,
        *,
        keep_whole: Collection[str] = (),
        never: Collection[frozenset[str]] = frozenset(),
        skip: Collection[str] = (),
        max_words: int = 4,
        max_chars: int = 40,
    ) -> None:
        self.conn = conn
        self.keep_whole = frozenset(keep_whole)
        self.never = frozenset(never)
        self.skip = frozenset(skip)
        self.max_words, self.max_chars = max_words, max_chars
        self.dropped: list[tuple[str, str]] = []
        self.counts: Counter = Counter()
        nodes = conn.execute("SELECT name, mine, seen_count FROM skill_vocab").fetchall()
        aliases = conn.execute("SELECT alias, canonical FROM skill_alias").fetchall()
        self._canon: dict[str, str] = {r["name"]: r["name"] for r in nodes}
        self._canon.update({a["alias"]: a["canonical"] for a in aliases})
        self._by_key: dict[str, str] = {}
        # Which node a spelling key leads to when several share it: the owner's skill, then the most seen, shorter.
        ranked = sorted(nodes, key=lambda r: (not r["mine"], -r["seen_count"], len(r["name"]), r["name"]))
        self._rank = {r["name"]: i for i, r in enumerate(ranked)}
        for r in ranked:
            self._index(r["name"], r["name"])
        for a in aliases:
            self._index(a["alias"], a["canonical"])

    # ── public ────────────────────────────────────────────────────────────────
    def resolve(self, raw_name: str) -> list[str]:
        """The canonical names ``raw_name`` stands for: usually one, several for a list, none for junk or a generic
        word. Every name returned has had its sighting counted."""
        name = normalize(raw_name)
        if not name or name in self.skip:
            return []
        if name in self._canon:
            return [self._touch(self._canon[name])]
        m = _BRACKET.fullmatch(name)
        if m:
            outer, inner = m.group(1).strip(), m.group(2).strip()
            if is_acronym(inner, outer) or is_acronym(outer, inner):
                return self._one_of(name, outer, inner)
            parts = self._split(outer) + self._split(_EXAMPLES_LEAD.sub("", inner))
        else:
            parts = self._split(name)
        out: list[str] = []
        for part in parts:
            canonical = self._one(part)
            if canonical and canonical not in out:
                out.append(canonical)
        return out

    def find(self, name: str) -> tuple[str, str] | None:
        """``(canonical, how)`` for a name we already know, without adding anything: ``exact``, ``version`` or
        ``spelling``. None for a name that would be new."""
        if name in self._canon:
            return self._canon[name], "exact"
        base = version_base(name)
        if base and base not in self.keep_whole and name not in self.keep_whole and base in self._canon:
            canonical = self._canon[base]
            if not self._never(name, canonical):
                return canonical, "version"
        # Two keys can lead to two nodes ("neural network" / "neural-networks"): the higher-priority node wins.
        hits = {self._by_key[k] for k in _spelling_keys(name) if k in self._by_key}
        for canonical in sorted(hits, key=lambda c: (self._rank.get(c, len(self._rank)), c)):
            if not self._never(name, canonical):
                return canonical, "spelling"
        return None

    def junk_reason(self, name: str) -> str | None:
        """Why ``name`` is not a usable skill name, or None."""
        if not any(ch.isalpha() for ch in name):
            return "no letters"
        if not latin_only(name):
            return "non-Latin letters"
        if len(name.split()) > self.max_words:
            return "too many words"
        if len(name) > self.max_chars:
            return "too long"
        return None

    # ── the steps ─────────────────────────────────────────────────────────────
    def _split(self, name: str) -> list[str]:
        """A list split into skills. Comma lists keep only the parts we recognise ("building large, distributed
        systems" -> distributed systems); with none recognised the name stays whole. "and", "&", "or" and "/"
        split only when every part is known."""
        if name in self.keep_whole or name in self._canon:
            return [name]
        pieces = [p for p in _LIST_SEP.split(name) if p]
        out = [part for piece in pieces for part in self._split_words(piece)]
        if len(pieces) > 1:
            out = [p for p in out if p in self.skip or self.find(p)] or [name]
        if len(out) > 1:
            self.counts["split"] += 1
        return out

    def _split_words(self, piece: str) -> list[str]:
        if piece in self.keep_whole or piece in self._canon:
            return [piece]
        for sep in _WORD_SEPS:
            parts = [p.strip() for p in sep.split(piece) if p.strip()]
            if len(parts) > 1 and all(p in self.skip or self.find(p) for p in parts):
                return parts
        return [piece]

    def _one(self, part: str) -> str | None:
        if part in self.skip:
            return None
        found = self.find(part)
        if found:
            canonical, how = found
            if canonical != part:
                self._alias(part, canonical, how)
            return self._touch(canonical)
        reason = self.junk_reason(part)
        if reason:
            self.dropped.append((part, reason))
            self.counts["dropped"] += 1
            return None
        self.counts["new"] += 1
        self._canon[part] = part
        self._index(part, part)
        return self._touch(part)

    def _one_of(self, full: str, outer: str, inner: str) -> list[str]:
        """``outer (inner)`` where one is the other's acronym: one skill, every form an alias of it. A form we
        already know wins (the acronym first); otherwise the spelled-out name becomes the node."""
        known = next((f[0] for f in map(self.find, (inner, outer)) if f and f[0] not in self.skip), None)
        canonical = self._touch(known) if known else self._one(outer)  # _one counts the sighting itself
        if canonical is None:
            return []
        for form in (inner, outer, full):
            if form != canonical and form not in self._canon:
                self._alias(form, canonical, "acronym")
        return [canonical]

    # ── bookkeeping ───────────────────────────────────────────────────────────
    def _never(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self.never

    def _index(self, name: str, canonical: str) -> None:
        for key in _spelling_keys(name):
            self._by_key.setdefault(key, canonical)

    def _alias(self, alias: str, canonical: str, how: str) -> None:
        if alias in self._canon or self._never(alias, canonical):
            return
        add_alias(self.conn, alias, canonical)
        self._canon[alias] = canonical
        self._index(alias, canonical)
        self.counts[how] += 1

    def _touch(self, canonical: str) -> str:
        touch_skill_vocab(self.conn, canonical)
        return canonical


def resolve_skill(conn, raw_name: str) -> str:
    """Fold ``raw_name`` and resolve it through the alias table only, recording the sighting -- the old exact-only
    path, kept for callers that hold no ``Resolver`` (tests, one-off tools). Extraction uses ``Resolver``."""
    name = normalize(raw_name)
    canonical = resolve_alias(conn, name) or name
    touch_skill_vocab(conn, canonical)
    return canonical
