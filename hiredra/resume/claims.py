"""The claim gate: generated text may say only what the master profile says (hard rule 4; plan decision F1).

Pure code, no model. Used for the résumé, the cover letter and drafted form answers. Each piece of text is checked
against its source: the profile entry it rewrites (``ref``, the entry's ``###`` heading), or, for a summary or a
letter (no ref), the whole profile.

- ``bad_ref``: the ref is not a real profile heading.
- ``new_number``: a number (``116,712``, ``20%``, ``50x``, ``0.795``) that the source text does not contain.
- ``unsupported_skill``: a skill name from the graph's vocabulary that is not in the source text and is not one of
  the owner's Strong or Working skills. A Familiar skill counts as unsupported unless the source text itself names
  it (an entry may say what it used; a summary may not claim it).

The result is ``{"hits": [{"rule", "where", "detail"}], "pass": bool}``. Dashes are not a gate failure: the
renderer replaces them (``word.clean``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hiredra.resume.skills import claimable, fold
from hiredra.resume.source import Entry, ProfileText

_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(\s*%|x\b)?", re.IGNORECASE)
_TOKEN_EDGE = re.compile(r"^[^\w+#]+|[^\w+#]+$")
#: Vocabulary names that are everyday words too: never a claim on their own.
_IGNORE = frozenset({
    "ai", "ml", "data", "design", "testing", "security", "documentation", "debugging", "communication",
    "collaboration", "monitoring", "logging", "deployment", "performance", "automation", "integration",
    "development", "engineering", "software", "programming", "research", "analysis", "architecture", "systems",
    "services", "platform", "tools", "agents", "models", "apis", "api", "cloud", "backend", "frontend", "go",
})  # fmt: skip


#: Words that only say what kind of thing a skill is ("agent systems", "evaluation infrastructure"): a phrase is
#: judged by its other words.
_FILLER = frozenset({
    "system", "application", "infrastructure", "engineering", "platform", "solution", "service", "development",
    "tool", "tooling", "pipeline", "workflow", "deployment", "framework", "stack", "capability", "practice",
    "technique", "method", "approach", "experience", "product", "feature", "environment",
})  # fmt: skip


def singular(phrase: str) -> str:
    """Each word's plural "s" dropped ("mcp servers" -> "mcp server", "sdks" -> "sdk"); "ss" and short words kept."""
    return " ".join(w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in phrase.split())


def numbers(text: str) -> set[str]:
    """Every number in ``text``, normalised ("116,712" -> "116712", "20 %" -> "20%", "50X" -> "50x")."""
    return {m.group(1).replace(",", "") + (m.group(2) or "").replace(" ", "").lower() for m in _NUMBER.finditer(text)}


def _plain(n: str) -> str:
    return n.rstrip("%x")


def entry_text(e: Entry) -> str:
    return " ".join([e.heading, e.intro, *e.bullets, e.stack])


def profile_text(p: ProfileText) -> str:
    return " ".join([p.name, *p.education, *(entry_text(e) for e in p.entries)])


@dataclass
class Gate:
    """The gate for one profile: built once, then ``check`` per piece."""

    profile: ProfileText
    vocabulary: set[str]  # folded skill names and aliases from the graph
    aliases: dict[str, str] = field(default_factory=dict)  # folded alias -> folded canonical name
    ignore: frozenset[str] = frozenset()  # everyday words that are graph names too (extraction's generic list)
    banned: tuple[str, ...] = ()  # phrases the profile's standing rules forbid ("full-stack")
    ok_skills: set[str] = field(init=False)
    familiar: set[str] = field(init=False)
    whole: str = field(init=False)

    def __post_init__(self):
        self.ok_skills = {singular(fold(t)) for t in claimable(self.profile.skills)}
        self.familiar = {singular(fold(t)) for t in self.profile.skills.tiers["familiar"]}
        self.ignore = frozenset(_IGNORE | {fold(w) for w in self.ignore})
        self.ok_words = {w for s in self.ok_skills for w in s.split()}
        self.whole = profile_text(self.profile)
        self.headings = {e.heading: e for e in self.profile.entries}

    def skills_in(self, text: str) -> set[str]:
        """Vocabulary names that appear in ``text`` as whole words (1 to 4 word phrases)."""
        words = [w for w in (_TOKEN_EDGE.sub("", t) for t in fold(text.replace("**", "")).split()) if w]
        found = set()
        for size in range(1, 5):
            for i in range(len(words) - size + 1):
                phrase = " ".join(words[i : i + size])
                if len(phrase) >= 3 and phrase in self.vocabulary and phrase not in self.ignore:
                    found.add(phrase)
        return found

    def _forms(self, phrase: str) -> set[str]:
        """The spellings a found name may match by: itself, singular, and its canonical name if it is an alias."""
        forms = {phrase, singular(phrase)}
        canonical = self.aliases.get(phrase) or self.aliases.get(singular(phrase))
        if canonical:
            forms |= {canonical, singular(canonical)}
        return forms

    def check(self, where: str, text: str, ref: str | None = None, context: str = "") -> list[dict]:
        """``context`` is text the piece may echo without it being a claim: the job's title and company."""
        hits = []
        if ref is not None and ref not in self.headings:
            return [{"rule": "bad_ref", "where": where, "detail": ref}]
        low = text.casefold()
        hits += [{"rule": "banned_phrase", "where": where, "detail": b} for b in self.banned if b.casefold() in low]
        # "Amazon.com" in the context must cover "Amazon" in the text: punctuation splits context words.
        context = re.sub(r"[^\w+#]+", " ", context)
        source = (entry_text(self.headings[ref]) if ref is not None else self.whole) + " " + context
        have = numbers(source) | {_plain(n) for n in numbers(source)}
        for n in sorted(numbers(text)):
            if n not in have and _plain(n) not in have:
                hits.append({"rule": "new_number", "where": where, "detail": n})
        folded = " ".join(w for w in (_TOKEN_EDGE.sub("", t) for t in fold(source.replace("**", "")).split()) if w)
        source_words = f" {folded} {singular(folded)} "  # the same word cleaning as the text: "(Gemma" -> "gemma"
        for skill in sorted(self.skills_in(text)):
            forms = self._forms(skill)
            familiar_only = any(f in self.familiar for f in forms) and not any(f in self.ok_skills for f in forms)
            if familiar_only and ref is None:  # a summary or a letter never claims a Familiar skill
                hits.append({"rule": "unsupported_skill", "where": where, "detail": skill})
                continue
            if any(f" {f} " in source_words for f in forms):
                continue  # the entry itself names it
            if familiar_only:
                hits.append({"rule": "unsupported_skill", "where": where, "detail": skill})
                continue
            if any(f in self.ok_skills for f in forms):
                continue
            words = set(source_words.split())
            content = [w for w in singular(skill).split() if w not in _FILLER]
            if all(w in self.ok_words or w in words for w in content):
                continue  # a broad phrase made of the owner's own skills ("agent systems", "llm applications")
            hits.append({"rule": "unsupported_skill", "where": where, "detail": skill})
        return hits


def report(hits: list[dict]) -> dict:
    return {"hits": hits, "pass": not hits}
