"""JD-body HTML -> plain text. Dependency-free.

Greenhouse (and several other ATS) serve posting bodies as *double-encoded*
HTML inside the list JSON: the string carries entity-escaped markup
("&lt;p&gt;"), so one unescape reveals the real tags, and the text-level
entities ("&amp;", "&#39;") only decode safely once the tags are gone.

Good enough to feed the seniority-gate regex (build 2b) and the skill-extraction
LLM (build 3); it is not a general HTML renderer.
"""

from __future__ import annotations

import html
import re

_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.I | re.S)
_NEWLINE_TAGS = re.compile(
    r"<br\s*/?>"
    r"|</(?:p|div|li|ul|ol|tr|table|h[1-6]|section|article|blockquote)\s*>",
    re.I,
)
_TAG = re.compile(r"<[^>]+>")
_INLINE_WS = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def html_to_text(value: str | None) -> str:
    """Best-effort plain text from (possibly double-encoded) HTML. ``""`` for falsy input."""
    if not value:
        return ""
    text = html.unescape(str(value))  # entity-escaped markup -> real tags
    text = _SCRIPT_STYLE.sub("", text)
    text = _NEWLINE_TAGS.sub("\n", text)
    text = _TAG.sub("", text)  # drop remaining tags
    text = html.unescape(text)  # now decode the text-level entities
    text = _INLINE_WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()
