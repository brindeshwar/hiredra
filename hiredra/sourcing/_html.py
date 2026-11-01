"""JD-body HTML -> plain text. Dependency-free.

Greenhouse (and several other ATS) serve posting bodies as *double-encoded*
HTML inside the list JSON: the string carries entity-escaped markup
("&lt;p&gt;"), so one unescape reveals the real tags, and the text-level
entities ("&amp;", "&#39;") only decode safely once the tags are gone.

Good enough to feed the seniority-gate regex (build 2b) and the skill-extraction
LLM (build 3); it is not a general HTML renderer.

Also the schema.org JobPosting a server-rendered job page carries (``ld_job_posting``,
ported from freehire's ``jsonld.go``, MIT), for the adapters that read job pages (iCIMS).
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

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
    text = str(value)
    if "<" not in text:  # double-encoded (Greenhouse): entity-escaped markup -> real tags. Real tags present means
        text = html.unescape(text)  # single-encoded, where "&lt;String&gt;" is text ("List<String>"), kept
    text = _SCRIPT_STYLE.sub("", text)
    text = _NEWLINE_TAGS.sub("\n", text)
    text = _TAG.sub("", text)  # drop remaining tags
    text = html.unescape(text)  # now decode the text-level entities
    text = _INLINE_WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


_LD_JSON = re.compile(r"""<script\b[^>]*\btype=["']application/ld\+json["'][^>]*>(.*?)</script\s*>""", re.I | re.S)


def _job_postings(node: Any) -> list[dict]:
    """The schema.org JobPosting nodes in one ld+json value: a bare object, a list, an ``@graph`` wrapper, or a page
    whose ``mainEntity`` is the posting (freehire's ``jobPostingNodes``)."""
    if isinstance(node, list):
        return [n for n in node if isinstance(n, dict) and str(n.get("@type", "")).lower() == "jobposting"]
    if not isinstance(node, dict):
        return []
    if str(node.get("@type", "")).lower() == "jobposting":
        return [node]
    return _job_postings(node.get("@graph")) or _job_postings([node.get("mainEntity")])


def ld_job_posting(page: str | None) -> dict | None:
    """The first schema.org JobPosting a server-rendered page carries in an ``application/ld+json`` block, else
    ``None``. Raw control characters inside its strings are accepted, as freehire's decoder accepts them."""
    for block in _LD_JSON.findall(page or ""):
        try:
            value = json.loads(block, strict=False)
        except ValueError:
            continue
        if postings := _job_postings(value):
            return postings[0]
    return None


def itemprop_html(page: str | None, prop: str) -> list[str]:
    """The inner HTML of each element carrying ``itemprop="<prop>"`` (schema.org microdata), up to its matching
    close tag; a self-closing or never-closed element gives nothing. For pages that mark a posting up in microdata
    instead of ld+json (SuccessFactors)."""
    page = page or ""
    found: list[str] = []
    opening = re.compile(rf"""<([A-Za-z][A-Za-z0-9]*)\b[^>]*\bitemprop=["']{re.escape(prop)}["'][^>]*>""", re.I)
    for m in opening.finditer(page):
        if m.group(0).endswith("/>"):
            continue
        tags = re.compile(rf"<(/?){m.group(1)}\b[^>]*?(/?)>", re.I)
        depth = 1
        for t in tags.finditer(page, m.end()):
            depth += -1 if t.group(1) else 0 if t.group(2) else 1
            if depth == 0:
                found.append(page[m.end() : t.start()])
                break
    return found


def meta_property(page: str | None, prop: str) -> str | None:
    """The ``content`` of the first ``<meta property="<prop>">`` (Open Graph), unescaped."""
    for tag in re.findall(r"<meta\b[^>]*>", page or "", re.I):
        if re.search(rf"""\bproperty=["']{re.escape(prop)}["']""", tag, re.I):
            m = re.search(r"""\bcontent=(["'])(.*?)\1""", tag, re.S)
            return (html.unescape(m.group(2)).strip() or None) if m else None
    return None
