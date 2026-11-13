"""Render a résumé or a cover letter as an editable Word file (owner decision, 2026-11-13: .docx is the main
output, with a PDF copy made from it -- see ``hiredra.resume.pdf``).

The layout follows the master profile's standing rules: one column, Arial, explicit font sizes, no tables, text
boxes or images (ATS-safe), justified body text, important terms in bold. Everything that decides how much fits on
a page is in ``Layout``, so the page-fit step can tighten it without touching this code. The owner's example file
will replace these defaults; the payload does not change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt

from hiredra.resume.payload import Candidate, CoverLetter, CvPayload

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_PAGES = {"a4": (Mm(210), Mm(297)), "letter": (Mm(215.9), Mm(279.4))}


@dataclass(frozen=True)
class Layout:
    font: str = "Arial"
    body_pt: float = 10.0
    name_pt: float = 16.0
    heading_pt: float = 10.5
    margin_mm: float = 13.0
    line_spacing: float = 1.0
    space_after_pt: float = 2.0  # after a bullet or a body line
    section_gap_pt: float = 6.0  # before a section heading

    def tighter(self) -> Layout | None:
        """The next smaller layout for the page-fit loop, or None when it would become unreadable."""
        if self.body_pt <= 8.5:
            return None
        return Layout(
            font=self.font,
            body_pt=self.body_pt - 0.5,
            name_pt=max(13.0, self.name_pt - 1.0),
            heading_pt=self.heading_pt - 0.5,
            margin_mm=max(10.0, self.margin_mm - 1.0),
            line_spacing=self.line_spacing,
            space_after_pt=max(0.0, self.space_after_pt - 1.0),
            section_gap_pt=max(3.0, self.section_gap_pt - 1.0),
        )


def clean(text: str) -> str:
    """The standing rules a renderer can enforce by itself: no em or en dashes, single spaces."""
    return " ".join(text.replace("—", ", ").replace("–", "-").split()).replace(" ,", ",")


class _Writer:
    def __init__(self, layout: Layout, page: str):
        self.l = layout
        self.doc = Document()
        section = self.doc.sections[0]
        section.page_width, section.page_height = _PAGES.get(page, _PAGES["a4"])
        for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(section, side, Mm(layout.margin_mm))
        self.width = section.page_width - section.left_margin - section.right_margin
        normal = self.doc.styles["Normal"]
        normal.font.name = layout.font
        normal.font.size = Pt(layout.body_pt)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), layout.font)

    def para(self, text: str = "", *, size: float | None = None, bold: bool = False, italic: bool = False,
             align=None, before: float = 0.0, after: float | None = None, style: str | None = None):  # fmt: skip
        p = self.doc.add_paragraph(style=style)
        fmt = p.paragraph_format
        fmt.space_before = Pt(before)
        fmt.space_after = Pt(self.l.space_after_pt if after is None else after)
        fmt.line_spacing = self.l.line_spacing
        if align is not None:
            p.alignment = align
        if text:
            self.runs(p, text, size=size, bold=bold, italic=italic)
        return p

    def runs(self, p, text: str, *, size: float | None = None, bold: bool = False, italic: bool = False) -> None:
        """Add ``text`` to ``p``; ``**term**`` becomes bold type."""
        pos = 0
        text = clean(text)
        for m in _BOLD.finditer(text):
            self._run(p, text[pos : m.start()], size, bold, italic)
            self._run(p, m.group(1), size, True, italic)
            pos = m.end()
        self._run(p, text[pos:], size, bold, italic)

    def _run(self, p, text: str, size, bold: bool, italic: bool) -> None:
        if not text:
            return
        run = p.add_run(text)
        run.bold, run.italic = bold or None, italic or None
        run.font.name = self.l.font
        run.font.size = Pt(size or self.l.body_pt)

    def heading(self, title: str) -> None:
        p = self.para(title.upper(), size=self.l.heading_pt, bold=True, before=self.l.section_gap_pt, after=2.0)
        border = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        for key, value in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"), ("w:color", "000000")):
            bottom.set(qn(key), value)
        border.append(bottom)
        p._p.get_or_add_pPr().append(border)

    def split_line(self, left: str, right: str, *, left_bold: bool = True) -> None:
        """One line with ``left`` at the margin and ``right`` (dates) at the right edge: a tab stop, not a table."""
        p = self.para(after=0.0)
        p.paragraph_format.tab_stops.add_tab_stop(self.width, WD_TAB_ALIGNMENT.RIGHT)
        self.runs(p, left, bold=left_bold)
        if right:
            p.add_run("\t")
            self.runs(p, right)

    def bullets(self, items: list[str]) -> None:
        for item in items:
            p = self.para(item, align=WD_ALIGN_PARAGRAPH.JUSTIFY, style="List Bullet")
            p.paragraph_format.left_indent = Mm(5)
            p.paragraph_format.first_line_indent = Mm(-4)

    def header(self, c: Candidate) -> None:
        self.para(c.name, size=self.l.name_pt, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, after=1.0)
        contact = [x for x in (c.location, c.phone, c.email, c.linkedin, c.github, c.portfolio) if x]
        if contact:
            self.para(" | ".join(contact), size=self.l.body_pt - 0.5, align=WD_ALIGN_PARAGRAPH.CENTER)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))
        return path


def render_cv(payload: CvPayload, path: Path | str, layout: Layout | None = None) -> Path:
    """Write the résumé as a .docx. Sections with nothing in them are left out."""
    w = _Writer(layout or Layout(), payload.page_format)
    w.header(payload.candidate)
    if payload.summary:
        w.heading("Summary")
        w.para(payload.summary, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    if payload.experience:
        w.heading("Experience")
        for e in payload.experience:
            where = ", ".join(x for x in (e.company, e.location) if x)
            w.split_line(f"**{e.role}**, {where}", e.dates, left_bold=False)
            w.bullets(e.bullets)
    if payload.projects:
        w.heading("Projects")
        for pr in payload.projects:
            w.split_line(f"**{pr.name}**" + (f" ({pr.tech})" if pr.tech else ""), pr.url, left_bold=False)
            if pr.description:
                w.para(pr.description, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
            w.bullets(pr.bullets)
    if payload.education:
        w.heading("Education")
        for ed in payload.education:
            where = ", ".join(x for x in (ed.org, ed.location) if x)
            w.split_line(f"**{ed.title}**" + (f", {where}" if where else ""), ed.year, left_bold=False)
            if ed.description:
                w.para(ed.description, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    if payload.skills:
        w.heading("Skills")
        for group in payload.skills:
            w.para(f"**{group.category}:** {', '.join(group.items)}")
    return w.save(path)


def render_cover(letter: CoverLetter, path: Path | str, layout: Layout | None = None) -> Path:
    """Write the cover letter as a .docx: the same header as the résumé, then a plain one-page letter."""
    layout = layout or Layout(body_pt=10.5, margin_mm=20.0, space_after_pt=8.0)
    w = _Writer(layout, "a4")
    w.header(letter.candidate)
    if letter.date:
        w.para(letter.date, before=10.0)
    w.para(letter.company, before=4.0, after=0.0)
    w.para(f"Re: {letter.role}", bold=True)
    w.para(letter.greeting, before=6.0)
    for text in letter.paragraphs:
        w.para(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    w.para(letter.closing, before=6.0, after=0.0)
    w.para(letter.candidate.name, bold=True)
    return w.save(path)
