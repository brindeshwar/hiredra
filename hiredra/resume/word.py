"""Render a résumé or a cover letter as an editable Word file (owner decision, 2026-11-13: .docx is the main
output, with a PDF copy made from it -- see ``hiredra.resume.pdf``).

The layout is the owner's example résumé, measured from the file itself: US Letter, narrow margins, Arial 9.5 pt;
a centred name and one contact line with short clickable links; sections in the order Summary, Technical Skills,
Experience, Projects, Education, each under a ruled heading; an experience entry as "Company | Role" with bold
dates at the right edge, an italic line on the company, then justified bullets; a project as a clickable
"Name: Tagline | tech". One column, no tables, text boxes or images (ATS-safe).

Everything that decides how much fits on a page is in ``Layout``, so the page-fit step can tighten it without
touching this code. ``**term**`` in any text becomes bold type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor, Twips

from hiredra.resume.payload import Candidate, CoverLetter, CvPayload

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_PAGES = {"a4": (Mm(210), Mm(297)), "letter": (Mm(215.9), Mm(279.4))}
_LINK_BLUE = RGBColor(0x05, 0x63, 0xC1)


@dataclass(frozen=True)
class Layout:
    """The example's measurements. Points unless the name says otherwise."""

    font: str = "Arial"
    body_pt: float = 9.5
    name_pt: float = 18.0
    contact_pt: float = 9.0
    heading_pt: float = 10.5
    side_mm: float = 12.2
    top_mm: float = 8.8
    bottom_mm: float = 7.9
    heading_before: float = 4.5
    section_gap: float = 11.0  # the example's blank line before every section but the first
    heading_after: float = 2.0
    entry_before: float = 2.5
    bullet_after: float = 0.5

    def tighter(self) -> Layout | None:
        """The next smaller layout for the page-fit loop, or None when it would become unreadable. The blank
        lines between sections go first; type gets smaller only after that."""
        if self.section_gap > 0:
            return replace(self, section_gap=max(0.0, self.section_gap - 5.5))
        if self.body_pt <= 8.5:
            return None
        return replace(
            self,
            body_pt=self.body_pt - 0.5,
            contact_pt=self.contact_pt - 0.5,
            heading_pt=self.heading_pt - 0.5,
            name_pt=self.name_pt - 1.0,
            heading_before=max(3.0, self.heading_before - 0.75),
            entry_before=max(1.5, self.entry_before - 0.5),
        )


def clean(text: str) -> str:
    """The standing rules a renderer can enforce by itself: no em or en dashes, single spaces."""
    return " ".join(text.replace("—", ", ").replace("–", "-").split()).replace(" ,", ",")


def _href(link: str) -> str:
    return link if link.startswith(("http://", "https://", "mailto:")) else f"https://{link}"


class _Writer:
    def __init__(self, layout: Layout, page: str):
        self.l = layout
        self.headings = 0
        self.doc = Document()
        section = self.doc.sections[0]
        section.page_width, section.page_height = _PAGES.get(page, _PAGES["letter"])
        section.left_margin = section.right_margin = Mm(layout.side_mm)
        section.top_margin, section.bottom_margin = Mm(layout.top_mm), Mm(layout.bottom_mm)
        self.width = section.page_width - section.left_margin - section.right_margin
        normal = self.doc.styles["Normal"]
        normal.font.name = layout.font
        normal.font.size = Pt(layout.body_pt)
        normal.element.rPr.rFonts.set(qn("w:eastAsia"), layout.font)
        normal.paragraph_format.space_after = Pt(0)
        normal.paragraph_format.line_spacing = 1.0

    def para(self, text: str = "", *, size: float | None = None, bold: bool = False, italic: bool = False,
             align=None, before: float = 0.0, after: float = 0.0, style: str | None = None):  # fmt: skip
        p = self.doc.add_paragraph(style=style)
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(before), Pt(after)
        p.paragraph_format.line_spacing = 1.0
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

    def _run(self, p, text: str, size, bold: bool, italic: bool):
        if not text:
            return None
        run = p.add_run(text)
        run.bold, run.italic = bold or None, italic or None
        run.font.name = self.l.font
        run.font.size = Pt(size or self.l.body_pt)
        return run

    def link(self, p, text: str, url: str, *, size: float | None = None, bold: bool = False) -> None:
        """A real hyperlink (blue, underlined), as in the example's header and project titles."""
        rel = self.doc.part.relate_to(_href(url), RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
        hyperlink = OxmlElement("w:hyperlink")
        hyperlink.set(qn("r:id"), rel)
        run = self._run(p, clean(text.replace("**", "")), size, bold, False)
        run.font.underline = True
        run.font.color.rgb = _LINK_BLUE
        p._p.remove(run._r)
        hyperlink.append(run._r)
        p._p.append(hyperlink)

    def heading(self, title: str) -> None:
        gap = self.l.section_gap if self.headings else 0.0
        self.headings += 1
        p = self.para(title.upper(), size=self.l.heading_pt, bold=True, before=self.l.heading_before + gap,
                      after=self.l.heading_after)  # fmt: skip
        border = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        for key, value in (("w:val", "single"), ("w:sz", "6"), ("w:space", "1"), ("w:color", "000000")):
            bottom.set(qn(key), value)
        border.append(bottom)
        p._p.get_or_add_pPr().append(border)

    def entry_line(self, before: float):
        """A heading line whose right-hand text (dates) sits at the right edge: a tab stop, not a table."""
        p = self.para(before=before)
        p.paragraph_format.tab_stops.add_tab_stop(self.width, WD_TAB_ALIGNMENT.RIGHT)
        return p

    def right(self, p, text: str) -> None:
        if text:
            p.add_run("\t")
            self.runs(p, text, bold=True)

    def bullets(self, items: list[str]) -> None:
        for item in items:
            p = self.para(item, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=self.l.bullet_after, style="List Bullet")
            p.paragraph_format.left_indent = Twips(250)
            p.paragraph_format.first_line_indent = Twips(-180)

    def header(self, c: Candidate) -> None:
        self.para(c.name, size=self.l.name_pt, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
        p = self.para(align=WD_ALIGN_PARAGRAPH.CENTER, after=1.0)
        parts = [(x, None) for x in (c.email, c.phone, c.location) if x]
        parts += [(link.split("/")[0], link) for link in (c.linkedin, c.github, c.portfolio) if link]
        for i, (text, url) in enumerate(parts):
            if i:
                self._run(p, " | ", self.l.contact_pt, False, False)
            if url:
                self.link(p, text.removeprefix("https://").removeprefix("www."), url, size=self.l.contact_pt)
            else:
                self._run(p, text, self.l.contact_pt, False, False)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(path))
        return path


def render_cv(payload: CvPayload, path: Path | str, layout: Layout | None = None) -> Path:
    """Write the résumé as a .docx in the example's section order. Sections with nothing in them are left out."""
    w = _Writer(layout or Layout(), payload.page_format)
    lay = w.l
    w.header(payload.candidate)
    if payload.summary:
        w.heading("Summary")
        w.para(payload.summary, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    if payload.skills:
        w.heading("Technical Skills")
        for group in payload.skills:
            w.para(f"**{group.category}:** {', '.join(group.items)}")
    if payload.experience:
        w.heading("Experience")
        for i, e in enumerate(payload.experience):
            p = w.entry_line(lay.entry_before + (1.0 if i else 0.0))
            w.runs(p, f"**{e.company}** | {e.role}" + (f", {e.location}" if e.location else ""))
            w.right(p, e.dates)
            if e.about:
                w.para(e.about, italic=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=lay.bullet_after)
            w.bullets(e.bullets)
    if payload.projects:
        w.heading("Projects")
        for i, pr in enumerate(payload.projects):
            p = w.entry_line(lay.entry_before - (1.0 if not i else 0.0))
            title = pr.name + (f": {pr.tagline}" if pr.tagline else "")
            if pr.url:
                w.link(p, title, pr.url, bold=True)
            else:
                w.runs(p, title, bold=True)
            if pr.tech:
                w.runs(p, f" | {pr.tech}")
            if pr.description:
                w.para(pr.description, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=lay.bullet_after)
            w.bullets(pr.bullets)
    if payload.education:
        w.heading("Education")
        for ed in payload.education:
            p = w.entry_line(lay.entry_before - 1.0)
            where = ", ".join(x for x in (ed.org, ed.location) if x)
            w.runs(p, f"**{ed.title}**" + (f" | {where}" if where else ""))
            w.right(p, ed.year)
            if ed.description:
                w.para(ed.description)
    return w.save(path)


def render_cover(letter: CoverLetter, path: Path | str, layout: Layout | None = None) -> Path:
    """Write the cover letter as a .docx: the same header as the résumé, then a plain one-page letter."""
    layout = layout or Layout(body_pt=10.5, contact_pt=9.5, side_mm=22.0, top_mm=18.0, bottom_mm=18.0)
    w = _Writer(layout, "letter")
    w.header(letter.candidate)
    if letter.date:
        w.para(letter.date, before=14.0)
    w.para(letter.company, before=6.0)
    w.para(f"Re: {letter.role}", bold=True, after=8.0)
    w.para(letter.greeting, after=8.0)
    for text in letter.paragraphs:
        w.para(text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=8.0)
    w.para(letter.closing, before=6.0)
    w.para(letter.candidate.name, bold=True)
    return w.save(path)
