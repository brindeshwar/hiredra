"""Render a payload into its files, and make it fit one page (plan unit S8.5, the owner's piece).

``write_cv`` is the loop: render the .docx, convert it to a PDF, count the pages. Over one page, it first tightens
the layout step by step (``Layout.tighter``); when the layout can go no smaller, it drops the last bullet of the
entry with the most bullets, and tries again. It never touches the wording: a bullet is kept whole or left out.
Without a PDF converter on the machine the .docx is still written and the result says the page count is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hiredra.resume.payload import CoverLetter, CvPayload
from hiredra.resume.pdf import PdfUnavailable, page_count, to_pdf
from hiredra.resume.word import Layout, render_cover, render_cv

MAX_TRIES = 30


@dataclass
class Rendered:
    docx: Path
    pdf: Path | None
    pages: int | None  # None: no PDF could be made, so the page count is unknown
    layout: Layout
    dropped: int = 0  # bullets left out to fit the page
    note: str = ""


def _drop_one_bullet(payload: CvPayload) -> bool:
    """Remove the last bullet of the entry that has the most (projects before experience on a tie); an entry
    keeps at least one. False when nothing more can go."""
    entries = [*payload.projects[::-1], *payload.experience[::-1]]
    longest = max(entries, key=lambda e: len(e.bullets), default=None)
    if longest is None or len(longest.bullets) <= 1:
        return False
    longest.bullets.pop()
    return True


def write_cv(payload: CvPayload, out_dir: Path | str, *, max_pages: int = 1, name: str = "cv") -> Rendered:
    """Write ``<name>.docx`` and ``<name>.pdf`` into ``out_dir``, fitted to ``max_pages``."""
    out_dir = Path(out_dir)
    payload = payload.model_copy(deep=True)
    layout, dropped = Layout(), 0
    for _ in range(MAX_TRIES):
        docx = render_cv(payload, out_dir / f"{name}.docx", layout)
        try:
            pdf = to_pdf(docx)
        except PdfUnavailable as exc:
            return Rendered(docx, None, None, layout, dropped, str(exc))
        pages = page_count(pdf)
        if pages <= max_pages:
            return Rendered(docx, pdf, pages, layout, dropped)
        tighter = layout.tighter()
        if tighter is not None:
            layout = tighter
        elif _drop_one_bullet(payload):
            dropped += 1
        else:
            return Rendered(docx, pdf, pages, layout, dropped, f"still {pages} pages at the smallest layout")
    return Rendered(docx, pdf, pages, layout, dropped, f"gave up after {MAX_TRIES} tries at {pages} pages")


def write_cover(letter: CoverLetter, out_dir: Path | str, *, name: str = "cover") -> Rendered:
    docx = render_cover(letter, Path(out_dir) / f"{name}.docx")
    try:
        pdf = to_pdf(docx)
    except PdfUnavailable as exc:
        return Rendered(docx, None, None, Layout(), note=str(exc))
    return Rendered(docx, pdf, page_count(pdf), Layout())
