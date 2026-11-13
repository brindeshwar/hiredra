"""The PDF copy of a Word file, and its page count.

The .docx is the document of record; the PDF is what a form upload wants. Conversion needs a program that can lay
out a Word file, and it uses the first one this machine has:

1. LibreOffice (``soffice``), free and available on every platform;
2. Microsoft Word, on Windows, driven through its own automation interface.

With neither, ``to_pdf`` raises ``PdfUnavailable``: the .docx is still written, and the packet is marked as having
no PDF. ``page_count`` reads the PDF with pypdf, which is what the page-fit step measures.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader

_SOFFICE_PATHS = (
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    "/usr/bin/soffice",
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
)
#: Word's own "export as PDF" (17 = wdExportFormatPDF), opened read-only and never shown.
_WORD_SCRIPT = (
    "$ErrorActionPreference = 'Stop'; $w = New-Object -ComObject Word.Application; $w.Visible = $false; "
    "try {{ $d = $w.Documents.Open('{src}', $false, $true); "
    "try {{ $d.ExportAsFixedFormat('{dst}', 17) }} finally {{ $d.Close($false) }} }} finally {{ $w.Quit() }}"
)


class PdfUnavailable(RuntimeError):
    """No program on this machine can turn a Word file into a PDF, or the conversion failed."""


def _soffice() -> str | None:
    return shutil.which("soffice") or next((p for p in _SOFFICE_PATHS if Path(p).exists()), None)


def converter() -> str | None:
    """Which converter this machine has: ``libreoffice``, ``word`` or None."""
    if _soffice():
        return "libreoffice"
    if sys.platform == "win32" and shutil.which("powershell"):
        probe = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "[bool](Get-ItemProperty 'HKLM:\\SOFTWARE\\Classes\\Word.Application' -ErrorAction SilentlyContinue)"],
            capture_output=True, text=True, timeout=30,
        )  # fmt: skip
        if probe.stdout.strip().lower() == "true":
            return "word"
    return None


def to_pdf(docx_path: Path | str, pdf_path: Path | str | None = None, *, timeout: int = 180) -> Path:
    """Convert ``docx_path`` to a PDF beside it (or at ``pdf_path``)."""
    src = Path(docx_path).resolve()
    dst = Path(pdf_path).resolve() if pdf_path else src.with_suffix(".pdf")
    which = converter()
    if which is None:
        raise PdfUnavailable("no LibreOffice or Microsoft Word on this machine: the .docx is kept, no PDF is made")
    try:
        if which == "libreoffice":
            subprocess.run(
                [_soffice(), "--headless", "--convert-to", "pdf", "--outdir", str(dst.parent), str(src)],
                check=True, capture_output=True, timeout=timeout,
            )  # fmt: skip
            made = dst.parent / (src.stem + ".pdf")
            if made != dst:
                made.replace(dst)
        else:
            script = _WORD_SCRIPT.format(src=str(src).replace("'", "''"), dst=str(dst).replace("'", "''"))
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", script], check=True, capture_output=True, timeout=timeout
            )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        detail = getattr(exc, "stderr", b"") or b""
        raise PdfUnavailable(
            f"{which} could not convert {src.name}: {detail.decode(errors='replace')[:300] or exc}"
        ) from exc
    if not dst.exists():
        raise PdfUnavailable(f"{which} reported success but wrote no PDF for {src.name}")
    return dst


def page_count(pdf_path: Path | str) -> int:
    return len(PdfReader(str(pdf_path)).pages)
