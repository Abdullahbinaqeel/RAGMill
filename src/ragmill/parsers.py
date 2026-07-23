"""
Format-specific text extractors.

Each extractor takes a file path and returns its plain-text content. All
third-party imports are lazy so the core package stays installable with zero
dependencies — a format's extractor only needs its extra when that format is
actually encountered.

Extras:
  - pdf    → pypdf                          (.pdf, digital text layer)
  - docx   → python-docx                    (.docx)
  - office → beautifulsoup4, striprtf,      (.html/.htm, .rtf, .xlsx, .pptx)
             openpyxl, python-pptx
  - ocr    → pytesseract, pillow            (.png/.jpg/... and scanned .pdf)
             + system binaries: `tesseract`, and `pdftoppm` (poppler) for PDFs

OCR is English-only by default (lang="eng").
"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_OCR_LANG = "eng"
# A PDF whose extracted text layer is shorter than this is treated as
# scanned/image-only and routed to OCR (when enabled).
PDF_OCR_MIN_CHARS = 10


# ── PDF (digital text layer, with OCR fallback for scans) ────────────────────


def extract_pdf_text(
    path: str,
    *,
    enable_ocr: bool = True,
    ocr_lang: str = DEFAULT_OCR_LANG,
    min_chars: int = PDF_OCR_MIN_CHARS,
) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF support requires the 'pdf' extra. Install it with: pip install ragmill[pdf]"
        ) from exc

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(page.strip() for page in pages if page.strip())

    if len(text) >= min_chars or not enable_ocr:
        return text

    # No usable text layer → scanned/image PDF. Rasterise then OCR.
    return _ocr_pdf(path, ocr_lang)


def _ocr_pdf(path: str, ocr_lang: str) -> str:
    if not shutil.which("pdftoppm"):
        raise RuntimeError(
            "Scanned PDF needs poppler's `pdftoppm` to rasterise pages. "
            "Install it with: brew install poppler (or apt-get install poppler-utils)."
        )
    logger.info("%s: no text layer, running OCR (%s)…", os.path.basename(path), ocr_lang)
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "page")
        subprocess.run(
            ["pdftoppm", "-png", "-r", "300", str(path), prefix],
            check=True,
            capture_output=True,
        )
        parts = [_ocr_image_file(png, ocr_lang) for png in sorted(Path(tmp).glob("page*.png"))]
    return "\n\n".join(p for p in parts if p).strip()


# ── DOCX ──────────────────────────────────────────────────────────────────────


def extract_docx_text(path: str) -> str:
    try:
        import docx
    except ImportError as exc:
        raise ImportError(
            "DOCX support requires the 'docx' extra. Install it with: pip install ragmill[docx]"
        ) from exc

    document = docx.Document(path)
    parts = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


# ── Images + shared OCR helper ────────────────────────────────────────────────


def _ocr_image_file(image_path, ocr_lang: str) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "OCR support requires the 'ocr' extra. Install it with: pip install ragmill[ocr] "
            "(and the system `tesseract` binary: brew install tesseract)."
        ) from exc
    if not shutil.which("tesseract"):
        raise RuntimeError(
            "The `tesseract` OCR binary is not on PATH. Install it with: brew install tesseract "
            "(or apt-get install tesseract-ocr)."
        )
    with Image.open(image_path) as img:
        return pytesseract.image_to_string(img, lang=ocr_lang).strip()


def extract_image_text(path: str, *, ocr_lang: str = DEFAULT_OCR_LANG) -> str:
    return _ocr_image_file(path, ocr_lang)


# ── HTML ──────────────────────────────────────────────────────────────────────


def extract_html_text(path: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "HTML support requires the 'office' extra. Install it with: pip install ragmill[office]"
        ) from exc
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n").strip()


# ── RTF ───────────────────────────────────────────────────────────────────────


def extract_rtf_text(path: str) -> str:
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError as exc:
        raise ImportError(
            "RTF support requires the 'office' extra. Install it with: pip install ragmill[office]"
        ) from exc
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return rtf_to_text(f.read()).strip()


# ── XLSX ──────────────────────────────────────────────────────────────────────


def extract_xlsx_text(path: str) -> str:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise ImportError(
            "XLSX support requires the 'office' extra. Install it with: pip install ragmill[office]"
        ) from exc

    workbook = load_workbook(filename=path, read_only=True, data_only=True)
    parts = []
    for sheet in workbook.worksheets:
        parts.append(f"# Sheet: {sheet.title}")
        for row in sheet.iter_rows(values_only=True):
            cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
            if cells:
                parts.append(" | ".join(cells))
    workbook.close()
    return "\n".join(parts).strip()


# ── PPTX ──────────────────────────────────────────────────────────────────────


def extract_pptx_text(path: str) -> str:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise ImportError(
            "PPTX support requires the 'office' extra. Install it with: pip install ragmill[office]"
        ) from exc

    prs = Presentation(path)
    parts = []
    for index, slide in enumerate(prs.slides, 1):
        parts.append(f"# Slide {index}")
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    parts.append(text)
    return "\n\n".join(parts).strip()


# ── CSV / TSV (no extra needed — stdlib) ─────────────────────────────────────


def extract_csv_text(path: str) -> str:
    import csv

    dialect = "excel-tab" if os.path.splitext(path)[1].lower() == ".tsv" else "excel"
    rows = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f, dialect=dialect):
            cells = [c.strip() for c in row if c.strip()]
            if cells:
                rows.append(" | ".join(cells))
    return "\n".join(rows).strip()
