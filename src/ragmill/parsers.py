"""
Format-specific text extractors.

Each extractor takes a file path and returns its plain-text content.
PDF and DOCX support are optional extras — the imports are lazy so the
core package stays installable with zero dependencies.
"""


def extract_pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF support requires the 'pdf' extra. Install it with: pip install ragmill[pdf]"
        ) from exc

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(page.strip() for page in pages if page.strip())


def extract_docx_text(path: str) -> str:
    try:
        import docx
    except ImportError as exc:
        raise ImportError(
            "DOCX support requires the 'docx' extra. Install it with: pip install ragmill[docx]"
        ) from exc

    document = docx.Document(path)
    paragraphs = [p.text.strip() for p in document.paragraphs]
    return "\n\n".join(p for p in paragraphs if p)
