import pytest


@pytest.fixture
def sample_docs_dir(tmp_path):
    (tmp_path / "notes.txt").write_text("Plain text file content.", encoding="utf-8")
    (tmp_path / "readme.md").write_text("# Markdown\n\nSome markdown content.", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"\x00\x01\x02")

    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "report.pdf"
    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "PDF extracted content.")
    c.save()

    pytest.importorskip("docx")
    import docx

    docx_path = tmp_path / "letter.docx"
    document = docx.Document()
    document.add_paragraph("DOCX extracted content.")
    document.save(str(docx_path))

    return tmp_path
