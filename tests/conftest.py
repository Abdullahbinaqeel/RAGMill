import os

import pytest

_ENV_PREFIXES = ("RAGMILL_", "QDRANT_", "PINECONE_", "GEMINI_", "GOOGLE_")


@pytest.fixture(autouse=True)
def _snapshot_env():
    """Prevents env var mutations in one test from leaking into the next."""
    saved = dict(os.environ)
    yield
    for key in list(os.environ):
        if key not in saved and key.startswith(_ENV_PREFIXES):
            del os.environ[key]
    os.environ.update(saved)


@pytest.fixture
def sqlite_store():
    from ragmill.vector_store import SQLiteVectorStore

    store = SQLiteVectorStore()
    yield store
    store.close()


@pytest.fixture
def populated_store(sqlite_store, sample_docs_dir):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    from ragmill import RAGEngine
    from ragmill.embeddings import EmbeddingModel

    engine = RAGEngine()
    model = EmbeddingModel()
    chunks = engine.execute_pipeline(str(sample_docs_dir))
    if chunks:
        vectors = model.embed([c["content"] for c in chunks])
        sqlite_store.add(chunks, vectors)
    return sqlite_store


@pytest.fixture
def temp_env_file(tmp_path):
    """A not-yet-existing .env path under a tmp dir, for config/config-UI tests."""
    return tmp_path / ".env"


@pytest.fixture
def mock_llm():
    """A fake generate_answer(query, chunks, config=None) -> str. Callers monkeypatch it onto
    whichever module imported the real one (e.g. ragmill.server, ragmill.__main__)
    so tests never load the real model."""
    calls = []

    def _fake(query, chunks, config=None):
        calls.append((query, chunks))
        return f"MOCKED ANSWER for: {query}"

    _fake.calls = calls
    return _fake


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
