from ragmill import RAGEngine


def test_stream_directory_reads_txt_and_md(sample_docs_dir):
    engine = RAGEngine()
    results = {r["filename"]: r["raw_content"] for r in engine.stream_directory(str(sample_docs_dir))}

    assert results["notes.txt"] == "Plain text file content."
    assert "Some markdown content." in results["readme.md"]


def test_stream_directory_skips_unsupported_extensions(sample_docs_dir):
    engine = RAGEngine()
    filenames = [r["filename"] for r in engine.stream_directory(str(sample_docs_dir))]

    assert "ignored.bin" not in filenames


def test_stream_directory_extracts_pdf_text(sample_docs_dir):
    engine = RAGEngine()
    results = {r["filename"]: r["raw_content"] for r in engine.stream_directory(str(sample_docs_dir))}

    assert "PDF extracted content." in results["report.pdf"]


def test_stream_directory_extracts_docx_text(sample_docs_dir):
    engine = RAGEngine()
    results = {r["filename"]: r["raw_content"] for r in engine.stream_directory(str(sample_docs_dir))}

    assert "DOCX extracted content." in results["letter.docx"]


def test_stream_directory_raises_on_missing_path():
    engine = RAGEngine()
    try:
        list(engine.stream_directory("/definitely/not/a/real/path"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_execute_pipeline_produces_chunks_with_metadata(sample_docs_dir):
    engine = RAGEngine(chunk_size=200, overlap=20)
    payloads = engine.execute_pipeline(str(sample_docs_dir))

    assert len(payloads) > 0
    for payload in payloads:
        assert "source_file" in payload["metadata"]
        assert "filename" in payload["metadata"]
        assert "chunk_index" in payload["metadata"]
        assert payload["content"]
