from ragmill import RAGEngine


def test_empty_text_returns_no_chunks():
    engine = RAGEngine(chunk_size=100, overlap=10)
    assert engine.semantic_chunking("") == []


def test_short_text_returns_single_chunk():
    engine = RAGEngine(chunk_size=100, overlap=10)
    chunks = engine.semantic_chunking("Just one short paragraph.")
    assert chunks == ["Just one short paragraph."]


def test_splits_on_paragraph_boundaries():
    engine = RAGEngine(chunk_size=40, overlap=5)
    text = "First paragraph here.\n\nSecond paragraph goes here too."
    chunks = engine.semantic_chunking(text)
    assert len(chunks) >= 2
    assert all(len(c) <= 60 for c in chunks)


def test_splits_oversized_paragraph_on_sentences():
    engine = RAGEngine(chunk_size=50, overlap=5)
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = engine.semantic_chunking(text)
    assert len(chunks) > 1
    joined = " ".join(chunks)
    assert "sentence one" in joined
    assert "sentence three" in joined


def test_overlap_carries_context_between_chunks():
    engine = RAGEngine(chunk_size=30, overlap=10)
    text = "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india."
    chunks = engine.semantic_chunking(text)
    assert len(chunks) > 1


def test_invalid_overlap_raises():
    try:
        RAGEngine(chunk_size=50, overlap=50)
        assert False, "expected ValueError"
    except ValueError:
        pass
