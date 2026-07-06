import numpy as np
import pytest

from ragmill.vector_store import VectorStore


def _payload(filename, chunk_index, content, source_file=None, modified_at=None):
    return {
        "metadata": {
            "source_file": source_file or f"/docs/{filename}",
            "filename": filename,
            "chunk_index": chunk_index,
            "character_length": len(content),
            "modified_at": modified_at,
        },
        "content": content,
    }


def _normalized(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_empty_store_search_returns_nothing():
    store = VectorStore()
    results = store.search(_normalized([1.0, 0.0, 0.0]), top_k=5)
    assert results == []


def test_add_and_count():
    store = VectorStore()
    payloads = [_payload("a.txt", 0, "alpha"), _payload("a.txt", 1, "beta")]
    embeddings = np.stack([_normalized([1, 0, 0]), _normalized([0, 1, 0])])

    store.add(payloads, embeddings)

    assert store.count() == 2


def test_search_ranks_closest_vector_first():
    store = VectorStore()
    payloads = [
        _payload("a.txt", 0, "points toward x"),
        _payload("b.txt", 0, "points toward y"),
        _payload("c.txt", 0, "points toward z"),
    ]
    embeddings = np.stack([
        _normalized([1, 0, 0]),
        _normalized([0, 1, 0]),
        _normalized([0, 0, 1]),
    ])
    store.add(payloads, embeddings)

    query = _normalized([0.9, 0.1, 0])
    results = store.search(query, top_k=2)

    assert results[0]["metadata"]["filename"] == "a.txt"
    assert results[0]["score"] > results[1]["score"]
    assert len(results) == 2


def test_add_raises_on_mismatched_lengths():
    store = VectorStore()
    payloads = [_payload("a.txt", 0, "alpha")]
    embeddings = np.stack([_normalized([1, 0, 0]), _normalized([0, 1, 0])])

    with pytest.raises(ValueError):
        store.add(payloads, embeddings)


def test_persists_across_reconnect(tmp_path):
    db_path = tmp_path / "store.db"
    store = VectorStore(db_path)
    store.add([_payload("a.txt", 0, "alpha")], np.stack([_normalized([1, 0, 0])]))
    store.close()

    reopened = VectorStore(db_path)
    assert reopened.count() == 1


def test_search_filters_by_filename():
    store = VectorStore()
    payloads = [
        _payload("a.txt", 0, "points toward x"),
        _payload("b.txt", 0, "points toward x too"),
    ]
    embeddings = np.stack([_normalized([1, 0, 0]), _normalized([1, 0, 0])])
    store.add(payloads, embeddings)

    results = store.search(_normalized([1, 0, 0]), top_k=5, filename="b.txt")

    assert len(results) == 1
    assert results[0]["metadata"]["filename"] == "b.txt"


def test_search_filters_by_source_file():
    store = VectorStore()
    payloads = [
        _payload("a.txt", 0, "alpha", source_file="/docs/project1/a.txt"),
        _payload("a.txt", 0, "alpha again", source_file="/docs/project2/a.txt"),
    ]
    embeddings = np.stack([_normalized([1, 0, 0]), _normalized([1, 0, 0])])
    store.add(payloads, embeddings)

    results = store.search(_normalized([1, 0, 0]), top_k=5, source_file="/docs/project2/a.txt")

    assert len(results) == 1
    assert results[0]["metadata"]["source_file"] == "/docs/project2/a.txt"


def test_search_filters_by_modified_at_range():
    store = VectorStore()
    payloads = [
        _payload("old.txt", 0, "old content", modified_at=1000.0),
        _payload("new.txt", 0, "new content", modified_at=2000.0),
    ]
    embeddings = np.stack([_normalized([1, 0, 0]), _normalized([1, 0, 0])])
    store.add(payloads, embeddings)

    only_new = store.search(_normalized([1, 0, 0]), top_k=5, modified_after=1500.0)
    only_old = store.search(_normalized([1, 0, 0]), top_k=5, modified_before=1500.0)

    assert [r["metadata"]["filename"] for r in only_new] == ["new.txt"]
    assert [r["metadata"]["filename"] for r in only_old] == ["old.txt"]


def test_file_state_roundtrip():
    store = VectorStore()
    assert store.get_file_state("/docs/a.txt") is None

    store.upsert_file_state("/docs/a.txt", "hash1", 1000.0)
    state = store.get_file_state("/docs/a.txt")
    assert state == {"content_hash": "hash1", "modified_at": 1000.0}

    store.upsert_file_state("/docs/a.txt", "hash2", 2000.0)
    assert store.get_file_state("/docs/a.txt") == {"content_hash": "hash2", "modified_at": 2000.0}


def test_delete_by_source_removes_chunks_and_file_state():
    store = VectorStore()
    store.add([_payload("a.txt", 0, "alpha", source_file="/docs/a.txt")], np.stack([_normalized([1, 0, 0])]))
    store.upsert_file_state("/docs/a.txt", "hash1", 1000.0)

    store.delete_by_source("/docs/a.txt")

    assert store.count() == 0
    assert store.get_file_state("/docs/a.txt") is None


def test_delete_missing_sources_removes_only_stale_entries():
    store = VectorStore()
    store.upsert_file_state("/docs/keep.txt", "hash1", 1000.0)
    store.upsert_file_state("/docs/stale.txt", "hash2", 1000.0)
    store.add([_payload("stale.txt", 0, "gone", source_file="/docs/stale.txt")], np.stack([_normalized([1, 0, 0])]))

    removed = store.delete_missing_sources({"/docs/keep.txt"})

    assert removed == 1
    assert store.get_file_state("/docs/stale.txt") is None
    assert store.get_file_state("/docs/keep.txt") is not None
    assert store.count() == 0
