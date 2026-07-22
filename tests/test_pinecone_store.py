"""Integration tests for PineconeVectorStore against a live Pinecone project.

Skipped entirely unless both the 'pinecone' extra is installed and
PINECONE_API_KEY is set. Uses a uniquely-named index per test session
(index creation is slow, so one index is shared across this file's
tests rather than per-test) and cleans up afterward.
"""

import os
import time
import uuid

import numpy as np
import pytest

pytest.importorskip("pinecone")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("PINECONE_API_KEY"),
        reason="requires PINECONE_API_KEY for a live Pinecone project",
    ),
]


def _normalized(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _payload(filename, chunk_index, content, source_file=None, modified_at=None):
    return {
        "metadata": {
            "source_file": source_file or f"/docs/{filename}",
            "filename": filename,
            "chunk_index": chunk_index,
            "modified_at": modified_at,
        },
        "content": content,
    }


@pytest.fixture(scope="module")
def pinecone_index_name():
    return f"ragmill-test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def pinecone_store(pinecone_index_name):
    from ragmill.pinecone_store import PineconeVectorStore

    store = PineconeVectorStore(
        api_key=os.environ["PINECONE_API_KEY"],
        index_name=pinecone_index_name,
    )
    yield store
    # clear this test's data (namespace-scoped) rather than deleting the
    # whole index, which is slow to create/delete repeatedly
    try:
        store.index.delete(delete_all=True)
        store.index.delete(delete_all=True, namespace="__ragmill_file_state__")
    except Exception:
        pass


def test_add_and_count(pinecone_store):
    payloads = [_payload("a.txt", 0, "alpha content"), _payload("b.txt", 0, "beta content")]
    vectors = np.stack([_normalized([1, 0, 0] + [0] * 381), _normalized([0, 1, 0] + [0] * 381)])
    pinecone_store.add(payloads, vectors)
    time.sleep(2)  # Pinecone upserts are eventually consistent
    assert pinecone_store.count() == 2


def test_search_ranks_by_similarity(pinecone_store):
    payloads = [_payload("a.txt", 0, "alpha"), _payload("b.txt", 0, "beta")]
    vectors = np.stack([_normalized([1, 0, 0] + [0] * 381), _normalized([0, 1, 0] + [0] * 381)])
    pinecone_store.add(payloads, vectors)
    time.sleep(2)

    results = pinecone_store.search(_normalized([0.9, 0.1, 0] + [0] * 381), top_k=2)
    assert len(results) == 2
    assert results[0]["metadata"]["filename"] == "a.txt"


def test_search_filters_by_filename(pinecone_store):
    payloads = [_payload("a.txt", 0, "alpha"), _payload("b.txt", 0, "beta")]
    vectors = np.stack([_normalized([1, 0, 0] + [0] * 381), _normalized([0, 1, 0] + [0] * 381)])
    pinecone_store.add(payloads, vectors)
    time.sleep(2)

    results = pinecone_store.search(_normalized([1, 1, 0] + [0] * 381), top_k=5, filename="b.txt")
    assert all(r["metadata"]["filename"] == "b.txt" for r in results)


def test_search_filters_by_modified_at_range(pinecone_store):
    payloads = [
        _payload("old.txt", 0, "old", modified_at=100.0),
        _payload("new.txt", 0, "new", modified_at=200.0),
    ]
    vectors = np.stack([_normalized([1, 0, 0] + [0] * 381), _normalized([0, 1, 0] + [0] * 381)])
    pinecone_store.add(payloads, vectors)
    time.sleep(2)

    results = pinecone_store.search(
        _normalized([1, 1, 0] + [0] * 381), top_k=5, modified_after=150.0
    )
    filenames = {r["metadata"]["filename"] for r in results}
    assert filenames == {"new.txt"}


def test_search_results_include_modified_at(pinecone_store):
    payloads = [_payload("a.txt", 0, "alpha", modified_at=123.0)]
    vectors = np.stack([_normalized([1, 0, 0] + [0] * 381)])
    pinecone_store.add(payloads, vectors)
    time.sleep(2)

    results = pinecone_store.search(_normalized([1, 0, 0] + [0] * 381), top_k=1)
    assert "modified_at" in results[0]["metadata"]


def test_delete_by_source(pinecone_store):
    payloads = [_payload("a.txt", 0, "alpha", source_file="/docs/a.txt")]
    vectors = np.stack([_normalized([1, 0, 0] + [0] * 381)])
    pinecone_store.add(payloads, vectors)
    time.sleep(2)
    assert pinecone_store.count() == 1

    pinecone_store.delete_by_source("/docs/a.txt")
    time.sleep(2)
    assert pinecone_store.count() == 0


def test_file_state_roundtrip(pinecone_store):
    assert pinecone_store.get_file_state("/docs/a.txt") is None
    pinecone_store.upsert_file_state("/docs/a.txt", "hash123", 1234567890.0)
    time.sleep(2)
    state = pinecone_store.get_file_state("/docs/a.txt")
    assert state == {"content_hash": "hash123", "modified_at": 1234567890.0}
