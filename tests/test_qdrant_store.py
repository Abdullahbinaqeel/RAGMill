"""Integration tests for QdrantVectorStore against a live cluster.

Skipped entirely unless both the 'qdrant' extra is installed and
RAGMILL_QDRANT_URL is set (RAGMILL_QDRANT_API_KEY is optional, for
cloud clusters that require it). Uses a uniquely-named collection per
test run and tears it down afterward so repeated runs stay clean and
nothing is left behind on a shared/live cluster.
"""

import os
import uuid

import numpy as np
import pytest

pytest.importorskip("qdrant_client")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("RAGMILL_QDRANT_URL"),
        reason="requires RAGMILL_QDRANT_URL (and optionally RAGMILL_QDRANT_API_KEY) for a live Qdrant cluster",
    ),
]


def _normalized(vec):
    """Pads to the store's fixed 384-dim schema, then L2-normalizes."""
    vec = list(vec) + [0.0] * (384 - len(vec))
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture(scope="module")
def _qdrant_collection():
    """One collection for the whole module — creating/deleting a collection
    per test hits collection-creation rate limits on free-tier cloud clusters."""
    from ragmill.qdrant_store import QdrantVectorStore

    collection = f"ragmill_test_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(
        url=os.environ["RAGMILL_QDRANT_URL"],
        api_key=os.getenv("RAGMILL_QDRANT_API_KEY"),
        collection_name=collection,
    )
    yield store
    try:
        store.client.delete_collection(collection)
        store.client.delete_collection(store._state_collection)
    finally:
        store.close()


@pytest.fixture
def qdrant_store(_qdrant_collection):
    """Clears both collections' points before each test for isolation,
    without re-creating (and re-rate-limiting) the collection itself."""
    from qdrant_client.http import models

    store = _qdrant_collection
    empty_filter = models.Filter()  # no conditions => matches every point
    for name in (store.collection_name, store._state_collection):
        store.client.delete(
            collection_name=name, points_selector=models.FilterSelector(filter=empty_filter)
        )
    return store


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


def test_add_and_count(qdrant_store):
    payloads = [_payload("a.txt", 0, "alpha content"), _payload("b.txt", 0, "beta content")]
    vectors = np.stack([_normalized([1, 0, 0]), _normalized([0, 1, 0])])
    qdrant_store.add(payloads, vectors)
    assert qdrant_store.count() == 2


def test_search_ranks_by_similarity(qdrant_store):
    payloads = [_payload("a.txt", 0, "alpha"), _payload("b.txt", 0, "beta")]
    vectors = np.stack([_normalized([1, 0, 0]), _normalized([0, 1, 0])])
    qdrant_store.add(payloads, vectors)

    results = qdrant_store.search(_normalized([0.9, 0.1, 0]), top_k=2)
    assert len(results) == 2
    assert results[0]["metadata"]["filename"] == "a.txt"
    assert results[0]["score"] > results[1]["score"]


def test_search_filters_by_filename(qdrant_store):
    payloads = [_payload("a.txt", 0, "alpha"), _payload("b.txt", 0, "beta")]
    vectors = np.stack([_normalized([1, 0, 0]), _normalized([0, 1, 0])])
    qdrant_store.add(payloads, vectors)

    results = qdrant_store.search(_normalized([1, 0, 0]), top_k=5, filename="b.txt")
    assert all(r["metadata"]["filename"] == "b.txt" for r in results)


def test_search_filters_by_modified_at_range(qdrant_store):
    payloads = [
        _payload("old.txt", 0, "old", modified_at=100.0),
        _payload("new.txt", 0, "new", modified_at=200.0),
    ]
    vectors = np.stack([_normalized([1, 0, 0]), _normalized([0, 1, 0])])
    qdrant_store.add(payloads, vectors)

    results = qdrant_store.search(_normalized([1, 1, 0]), top_k=5, modified_after=150.0)
    filenames = {r["metadata"]["filename"] for r in results}
    assert filenames == {"new.txt"}


def test_delete_by_source(qdrant_store):
    payloads = [_payload("a.txt", 0, "alpha", source_file="/docs/a.txt")]
    vectors = np.stack([_normalized([1, 0, 0])])
    qdrant_store.add(payloads, vectors)
    assert qdrant_store.count() == 1

    qdrant_store.delete_by_source("/docs/a.txt")
    assert qdrant_store.count() == 0


def test_delete_missing_sources(qdrant_store):
    payloads = [
        _payload("a.txt", 0, "alpha", source_file="/docs/a.txt"),
        _payload("b.txt", 0, "beta", source_file="/docs/b.txt"),
    ]
    vectors = np.stack([_normalized([1, 0, 0]), _normalized([0, 1, 0])])
    qdrant_store.add(payloads, vectors)

    deleted = qdrant_store.delete_missing_sources({"/docs/a.txt"})
    assert deleted == 1
    assert qdrant_store.count() == 1


def test_file_state_roundtrip(qdrant_store):
    assert qdrant_store.get_file_state("/docs/a.txt") is None
    qdrant_store.upsert_file_state("/docs/a.txt", "hash123", 1234567890.0)
    state = qdrant_store.get_file_state("/docs/a.txt")
    assert state == {"content_hash": "hash123", "modified_at": 1234567890.0}


def test_scroll_pagination(qdrant_store):
    payloads = [_payload(f"f{i}.txt", 0, f"content {i}") for i in range(5)]
    vectors = np.stack([_normalized([float(i), 1.0, 0.0]) for i in range(5)])
    qdrant_store.add(payloads, vectors)

    seen = []
    cursor = None
    while True:
        records, cursor = qdrant_store.scroll(cursor=cursor, limit=2)
        seen.extend(records)
        if cursor is None:
            break
    assert len(seen) == 5
