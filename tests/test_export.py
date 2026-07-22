"""Deeper tests for ragmill.export (export_store/import_store), beyond the
single roundtrip smoke test in test_abstractions.py."""

import base64
import json

import numpy as np
import pytest

from ragmill.export import export_store, import_store, CHUNK_SCHEMA_VERSION
from ragmill.vector_store import SQLiteVectorStore


class _FakeStore:
    """A minimal scroll()-only store for exercising export_store's
    batching/skip logic without depending on a real backend's scroll shape."""

    def __init__(self, records, page_size=None):
        self._records = records
        self._page_size = page_size or len(records) or 1

    def scroll(self, cursor=None, limit=100):
        start = cursor or 0
        page = self._records[start : start + self._page_size]
        # each returned record must be independent so callers popping keys
        # (export.py pops "embedding") doesn't corrupt our source data
        page = [dict(r) for r in page]
        next_cursor = start + self._page_size
        if next_cursor >= len(self._records):
            next_cursor = None
        return page, next_cursor


def _record(filename, content, embedding=(0.1, 0.2, 0.3)):
    rec = {
        "metadata": {
            "source_file": f"/docs/{filename}",
            "filename": filename,
            "chunk_index": 0,
            "modified_at": 123.0,
        },
        "content": content,
    }
    if embedding is not None:
        rec["embedding"] = list(embedding)
    return rec


def test_export_writes_exact_jsonl_schema(tmp_path):
    store = _FakeStore([_record("a.txt", "alpha content")])
    path = str(tmp_path / "out.jsonl")
    written = export_store(path, store)
    assert written == 1

    with open(path) as f:
        line = json.loads(f.readline())

    assert set(line.keys()) == {"metadata", "content", "embedding_b64", "schema_version"}
    assert line["schema_version"] == CHUNK_SCHEMA_VERSION == 1
    assert line["content"] == "alpha content"
    assert line["metadata"]["filename"] == "a.txt"

    decoded = np.frombuffer(base64.b64decode(line["embedding_b64"]), dtype=np.float32)
    assert np.allclose(decoded, [0.1, 0.2, 0.3], atol=1e-6)


def test_export_skips_records_missing_embedding(tmp_path):
    store = _FakeStore(
        [
            _record("a.txt", "has embedding", embedding=(1.0, 0.0, 0.0)),
            _record("b.txt", "no embedding", embedding=None),
            _record("c.txt", "also has embedding", embedding=(0.0, 1.0, 0.0)),
        ]
    )
    path = str(tmp_path / "out.jsonl")
    written = export_store(path, store)

    assert written == 2
    with open(path) as f:
        filenames = [json.loads(line)["metadata"]["filename"] for line in f]
    assert filenames == ["a.txt", "c.txt"]


def test_export_empty_store_writes_empty_file(tmp_path):
    store = _FakeStore([])
    path = str(tmp_path / "out.jsonl")
    written = export_store(path, store)
    assert written == 0
    assert open(path).read() == ""


@pytest.mark.parametrize("batch_size,n_records", [(1, 3), (3, 3), (5, 3), (2, 4)])
def test_export_batch_size_boundaries(tmp_path, batch_size, n_records):
    records = [_record(f"f{i}.txt", f"content {i}") for i in range(n_records)]
    store = _FakeStore(records, page_size=batch_size)
    path = str(tmp_path / "out.jsonl")
    written = export_store(path, store, batch_size=batch_size)
    assert written == n_records
    with open(path) as f:
        assert sum(1 for _ in f) == n_records


def test_import_batch_size_boundaries(tmp_path):
    lines = []
    for i in range(5):
        vec = np.array([float(i), 0.0, 0.0], dtype=np.float32)
        lines.append(
            json.dumps(
                {
                    "metadata": {
                        "source_file": f"/d/f{i}.txt",
                        "filename": f"f{i}.txt",
                        "chunk_index": 0,
                        "modified_at": None,
                    },
                    "content": f"content {i}",
                    "embedding_b64": base64.b64encode(vec.tobytes()).decode("ascii"),
                    "schema_version": 1,
                }
            )
        )
    path = tmp_path / "in.jsonl"
    path.write_text("\n".join(lines) + "\n")

    store = SQLiteVectorStore()
    imported = import_store(str(path), store, batch_size=2)  # 5 records, batch of 2 -> 2+2+1
    assert imported == 5
    assert store.count() == 5
    store.close()


def test_import_missing_embedding_b64_raises(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"metadata": {}, "content": "x"}) + "\n")
    store = SQLiteVectorStore()
    with pytest.raises(ValueError, match="embedding_b64"):
        import_store(str(path), store)
    store.close()


def test_import_recomputes_character_length_from_content(tmp_path):
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    path = tmp_path / "in.jsonl"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "source_file": "/d/a.txt",
                    "filename": "a.txt",
                    "chunk_index": 0,
                    "character_length": 9999,
                    "modified_at": None,
                },  # deliberately wrong
                "content": "short",
                "embedding_b64": base64.b64encode(vec.tobytes()).decode("ascii"),
                "schema_version": 1,
            }
        )
        + "\n"
    )

    store = SQLiteVectorStore()
    import_store(str(path), store)
    results = store.search(vec, top_k=1)
    assert results[0]["content"] == "short"
    store.close()


def test_full_roundtrip_via_real_sqlite_store(tmp_path):
    src = SQLiteVectorStore()
    payloads = [
        {
            "metadata": {
                "source_file": "/docs/a.txt",
                "filename": "a.txt",
                "chunk_index": 0,
                "character_length": 5,
                "modified_at": None,
            },
            "content": "alpha",
        },
        {
            "metadata": {
                "source_file": "/docs/b.txt",
                "filename": "b.txt",
                "chunk_index": 0,
                "character_length": 4,
                "modified_at": None,
            },
            "content": "beta",
        },
    ]
    vectors = np.stack(
        [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        ]
    )
    src.add(payloads, vectors)

    path = str(tmp_path / "roundtrip.jsonl")
    written = export_store(path, src)
    assert written == 2

    dst = SQLiteVectorStore()
    imported = import_store(path, dst)
    assert imported == 2
    assert dst.count() == src.count() == 2

    src.close()
    dst.close()
