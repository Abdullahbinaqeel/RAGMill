"""Tests for the new abstraction layer, config, export/import, and server."""

import json
import os
import tempfile

import numpy as np
import pytest

from ragmill import RAGEngine
from ragmill.vector_store import BaseVectorStore, SQLiteVectorStore, VectorStore, store_from_config
from ragmill.config import RAGMillConfig
from ragmill.export import export_store, import_store

# ── Helpers ────────────────────────────────────────────────────────────────


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


# ── BaseVectorStore ABC ────────────────────────────────────────────────────


def test_base_vector_store_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseVectorStore()  # abstract


def test_sqlite_vector_store_is_valid_subclass():
    assert issubclass(SQLiteVectorStore, BaseVectorStore)
    assert issubclass(VectorStore, BaseVectorStore)


def test_vector_store_alias_still_works():
    store = VectorStore()
    assert isinstance(store, SQLiteVectorStore)
    assert isinstance(store, BaseVectorStore)


# ── Config ─────────────────────────────────────────────────────────────────


def test_config_from_env_defaults():
    # Clear any pre-existing RAGMILL_ vars for this test
    for k in list(os.environ):
        if k.startswith("RAGMILL_"):
            del os.environ[k]
    cfg = RAGMillConfig.from_env()
    assert cfg.store_type == "sqlite"
    assert cfg.chunk_size == 500
    assert cfg.overlap == 50
    assert cfg.server_host == "127.0.0.1"
    assert cfg.server_port == 8000


def test_config_from_env_custom():
    os.environ["RAGMILL_STORE_TYPE"] = "qdrant"
    os.environ["RAGMILL_CHUNK_SIZE"] = "300"
    os.environ["RAGMILL_OVERLAP"] = "30"
    os.environ["RAGMILL_QDRANT_URL"] = "http://localhost:6333"
    os.environ["RAGMILL_PORT"] = "9000"
    try:
        cfg = RAGMillConfig.from_env()
        assert cfg.store_type == "qdrant"
        assert cfg.chunk_size == 300
        assert cfg.overlap == 30
        assert cfg.qdrant_url == "http://localhost:6333"
        assert cfg.server_port == 9000
    finally:
        del os.environ["RAGMILL_STORE_TYPE"]
        del os.environ["RAGMILL_CHUNK_SIZE"]
        del os.environ["RAGMILL_OVERLAP"]
        del os.environ["RAGMILL_QDRANT_URL"]
        del os.environ["RAGMILL_PORT"]


# ── store_from_config ──────────────────────────────────────────────────────


def test_store_from_config_sqlite():
    cfg = RAGMillConfig(store_type="sqlite", sqlite_path=":memory:")
    store = store_from_config(cfg)
    assert isinstance(store, SQLiteVectorStore)


def test_store_from_config_unknown_raises():
    cfg = RAGMillConfig(store_type="nonexistent")
    with pytest.raises(ValueError, match="Unknown store_type"):
        store_from_config(cfg)


# ── Export / import ─────────────────────────────────────────────────────────


def test_export_import_roundtrip(tmp_path):
    store = SQLiteVectorStore()
    payloads = [
        _payload("a.txt", 0, "alpha vector content"),
        _payload("b.txt", 0, "beta vector content"),
    ]
    embeddings = np.stack([_normalized([1, 0, 0]), _normalized([0, 1, 0])])
    store.add(payloads, embeddings)

    export_path = str(tmp_path / "export.jsonl")
    written = export_store(export_path, store)
    assert written == 2

    # Verify file content
    with open(export_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 2
    assert "embedding_b64" in lines[0]
    assert lines[0]["metadata"]["filename"] == "a.txt"

    # Import into a fresh store
    store2 = SQLiteVectorStore()
    imported = import_store(export_path, store2)
    assert imported == 2
    assert store2.count() == 2

    # Search should work on the re-imported data
    query = _normalized([0.9, 0.1, 0])
    results = store2.search(query, top_k=1)
    assert len(results) == 1
    assert "alpha" in results[0]["content"]


# ── L3: .env is gitignored ────────────────────────────────────────────────


def test_gitignore_includes_env():
    gitignore_path = os.path.join(os.path.dirname(__file__), "..", ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path) as f:
            content = f.read()
        assert ".env" in content
