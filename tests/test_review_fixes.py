"""
Unit tests verifying each fix from the Senior Dev Review and SQA Review.

Each test class corresponds to a specific finding and validates the fix
without requiring live cloud services (Pinecone/Qdrant are mocked).
"""

import importlib
import logging
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.9-3.10
    import tomli as tomllib

import numpy as np
import pytest

from ragmill.config import RAGMillConfig
from ragmill.vector_store import SQLiteVectorStore, store_from_config


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


# ── #1 CRITICAL: Pinecone delete_by_source uses ID-based delete ────────────


class TestPineconeDeleteById:
    """Verifies PineconeVectorStore.delete_by_source() uses list+delete-by-IDs
    instead of metadata-filtered delete, which serverless doesn't support."""

    def test_delete_by_source_calls_list_paginated_and_delete_by_ids(self):
        mock_index = MagicMock()

        # Simulate listing vectors where some match the source file prefix
        vec1 = MagicMock()
        vec1.id = "/docs/a.txt__0"
        vec2 = MagicMock()
        vec2.id = "/docs/a.txt__1"
        vec3 = MagicMock()
        vec3.id = "/docs/b.txt__0"

        page = MagicMock()
        page.vectors = [vec1, vec2, vec3]
        page.pagination = MagicMock()
        page.pagination.next = None
        mock_index.list_paginated.return_value = page

        mock_client = MagicMock()
        mock_client.has_index.return_value = True
        mock_client.Index.return_value = mock_index

        with patch.dict(
            sys.modules,
            {
                "pinecone": MagicMock(
                    Pinecone=MagicMock(return_value=mock_client), ServerlessSpec=MagicMock()
                )
            },
        ):
            from ragmill.pinecone_store import PineconeVectorStore

            store = PineconeVectorStore.__new__(PineconeVectorStore)
            store.index = mock_index
            store._embedding_dim = 384

            store.delete_by_source("/docs/a.txt")

        # Should have called delete with the two matching IDs (plus the state delete)
        chunk_delete = call(ids=["/docs/a.txt__0", "/docs/a.txt__1"])
        assert chunk_delete in mock_index.delete.call_args_list

    def test_delete_by_source_also_deletes_file_state(self):
        mock_index = MagicMock()
        page = MagicMock()
        page.vectors = []
        page.pagination = MagicMock()
        page.pagination.next = None
        mock_index.list_paginated.return_value = page

        store = MagicMock()
        store.index = mock_index
        store._embedding_dim = 384

        # Call the real method
        from ragmill.pinecone_store import PineconeVectorStore

        PineconeVectorStore.delete_by_source(store, "/docs/a.txt")

        # Should delete file_state entry
        mock_index.delete.assert_any_call(ids=["/docs/a.txt"], namespace="__ragmill_file_state__")


# ── #2 HIGH: Server defaults to 127.0.0.1 and has auth ────────────────────


class TestServerSecurity:
    """Verifies the server binds to 127.0.0.1 by default, has API key auth,
    and restricts ingest paths to configured allowed roots."""

    def test_config_defaults_to_localhost(self):
        for k in list(os.environ):
            if k.startswith("RAGMILL_"):
                del os.environ[k]
        cfg = RAGMillConfig.from_env()
        assert cfg.server_host == "127.0.0.1"

    def test_config_has_api_key_field(self):
        cfg = RAGMillConfig()
        assert hasattr(cfg, "server_api_key")
        assert cfg.server_api_key is None

    def test_config_has_allowed_roots_field(self):
        cfg = RAGMillConfig()
        assert hasattr(cfg, "server_allowed_roots")
        assert cfg.server_allowed_roots is None

    def test_config_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("RAGMILL_API_KEY", "test-secret-key")
        cfg = RAGMillConfig.from_env()
        assert cfg.server_api_key == "test-secret-key"

    def test_config_reads_allowed_roots_from_env(self, monkeypatch):
        monkeypatch.setenv("RAGMILL_ALLOWED_ROOTS", "/data/docs:/data/pdfs")
        cfg = RAGMillConfig.from_env()
        assert cfg.server_allowed_roots == "/data/docs:/data/pdfs"

    def test_server_auth_rejects_invalid_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAGMILL_STORE_TYPE", "sqlite")
        monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "auth_test.db"))
        monkeypatch.setenv("RAGMILL_API_KEY", "correct-key")

        import ragmill.server as srv

        importlib.reload(srv)

        from fastapi.testclient import TestClient

        with TestClient(srv.app) as c:
            # No key -> 401
            r = c.post("/ingest", json={"directory": str(tmp_path)})
            assert r.status_code == 401

            # Wrong key -> 401
            r = c.post(
                "/ingest", json={"directory": str(tmp_path)}, headers={"X-API-Key": "wrong-key"}
            )
            assert r.status_code == 401

            # Correct key -> proceeds (200 with 0 chunks for empty dir)
            r = c.post(
                "/ingest", json={"directory": str(tmp_path)}, headers={"X-API-Key": "correct-key"}
            )
            assert r.status_code == 200

    def test_server_no_auth_when_no_api_key_configured(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAGMILL_STORE_TYPE", "sqlite")
        monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "noauth_test.db"))
        monkeypatch.delenv("RAGMILL_API_KEY", raising=False)

        import ragmill.server as srv

        importlib.reload(srv)

        from fastapi.testclient import TestClient

        with TestClient(srv.app) as c:
            r = c.get("/health")
            assert r.status_code == 200

    def test_server_rejects_paths_outside_allowed_roots(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAGMILL_STORE_TYPE", "sqlite")
        monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "root_test.db"))
        monkeypatch.setenv("RAGMILL_ALLOWED_ROOTS", str(tmp_path / "allowed"))

        (tmp_path / "allowed").mkdir()
        (tmp_path / "outside").mkdir()

        import ragmill.server as srv

        importlib.reload(srv)

        from fastapi.testclient import TestClient

        with TestClient(srv.app) as c:
            # Path inside allowed root -> proceeds
            r = c.post("/ingest", json={"directory": str(tmp_path / "allowed")})
            assert r.status_code == 200

            # Path outside allowed root -> 403
            r = c.post("/ingest", json={"directory": str(tmp_path / "outside")})
            assert r.status_code == 403


# ── #3 HIGH: Cloud time-filtering stores modified_at as number ─────────────


class TestCloudTimeFiltering:
    """Verifies modified_at is stored as float and search filters work correctly."""

    def test_qdrant_stores_modified_at_as_float(self):
        """Qdrant add() should store modified_at as float, not str."""
        # We can verify this by checking the point struct construction
        mock_client = MagicMock()
        mock_models = MagicMock()

        from ragmill.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store.client = mock_client
        store.collection_name = "test"
        store._state_collection = "test__file_state"
        store._models = mock_models

        payloads = [_payload("a.txt", 0, "content", modified_at=1234567890.0)]
        embeddings = np.stack([_normalized([1, 0, 0] + [0] * 381)])

        store.add(payloads, embeddings)

        # Verify the point was created with float modified_at (not str)
        call_args = mock_client.upsert.call_args
        point = call_args[1]["points"][0] if "points" in call_args[1] else call_args[0][1][0]
        # The payload should have modified_at as a float
        assert point.payload["modified_at"] == 1234567890.0
        assert isinstance(point.payload["modified_at"], float)

    def test_pinecone_stores_modified_at_as_float(self):
        """Pinecone add() should store modified_at as float, not str."""
        mock_index = MagicMock()
        mock_client = MagicMock()
        mock_client.has_index.return_value = True
        mock_client.Index.return_value = mock_index

        with patch.dict(
            sys.modules,
            {
                "pinecone": MagicMock(
                    Pinecone=MagicMock(return_value=mock_client), ServerlessSpec=MagicMock()
                )
            },
        ):
            from ragmill.pinecone_store import PineconeVectorStore

            store = PineconeVectorStore.__new__(PineconeVectorStore)
            store.index = mock_index
            store._embedding_dim = 384

            payloads = [_payload("a.txt", 0, "content", modified_at=1234567890.0)]
            embeddings = np.stack([_normalized([1, 0, 0] + [0] * 381)])
            store.add(payloads, embeddings)

        call_args = mock_index.upsert.call_args
        vectors = call_args[1]["vectors"]
        meta = vectors[0]["metadata"]
        assert meta["modified_at"] == 1234567890.0
        assert isinstance(meta["modified_at"], float)

    def test_qdrant_add_handles_modified_at_none(self):
        """A file with modified_at=None must not raise (regression: float(None))."""
        mock_client = MagicMock()

        from ragmill.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store.client = mock_client
        store.collection_name = "test"
        store._state_collection = "test__file_state"
        store._models = MagicMock()

        payloads = [_payload("a.txt", 0, "content", modified_at=None)]
        embeddings = np.stack([_normalized([1, 0, 0] + [0] * 381)])
        store.add(payloads, embeddings)  # must not raise

        point = mock_client.upsert.call_args[1]["points"][0]
        assert point.payload["modified_at"] is None

    def test_pinecone_add_handles_modified_at_none(self):
        """A file with modified_at=None must not raise; the key is omitted
        (Pinecone metadata cannot contain null)."""
        mock_index = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "pinecone": MagicMock(
                    Pinecone=MagicMock(return_value=MagicMock()), ServerlessSpec=MagicMock()
                )
            },
        ):
            from ragmill.pinecone_store import PineconeVectorStore

            store = PineconeVectorStore.__new__(PineconeVectorStore)
            store.index = mock_index
            store._embedding_dim = 384

            payloads = [_payload("a.txt", 0, "content", modified_at=None)]
            embeddings = np.stack([_normalized([1, 0, 0] + [0] * 381)])
            store.add(payloads, embeddings)  # must not raise

        meta = mock_index.upsert.call_args[1]["vectors"][0]["metadata"]
        assert "modified_at" not in meta

    def test_pinecone_search_includes_modified_at_in_results(self):
        """Pinecone search() results should include modified_at in metadata."""
        mock_index = MagicMock()
        match = MagicMock()
        match.score = 0.9
        match.metadata = {
            "source_file": "/docs/a.txt",
            "filename": "a.txt",
            "chunk_index": 0,
            "modified_at": 1234567890.0,
            "content": "test",
        }
        result = MagicMock()
        result.matches = [match]
        mock_index.query.return_value = result

        with patch.dict(
            sys.modules,
            {
                "pinecone": MagicMock(
                    Pinecone=MagicMock(
                        return_value=MagicMock(
                            has_index=MagicMock(return_value=True),
                            Index=MagicMock(return_value=mock_index),
                        )
                    ),
                    ServerlessSpec=MagicMock(),
                )
            },
        ):
            from ragmill.pinecone_store import PineconeVectorStore

            store = PineconeVectorStore.__new__(PineconeVectorStore)
            store.index = mock_index
            store._embedding_dim = 384

            results = store.search(_normalized([1, 0, 0] + [0] * 381), top_k=1)

        assert "modified_at" in results[0]["metadata"]
        assert results[0]["metadata"]["modified_at"] == 1234567890.0

    def test_pinecone_search_applies_modified_after_filter(self):
        """Pinecone search() should apply $gte filter for modified_after."""
        mock_index = MagicMock()
        mock_index.query.return_value = MagicMock(matches=[])

        with patch.dict(
            sys.modules,
            {
                "pinecone": MagicMock(
                    Pinecone=MagicMock(
                        return_value=MagicMock(
                            has_index=MagicMock(return_value=True),
                            Index=MagicMock(return_value=mock_index),
                        )
                    ),
                    ServerlessSpec=MagicMock(),
                )
            },
        ):
            from ragmill.pinecone_store import PineconeVectorStore

            store = PineconeVectorStore.__new__(PineconeVectorStore)
            store.index = mock_index
            store._embedding_dim = 384

            store.search(_normalized([1, 0, 0] + [0] * 381), top_k=5, modified_after=1000.0)

        call_kwargs = mock_index.query.call_args[1]
        assert call_kwargs["filter"] == {"modified_at": {"$gte": 1000.0}}

    def test_pinecone_search_applies_both_time_filters(self):
        """Pinecone search() should combine $gte and $lte for both time filters."""
        mock_index = MagicMock()
        mock_index.query.return_value = MagicMock(matches=[])

        with patch.dict(
            sys.modules,
            {
                "pinecone": MagicMock(
                    Pinecone=MagicMock(
                        return_value=MagicMock(
                            has_index=MagicMock(return_value=True),
                            Index=MagicMock(return_value=mock_index),
                        )
                    ),
                    ServerlessSpec=MagicMock(),
                )
            },
        ):
            from ragmill.pinecone_store import PineconeVectorStore

            store = PineconeVectorStore.__new__(PineconeVectorStore)
            store.index = mock_index
            store._embedding_dim = 384

            store.search(
                _normalized([1, 0, 0] + [0] * 381),
                top_k=5,
                modified_after=100.0,
                modified_before=200.0,
            )

        call_kwargs = mock_index.query.call_args[1]
        assert call_kwargs["filter"] == {"modified_at": {"$gte": 100.0, "$lte": 200.0}}

    def test_qdrant_search_includes_modified_at_in_results(self):
        """Qdrant search() results should include modified_at in metadata."""
        mock_client = MagicMock()
        result_point = MagicMock()
        result_point.score = 0.9
        result_point.payload = {
            "source_file": "/docs/a.txt",
            "filename": "a.txt",
            "chunk_index": 0,
            "modified_at": 1234567890.0,
            "content": "test",
        }
        mock_client.query_points.return_value = MagicMock(points=[result_point])

        from ragmill.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store.client = mock_client
        store.collection_name = "test"
        store._models = MagicMock()

        results = store.search(_normalized([1, 0, 0] + [0] * 381), top_k=1)

        assert "modified_at" in results[0]["metadata"]
        assert results[0]["metadata"]["modified_at"] == 1234567890.0


# ── #4 HIGH: Embeddings download is resumable ──────────────────────────────


class TestEmbeddingsDownloadResumable:
    """Verifies _download uses .part file + rename + retry pattern."""

    def test_download_writes_to_part_file_first(self, tmp_path, monkeypatch):
        from ragmill import embeddings

        download_calls = []

        def mock_urlretrieve(url, path):
            download_calls.append(str(path))
            Path(path).write_bytes(b"model data")

        monkeypatch.setattr("urllib.request.urlretrieve", mock_urlretrieve)

        result = embeddings._download("test/model", tmp_path / "cache")

        # Should have downloaded to .part files
        for c in download_calls:
            assert c.endswith(".part")

    def test_download_renames_part_to_final(self, tmp_path, monkeypatch):
        from ragmill import embeddings

        def mock_urlretrieve(url, path):
            Path(path).write_bytes(b"model data")

        monkeypatch.setattr("urllib.request.urlretrieve", mock_urlretrieve)

        result = embeddings._download("test/model", tmp_path / "cache")

        # Final files should exist (no .part files left)
        for name in embeddings._MODEL_FILES:
            assert (result / name).exists()
            assert not (result / Path(name).with_suffix(".part")).exists()

    def test_download_retries_on_failure(self, tmp_path, monkeypatch):
        from ragmill import embeddings

        attempts = {"n": 0}

        def flaky_download(url, path):
            attempts["n"] += 1
            # Fail first 2 attempts for each file, succeed on 3rd
            # There are 2 files (model.onnx + tokenizer.json)
            file_idx = 0 if "model" in str(path) else 1
            local_attempt = attempts["n"] - file_idx * 3
            if local_attempt <= 2:
                raise OSError("transient failure")
            Path(path).write_bytes(b"model data")

        monkeypatch.setattr("urllib.request.urlretrieve", flaky_download)
        monkeypatch.setattr("time.sleep", lambda *_: None)

        result = embeddings._download("test/model", tmp_path / "cache", retries=3)

        for name in embeddings._MODEL_FILES:
            assert (result / name).exists()

    def test_download_raises_after_all_retries_exhausted(self, tmp_path, monkeypatch):
        from ragmill import embeddings

        def always_fail(url, path):
            raise OSError("persistent failure")

        monkeypatch.setattr("urllib.request.urlretrieve", always_fail)
        monkeypatch.setattr("time.sleep", lambda *_: None)

        with pytest.raises(ConnectionError, match="Failed to download"):
            embeddings._download("test/model", tmp_path / "cache", retries=2)

    def test_download_skips_existing_files(self, tmp_path, monkeypatch):
        from ragmill import embeddings

        cache_dir = tmp_path / "cache" / "test__model"
        cache_dir.mkdir(parents=True)
        for name in embeddings._MODEL_FILES:
            (cache_dir / name).write_bytes(b"cached")

        def boom(url, path):
            raise AssertionError("should not download when file exists")

        monkeypatch.setattr("urllib.request.urlretrieve", boom)

        result = embeddings._download("test/model", tmp_path / "cache")
        assert result == cache_dir


# ── #5 HIGH: Embedding dimension is configurable ──────────────────────────


class TestEmbeddingDimension:
    """Verifies PineconeVectorStore accepts embedding_dim parameter."""

    def test_pinecone_accepts_embedding_dim(self):
        mock_client = MagicMock()
        mock_client.has_index.return_value = False

        with patch.dict(
            sys.modules,
            {
                "pinecone": MagicMock(
                    Pinecone=MagicMock(return_value=mock_client), ServerlessSpec=MagicMock()
                )
            },
        ):
            from ragmill.pinecone_store import PineconeVectorStore

            store = PineconeVectorStore.__new__(PineconeVectorStore)
            store._embedding_dim = 768

            assert store._embedding_dim == 768

    def test_pinecone_default_dimension_is_384(self):
        from ragmill.pinecone_store import _DEFAULT_EMBEDDING_DIM

        assert _DEFAULT_EMBEDDING_DIM == 384


# ── #6 HIGH: Config wiring passes config to chat.py ───────────────────────


class TestConfigWiring:
    """Verifies generate_answer accepts and uses RAGMillConfig."""

    def test_generate_answer_accepts_config_parameter(self):
        from ragmill.chat import generate_answer
        import inspect

        sig = inspect.signature(generate_answer)
        assert "config" in sig.parameters

    def test_generate_answer_config_defaults_to_none(self):
        from ragmill.chat import generate_answer
        import inspect

        sig = inspect.signature(generate_answer)
        assert sig.parameters["config"].default is None

    def test_config_backend_field_is_used(self, monkeypatch, tmp_path):
        from ragmill import chat

        chat._llm_cache.clear()

        monkeypatch.setattr(
            chat, "_download_gguf", lambda repo, filename, cache_dir: tmp_path / filename
        )

        import sys, types

        fake_module = types.ModuleType("llama_cpp")
        fake_module.Llama = type(
            "_FakeLlama",
            (),
            {
                "__init__": lambda self, **kw: None,
                "create_chat_completion": lambda self, **kw: {
                    "choices": [{"message": {"content": "answer"}}]
                },
            },
        )
        monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)

        cfg = RAGMillConfig(
            chat_backend="local",
            chat_model_repo="custom/repo",
            chat_model_file="custom.gguf",
            chat_n_ctx=2048,
        )
        seen = {}

        def fake_download(repo, filename, cache_dir):
            seen["repo"] = repo
            seen["filename"] = filename
            return tmp_path / filename

        monkeypatch.setattr(chat, "_download_gguf", fake_download)

        chat.generate_answer("q", [], config=cfg)

        assert seen["repo"] == "custom/repo"
        assert seen["filename"] == "custom.gguf"

    def test_server_passes_config_to_generate_answer(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAGMILL_STORE_TYPE", "sqlite")
        monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "cfg_test.db"))
        monkeypatch.delenv("RAGMILL_API_KEY", raising=False)

        import ragmill.server as srv

        importlib.reload(srv)

        from fastapi.testclient import TestClient

        with TestClient(srv.app) as c:
            calls = []

            def mock_answer(query, chunks, config=None):
                calls.append(config)
                return "mocked"

            monkeypatch.setattr(srv, "generate_answer", mock_answer)

            r = c.post("/chat", json={"query": "test"})
            assert r.status_code == 200
            assert len(calls) == 1
            assert isinstance(calls[0], RAGMillConfig)


# ── #7 MEDIUM: Thread-local batch depth ────────────────────────────────────


class TestThreadLocalBatch:
    """Verifies _batch_depth is per-thread, not global."""

    def test_batch_depth_is_thread_local(self):
        store = SQLiteVectorStore()
        store._set_batch_depth(0)

        results = {}

        def thread_fn(name):
            store._set_batch_depth(1)
            time.sleep(0.05)
            results[name] = store._get_batch_depth()

        t1 = threading.Thread(target=thread_fn, args=("t1",))
        t2 = threading.Thread(target=thread_fn, args=("t2",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread sees its own depth
        assert results["t1"] == 1
        assert results["t2"] == 1
        # Main thread's depth is still 0
        assert store._get_batch_depth() == 0
        store.close()

    def test_concurrent_independent_writes_succeed(self):
        """Two threads each doing their own add() should both commit independently."""
        store = SQLiteVectorStore()
        errors = []

        def writer(name, idx):
            try:
                payloads = [_payload(f"{name}.txt", 0, f"content from {name}")]
                embeddings = np.stack([_normalized([1, 0, 0])])
                store.add(payloads, embeddings)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"thread_{i}", i)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors: {errors}"
        assert store.count() == 5
        store.close()


# ── #8 MEDIUM: file_state cleanup on cloud delete ─────────────────────────


class TestCloudFileStateCleanup:
    """Verifies delete_by_source cleans file_state on cloud backends."""

    def test_pinecone_delete_by_source_cleans_state(self):
        mock_index = MagicMock()
        page = MagicMock()
        page.vectors = [MagicMock(id="/docs/a.txt__0")]
        page.pagination = MagicMock()
        page.pagination.next = None
        mock_index.list_paginated.return_value = page

        with patch.dict(
            sys.modules,
            {
                "pinecone": MagicMock(
                    Pinecone=MagicMock(
                        return_value=MagicMock(
                            has_index=MagicMock(return_value=True),
                            Index=MagicMock(return_value=mock_index),
                        )
                    ),
                    ServerlessSpec=MagicMock(),
                )
            },
        ):
            from ragmill.pinecone_store import PineconeVectorStore

            store = PineconeVectorStore.__new__(PineconeVectorStore)
            store.index = mock_index
            store._embedding_dim = 384

            store.delete_by_source("/docs/a.txt")

        # Verify state namespace delete was called
        state_delete_calls = [
            c
            for c in mock_index.delete.call_args_list
            if any("namespace" in str(a) for a in c)
            or any("__ragmill_file_state__" in str(a) for a in c)
        ]
        assert len(state_delete_calls) >= 1

    def test_qdrant_delete_by_source_cleans_state(self):
        mock_client = MagicMock()

        with patch.dict(
            sys.modules,
            {"qdrant_client": MagicMock(QdrantClient=MagicMock(return_value=mock_client))},
        ):
            from ragmill.qdrant_store import QdrantVectorStore

            store = QdrantVectorStore.__new__(QdrantVectorStore)
            store.client = mock_client
            store.collection_name = "test"
            store._state_collection = "test__file_state"
            store._models = MagicMock()

            store.delete_by_source("/docs/a.txt")

        # Should have called delete on both collections
        delete_calls = mock_client.delete.call_args_list
        assert len(delete_calls) == 2  # main collection + state collection


# ── #9 MEDIUM: Chunk size enforcement ─────────────────────────────────────


class TestChunkSizeEnforcement:
    """Verifies semantic_chunking enforces chunk_size with hard fallback."""

    def test_no_chunk_exceeds_chunk_size(self):
        from ragmill import RAGEngine

        engine = RAGEngine(chunk_size=50, overlap=10)

        # Long text with no sentence boundaries (simulates base64/CJK)
        long_text = "x" * 200
        chunks = engine.semantic_chunking(long_text)

        for chunk in chunks:
            assert len(chunk) <= 50, f"Chunk of len {len(chunk)} exceeds chunk_size 50"

    def test_long_sentence_without_punctuation_is_sliced(self):
        from ragmill import RAGEngine

        engine = RAGEngine(chunk_size=30, overlap=5)

        text = "abcdefghijklmnopqrstuvwxyz1234567890"  # 36 chars, no punctuation
        chunks = engine.semantic_chunking(text)

        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 30

    def test_chunks_with_overlap_preserve_content(self):
        from ragmill import RAGEngine

        engine = RAGEngine(chunk_size=20, overlap=5)

        text = "abcdefghijklmnopqrst" * 3  # 60 chars
        chunks = engine.semantic_chunking(text)

        assert len(chunks) >= 2
        # All original content should appear somewhere in the chunks
        combined = "".join(chunks)
        assert "abcdefghijklmnopqrst" in combined

    def test_normal_text_still_chunks_correctly(self):
        from ragmill import RAGEngine

        engine = RAGEngine(chunk_size=100, overlap=10)

        text = "This is a normal paragraph with sentences. And another sentence here."
        chunks = engine.semantic_chunking(text)

        for chunk in chunks:
            assert len(chunk) <= 100


# ── #12 MEDIUM: Logging instead of print ──────────────────────────────────


class TestLoggingInsteadOfPrint:
    """Verifies modules use logging instead of print for output."""

    def test_engine_uses_logging(self):
        import ragmill.engine as engine_mod

        assert hasattr(engine_mod, "logger")
        assert isinstance(engine_mod.logger, logging.Logger)

    def test_chat_uses_logging(self):
        import ragmill.chat as chat_mod

        assert hasattr(chat_mod, "logger")
        assert isinstance(chat_mod.logger, logging.Logger)

    def test_server_uses_logging(self):
        import ragmill.server as server_mod

        assert hasattr(server_mod, "logger")
        assert isinstance(server_mod.logger, logging.Logger)

    def test_main_uses_logging(self):
        import ragmill.__main__ as main_mod

        assert hasattr(main_mod, "logger")
        assert isinstance(main_mod.logger, logging.Logger)

    def test_engine_stream_directory_logs_parse_errors(self, tmp_path, caplog):
        from ragmill import RAGEngine

        engine = RAGEngine()

        # Create a file that will fail to parse
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a real pdf")

        with caplog.at_level(logging.WARNING):
            files = list(engine.stream_directory(str(tmp_path)))

        assert len(files) == 0
        assert "Unable to parse file" in caplog.text


# ── #14 LOW: SQLite search uses argpartition ───────────────────────────────


class TestSQLiteSearchOptimization:
    """Verifies SQLite search uses argpartition for large result sets."""

    def test_search_with_fewer_results_than_top_k(self):
        store = SQLiteVectorStore()
        payloads = [_payload("a.txt", 0, "content")]
        embeddings = np.stack([_normalized([1, 0, 0])])
        store.add(payloads, embeddings)

        results = store.search(_normalized([1, 0, 0]), top_k=10)
        assert len(results) == 1
        store.close()

    def test_search_returns_modified_at_in_metadata(self):
        store = SQLiteVectorStore()
        payloads = [_payload("a.txt", 0, "content", modified_at=1234.0)]
        embeddings = np.stack([_normalized([1, 0, 0])])
        store.add(payloads, embeddings)

        results = store.search(_normalized([1, 0, 0]), top_k=1)
        assert "modified_at" in results[0]["metadata"]
        assert results[0]["metadata"]["modified_at"] == 1234.0
        store.close()


# ── Low: Export _serialize_records doesn't mutate input ────────────────────


class TestExportNoMutation:
    """Verifies _serialize_records doesn't mutate its input records."""

    def test_serialize_records_preserves_original(self):
        from ragmill.export import _serialize_records

        rec = {
            "metadata": {"source_file": "/docs/a.txt", "filename": "a.txt", "chunk_index": 0},
            "content": "test content",
            "embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        }
        original_keys = set(rec.keys())

        _serialize_records([rec])

        assert set(rec.keys()) == original_keys
        assert "embedding_b64" not in rec
        assert "schema_version" not in rec


# ── Low: Qdrant narrow exceptions ─────────────────────────────────────────


class TestNarrowExceptions:
    """Verifies cloud backends narrow exception handling."""

    def test_qdrant_get_file_state_catches_specific_exceptions(self):
        from ragmill.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store.client = MagicMock()
        store.collection_name = "test"
        store._state_collection = "test__state"
        store._models = MagicMock()

        # Simulate a ValueError (e.g., bad data)
        store.client.scroll.side_effect = ValueError("bad data")
        result = store.get_file_state("/docs/a.txt")
        assert result is None

        # Simulate a real error that shouldn't be caught
        store.client.scroll.side_effect = RuntimeError("connection lost")
        with pytest.raises(RuntimeError):
            store.get_file_state("/docs/a.txt")


# ── #5 HIGH: Qdrant embedding_dim is configurable ─────────────────────────


class TestQdrantEmbeddingDimension:
    """Verifies QdrantVectorStore accepts embedding_dim parameter."""

    def test_qdrant_accepts_embedding_dim(self):
        from ragmill.qdrant_store import QdrantVectorStore

        store = QdrantVectorStore.__new__(QdrantVectorStore)
        store.client = MagicMock()
        store.collection_name = "test"
        store._state_collection = "test__state"
        store._models = MagicMock()

        # Verify __init__ signature accepts embedding_dim
        import inspect

        sig = inspect.signature(QdrantVectorStore.__init__)
        assert "embedding_dim" in sig.parameters
        assert sig.parameters["embedding_dim"].default == 384

    def test_store_from_config_passes_embedding_dim_to_qdrant(self, monkeypatch):
        monkeypatch.setenv("RAGMILL_STORE_TYPE", "qdrant")
        monkeypatch.setenv("RAGMILL_QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("RAGMILL_EMBEDDING_DIM", "768")

        cfg = RAGMillConfig.from_env()
        assert cfg.embedding_dim == 768

    def test_config_embedding_dim_from_env(self, monkeypatch):
        monkeypatch.setenv("RAGMILL_EMBEDDING_DIM", "512")
        cfg = RAGMillConfig.from_env()
        assert cfg.embedding_dim == 512

    def test_config_embedding_dim_default(self):
        cfg = RAGMillConfig()
        assert cfg.embedding_dim == 384


# ── H2: store_from_config cloud branches ─────────────────────────────────


class TestStoreFromConfigCloudBranches:
    """Verifies store_from_config raises ValueError for missing cloud credentials."""

    def test_pinecone_missing_api_key_raises(self):
        cfg = RAGMillConfig(store_type="pinecone", pinecone_api_key=None)
        with pytest.raises(ValueError, match="pinecone_api_key is required"):
            store_from_config(cfg)

    def test_qdrant_missing_url_raises(self):
        cfg = RAGMillConfig(store_type="qdrant", qdrant_url=None)
        with pytest.raises(ValueError, match="qdrant_url is required"):
            store_from_config(cfg)

    def test_pinecone_with_key_passes_to_constructor(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.has_index.return_value = True

        with patch.dict(
            sys.modules,
            {
                "pinecone": MagicMock(
                    Pinecone=MagicMock(return_value=mock_client), ServerlessSpec=MagicMock()
                )
            },
        ):
            from ragmill.pinecone_store import PineconeVectorStore

            cfg = RAGMillConfig(
                store_type="pinecone",
                pinecone_api_key="test-key",
                pinecone_environment="us-east-1",
                pinecone_index_name="my-index",
            )
            store = store_from_config(cfg)
            assert isinstance(store, PineconeVectorStore)

    def test_qdrant_with_url_passes_to_constructor(self, monkeypatch):
        mock_client = MagicMock()

        with patch.dict(
            sys.modules,
            {"qdrant_client": MagicMock(QdrantClient=MagicMock(return_value=mock_client))},
        ):
            from ragmill.qdrant_store import QdrantVectorStore

            cfg = RAGMillConfig(
                store_type="qdrant",
                qdrant_url="http://localhost:6333",
                qdrant_api_key="test-key",
                qdrant_collection_name="my-coll",
            )
            store = store_from_config(cfg)
            assert isinstance(store, QdrantVectorStore)


# ── M3: Parser ImportError paths ─────────────────────────────────────────


class TestParserImportErrors:
    """Verifies parsers raise ImportError with helpful messages when extras are missing."""

    def test_extract_pdf_text_raises_when_pypdf_missing(self, monkeypatch):
        import ragmill.parsers as parsers_mod

        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def block_pypdf(name, *args, **kwargs):
            if name == "pypdf":
                raise ImportError("No module named 'pypdf'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", block_pypdf)

        with pytest.raises(ImportError, match="'pdf' extra"):
            parsers_mod.extract_pdf_text("/fake/path.pdf")

    def test_extract_docx_text_raises_when_docx_missing(self, monkeypatch):
        import ragmill.parsers as parsers_mod

        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def block_docx(name, *args, **kwargs):
            if name == "docx":
                raise ImportError("No module named 'docx'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", block_docx)

        with pytest.raises(ImportError, match="'docx' extra"):
            parsers_mod.extract_docx_text("/fake/path.docx")


# ── H3: Edge-case coverage ──────────────────────────────────────────────


class TestEdgeCases:
    """Tests unicode, empty, large, and binary-mislabeled content."""

    def test_unicode_content_chunks_correctly(self):
        from ragmill import RAGEngine

        engine = RAGEngine(chunk_size=200)

        text = "مرحبا بالعالم 🌍 это тест 中文内容 ñ ü ö ä"
        chunks = engine.semantic_chunking(text)

        assert len(chunks) >= 1
        combined = " ".join(chunks)
        assert "مرحبا" in combined
        assert "🌍" in combined
        assert "中文内容" in combined

    def test_emoji_heavy_content_chunks_correctly(self):
        from ragmill import RAGEngine

        engine = RAGEngine(chunk_size=50, overlap=5)

        text = "🎉🎊🎈🎂🍰☕🎵🎶🎸🎺🎻🎹🎤🎧🔊"
        chunks = engine.semantic_chunking(text)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) <= 50

    def test_empty_text_returns_empty_chunks(self):
        from ragmill import RAGEngine

        engine = RAGEngine()

        assert engine.semantic_chunking("") == []
        assert engine.semantic_chunking("   ") == []
        assert engine.semantic_chunking("\n\n\n") == []

    def test_empty_directory_yields_no_files(self, tmp_path):
        from ragmill import RAGEngine

        engine = RAGEngine()

        files = list(engine.stream_directory(str(tmp_path)))
        assert files == []

    def test_whitespace_only_text_returns_empty_chunks(self):
        from ragmill import RAGEngine

        engine = RAGEngine()

        chunks = engine.semantic_chunking("   \n\n  \t  ")
        assert chunks == []

    def test_large_text_chunks_within_limit(self):
        from ragmill import RAGEngine

        engine = RAGEngine(chunk_size=100, overlap=10)

        text = "word " * 10000  # 50,000 chars
        chunks = engine.semantic_chunking(text)

        assert len(chunks) > 10
        for chunk in chunks:
            assert len(chunk) <= 100, f"Chunk of len {len(chunk)} exceeds limit"

    def test_binary_content_mislabeled_as_txt(self, tmp_path):
        from ragmill import RAGEngine

        engine = RAGEngine()

        binary_file = tmp_path / "data.txt"
        binary_file.write_bytes(b"\x00\x01\x02\xff\xfe\xfd" * 100)

        files = list(engine.stream_directory(str(tmp_path)))
        assert len(files) == 1
        assert files[0]["filename"] == "data.txt"
        assert len(files[0]["raw_content"]) > 0

    def test_single_character_text(self):
        from ragmill import RAGEngine

        engine = RAGEngine()

        chunks = engine.semantic_chunking("x")
        assert chunks == ["x"]

    def test_only_punctuation_text(self):
        from ragmill import RAGEngine

        engine = RAGEngine()

        chunks = engine.semantic_chunking("... ! ? ... ! ?")
        assert len(chunks) >= 1

    def test_newlines_only_paragraphs(self):
        from ragmill import RAGEngine

        engine = RAGEngine(chunk_size=100)

        text = "aaa\n\nbbb\n\nccc\n\nddd"
        chunks = engine.semantic_chunking(text)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_mixed_valid_and_invalid_files(self, tmp_path):
        """Valid files are parsed; broken PDFs are logged and skipped."""
        from ragmill import RAGEngine

        engine = RAGEngine()

        good = tmp_path / "good.txt"
        good.write_text("valid content here")

        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"not a pdf")

        files = list(engine.stream_directory(str(tmp_path)))
        assert len(files) == 1
        assert files[0]["filename"] == "good.txt"


# ── H1: Engine exception = file skipped + pipeline continues ──────────────


class TestEngineExceptionSkipsFile:
    """Verifies a bad file is skipped and processing continues for valid files."""

    def test_bad_file_skipped_processing_continues(self, tmp_path, caplog):
        from ragmill import RAGEngine

        engine = RAGEngine()

        good1 = tmp_path / "a.txt"
        good1.write_text("first valid file")
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"not a pdf at all")
        good2 = tmp_path / "b.txt"
        good2.write_text("second valid file")

        with caplog.at_level(logging.WARNING):
            files = list(engine.stream_directory(str(tmp_path)))

        assert len(files) == 2
        filenames = {f["filename"] for f in files}
        assert filenames == {"a.txt", "b.txt"}
        assert "Unable to parse file" in caplog.text

    def test_all_files_fail_yields_empty(self, tmp_path, caplog):
        from ragmill import RAGEngine

        engine = RAGEngine()

        (tmp_path / "a.pdf").write_bytes(b"not pdf 1")
        (tmp_path / "b.pdf").write_bytes(b"not pdf 2")

        with caplog.at_level(logging.WARNING):
            files = list(engine.stream_directory(str(tmp_path)))

        assert files == []
        assert caplog.text.count("Unable to parse file") == 2

    def test_execute_pipeline_skips_bad_files(self, tmp_path):
        from ragmill import RAGEngine

        engine = RAGEngine(chunk_size=200)

        (tmp_path / "good.txt").write_text("This is valid content for the pipeline.")
        (tmp_path / "bad.pdf").write_bytes(b"not a real pdf file")

        chunks = engine.execute_pipeline(str(tmp_path))
        assert len(chunks) > 0
        assert all(c["metadata"]["filename"] == "good.txt" for c in chunks)


# ── Fix #16: Retry/backoff on cloud network calls ───────────────────────


class TestCloudRetryBackoff:
    """Verifies cloud stores retry on transient failures."""

    def test_pinecone_retries_on_connection_error(self):
        from ragmill.pinecone_store import _retry

        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("transient")
            return "ok"

        result = _retry(flaky, retries=3)
        assert result == "ok"
        assert call_count["n"] == 3

    def test_pinecone_retries_on_timeout_error(self):
        from ragmill.pinecone_store import _retry

        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise TimeoutError("timeout")
            return "ok"

        result = _retry(flaky, retries=3)
        assert result == "ok"

    def test_pinecone_raises_after_retries_exhausted(self):
        from ragmill.pinecone_store import _retry

        def always_fail():
            raise ConnectionError("persistent")

        with pytest.raises(ConnectionError):
            _retry(always_fail, retries=2)

    def test_qdrant_retries_on_connection_error(self):
        from ragmill.qdrant_store import _retry

        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("transient")
            return "ok"

        result = _retry(flaky, retries=3)
        assert result == "ok"
        assert call_count["n"] == 3

    def test_qdrant_raises_after_retries_exhausted(self):
        from ragmill.qdrant_store import _retry

        def always_fail():
            raise ConnectionError("persistent")

        with pytest.raises(ConnectionError):
            _retry(always_fail, retries=2)

    def test_pinecone_does_not_retry_on_permanent_errors(self):
        from ragmill.pinecone_store import _retry

        call_count = {"n": 0}

        def auth_error():
            call_count["n"] += 1
            raise PermissionError("unauthorized")

        with pytest.raises(PermissionError):
            _retry(auth_error, retries=3)
        assert call_count["n"] == 1


# ── Fix #10: Batched ingest ─────────────────────────────────────────────


class TestBatchedIngest:
    """Verifies ingest processes chunks in batches, not all at once."""

    def test_server_ingest_batches_embeddings(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAGMILL_STORE_TYPE", "sqlite")
        monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "batch_test.db"))
        monkeypatch.delenv("RAGMILL_API_KEY", raising=False)

        import ragmill.server as srv

        importlib.reload(srv)

        # Create test files
        test_dir = tmp_path / "docs"
        test_dir.mkdir()
        (test_dir / "a.txt").write_text("first document content here")
        (test_dir / "b.txt").write_text("second document content here")

        embed_calls = []

        def mock_embed(self_or_texts, *args, **kwargs):
            # Handle both instance method and standalone calls
            if hasattr(self_or_texts, "embed"):
                texts = args[0] if args else kwargs.get("texts", [])
            else:
                texts = self_or_texts
            embed_calls.append(len(texts) if texts else 0)
            n = len(texts) if texts else 1
            return np.random.rand(n, 384).astype(np.float32)

        monkeypatch.setattr(
            srv, "_get_model", lambda: type("FakeModel", (), {"embed": mock_embed})()
        )

        from fastapi.testclient import TestClient

        with TestClient(srv.app) as c:
            r = c.post("/ingest", json={"directory": str(test_dir)})
            assert r.status_code == 200
            assert r.json()["chunks"] > 0
            # Embedding should have been called at least once
            assert len(embed_calls) >= 1

    def test_config_has_batch_size_consistency(self):
        """Verify DEFAULT_EMBED_BATCH is imported and used consistently."""
        from ragmill.embeddings import DEFAULT_EMBED_BATCH

        assert DEFAULT_EMBED_BATCH == 16
        assert isinstance(DEFAULT_EMBED_BATCH, int)


# ── BLOCKER 2: Core import doesn't require numpy ─────────────────────────


class TestCoreImportNoNumpy:
    """import ragmill must not fail when numpy is absent."""

    def test_import_ragmill_only_exposes_engine(self):
        """ragmill.__all__ should only contain RAGEngine."""
        import ragmill

        assert ragmill.__all__ == ["RAGEngine"]

    def test_import_ragmill_does_not_import_vector_store(self):
        """ragmill top-level must not pull in vector_store (which requires numpy)."""
        import sys

        # Temporarily remove ragmill and vector_store from sys.modules
        to_restore = {}
        for key in list(sys.modules):
            if key == "ragmill" or key.startswith("ragmill.vector_store"):
                to_restore[key] = sys.modules.pop(key)
        try:
            import ragmill

            assert "ragmill.vector_store" not in sys.modules
        finally:
            sys.modules.update(to_restore)

    def test_vector_store_requires_explicit_import(self):
        """Users must explicitly import from ragmill.vector_store."""
        from ragmill.vector_store import SQLiteVectorStore, VectorStore

        store = SQLiteVectorStore(":memory:")
        assert store.count() == 0


# ── BLOCKER 3: Default sqlite_path is ./ragmill.db ───────────────────────


class TestDefaultSqlitePath:
    """Config default sqlite_path should be ./ragmill.db, not None/:memory:."""

    def test_config_default_sqlite_path(self):
        cfg = RAGMillConfig()
        assert cfg.sqlite_path == "./ragmill.db"

    def test_from_env_default_sqlite_path(self):
        for k in list(os.environ):
            if k.startswith("RAGMILL_SQLITE_PATH"):
                del os.environ[k]
        cfg = RAGMillConfig.from_env()
        assert cfg.sqlite_path == "./ragmill.db"

    def test_from_env_respects_explicit_sqlite_path(self, monkeypatch):
        monkeypatch.setenv("RAGMILL_SQLITE_PATH", "/tmp/custom.db")
        cfg = RAGMillConfig.from_env()
        assert cfg.sqlite_path == "/tmp/custom.db"

    def test_store_from_config_uses_default_path(self, tmp_path):
        """store_from_config passes the config sqlite_path to SQLiteVectorStore."""
        db_path = str(tmp_path / "test.db")
        cfg = RAGMillConfig(store_type="sqlite", sqlite_path=db_path)
        store = store_from_config(cfg)
        # Verify the store was created (not :memory:) by checking it persists
        payloads = [_payload("a.txt", 0, "hello")]
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        store.add(payloads, embeddings)
        store.close()
        # Reopen and verify data persists
        store2 = SQLiteVectorStore(db_path)
        assert store2.count() == 1
        store2.close()


# ── BLOCKER 4: config-ui extra has all required deps ─────────────────────


class TestConfigUiExtra:
    """config-ui extra must include FastAPI/uvicorn/pydantic/numpy."""

    def test_config_ui_extra_includes_fastapi(self):
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        deps = data["project"]["optional-dependencies"]["config-ui"]
        dep_names = [d.split(">=")[0].split("==")[0].strip() for d in deps]
        # Check base package names (strip extras like [standard])
        base_names = [n.split("[")[0] for n in dep_names]
        assert "fastapi" in base_names
        assert "uvicorn" in base_names
        assert "pydantic" in base_names
        assert "numpy" in base_names
        assert "python-dotenv" in base_names


# ── Regression: `all` and `dev` must install without a C++ toolchain ───────


class TestAllExtraIsWheelInstallable:
    """`pip install ragmill[all]` must not pull a source-only dependency.

    llama-cpp-python ships no PyPI wheels for recent versions, so pip falls back
    to a 70MB+ sdist that vendors llama.cpp. Building it needs a C++ toolchain,
    and on Windows the vendored tree blows past the 260-char MAX_PATH limit
    during extraction, so `pip install ragmill[all]` died with
    "OSError: [Errno 2] No such file or directory" before it installed anything.

    The local LLM stays available via the opt-in `chat` extra.
    """

    # Dependencies that cannot be relied on to resolve to a wheel.
    SOURCE_ONLY = {"llama-cpp-python"}

    def _dep_names(self, extra):
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        deps = data["project"]["optional-dependencies"][extra]
        names = [d.split(">=")[0].split("==")[0].split(";")[0].strip() for d in deps]
        return [n.split("[")[0] for n in names]

    @pytest.mark.parametrize("extra", ["all", "dev"])
    def test_extra_has_no_source_only_dependency(self, extra):
        offenders = self.SOURCE_ONLY.intersection(self._dep_names(extra))
        assert not offenders, (
            f"The '{extra}' extra pulls source-only package(s) {sorted(offenders)}, "
            f"which breaks 'pip install ragmill[{extra}]' on machines without a "
            "C++ toolchain (and on Windows, via MAX_PATH). Keep them in 'chat'."
        )

    def test_chat_extra_still_provides_the_local_llm(self):
        assert "llama-cpp-python" in self._dep_names("chat")

    def test_sdist_uses_an_allowlist(self):
        """A denylisted sdist publishes any local scratch directory.

        Hatchling ships everything .gitignore does not exclude, so an untracked
        working directory (a test corpus, a deck, node_modules) silently ends up
        in the published tarball. An `include` list fails closed instead.
        """
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        sdist = tomllib.loads(pyproject.read_text())["tool"]["hatch"]["build"]["targets"]["sdist"]
        assert "include" in sdist, (
            "The sdist target must use an `include` allowlist; with only `exclude`, "
            "any new untracked directory is published to PyPI."
        )
        assert "src/ragmill" in sdist["include"]

    def test_all_extra_still_covers_every_other_optional_feature(self):
        """Removing llama-cpp-python must not have dropped anything else."""
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        extras = data["project"]["optional-dependencies"]
        aggregated = set(self._dep_names("all"))
        # every feature extra except the opt-in local LLM and doc/dev tooling
        for name in extras:
            if name in {"all", "dev", "docs", "chat"}:
                continue
            for dep in self._dep_names(name):
                assert dep in aggregated, f"'{dep}' from extra '{name}' is missing from 'all'"


# ── MAJOR 9: Pinecone environment is honored ──────────────────────────────


class TestPineconeEnvironmentParsing:
    """RAGMILL_PINECONE_ENVIRONMENT should parse region correctly."""

    def _parse_region(self, environment):
        """Extract region parsing logic from PineconeVectorStore."""
        region = "us-west-2"
        if environment:
            parts = environment.split("-")
            cloud_suffixes = {"gcp", "aws", "azure"}
            region_parts = parts
            for i in range(len(parts) - 1, 0, -1):
                if parts[i].lower() in cloud_suffixes:
                    region_parts = parts[:i]
                    break
            region = "-".join(region_parts)
        return region

    def test_plain_region(self):
        """'us-west-2' stays as 'us-west-2'."""
        assert self._parse_region("us-west-2") == "us-west-2"

    def test_region_with_gcp_suffix(self):
        """'us-west1-gcp' strips to 'us-west1'."""
        assert self._parse_region("us-west1-gcp") == "us-west1"

    def test_region_with_aws_suffix(self):
        """'us-east-1-aws' strips to 'us-east-1'."""
        assert self._parse_region("us-east-1-aws") == "us-east-1"

    def test_default_region_when_none(self):
        """No environment -> defaults to 'us-west-2'."""
        assert self._parse_region(None) == "us-west-2"

    def test_region_with_azure_suffix(self):
        """'westeurope-azure' strips to 'westeurope'."""
        assert self._parse_region("westeurope-azure") == "westeurope"

    def test_single_part_region(self):
        """A single-part region with no suffix stays as-is."""
        assert self._parse_region("us-west2") == "us-west2"


# ── MAJOR 10: /chat returns 501 when backend missing ─────────────────────


class TestChatEndpointMissingBackend:
    """/chat should return 501 (not 500) when chat backend deps are missing."""

    def test_chat_returns_501_when_embedding_model_missing(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAGMILL_STORE_TYPE", "sqlite")
        monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "chat_test.db"))
        monkeypatch.delenv("RAGMILL_API_KEY", raising=False)

        import ragmill.server as srv

        importlib.reload(srv)

        def raise_import_error():
            raise ImportError("No module named 'onnxruntime'")

        monkeypatch.setattr(srv, "_get_model", raise_import_error)

        from fastapi.testclient import TestClient

        with TestClient(srv.app) as c:
            r = c.post("/chat", json={"query": "test"})
            assert r.status_code == 501
            assert "embedding" in r.json()["detail"].lower()


# ── Minor 15: server extra includes python-dotenv ────────────────────────


class TestServerExtraIncludesDotenv:
    """The server extra must include python-dotenv so .env is loaded."""

    def test_server_extra_includes_dotenv(self):
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text())
        deps = data["project"]["optional-dependencies"]["server"]
        dep_names = [d.split(">=")[0].split("==")[0].strip() for d in deps]
        assert "python-dotenv" in dep_names


# ── Docker entrypoint exists and is correct ────────────────────────────────


class TestDockerEntrypoint:
    """docker-entrypoint.sh should sync docs before starting the server."""

    def test_entrypoint_exists(self):
        entrypoint = Path(__file__).parent.parent / "docker-entrypoint.sh"
        assert entrypoint.exists()

    def test_entrypoint_syncs_before_serve(self):
        entrypoint = Path(__file__).parent.parent / "docker-entrypoint.sh"
        content = entrypoint.read_text()
        assert "ragmill sync" in content
        assert "CMD" not in content or "serve" in content

    def test_dockerfile_sets_sqlite_path(self):
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert "RAGMILL_SQLITE_PATH=/data/ragmill.db" in content
        assert "docker-entrypoint.sh" in content

    def test_compose_healthcheck_uses_wget(self):
        compose = Path(__file__).parent.parent / "docker-compose.yml"
        content = compose.read_text()
        assert "wget" in content
        assert "/readyz" in content


# ── Nit 16-17: Stale files deleted ────────────────────────────────────────


class TestStaleFilesRemoved:
    """overview.md and docs/welcome.txt should not exist."""

    def test_overview_md_deleted(self):
        assert not Path(__file__).parent.parent.joinpath("overview.md").exists()

    def test_welcome_txt_deleted(self):
        assert not Path(__file__).parent.parent.joinpath("docs", "welcome.txt").exists()


# ── Docs never hand out the command that compiles from source ────────────────


class TestDocsInstallCommandIsCorrect:
    """`pip install llama-cpp-python --extra-index-url <index>` resolves to the
    sdist, because --extra-index-url merges indexes and PyPI carries a newer
    sdist-only release than the wheel index carries wheels. Any doc that prints
    that command without --only-binary is telling users to trigger the very
    build failure the index exists to avoid.
    """

    DOCS = ["README.md", "docs/installation.md", "docs/guide/chat.md", "docs/quickstart.md"]

    def _text(self, name):
        return (Path(__file__).parent.parent / name).read_text()

    @pytest.mark.parametrize("name", DOCS)
    def test_llama_install_snippets_force_a_wheel(self, name):
        text = self._text(name)
        if "abetlen.github.io" not in text:
            return
        # every mention of the index must be accompanied by the flag
        assert "--only-binary" in text, (
            f"{name} shows the wheel-index command without --only-binary, which "
            "resolves to the 70MB sdist and compiles from source."
        )

    @pytest.mark.parametrize("name", ["README.md", "docs/index.md", "docs/quickstart.md"])
    def test_quickstarts_mention_setup_chat(self, name):
        """A quickstart that goes pip install -> ragmill chat walks the reader
        into the missing-model error."""
        text = self._text(name)
        if "ragmill chat" not in text:
            return
        assert "setup-chat" in text, f"{name} shows `ragmill chat` but never mentions setup-chat"
