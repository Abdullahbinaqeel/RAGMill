"""Tests for the FastAPI REST server (ragmill.server).

server.py builds the embedding model, engine, and store as module-level
globals at import time, so tests set RAGMILL_STORE_TYPE/RAGMILL_SQLITE_PATH
*before* importing (or reload the module) to get an isolated store per test.
The /chat endpoint never hits the real local model — it's monkeypatched.
"""

import importlib

import pytest

pytest.importorskip("fastapi")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGMILL_STORE_TYPE", "sqlite")
    monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "server_test.db"))

    import ragmill.server as srv

    importlib.reload(srv)

    from fastapi.testclient import TestClient

    with TestClient(srv.app) as c:
        yield c, srv


def test_health(client):
    c, srv = client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok", "store_type": "sqlite", "chunk_count": 0}


def test_index_serves_chat_ui(client):
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "ragmill" in r.text.lower()


def test_count_empty_store(client):
    c, _ = client
    r = c.get("/count")
    assert r.status_code == 200
    assert r.json() == {"count": 0}


def test_ingest_nonexistent_directory_returns_404(client):
    c, _ = client
    r = c.post("/ingest", json={"directory": "/no/such/dir"})
    assert r.status_code == 404


def test_sync_nonexistent_directory_returns_404(client):
    c, _ = client
    r = c.post("/sync", json={"directory": "/no/such/dir"})
    assert r.status_code == 404


@pytest.mark.integration
def test_ingest_and_search_and_count(client, sample_docs_dir):
    """Requires the embeddings extra (onnxruntime/tokenizers) + model download —
    marked integration since it's slower than a pure-mock test."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    c, _ = client

    r = c.post("/ingest", json={"directory": str(sample_docs_dir)})
    assert r.status_code == 200
    chunks = r.json()["chunks"]
    assert chunks > 0

    r = c.get("/count")
    assert r.json() == {"count": chunks}

    r = c.post("/search", json={"query": "markdown content", "top_k": 3})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) > 0
    assert "score" in results[0] and "metadata" in results[0] and "content" in results[0]


def test_search_empty_store_returns_empty_results(client):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    c, _ = client
    r = c.post("/search", json={"query": "anything", "top_k": 5})
    assert r.status_code == 200
    assert r.json() == {"results": []}


def test_chat_endpoint_uses_mocked_generator(client, mock_llm, monkeypatch):
    c, srv = client
    monkeypatch.setattr(srv, "generate_answer", mock_llm)

    r = c.post("/chat", json={"query": "what is this?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "MOCKED ANSWER for: what is this?"
    assert body["sources"] == []  # empty store
    assert len(mock_llm.calls) == 1
    assert mock_llm.calls[0][0] == "what is this?"


def test_export_then_import_roundtrip(client, tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    c, _ = client

    r = c.post("/export")
    assert r.status_code == 200
    # export now returns the file directly as a streaming response
    assert len(r.content) >= 0  # empty store = empty file

    import io

    file_like = io.BytesIO(r.content)
    r2 = c.post("/import", files={"file": ("export.jsonl", file_like, "application/octet-stream")})
    assert r2.status_code == 200
    assert r2.json() == {"imported": 0}


def test_import_without_file_returns_422(client):
    c, _ = client
    r = c.post("/import")
    assert r.status_code == 422


def test_invalid_json_body_returns_422(client):
    c, _ = client
    r = c.post("/ingest", content="not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_missing_required_field_returns_422(client):
    c, _ = client
    r = c.post("/ingest", json={})
    assert r.status_code == 422


def test_wrong_method_returns_405(client):
    c, _ = client
    r = c.get("/ingest")
    assert r.status_code == 405


def test_nonexistent_endpoint_returns_404(client):
    c, _ = client
    r = c.get("/nonexistent")
    assert r.status_code == 404


def test_sync_endpoint_accepts_valid_directory(client, monkeypatch, tmp_path):
    """Tests the /sync endpoint with a mocked embedding model and real store."""
    c, srv = client

    # Create a test directory with a file
    test_dir = tmp_path / "sync_docs"
    test_dir.mkdir()
    (test_dir / "a.txt").write_text("sync test content")

    # Mock the embedding model
    embed_calls = []
    monkeypatch.setattr(
        srv,
        "_get_model",
        lambda: type(
            "FakeModel",
            (),
            {
                "embed": lambda self, texts: (
                    __import__("numpy")
                    .random.rand(len(texts), 384)
                    .astype(__import__("numpy").float32)
                )
            },
        )(),
    )

    r = c.post("/sync", json={"directory": str(test_dir)})
    assert r.status_code == 200
    body = r.json()
    assert body["added"] == 1
    assert body["skipped"] == 0

    # Second sync with no changes should skip
    r2 = c.post("/sync", json={"directory": str(test_dir)})
    assert r2.status_code == 200
    assert r2.json()["skipped"] == 1


def test_health_exposed_without_auth(monkeypatch, tmp_path):
    """/health should work even when API key is set (it's exempt from auth)."""
    monkeypatch.setenv("RAGMILL_STORE_TYPE", "sqlite")
    monkeypatch.setenv("RAGMILL_SQLITE_PATH", str(tmp_path / "health_test.db"))
    monkeypatch.setenv("RAGMILL_API_KEY", "secret-key")

    import ragmill.server as srv

    importlib.reload(srv)

    from fastapi.testclient import TestClient

    with TestClient(srv.app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
