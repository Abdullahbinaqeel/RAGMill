"""Tests for the standalone config/setup UI (ragmill.config_ui)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("dotenv")


@pytest.fixture
def ui_client(monkeypatch, temp_env_file):
    monkeypatch.setenv("RAGMILL_ENV_PATH", str(temp_env_file))
    from fastapi.testclient import TestClient
    from ragmill.config_ui import config_app

    with TestClient(config_app) as c:
        yield c, temp_env_file


def test_form_renders_expected_fields(ui_client):
    c, _ = ui_client
    r = c.get("/")
    assert r.status_code == 200
    for field in (
        "store_type",
        "qdrant_url",
        "qdrant_api_key",
        "qdrant_collection_name",
        "pinecone_api_key",
        "pinecone_environment",
        "pinecone_index_name",
        "chat_backend",
        "chat_model_repo",
        "chat_model_file",
        "chat_n_ctx",
        "gemini_api_key",
        "gemini_model",
        "openai_api_key",
        "openai_model",
        "server_host",
        "server_port",
    ):
        assert f'name="{field}"' in r.text


def test_save_writes_only_nonempty_fields(ui_client):
    c, env_path = ui_client
    r = c.post(
        "/save",
        json={
            "store_type": "qdrant",
            "qdrant_url": "https://example.qdrant.io",
            "qdrant_api_key": "",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "RAGMILL_STORE_TYPE" in body["saved"]
    assert "RAGMILL_QDRANT_URL" in body["saved"]
    assert "RAGMILL_QDRANT_API_KEY" not in body["saved"]  # empty string, not saved

    content = env_path.read_text()
    assert "RAGMILL_STORE_TYPE" in content
    assert "RAGMILL_QDRANT_URL" in content
    assert "RAGMILL_QDRANT_API_KEY" not in content


def test_save_preserves_unrelated_existing_lines(ui_client):
    c, env_path = ui_client
    env_path.write_text("SOME_OTHER_VAR=keepme\n# a comment\n")

    r = c.post("/save", json={"pinecone_api_key": "secret-key-123"})
    assert r.status_code == 200

    content = env_path.read_text()
    assert "SOME_OTHER_VAR=keepme" in content
    assert "# a comment" in content
    assert "RAGMILL_PINECONE_API_KEY" in content


def test_save_second_time_updates_in_place_not_duplicated(ui_client):
    c, env_path = ui_client
    c.post("/save", json={"server_port": "8000"})
    c.post("/save", json={"server_port": "9000"})

    content = env_path.read_text()
    assert content.count("RAGMILL_PORT") == 1
    assert "9000" in content
    assert "8000" not in content


def test_save_with_nothing_filled_in_saves_nothing(ui_client):
    c, env_path = ui_client
    r = c.post("/save", json={})
    assert r.status_code == 200
    assert r.json()["saved"] == []


def test_save_response_mentions_gitignore_and_restart(ui_client):
    c, _ = ui_client
    r = c.post("/save", json={"server_host": "127.0.0.1"})
    message = r.json()["message"].lower()
    assert "gitignore" in message
    assert "restart" in message


def test_save_gemini_backend_fields(ui_client):
    c, env_path = ui_client
    r = c.post(
        "/save",
        json={
            "chat_backend": "gemini",
            "gemini_api_key": "fake-gemini-key",
            "gemini_model": "gemini-custom",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "RAGMILL_CHAT_BACKEND" in body["saved"]
    assert "GEMINI_API_KEY" in body["saved"]
    assert "RAGMILL_GEMINI_MODEL" in body["saved"]

    content = env_path.read_text()
    assert "GEMINI_API_KEY" in content
    assert "fake-gemini-key" in content


def test_save_openai_backend_fields(ui_client):
    c, env_path = ui_client
    r = c.post(
        "/save",
        json={
            "chat_backend": "openai",
            "openai_api_key": "fake-openai-key",
            "openai_model": "gpt-custom",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "RAGMILL_CHAT_BACKEND" in body["saved"]
    assert "OPENAI_API_KEY" in body["saved"]
    assert "RAGMILL_OPENAI_MODEL" in body["saved"]

    content = env_path.read_text()
    assert "OPENAI_API_KEY" in content
    assert "fake-openai-key" in content
