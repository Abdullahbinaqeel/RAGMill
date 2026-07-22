"""Expanded RAGMillConfig tests: new chat-model fields + .env loading.

Baseline config tests (defaults, basic env-var overrides) already live in
test_abstractions.py — this file adds the new chat_model_* fields and the
config-UI -> .env -> RAGMillConfig.from_env() loop.
"""

import os

import pytest

from ragmill.config import RAGMillConfig


def test_chat_model_defaults():
    for k in list(os.environ):
        if k.startswith("RAGMILL_CHAT_"):
            del os.environ[k]
    cfg = RAGMillConfig.from_env()
    assert cfg.chat_model_repo == "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    assert cfg.chat_model_file == "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    assert cfg.chat_n_ctx == 4096


def test_chat_model_env_overrides(monkeypatch):
    monkeypatch.setenv("RAGMILL_CHAT_MODEL_REPO", "some/other-repo")
    monkeypatch.setenv("RAGMILL_CHAT_MODEL_FILE", "other-model.gguf")
    monkeypatch.setenv("RAGMILL_CHAT_N_CTX", "8192")

    cfg = RAGMillConfig.from_env()
    assert cfg.chat_model_repo == "some/other-repo"
    assert cfg.chat_model_file == "other-model.gguf"
    assert cfg.chat_n_ctx == 8192


def test_dataclass_defaults_match_from_env_defaults():
    plain = RAGMillConfig()
    assert plain.chat_model_repo == "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    assert plain.chat_model_file == "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    assert plain.chat_n_ctx == 4096


def test_chat_backend_defaults_to_local():
    for k in ("RAGMILL_CHAT_BACKEND", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(k, None)
    cfg = RAGMillConfig.from_env()
    assert cfg.chat_backend == "local"
    assert cfg.gemini_model == "gemini-flash-latest"
    assert cfg.openai_model == "gpt-4o-mini"
    assert cfg.gemini_api_key is None
    assert cfg.openai_api_key is None


def test_chat_backend_env_overrides(monkeypatch):
    monkeypatch.setenv("RAGMILL_CHAT_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("RAGMILL_GEMINI_MODEL", "gemini-custom")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setenv("RAGMILL_OPENAI_MODEL", "gpt-custom")

    cfg = RAGMillConfig.from_env()
    assert cfg.chat_backend == "gemini"
    assert cfg.gemini_api_key == "fake-gemini-key"
    assert cfg.gemini_model == "gemini-custom"
    assert cfg.openai_api_key == "fake-openai-key"
    assert cfg.openai_model == "gpt-custom"


def test_gemini_api_key_falls_back_to_google_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "fallback-key")
    cfg = RAGMillConfig.from_env()
    assert cfg.gemini_api_key == "fallback-key"


def test_from_env_loads_dotenv_file(monkeypatch, tmp_path):
    pytest.importorskip("dotenv")

    env_file = tmp_path / ".env"
    env_file.write_text("RAGMILL_STORE_TYPE=qdrant\nRAGMILL_QDRANT_URL=https://example.qdrant.io\n")

    monkeypatch.chdir(tmp_path)
    for k in ("RAGMILL_STORE_TYPE", "RAGMILL_QDRANT_URL"):
        monkeypatch.delenv(k, raising=False)

    cfg = RAGMillConfig.from_env()
    assert cfg.store_type == "qdrant"
    assert cfg.qdrant_url == "https://example.qdrant.io"


def test_from_env_real_env_var_wins_over_dotenv(monkeypatch, tmp_path):
    pytest.importorskip("dotenv")

    env_file = tmp_path / ".env"
    env_file.write_text("RAGMILL_STORE_TYPE=qdrant\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAGMILL_STORE_TYPE", "pinecone")

    cfg = RAGMillConfig.from_env()
    assert cfg.store_type == "pinecone"


def test_from_env_works_without_dotenv_installed(monkeypatch):
    """load_dotenv() is best-effort — a missing 'dotenv' package must not break from_env()."""
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("no dotenv installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    cfg = RAGMillConfig.from_env()
    assert cfg.store_type in ("sqlite", "qdrant", "pinecone")  # doesn't raise
