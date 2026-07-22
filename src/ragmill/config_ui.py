"""
Standalone setup UI for RAGMill configuration.

A small FastAPI app, entirely separate from server.py's REST API, meant
to be run once during setup — e.g. via `ragmill configure` — to fill in
optional cloud-backend credentials and other settings. Saves them into
a local .env file via python-dotenv so they're picked up automatically
by RAGMillConfig.from_env() on the next run. Nothing is ever written
into source code.

Binds to 127.0.0.1 by default (see __main__.py's cmd_configure) since
this page renders a credential-entry form.

Start with:
    ragmill configure
"""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

config_app = FastAPI(title="RAGMill Setup", version="0.3.1")

STATIC_DIR = Path(__file__).parent / "config_ui_static"


class SaveRequest(BaseModel):
    store_type: Optional[str] = None
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: Optional[str] = None
    chat_backend: Optional[str] = None
    chat_model_repo: Optional[str] = None
    chat_model_file: Optional[str] = None
    chat_n_ctx: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    server_host: Optional[str] = None
    server_port: Optional[str] = None


class SaveResponse(BaseModel):
    saved: List[str]
    env_path: str
    message: str


_FIELD_TO_ENV_VAR = {
    "store_type": "RAGMILL_STORE_TYPE",
    "qdrant_url": "RAGMILL_QDRANT_URL",
    "qdrant_api_key": "RAGMILL_QDRANT_API_KEY",
    "qdrant_collection_name": "RAGMILL_QDRANT_COLLECTION_NAME",
    "pinecone_api_key": "RAGMILL_PINECONE_API_KEY",
    "pinecone_environment": "RAGMILL_PINECONE_ENVIRONMENT",
    "pinecone_index_name": "RAGMILL_PINECONE_INDEX_NAME",
    "chat_backend": "RAGMILL_CHAT_BACKEND",
    "chat_model_repo": "RAGMILL_CHAT_MODEL_REPO",
    "chat_model_file": "RAGMILL_CHAT_MODEL_FILE",
    "chat_n_ctx": "RAGMILL_CHAT_N_CTX",
    "gemini_api_key": "GEMINI_API_KEY",
    "gemini_model": "RAGMILL_GEMINI_MODEL",
    "openai_api_key": "OPENAI_API_KEY",
    "openai_model": "RAGMILL_OPENAI_MODEL",
    "server_host": "RAGMILL_HOST",
    "server_port": "RAGMILL_PORT",
}


def _resolve_env_path() -> Path:
    """Re-reads RAGMILL_ENV_PATH on every call (not cached at import time)
    so tests and --env-path overrides don't require a module reload."""
    return Path(os.getenv("RAGMILL_ENV_PATH", "./.env")).resolve()


@config_app.get("/", response_class=HTMLResponse)
def form():
    return (STATIC_DIR / "index.html").read_text()


@config_app.post("/save", response_model=SaveResponse)
def save(request: SaveRequest):
    try:
        from dotenv import set_key
    except ImportError as exc:
        raise HTTPException(
            500,
            "Saving requires the 'config-ui' extra. Install it with: pip install ragmill[config-ui]",
        ) from exc

    env_path = _resolve_env_path()
    env_path.touch(exist_ok=True)

    saved = []
    for field, env_var in _FIELD_TO_ENV_VAR.items():
        value = getattr(request, field)
        if value is not None and str(value).strip() != "":
            set_key(str(env_path), env_var, str(value))
            saved.append(env_var)

    return SaveResponse(
        saved=saved,
        env_path=str(env_path),
        message=(
            f"Saved {len(saved)} value(s) to {env_path}. "
            "Make sure this file is listed in .gitignore. Restart `ragmill serve` "
            "(or any process reading RAGMillConfig) for the changes to take effect."
        ),
    )
